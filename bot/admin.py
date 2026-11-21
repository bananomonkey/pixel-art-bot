"""Админ-панель и рассылки. Доступ — по списку Telegram ID в ADMIN_IDS.

Хэндлеры регистрируются в начале цепочки главного роутера (см. register_admin_handlers),
чтобы фото админа во время создания поста попадало в рассылку, а не в пикселизацию.

Всё в памяти процесса: рассылки, база чатов и флоу создания поста
сбрасываются при перезапуске. Содержимое переписки не хранится —
для рассылки бот копирует сообщение админа (copy_message).
"""

import asyncio
import logging
import time

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from .broadcast import BroadcastStore, format_interval, parse_interval
from .config import (
    ADMIN_IDS,
    BROADCAST_MAX_ACTIVE,
    BROADCAST_MAX_INTERVAL,
    BROADCAST_MIN_INTERVAL,
    BROADCAST_SEND_RATE,
    BROADCAST_TICK,
    CHATS_MAX_ENTRIES,
)
from .state import ChatsRegistry

logger = logging.getLogger(__name__)

ADMIN_SET = frozenset(ADMIN_IDS)
chats = ChatsRegistry(max_entries=CHATS_MAX_ENTRIES)  # база чатов для рассылок
broadcasts = BroadcastStore(max_active=BROADCAST_MAX_ACTIVE)
stats = {"photos": 0, "sent": 0}  # технические счётчики, без user_id

# флоу создания рассылки: admin_id -> {"stage": post|mode|interval, "mode", "chat_id", "message_id"}
_flow: dict[int, dict] = {}

# фильтр-перехватчик сообщений админа во время флоу (фото админа без флоу — обычный пиксель-арт)
_ADMIN_CAPTURE = (
    (F.photo | F.document | F.animation | F.video | (F.text & ~F.text.startswith("/")))
    & F.from_user.id.in_(ADMIN_SET)
)


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in ADMIN_SET


def register_admin_handlers(router: Router) -> None:
    """Зарегистрировать админ-хэндлеры ПЕРВЫМИ в цепочке основного роутера."""
    router.message.register(cmd_admin, Command("admin"))
    router.message.register(cmd_cancel, Command("cancel"))
    router.message.register(admin_capture, _ADMIN_CAPTURE)
    router.callback_query.register(cb_ad_new, F.data == "ad_new")
    router.callback_query.register(cb_ad_once, F.data == "ad_once")
    router.callback_query.register(cb_ad_cycle, F.data == "ad_cycle")
    router.callback_query.register(cb_ad_abort, F.data == "ad_abort")
    router.callback_query.register(cb_ad_panel, F.data == "ad_panel")
    router.callback_query.register(cb_ad_list, F.data == "ad_list")
    router.callback_query.register(cb_ad_cancel, F.data.startswith("ad_cancel:"))


# ---------------------------------------------------------------- панель


def _panel_text() -> str:
    return (
        "🛠 Панель администратора\n\n"
        f"👥 Чатов в базе: {len(chats)}\n"
        f"🎨 Обработано фото: {stats['photos']}\n"
        f"📤 Отправлено рассылками: {stats['sent']}\n"
        f"📢 Активных рассылок: {len(broadcasts)}\n\n"
        "⚠️ База чатов и рассылки — в памяти процесса и сбрасываются при перезапуске."
    )


def _panel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📢 Новая рассылка", callback_data="ad_new")],
            [InlineKeyboardButton(text=f"📋 Рассылки ({len(broadcasts)})", callback_data="ad_list")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="ad_panel")],
        ]
    )


def _mode_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⏱ Разовая (через …)", callback_data="ad_once")],
            [InlineKeyboardButton(text="🔁 Циклическая (каждые …)", callback_data="ad_cycle")],
            [InlineKeyboardButton(text="❌ Отменить пост", callback_data="ad_abort")],
        ]
    )


# ---------------------------------------------------------------- команды


async def cmd_admin(message: Message) -> None:
    if not is_admin(message.from_user.id if message.from_user else None):
        await message.answer("⛔ Нет доступа")
        return
    chats.touch(message.chat.id)
    await message.answer(_panel_text(), reply_markup=_panel_kb())


async def cmd_cancel(message: Message) -> None:
    if not is_admin(message.from_user.id if message.from_user else None):
        await message.answer("⛔ Нет доступа")
        return
    flow = _flow.pop(message.from_user.id, None)  # type: ignore[union-attr]
    await message.answer("Действие отменено." if flow else "Нет активного действия.")


# ---------------------------------------------------------------- создание рассылки (флоу)


async def admin_capture(message: Message) -> None:
    """Перехватывает сообщения админа во время флоу; вне флоу пропускает дальше."""
    flow = _flow.get(message.from_user.id)  # type: ignore[union-attr]
    if flow is None:
        if message.text:
            await message.answer("Нет активного действия. Отправь фото или открой /admin")
            return
        raise SkipHandler  # фото/файл без флоу — обычная обработка пиксель-арта

    if flow["stage"] == "post":
        if message.media_group_id is not None:
            await message.answer("Пришли пост одним сообщением (без альбома).")
            return
        flow.update(stage="mode", chat_id=message.chat.id, message_id=message.message_id)
        await message.answer("✅ Пост сохранён. Когда отправлять?", reply_markup=_mode_kb())
        return

    if flow["stage"] == "mode":
        await message.answer("Выбери вариант кнопками выше или /cancel.")
        return

    # stage == "interval"
    seconds = parse_interval(message.text or "")
    if seconds is None:
        await message.answer("Не понял интервал. Пример: 30м, 1ч30м, 1д — или /cancel.")
        return
    if seconds < BROADCAST_MIN_INTERVAL:
        await message.answer(f"Минимум — {format_interval(BROADCAST_MIN_INTERVAL)}.")
        return
    if seconds > BROADCAST_MAX_INTERVAL:
        await message.answer(f"Максимум — {format_interval(BROADCAST_MAX_INTERVAL)}.")
        return
    mode = flow["mode"]
    created = broadcasts.add(flow["chat_id"], flow["message_id"], mode, seconds, time.monotonic())
    _flow.pop(message.from_user.id, None)  # type: ignore[union-attr]
    if created is None:
        await message.answer(f"Достигнут лимит активных рассылок ({BROADCAST_MAX_ACTIVE}).")
        return
    verb = "разово через" if mode == "once" else "циклично каждые"
    await message.answer(
        f"✅ Рассылка #{created.id}: {verb} {format_interval(seconds)}. "
        f"Чатов в базе: {len(chats)}. Список — /admin."
    )


# ---------------------------------------------------------------- кнопки панели


async def cb_ad_new(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    _flow[callback.from_user.id] = {"stage": "post"}
    await callback.message.answer(
        "📢 Пришли пост следующим сообщением — текст, фото с подписью, что угодно.\n"
        "Его получат все чаты в базе. /cancel — отмена."
    )
    await callback.answer()


async def cb_ad_once(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    flow = _flow.get(callback.from_user.id)
    if not flow or "chat_id" not in flow:
        await callback.answer("Сначала создай пост («📢 Новая рассылка»)", show_alert=True)
        return
    flow["stage"] = "interval"
    flow["mode"] = "once"
    await callback.message.answer("⏱ Через сколько отправить? Пример: 30м, 1ч30м, 1д. /cancel — отмена.")
    await callback.answer()


async def cb_ad_cycle(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    flow = _flow.get(callback.from_user.id)
    if not flow or "chat_id" not in flow:
        await callback.answer("Сначала создай пост («📢 Новая рассылка»)", show_alert=True)
        return
    flow["stage"] = "interval"
    flow["mode"] = "cycle"
    await callback.message.answer(
        "🔁 С каким интервалом отправлять? Пример: 30м, 2ч, 1д. /cancel — отмена."
    )
    await callback.answer()


async def cb_ad_abort(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    removed = _flow.pop(callback.from_user.id, None)
    await callback.answer("Пост отменён." if removed else "Нет активного действия.")


async def cb_ad_panel(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    try:
        await callback.message.edit_text(_panel_text(), reply_markup=_panel_kb())
    except TelegramBadRequest:
        pass  # «message is not modified» и подобное
    await callback.answer()


async def cb_ad_list(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    now = time.monotonic()
    rows = []
    for item in broadcasts.get_all():
        mark = "🔁 каждые" if item.mode == "cycle" else "⏱ разово через"
        wait = max(0, int(item.next_run - now))
        label = (
            f"❌ #{item.id} · {mark} {format_interval(item.interval)} · "
            f"осталось {format_interval(wait)} · отправлено {item.sent_total}"
        )
        rows.append([InlineKeyboardButton(text=label, callback_data=f"ad_cancel:{item.id}")])
    rows.append([InlineKeyboardButton(text="⬅️ В панель", callback_data="ad_panel")])
    text = "📢 Активные рассылки:" if broadcasts else "Активных рассылок нет."
    try:
        await callback.message.edit_text(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
    except TelegramBadRequest:
        pass
    await callback.answer()


async def cb_ad_cancel(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    try:
        broadcast_id = int((callback.data or "").split(":", 1)[1])
    except ValueError:
        await callback.answer()
        return
    removed = broadcasts.remove(broadcast_id)
    await callback.answer(f"Рассылка #{broadcast_id} отменена." if removed else "Не найдена.")


# ---------------------------------------------------------------- воркер рассылок


async def broadcast_worker(bot: Bot) -> None:
    """Фоновая задача: раз в BROADCAST_TICK секунд отправляет созревшие рассылки."""
    while True:
        await asyncio.sleep(BROADCAST_TICK)
        for item in broadcasts.due(time.monotonic()):
            try:
                sent, blocked = await _send_broadcast(bot, item)
            except Exception:  # noqa: BLE001 — воркер не должен умирать
                logger.exception("рассылка #%d завершилась ошибкой", item.id)
                sent, blocked = 0, 0
            stats["sent"] += sent
            if item.mode == "once":
                broadcasts.remove(item.id)
            else:
                broadcasts.reschedule(item, time.monotonic())
            if sent or blocked:
                logger.info(
                    "рассылка #%d: отправлено=%d, чатов исключено=%d, всего чатов=%d",
                    item.id, sent, blocked, len(chats),
                )


async def _send_broadcast(bot: Bot, item) -> tuple[int, int]:
    """Отправить пост во все чаты базы; вернуть (отправлено, исключено)."""
    sent = blocked = 0
    for chat_id in chats.chat_ids():
        try:
            await bot.copy_message(
                chat_id=chat_id, from_chat_id=item.chat_id, message_id=item.message_id
            )
            sent += 1
        except TelegramForbiddenError:
            chats.remove(chat_id)  # бот заблокирован — чат больше не актуален
            blocked += 1
        except TelegramRetryAfter as exc:  # flood wait от Telegram
            await asyncio.sleep(exc.retry_after + 1.0)
        except TelegramBadRequest as exc:
            # исходный пост недоступен (удалён админом) — прекращаем всю отправку
            logger.warning("рассылка #%d остановлена: %s", item.id, exc)
            break
        await asyncio.sleep(1.0 / BROADCAST_SEND_RATE)
    return sent, blocked
