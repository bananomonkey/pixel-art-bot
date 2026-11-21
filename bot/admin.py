"""Админ-панель и рассылки. Доступ — по списку Telegram ID в ADMIN_IDS.

Хэндлеры регистрируются в начале цепочки главного роутера (см. register_admin_handlers),
чтобы фото админа во время создания поста попадало в рассылку, а не в пикселизацию.

Всё в памяти процесса: рассылки, шаблоны, баны, база чатов, статистика и флоу
сбрасываются при перезапуске. Содержимое переписки не хранится —
для рассылки бот копирует сообщение админа (copy_message).
"""

import asyncio
import logging
import time
from datetime import date, timedelta

from aiogram import Bot, F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from .broadcast import (
    BroadcastStore,
    TemplateStore,
    format_interval,
    parse_interval,
    parse_target_days,
)
from .config import (
    ADMIN_IDS,
    BROADCAST_MAX_ACTIVE,
    BROADCAST_MAX_INTERVAL,
    BROADCAST_MIN_INTERVAL,
    BROADCAST_SEND_RATE,
    BROADCAST_TICK,
    CHATS_MAX_ENTRIES,
    DAILY_STATS_DAYS,
    TARGETED_MAX_DAYS,
    TEMPLATES_MAX,
)
from .state import ChatsRegistry

logger = logging.getLogger(__name__)

ADMIN_SET = frozenset(ADMIN_IDS)
chats = ChatsRegistry(max_entries=CHATS_MAX_ENTRIES)  # база чатов для рассылок
broadcasts = BroadcastStore(max_active=BROADCAST_MAX_ACTIVE)
templates = TemplateStore(max_items=TEMPLATES_MAX)
stats = {"photos": 0, "sent": 0}  # суммарные счётчики, без user_id

# дневная статистика: "ГГГГ-ММ-ДД" -> {"photos": int, "sent": int, "users": set}
_daily: dict[str, dict] = {}
_bans: set[int] = set()

# флоу создания рассылки/шаблона:
# admin_id -> {"stage": post|mode|days|interval, "purpose", "mode", "target_days", "chat_id", "message_id"}
_flow: dict[int, dict] = {}

# фильтр-перехватчик сообщений админа во время флоу (фото админа без флоу — обычный пиксель-арт)
_ADMIN_CAPTURE = (
    (F.photo | F.document | F.animation | F.video | (F.text & ~F.text.startswith("/")))
    & F.from_user.id.in_(ADMIN_SET)
)


def is_admin(user_id: int | None) -> bool:
    return user_id is not None and user_id in ADMIN_SET


def is_banned(user_id: int | None) -> bool:
    return user_id is not None and user_id in _bans


def register_admin_handlers(router: Router) -> None:
    """Зарегистрировать админ-хэндлеры ПЕРВЫМИ в цепочке основного роутера."""
    router.message.register(cmd_admin, Command("admin"))
    router.message.register(cmd_ban, Command("ban"))
    router.message.register(cmd_unban, Command("unban"))
    router.message.register(admin_capture, _ADMIN_CAPTURE)
    router.callback_query.register(cb_ad_new, F.data == "ad_new")
    router.callback_query.register(cb_ad_once, F.data == "ad_once")
    router.callback_query.register(cb_ad_cycle, F.data == "ad_cycle")
    router.callback_query.register(cb_ad_target, F.data == "ad_target")
    router.callback_query.register(cb_ad_abort, F.data == "ad_abort")
    router.callback_query.register(cb_ad_panel, F.data == "ad_panel")
    router.callback_query.register(cb_ad_list, F.data == "ad_list")
    router.callback_query.register(cb_ad_cancel, F.data.startswith("ad_cancel:"))
    router.callback_query.register(cb_ad_daily, F.data == "ad_daily")
    router.callback_query.register(cb_ad_tpl, F.data == "ad_tpl")
    router.callback_query.register(cb_ad_tpl_new, F.data == "ad_tpl_new")
    router.callback_query.register(cb_ad_tpl_send, F.data.startswith("ad_tpl_send:"))
    router.callback_query.register(cb_ad_tpl_del, F.data.startswith("ad_tpl_del:"))


# ---------------------------------------------------------------- дневная статистика


def _today_key() -> str:
    return date.today().isoformat()


def record_photo(user_id: int | None) -> None:
    stats["photos"] += 1
    day = _daily.setdefault(_today_key(), {"photos": 0, "sent": 0, "users": set()})
    day["photos"] += 1
    day["users"].add(user_id)


def record_sent(count: int) -> None:
    stats["sent"] += count
    day = _daily.setdefault(_today_key(), {"photos": 0, "sent": 0, "users": set()})
    day["sent"] += count


def prune_daily() -> None:
    """Оставить только последние DAILY_STATS_DAYS дней."""
    keep = sorted(_daily)[-DAILY_STATS_DAYS:]
    for key in list(_daily.keys()):
        if key not in keep:
            del _daily[key]


def daily_text() -> str:
    """Текстовая «графическая» статистика за последние 14 дней."""
    if not _daily:
        return "📈 Статистики пока нет."
    lines = ["📈 Статистика по дням (фото ▏пользователи ▏отправлено):"]
    days = sorted(_daily)[-14:]
    max_photos = max(_daily[key]["photos"] for key in days) or 1
    for key in days:
        day = _daily[key]
        label = date.fromisoformat(key).strftime("%d.%m")
        bar = "▇" * max(0, round(day["photos"] / max_photos * 10))
        lines.append(
            f"{label}: {day['photos']:>4} {bar:<10} ▏{len(day['users']):>3} ▏{day['sent']:>4}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- панель


def _panel_text() -> str:
    today = _daily.get(_today_key(), {"photos": 0})
    return (
        "🛠 Панель администратора\n\n"
        f"👥 Чатов в базе: {len(chats)}\n"
        f"🚫 Забанено: {len(_bans)}\n"
        f"🎨 Обработано фото: {stats['photos']} (сегодня: {today['photos']})\n"
        f"📤 Отправлено рассылками: {stats['sent']}\n"
        f"📢 Активных рассылок: {len(broadcasts)}\n"
        f"💾 Шаблонов: {len(templates)}\n\n"
        "Баны: /ban (реплаем или ID), /unban ID.\n"
        "⚠️ База чатов, рассылки, шаблоны и баны — в памяти процесса, "
        "сбрасываются при перезапуске."
    )


def _panel_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📢 Новая рассылка", callback_data="ad_new")],
            [
                InlineKeyboardButton(
                    text=f"📋 Рассылки ({len(broadcasts)})", callback_data="ad_list"
                ),
                InlineKeyboardButton(
                    text=f"💾 Шаблоны ({len(templates)})", callback_data="ad_tpl"
                ),
            ],
            [
                InlineKeyboardButton(text="📈 По дням", callback_data="ad_daily"),
                InlineKeyboardButton(text="🔄 Обновить", callback_data="ad_panel"),
            ],
        ]
    )


def _mode_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⏱ Разовая (через …)", callback_data="ad_once")],
            [InlineKeyboardButton(text="🔁 Циклическая (каждые …)", callback_data="ad_cycle")],
            [InlineKeyboardButton(text="🎯 Разово активным за N дней", callback_data="ad_target")],
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


async def cmd_ban(message: Message, command: CommandObject | None = None) -> None:
    if not is_admin(message.from_user.id if message.from_user else None):
        await message.answer("⛔ Нет доступа")
        return
    target = None
    if message.reply_to_message is not None and message.reply_to_message.from_user is not None:
        target = message.reply_to_message.from_user.id
    elif command is not None and command.args:
        arg = command.args.strip()
        if arg.isdigit():
            target = int(arg)
    if target is None:
        await message.answer("Использование: /ban реплаем на сообщение, или /ban <ID>")
        return
    _bans.add(target)
    chats.remove(target)
    await message.answer(f"🚫 Пользователь {target} забанен (всего: {len(_bans)}).")


async def cmd_unban(message: Message, command: CommandObject | None = None) -> None:
    if not is_admin(message.from_user.id if message.from_user else None):
        await message.answer("⛔ Нет доступа")
        return
    target = None
    if message.reply_to_message is not None and message.reply_to_message.from_user is not None:
        target = message.reply_to_message.from_user.id
    elif command is not None and command.args:
        arg = command.args.strip()
        if arg.isdigit():
            target = int(arg)
    if target is None:
        await message.answer("Использование: /unban реплаем на сообщение, или /unban <ID>")
        return
    if target in _bans:
        _bans.discard(target)
        await message.answer(f"✅ Пользователь {target} разбанен.")
    else:
        await message.answer(f"Пользователь {target} не был забанен.")


# ---------------------------------------------------------------- создание рассылки (флоу)


async def admin_capture(message: Message) -> None:
    """Перехватывает сообщения админа во время флоу; вне флоу пропускает."""
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
        if flow.get("purpose") == "template":
            created = templates.add(message.chat.id, message.message_id, time.time())
            _flow.pop(message.from_user.id, None)  # type: ignore[union-attr]
            if created is None:
                await message.answer(f"Лимит шаблонов ({TEMPLATES_MAX}) достигнут.")
            else:
                await message.answer(
                    f"💾 Шаблон #{created.id} сохранён. Отправить: /admin → Шаблоны."
                )
        else:
            await message.answer("✅ Пост сохранён. Когда отправлять?", reply_markup=_mode_kb())
        return

    if flow["stage"] == "mode":
        await message.answer("Выбери вариант кнопками выше или /cancel.")
        return

    if flow["stage"] == "days":
        days = parse_target_days(message.text or "")
        if days is None:
            await message.answer(f"Нужно число 1–{TARGETED_MAX_DAYS}. Пример: 7 — или /cancel.")
            return
        flow.update(stage="interval", mode="once", target_days=days)
        await message.answer(
            f"🎯 Разово активным за {int(days)} дн. Через сколько отправить? Пример: 30м — или /cancel."
        )
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
    created = broadcasts.add(
        flow["chat_id"],
        flow["message_id"],
        mode,
        seconds,
        time.monotonic(),
        target_days=flow.get("target_days"),
    )
    _flow.pop(message.from_user.id, None)  # type: ignore[union-attr]
    if created is None:
        await message.answer(f"Достигнут лимит активных рассылок ({BROADCAST_MAX_ACTIVE}).")
        return
    verb = "разово через" if mode == "once" else "циклично каждые"
    extra = f", активным за {int(created.target_days)} дн" if created.target_days else ""
    await message.answer(
        f"✅ Рассылка #{created.id}: {verb} {format_interval(seconds)}{extra}. "
        f"Чатов в базе: {len(chats)}. Список — /admin."
    )


# ---------------------------------------------------------------- кнопки панели


async def cb_ad_new(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    _flow[callback.from_user.id] = {"stage": "post", "purpose": "broadcast"}
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


async def cb_ad_target(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    flow = _flow.get(callback.from_user.id)
    if not flow or "chat_id" not in flow:
        await callback.answer("Сначала создай пост («📢 Новая рассылка»)", show_alert=True)
        return
    flow["stage"] = "days"
    await callback.message.answer(
        f"🎯 За сколько дней считать пользователя активным? Число 1–{TARGETED_MAX_DAYS}. /cancel — отмена."
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
        extra = f" · за {int(item.target_days)} дн" if item.target_days else ""
        label = (
            f"❌ #{item.id} · {mark} {format_interval(item.interval)}{extra} · "
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


async def cb_ad_daily(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    prune_daily()
    rows = [[InlineKeyboardButton(text="⬅️ В панель", callback_data="ad_panel")]]
    try:
        await callback.message.edit_text(
            daily_text(), reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
        )
    except TelegramBadRequest:
        pass
    await callback.answer()


# ---------------------------------------------------------------- шаблоны


async def cb_ad_tpl(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    rows = []
    for item in templates.get_all():
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"📤 #{item.id} · отправить", callback_data=f"ad_tpl_send:{item.id}"
                ),
                InlineKeyboardButton(
                    text="🗑", callback_data=f"ad_tpl_del:{item.id}"
                ),
            ]
        )
    rows.append([InlineKeyboardButton(text="➕ Новый шаблон", callback_data="ad_tpl_new")])
    rows.append([InlineKeyboardButton(text="⬅️ В панель", callback_data="ad_panel")])
    text = "💾 Шаблоны (отправка сразу всем чатам базы):" if templates else "Шаблонов нет."
    try:
        await callback.message.edit_text(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
    except TelegramBadRequest:
        pass
    await callback.answer()


async def cb_ad_tpl_new(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    _flow[callback.from_user.id] = {"stage": "post", "purpose": "template"}
    await callback.message.answer(
        "💾 Пришли пост для шаблона следующим сообщением. /cancel — отмена."
    )
    await callback.answer()


async def cb_ad_tpl_send(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    try:
        template_id = int((callback.data or "").split(":", 1)[1])
    except ValueError:
        await callback.answer()
        return
    item = templates.get(template_id)
    if item is None:
        await callback.answer("Шаблон не найден.", show_alert=True)
        return
    # отправка в фоне: разовая рассылка со срабатыванием на ближайшем тике воркера
    created = broadcasts.add(item.chat_id, item.message_id, "once", BROADCAST_TICK, time.monotonic())
    if created is None:
        await callback.answer("Лимит активных рассылок — подожди или отмени лишние.", show_alert=True)
        return
    await callback.answer(f"📤 Отправка шаблона #{template_id} запущена (в фоне).")


async def cb_ad_tpl_del(callback: CallbackQuery) -> None:
    if not is_admin(callback.from_user.id):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    try:
        template_id = int((callback.data or "").split(":", 1)[1])
    except ValueError:
        await callback.answer()
        return
    await callback.answer("Шаблон удалён." if templates.remove(template_id) else "Не найден.")


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
            record_sent(sent)
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
    """Отправить пост в целевые чаты; вернуть (отправлено, исключено)."""
    if item.target_days:
        targets = chats.recent_ids(time.time(), item.target_days)
    else:
        targets = chats.chat_ids()
    sent = blocked = 0
    for chat_id in targets:
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


def cancel_admin_flow(user_id: int) -> bool:
    """Отменить флоу админа; True если было что отменять."""
    return _flow.pop(user_id, None) is not None
