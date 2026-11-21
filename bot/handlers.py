"""Хэндлеры aiogram 3.x: приём фото, кнопки стиля, редактирование результата.

Никаких команд вводить не нужно: пришло фото -> сразу результат с кнопками.
Фото и результаты только в памяти (io.BytesIO), на диск не пишутся и не логируются.
"""

import asyncio
import io
import logging
import random
import time
from collections import OrderedDict
from dataclasses import replace

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)

from .config import (
    MAX_CONCURRENT_JOBS,
    MAX_FILE_SIZE,
    RATE_LIMIT_SECONDS,
    STATE_MAX_ENTRIES,
    STATE_TTL_SECONDS,
)
from .palettes import PALETTE_LABELS, PALETTE_ORDER
from .pixelart import (
    GRID_BY_SIZE,
    ImageTooLargeError,
    UnsupportedMediaError,
    decode_source,
    encode_source,
    render,
)
from .state import COLOR_STEPS, SIZE_LABELS, SIZE_ORDER, Settings, StateStore

logger = logging.getLogger(__name__)
router = Router()

# --- общие singleton'ы в памяти процесса ---
store = StateStore(max_entries=STATE_MAX_ENTRIES, ttl_seconds=STATE_TTL_SECONDS)
_jobs = asyncio.Semaphore(MAX_CONCURRENT_JOBS)  # не более 2 одновременных обработок
_last_photo_at: dict[int, float] = {}
_albums_seen: "OrderedDict[str, None]" = OrderedDict()

# --- реплики ---
TEXT_NOT_A_PHOTO = "Пришли фото, сделаю пиксель-арт"
TEXT_ANIMATED = "Пока только фото"
TEXT_STALE = "Исходное фото устарело, пришли его заново"
TEXT_TOO_BIG_FILE = "Файл слишком большой — до 20 МБ"
TEXT_TOO_BIG_DIMS = "Изображение слишком большое — максимум 4000×4000"
TEXT_RATE_LIMIT = "Слишком часто — не чаще одного фото в 3 секунды"
TEXT_ERROR = "Что-то пошло не так. Попробуй ещё раз."
TEXT_COLORS_ONLY_AUTO = "Число цветов меняется только в режиме «Авто»"
TEXT_COLORS_MAX = "Максимум — 64 цвета"
TEXT_COLORS_MIN = "Минимум — 8 цветов"


# ---------------------------------------------------------------- команды


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Пришли фото — сразу верну пиксель-арт 🎨\n\n"
        "Под результатом кнопки: размер пикселя, палитра, дизеринг, "
        "больше/меньше цветов (для «Авто»), скачивание PNG без сжатия "
        "и «Другой стиль»."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться:\n"
        "1. Пришли фото (или файл jpg/png/webp).\n"
        "2. Получишь пиксель-арт с настройками по умолчанию.\n"
        "3. Меняй стиль кнопками под фото — картинка перерисуется на месте.\n\n"
        "Палитры: Авто (8–64 цвета), Game Boy, NES, PICO-8, C64, Ч/Б, CGA.\n"
        "«📎 Скачать PNG» отправляет файл без пережатия Telegram."
    )


# ---------------------------------------------------------------- приём фото


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot) -> None:
    if message.from_user is None:
        return
    album = False
    if message.media_group_id is not None:
        album = _claim_album(message.media_group_id)
        if not album:
            return  # остальные фото альбома молча пропускаем
    if not _allow_rate(message.from_user.id):
        await message.answer(TEXT_RATE_LIMIT)
        return
    photo = message.photo[-1]  # наибольший размер
    data = await _download(bot, photo.file_id)
    await _process_new(message, bot, data, album=album)


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    if message.from_user is None:
        return
    document = message.document
    name = (document.file_name or "").lower()
    mime = document.mime_type or ""
    if mime == "image/gif" or name.endswith(".gif"):
        await message.answer(TEXT_ANIMATED)
        return
    if not (mime.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp"))):
        await message.answer(TEXT_NOT_A_PHOTO)
        return
    if document.file_size is not None and document.file_size > MAX_FILE_SIZE:
        await message.answer(TEXT_TOO_BIG_FILE)
        return
    if not _allow_rate(message.from_user.id):
        await message.answer(TEXT_RATE_LIMIT)
        return
    data = await _download(bot, document.file_id)
    if len(data) > MAX_FILE_SIZE:
        await message.answer(TEXT_TOO_BIG_FILE)
        return
    await _process_new(message, bot, data, album=False)


@router.message(F.animation | F.video | F.video_note)
async def on_animation(message: Message) -> None:
    await message.answer(TEXT_ANIMATED)


async def _download(bot: Bot, file_id: str) -> bytes:
    """Скачать файл Telegram целиком в память."""
    file = await bot.get_file(file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


async def _process_new(message: Message, bot: Bot, data: bytes, *, album: bool) -> None:
    """Обработать новое фото и отправить результат с кнопками."""
    settings = Settings()
    try:
        async with _jobs:
            source_img = await asyncio.to_thread(decode_source, data)
            source = await asyncio.to_thread(encode_source, source_img)
            del data, source_img  # память: держим только сжатый исходник
            png = await asyncio.to_thread(render, source, settings)
    except ImageTooLargeError:
        await message.answer(TEXT_TOO_BIG_DIMS)
        return
    except UnsupportedMediaError:
        await message.answer(TEXT_NOT_A_PHOTO)
        return

    caption = _caption(settings)
    if album:
        caption = "В альбоме обработал только первое фото.\n" + caption
    sent = await bot.send_photo(
        message.chat.id,
        BufferedInputFile(png, filename="pixel_art.png"),
        caption=caption,
        reply_markup=_keyboard(settings),
    )
    store.put(message.chat.id, sent.message_id, source, settings)
    logger.info("новое фото обработано, сетка=%s", GRID_BY_SIZE[settings.pixel_size])


# ---------------------------------------------------------------- кнопки


@router.callback_query(F.data.startswith("size:"))
async def cb_size(callback: CallbackQuery, bot: Bot) -> None:
    value = (callback.data or "").split(":", 1)[-1]
    if value not in SIZE_ORDER:
        await callback.answer()
        return
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    await _apply(callback, bot, chat_id, message_id, entry, replace(entry.settings, pixel_size=value))


@router.callback_query(F.data.startswith("pal:"))
async def cb_palette(callback: CallbackQuery, bot: Bot) -> None:
    value = (callback.data or "").split(":", 1)[-1]
    if value not in PALETTE_ORDER:
        await callback.answer()
        return
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    await _apply(callback, bot, chat_id, message_id, entry, replace(entry.settings, palette=value))


@router.callback_query(F.data == "dither")
async def cb_dither(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    flipped = replace(entry.settings, dither=not entry.settings.dither)
    await _apply(callback, bot, chat_id, message_id, entry, flipped)


@router.callback_query(F.data == "more")
async def cb_more_colors(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    if entry.settings.palette != "auto":
        await callback.answer(TEXT_COLORS_ONLY_AUTO, show_alert=True)
        return
    index = COLOR_STEPS.index(entry.settings.n_colors) if entry.settings.n_colors in COLOR_STEPS else 1
    if index >= len(COLOR_STEPS) - 1:
        await callback.answer(TEXT_COLORS_MAX, show_alert=True)
        return
    await _apply(
        callback, bot, chat_id, message_id, entry,
        replace(entry.settings, n_colors=COLOR_STEPS[index + 1]),
    )


@router.callback_query(F.data == "less")
async def cb_less_colors(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    if entry.settings.palette != "auto":
        await callback.answer(TEXT_COLORS_ONLY_AUTO, show_alert=True)
        return
    index = COLOR_STEPS.index(entry.settings.n_colors) if entry.settings.n_colors in COLOR_STEPS else 1
    if index <= 0:
        await callback.answer(TEXT_COLORS_MIN, show_alert=True)
        return
    await _apply(
        callback, bot, chat_id, message_id, entry,
        replace(entry.settings, n_colors=COLOR_STEPS[index - 1]),
    )


@router.callback_query(F.data == "download")
async def cb_download(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, _message_id, entry = loaded
    async with _jobs:
        png = await asyncio.to_thread(render, entry.source, entry.settings)
    await bot.send_document(
        chat_id,
        BufferedInputFile(png, filename=_filename(entry.settings)),
        caption="📎 PNG без сжатия",
    )
    await callback.answer()


@router.callback_query(F.data == "random")
async def cb_random(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    current = entry.settings
    settings = replace(
        current,
        palette=random.choice([key for key in PALETTE_ORDER if key != current.palette]),
        pixel_size=random.choice([key for key in SIZE_ORDER if key != current.pixel_size]),
    )
    await _apply(callback, bot, chat_id, message_id, entry, settings)


# ---------------------------------------------------------------- общие помощники


def _load(callback: CallbackQuery):
    """Найти запись состояния по сообщению с кнопками; None, если недоступна."""
    message = callback.message
    if not isinstance(message, Message):
        return None
    entry = store.get(message.chat.id, message.message_id)
    if entry is None:
        return None
    return message.chat.id, message.message_id, entry


async def _stale(callback: CallbackQuery) -> None:
    await callback.answer(TEXT_STALE, show_alert=True)


async def _apply(
    callback: CallbackQuery,
    bot: Bot,
    chat_id: int,
    message_id: int,
    entry,
    settings: Settings,
) -> None:
    """Перерисовать результат на месте (edit_message_media) и сохранить настройки."""
    async with _jobs:
        png = await asyncio.to_thread(render, entry.source, settings)
    try:
        await bot.edit_message_media(
            chat_id=chat_id,
            message_id=message_id,
            media=InputMediaPhoto(
                media=BufferedInputFile(png, filename="pixel_art.png"),
                caption=_caption(settings),
            ),
            reply_markup=_keyboard(settings),
        )
    except TelegramBadRequest as exc:
        if "not modified" in str(exc).lower():
            pass  # та же картинка и та же клавиатура — не ошибка
        else:
            logger.warning("edit_message_media не удался: %s", exc)
            await callback.answer(TEXT_ERROR, show_alert=True)
            return
    store.put(chat_id, message_id, entry.source, settings)  # обновит настройки и LRU/TTL
    await callback.answer()


def _keyboard(settings: Settings) -> InlineKeyboardMarkup:
    """Компактная inline-клавиатура с пометками ✓ у выбранных значений."""
    def mark(text: str, active: bool) -> str:
        return f"{text} ✓" if active else text

    size_row = [
        InlineKeyboardButton(
            text=mark(SIZE_LABELS[key], settings.pixel_size == key),
            callback_data=f"size:{key}",
        )
        for key in SIZE_ORDER
    ]
    palette_row = []
    for key in PALETTE_ORDER:
        label = f"Авто ({settings.n_colors})" if key == "auto" else PALETTE_LABELS[key]
        palette_row.append(
            InlineKeyboardButton(
                text=mark(label, settings.palette == key),
                callback_data=f"pal:{key}",
            )
        )
    options_row = [
        InlineKeyboardButton(
            text=f"Дизеринг: {'вкл' if settings.dither else 'выкл'}",
            callback_data="dither",
        ),
        InlineKeyboardButton(text="Больше цветов", callback_data="more"),
        InlineKeyboardButton(text="Меньше цветов", callback_data="less"),
    ]
    actions_row = [
        InlineKeyboardButton(text="📎 Скачать PNG", callback_data="download"),
        InlineKeyboardButton(text="🎲 Другой стиль", callback_data="random"),
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[size_row, palette_row, options_row, actions_row]
    )


def _caption(settings: Settings) -> str:
    palette_label = (
        f"Авто ({settings.n_colors})" if settings.palette == "auto" else PALETTE_LABELS[settings.palette]
    )
    dither_label = "вкл" if settings.dither else "выкл"
    return f"🎨 {GRID_BY_SIZE[settings.pixel_size]} px · {palette_label} · дизеринг {dither_label}"


def _filename(settings: Settings) -> str:
    return f"pixel_art_{settings.pixel_size}_{settings.palette}.png"


def _allow_rate(user_id: int) -> bool:
    """Антиспам в памяти: не более 1 фото в RATE_LIMIT_SECONDS на пользователя."""
    now = time.monotonic()
    last = _last_photo_at.get(user_id)
    if last is not None and now - last < RATE_LIMIT_SECONDS:
        return False
    _last_photo_at[user_id] = now
    if len(_last_photo_at) > 10_000:  # страховка от роста словаря
        cutoff = now - 300.0
        for uid in [uid for uid, ts in _last_photo_at.items() if ts < cutoff]:
            _last_photo_at.pop(uid, None)
    return True


def _claim_album(group_id: str) -> bool:
    """True для первого фото альбома, False для остальных (LRU на 100 групп)."""
    if group_id in _albums_seen:
        return False
    _albums_seen[group_id] = None
    _albums_seen.move_to_end(group_id)
    while len(_albums_seen) > 100:
        _albums_seen.popitem(last=False)
    return True


async def periodic_purge() -> None:
    """Фоновая чистка истёкших записей состояния (раз в 5 минут, экономия RAM)."""
    while True:
        await asyncio.sleep(300)
        try:
            removed = store.purge_expired()
            if removed:
                logger.info("очищено истёкших записей: %d", removed)
        except Exception:  # noqa: BLE001 — фон не должен падать
            logger.exception("ошибка очистки состояния")


# ---------------------------------------------------------------- глобальные ошибки


@router.error()
async def on_error(event: ErrorEvent) -> None:
    """Пользователю — понятная строка без трейсбэков; в лог — только тип ошибки."""
    exception = event.exception
    logger.error("необработанная ошибка: %s: %s", type(exception).__name__, exception)
    update = event.update
    try:
        if update.callback_query is not None:
            await update.callback_query.answer(TEXT_ERROR, show_alert=True)
        elif update.message is not None:
            await update.message.answer(TEXT_ERROR)
    except Exception:  # noqa: BLE001 — не роняем диспетчер на ответе об ошибке
        logger.warning("не удалось сообщить пользователю об ошибке")
