"""Хэндлеры aiogram 3.x: фото/стикеры/GIF, кнопки стиля, инлайн, свои палитры, любимый стиль.

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
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)

from .admin import (
    cancel_admin_flow,
    chats,
    is_banned,
    record_photo,
    register_admin_handlers,
)
from .config import (
    ADJUSTMENT_STEP,
    BG_TOLERANCE_RANGE,
    BG_TOLERANCE_STEP,
    CONTRAST_RANGE,
    DEFAULT_BG_TOLERANCE,
    DEFAULT_CONTRAST,
    DEFAULT_PADDING,
    DEFAULT_SATURATION,
    DEFAULT_SHARPNESS,
    GRID_MAX,
    GRID_MIN,
    MAX_CONCURRENT_JOBS,
    MAX_FILE_SIZE,
    PADDING_RANGE,
    PADDING_STEP,
    PALETTE_MAX_COLORS,
    RATE_LIMIT_SECONDS,
    SATURATION_RANGE,
    SHARPNESS_RANGE,
    SHARPNESS_STEP,
    STATE_MAX_ENTRIES,
    STATE_TTL_SECONDS,
    USER_STORE_MAX_ENTRIES,
)
from .pack import (
    MAX_PACK_PHOTOS,
    PACK_GRIDS,
    PackSettings,
    SEAM_MODES,
    add_photo,
    build_asset_pack,
    pop_photos,
    sessions as pack_sessions,
)
from .palettes import FIXED_PALETTES, PALETTE_LABELS, PALETTE_ORDER, parse_hex_colors
from .pixelart import (
    GRID_BY_SIZE,
    ImageTooLargeError,
    UnsupportedMediaError,
    build_demo_image,
    decode_source,
    encode_source,
    extract_first_frame,
    extract_palette,
    parse_grid,
    render,
    resize_for_sticker,
)
from .state import COLOR_STEPS, PerUserStore, SIZE_LABELS, SIZE_ORDER, Settings, StateStore

logger = logging.getLogger(__name__)
router = Router()
# админ-хэндлеры (/admin, /ban, перехват поста) — в начале цепочки,
# чтобы фото админа во время создания рассылки не уходило в пикселизацию
register_admin_handlers(router)

# --- общие singleton'ы в памяти процесса ---
store = StateStore(max_entries=STATE_MAX_ENTRIES, ttl_seconds=STATE_TTL_SECONDS)
favorites = PerUserStore(max_entries=USER_STORE_MAX_ENTRIES)  # user_id -> Settings
custom_palettes = PerUserStore(max_entries=USER_STORE_MAX_ENTRIES)  # user_id -> палитра
pack_settings = PerUserStore(max_entries=USER_STORE_MAX_ENTRIES)  # user_id -> PackSettings
_palette_flow: set[int] = set()  # пользователи, ожидающие ввода своей палитры
_grid_flow: dict[int, tuple[int, int]] = {}  # user_id -> (chat_id, message_id) цели
_recent_entry: dict[int, tuple[int, int]] = {}  # chat_id -> последнее (chat_id, message_id)
_jobs = asyncio.Semaphore(MAX_CONCURRENT_JOBS)  # не более 2 одновременных обработок
_last_photo_at: dict[int, float] = {}
_albums_seen: "OrderedDict[str, None]" = OrderedDict()

_demo_source: bytes | None = None

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

# кнопка -> (поле Settings, диапазон, шаг, значение по умолчанию)
_ADJUSTMENTS = {
    "con": ("contrast", CONTRAST_RANGE, ADJUSTMENT_STEP, DEFAULT_CONTRAST),
    "sat": ("saturation", SATURATION_RANGE, ADJUSTMENT_STEP, DEFAULT_SATURATION),
    "shp": ("sharpness", SHARPNESS_RANGE, SHARPNESS_STEP, DEFAULT_SHARPNESS),
}
# корректируемые кнопками ± целочисленные параметры кадра
_FRAME_STEPS = {
    "bgtol": ("bg_tolerance", BG_TOLERANCE_RANGE, BG_TOLERANCE_STEP, DEFAULT_BG_TOLERANCE, ""),
    "pad": ("padding", PADDING_RANGE, PADDING_STEP, DEFAULT_PADDING, "%"),
}
_VIEWS = ("main", "palettes", "adjust", "effects", "frame", "export")


# ---------------------------------------------------------------- команды


@router.message(Command("cancel"))
async def cmd_cancel(message: Message) -> None:
    """Отмена активного флоу (и админского, и создания своей палитры, и сетки)."""
    if message.from_user is None:
        return
    uid = message.from_user.id
    cancelled = cancel_admin_flow(uid)
    if uid in _palette_flow:
        _palette_flow.discard(uid)
        cancelled = True
    if uid in _grid_flow:
        _grid_flow.pop(uid, None)
        cancelled = True
    await message.answer("Действие отменено." if cancelled else "Нет активного действия.")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    chats.touch(message.chat.id)
    await message.answer(
        "Пришли фото, стикер или GIF — сразу верну пиксель-арт 🎨\n\n"
        "Под результатом кнопки: размер пикселя, палитра, дизеринг, коррекции, "
        "«🔖 Запомнить» (стиль для новых фото) и «📎 Скачать PNG» без сжатия.\n"
        "Напиши /demo, чтобы попробовать на примере."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться:\n"
        "1. Пришли фото, стикер или GIF (выйдет первый кадр).\n"
        "2. Получишь пиксель-арт; под ним — кнопки-категории.\n\n"
        "Категории кнопок:\n"
        "🎨 Палитры — Авто, Game Boy, NES, PICO-8, C64, Ч/Б, 1-bit, ZX, Sweetie, CGA, Свои\n"
        "🎛 Коррекции — дизеринг, контраст, насыщенность, резкость\n"
        "✨ Эффекты — обводка, CRT-сканлайны, зерно\n"
        "🖼 Кадр — ✂️ вырезать фон, 🎯 только объект, отступ\n"
        "📤 Экспорт — 🔖 запомнить стиль, 📎 PNG, 🧩 стикер 512, 💧 водяной знак\n\n"
        "Ассет-паки текстур: /pack → фото → /packdone; настройки — /packset.\n\n"
        "Команды: /palette, /mystyle, /demo, /grid N (свой уровень пикселизации)."
    )


@router.message(Command("palette"))
async def cmd_palette(message: Message) -> None:
    if message.from_user is None:
        return
    chats.touch(message.chat.id)
    _palette_flow.add(message.from_user.id)
    _grid_flow.pop(message.from_user.id, None)
    await message.answer(
        "🎨 Пришли следующим сообщением:\n"
        "• hex-цвета через пробел/запятую: #FF004D 00E436 7E2553\n"
        "• или фото — возьму основные цвета (до 64).\n"
        "/cancel — отмена."
    )


@router.message(Command("mystyle"))
async def cmd_mystyle(message: Message) -> None:
    if message.from_user is None:
        return
    chats.touch(message.chat.id)
    settings = favorites.get(message.from_user.id)
    if settings is None:
        await message.answer("Сохранённого стиля нет. Настрой стиль и нажми «🔖 Запомнить».")
        return
    await message.answer(
        f"🔖 Сохранённый стиль: {_describe_settings(settings)}\n"
        "Он применяется ко всем новым фото.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="🗑 Сбросить", callback_data="fav_clear")]]
        ),
    )


@router.message(Command("grid"))
async def cmd_grid(message: Message, bot: Bot, command: CommandObject | None = None) -> None:
    """Свой уровень пикселизации: /grid 48 применяет к последнему результату."""
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    target = _recent_entry.get(message.chat.id)
    args = (command.args if command else None) or ""
    grid = parse_grid(args)
    if grid is not None and target is not None:
        entry = store.get(*target)
        if entry is None:
            await bot.send_message(message.chat.id, TEXT_STALE)
            return
        settings = replace(entry.settings, pixel_size="custom", custom_grid=grid)
        ok = await _render_and_edit(bot, *target, entry, settings)
        await bot.send_message(
            message.chat.id, f"✅ Свой уровень: сетка {grid} px." if ok else TEXT_ERROR
        )
        return
    if target is not None and store.get(*target) is not None:
        _grid_flow[uid] = target
        _palette_flow.discard(uid)
        await bot.send_message(
            message.chat.id,
            f"📏 Пришли числом размер сетки {GRID_MIN}–{GRID_MAX} px "
            "(меньше = крупнее пиксели). /cancel — отмена.",
        )
        return
    await bot.send_message(
        message.chat.id,
        "Сначала пришли фото, затем /grid 48 — или кнопка «📏 Свой» под результатом.",
    )


@router.message(Command("demo"))
async def cmd_demo(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    if not _allow_rate(uid):
        await message.answer(TEXT_RATE_LIMIT)
        return
    source = await _get_demo_source()
    settings = Settings(
        pixel_size=random.choice(SIZE_ORDER),
        palette=random.choice(PALETTE_ORDER),
        dither=random.random() < 0.3,
    )
    async with _jobs:
        png = await asyncio.to_thread(render, source, settings, None)
    sent = await bot.send_photo(
        message.chat.id,
        BufferedInputFile(png, filename="pixel_demo.png"),
        caption="🎲 Демо: " + _caption(settings),
        reply_markup=_keyboard(settings),
    )
    store.put(message.chat.id, sent.message_id, source, settings)
    _recent_entry[message.chat.id] = (message.chat.id, sent.message_id)


# ---------------------------------------------------------------- свои палитры (флоу)


async def palette_capture(message: Message, bot: Bot) -> None:
    """Перехватывает ввод пользователя, создающего свою палитру."""
    uid = message.from_user.id if message.from_user else None
    if uid is None or uid not in _palette_flow:
        raise SkipHandler  # вне флоу — обычная обработка

    if message.media_group_id is not None:
        await message.answer("Пришли одним сообщением (без альбома).")
        return

    colors = None
    if message.text:
        colors = parse_hex_colors(message.text)
        if len(colors) < 2:
            await message.answer(
                "Не нашёл цветов (нужно хотя бы два). Пример: #FF004D 00E436 7E2553 — или /cancel."
            )
            return
    elif message.photo or (message.document and _is_image_doc(message.document)):
        try:
            if message.photo:
                data = await _download(bot, message.photo[-1].file_id)
            else:
                data = await _download(bot, message.document.file_id)
            async with _jobs:
                img = await asyncio.to_thread(decode_source, data)
                colors = await asyncio.to_thread(extract_palette, img, PALETTE_MAX_COLORS)
        except (UnsupportedMediaError, ImageTooLargeError):
            await message.answer(TEXT_NOT_A_PHOTO)
            return
        except TelegramBadRequest as exc:
            logger.warning("скачивание для палитры не удалось: %s", exc)
            await message.answer(TEXT_ERROR)
            return
    else:
        await message.answer("Пришли текст с hex-цветами или фото — или /cancel.")
        return

    custom_palettes.put(uid, colors[:PALETTE_MAX_COLORS])
    _palette_flow.discard(uid)
    await message.answer(
        f"✅ Своя палитра: {len(colors[:PALETTE_MAX_COLORS])} цветов. "
        "Нажми «Свои» под любым результатом."
    )


def _is_image_doc(document) -> bool:
    mime = document.mime_type or ""
    name = (document.file_name or "").lower()
    return mime.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp"))


async def grid_capture(message: Message, bot: Bot) -> None:
    """Перехватывает число — свой уровень пикселизации."""
    uid = message.from_user.id if message.from_user else None
    if uid is None or uid not in _grid_flow:
        raise SkipHandler  # вне флоу — обычная обработка
    grid = parse_grid(message.text or "")
    if grid is None:
        await message.answer(
            f"Нужно целое число {GRID_MIN}–{GRID_MAX}. Попробуй ещё раз или /cancel."
        )
        return
    target = _grid_flow.pop(uid)
    entry = store.get(*target)
    if entry is None:
        await message.answer(TEXT_STALE)
        return
    settings = replace(entry.settings, pixel_size="custom", custom_grid=grid)
    if await _render_and_edit(bot, *target, entry, settings):
        await message.answer(f"✅ Свой уровень: сетка {grid} px.")


# перехват до on_photo/on_document: сообщения пользователей в флоу своей палитры
router.message.register(
    palette_capture,
    F.photo | F.document | (F.text & ~F.text.startswith("/")),
)
# перехват ввода своего уровня пикселизации (текст-число)
router.message.register(
    grid_capture,
    F.text & ~F.text.startswith("/"),
)


# ---------------------------------------------------------------- приём фото/стикеров/GIF


def _user_ok(message: Message) -> int | None:
    """ID пользователя, если можно обрабатывать (не бан, не канал)."""
    if message.from_user is None or is_banned(message.from_user.id):
        return None
    return message.from_user.id


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    if uid in pack_sessions:  # режим ассет-пака: фото в буфер, без пикселизации
        data = await _download(bot, message.photo[-1].file_id)
        count = add_photo(uid, data)
        if count is None:
            await bot.send_message(
                message.chat.id, f"🧱 Лимит пака — {MAX_PACK_PHOTOS} фото. /packdone — собрать."
            )
        else:
            await bot.send_message(message.chat.id, f"🧱 В паке: {count}/{MAX_PACK_PHOTOS}")
        return
    album = False
    if message.media_group_id is not None:
        album = _claim_album(message.media_group_id)
        if not album:
            return  # остальные фото альбома молча пропускаем
    if not _allow_rate(uid):
        await message.answer(TEXT_RATE_LIMIT)
        return
    photo = message.photo[-1]  # наибольший размер
    data = await _download(bot, photo.file_id)
    await _process_new(message, bot, data, album=album)


@router.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    document = message.document
    name = (document.file_name or "").lower()
    mime = document.mime_type or ""
    is_gif = mime == "image/gif" or name.endswith(".gif")
    if not is_gif and not _is_image_doc(document):
        await message.answer(TEXT_NOT_A_PHOTO)
        return
    if document.file_size is not None and document.file_size > MAX_FILE_SIZE:
        await message.answer(TEXT_TOO_BIG_FILE)
        return
    if uid in pack_sessions:  # режим ассет-пака: файл в буфер
        data = await _download(bot, document.file_id)
        count = add_photo(uid, data)
        if count is None:
            await bot.send_message(
                message.chat.id, f"🧱 Лимит пака — {MAX_PACK_PHOTOS} фото. /packdone — собрать."
            )
        else:
            await bot.send_message(message.chat.id, f"🧱 В паке: {count}/{MAX_PACK_PHOTOS}")
        return
    if not _allow_rate(uid):
        await message.answer(TEXT_RATE_LIMIT)
        return
    data = await _download(bot, document.file_id)
    if len(data) > MAX_FILE_SIZE:
        await message.answer(TEXT_TOO_BIG_FILE)
        return
    await _process_new(message, bot, data, first_frame=is_gif)


@router.message(F.sticker)
async def on_sticker(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    sticker = message.sticker
    if sticker.is_animated or sticker.is_video:
        await message.answer(TEXT_ANIMATED)
        return
    if not _allow_rate(uid):
        await message.answer(TEXT_RATE_LIMIT)
        return
    data = await _download(bot, sticker.file_id)
    await _process_new(message, bot, data)


@router.message(F.animation)
async def on_animation(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    if uid in pack_sessions:
        await message.answer("🧱 Для пака нужны фото (не анимации).")
        return
    if not _allow_rate(uid):
        await message.answer(TEXT_RATE_LIMIT)
        return
    data = await _download(bot, message.animation.file_id)
    await _process_new(message, bot, data, first_frame=True)


async def _download(bot: Bot, file_id: str) -> bytes:
    """Скачать файл Telegram целиком в память."""
    file = await bot.get_file(file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


async def _process_new(
    message: Message, bot: Bot, data: bytes, *, album: bool = False, first_frame: bool = False
) -> None:
    """Обработать новое фото и отправить результат с кнопками."""
    uid = message.from_user.id if message.from_user else None
    settings = favorites.get(uid) if uid is not None else None
    custom = None
    if settings is not None and settings.palette == "custom":
        custom = custom_palettes.get(uid) if uid is not None else None
        if custom is None:  # любимый стиль ссылается на удалённую палитру
            settings = Settings()
    elif settings is None:
        settings = Settings()
    try:
        async with _jobs:
            if first_frame:
                data = await asyncio.to_thread(extract_first_frame, data)
            source_img = await asyncio.to_thread(decode_source, data)
            source = await asyncio.to_thread(encode_source, source_img)
            del data, source_img  # память: держим только сжатый исходник
            png = await asyncio.to_thread(render, source, settings, custom)
    except ImageTooLargeError:
        await message.answer(TEXT_TOO_BIG_DIMS)
        return
    except UnsupportedMediaError:
        await message.answer(TEXT_ANIMATED if first_frame else TEXT_NOT_A_PHOTO)
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
    store.put(message.chat.id, sent.message_id, source, settings, custom_palette=custom)
    _recent_entry[message.chat.id] = (message.chat.id, sent.message_id)
    record_photo(uid)
    logger.info("новое фото обработано, сетка=%s", GRID_BY_SIZE.get(settings.pixel_size, settings.custom_grid))


# ---------------------------------------------------------------- кнопки стиля


@router.callback_query(F.data == "size:custom")
async def cb_size_custom(callback: CallbackQuery) -> None:
    """«📏 Свой»: запросить у пользователя число — свой уровень пикселизации."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    uid = callback.from_user.id
    _grid_flow[uid] = (loaded[0], loaded[1])
    _palette_flow.discard(uid)
    await callback.answer(
        f"Пришли следующим сообщением число {GRID_MIN}–{GRID_MAX} — размер сетки в px "
        "(меньше = крупнее пиксели)",
        show_alert=True,
    )


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
    if value not in PALETTE_ORDER and value != "custom":
        await callback.answer()
        return
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    if value == "custom":
        palette = custom_palettes.get(callback.from_user.id)
        if palette is None:
            # палитры нет — запускаем создание, картинку не трогаем
            _palette_flow.add(callback.from_user.id)
            await callback.answer(
                "Своей палитры нет. Пришли следующим сообщением hex-цвета или фото — создам.",
                show_alert=True,
            )
            return
        await _apply(
            callback, bot, chat_id, message_id, entry,
            replace(entry.settings, palette="custom"),
            custom_palette=palette,
        )
        return
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


@router.callback_query(F.data.startswith("adj:"))
async def cb_adjust(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    try:
        _, kind, sign = (callback.data or "").split(":")
        field, (lo, hi), step, _default = _ADJUSTMENTS[kind]
    except (KeyError, ValueError):
        await callback.answer()
        return
    current = getattr(entry.settings, field)
    new_value = round(max(lo, min(hi, current + (step if sign == "p" else -step))), 2)
    if new_value == current:
        await callback.answer("Предел достигнут", show_alert=True)
        return
    await _apply(callback, bot, chat_id, message_id, entry, replace(entry.settings, **{field: new_value}))


@router.callback_query(F.data == "adj_reset")
async def cb_adjust_reset(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    settings = replace(
        entry.settings,
        dither=False,
        contrast=DEFAULT_CONTRAST,
        saturation=DEFAULT_SATURATION,
        sharpness=DEFAULT_SHARPNESS,
    )
    await _apply(callback, bot, chat_id, message_id, entry, settings, toast="Коррекции сброшены")


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


@router.callback_query(F.data == "fav_save")
async def cb_fav_save(callback: CallbackQuery) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    _chat_id, _message_id, entry = loaded
    favorites.put(callback.from_user.id, replace(entry.settings))
    await callback.answer("🔖 Стиль запомнен — будет применяться к новым фото (/mystyle — сброс)")


@router.callback_query(F.data == "fav_clear")
async def cb_fav_clear(callback: CallbackQuery) -> None:
    favorites.drop(callback.from_user.id)
    await callback.answer("Стиль сброшен — новые фото снова в настройках по умолчанию.")


@router.callback_query(F.data == "demo")
async def cb_demo(callback: CallbackQuery, bot: Bot) -> None:
    uid = callback.from_user.id if callback.from_user else None
    if uid is None or is_banned(uid):
        await callback.answer()
        return
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    _chat_id, _message_id, entry = loaded
    if not _allow_rate(uid):
        await callback.answer(TEXT_RATE_LIMIT)
        return
    source = await _get_demo_source()
    async with _jobs:
        png = await asyncio.to_thread(render, source, entry.settings, entry.custom_palette)
    sent = await bot.send_photo(
        loaded[0],
        BufferedInputFile(png, filename="pixel_demo.png"),
        caption="🎲 Демо: " + _caption(entry.settings),
        reply_markup=_keyboard(entry.settings),
    )
    store.put(loaded[0], sent.message_id, source, entry.settings, custom_palette=entry.custom_palette)
    _recent_entry[loaded[0]] = (loaded[0], sent.message_id)
    await callback.answer()


@router.callback_query(F.data == "download")
async def cb_download(callback: CallbackQuery, bot: Bot) -> None:
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, _message_id, entry = loaded
    async with _jobs:
        png = await asyncio.to_thread(
            render, entry.source, entry.settings, entry.custom_palette, flatten=False
        )
        if entry.settings.sticker:
            png = await asyncio.to_thread(resize_for_sticker, png)
    if entry.settings.sticker:
        await bot.send_document(
            chat_id,
            BufferedInputFile(png, filename="pixel_sticker.png"),
            caption="🧩 Стикер 512 px"
            + (" с прозрачным фоном" if entry.settings.cut_bg else ""),
        )
    else:
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


# ---------------------------------------------------------------- категории клавиатуры


@router.callback_query(F.data.startswith("menu:"))
async def cb_menu(callback: CallbackQuery, bot: Bot) -> None:
    """Открыть категорию — перерисовывается только клавиатура, без рендера."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    view = (callback.data or "").split(":", 1)[1]
    if view not in _VIEWS:
        await callback.answer()
        return
    try:
        await bot.edit_message_reply_markup(
            chat_id=chat_id, message_id=message_id, reply_markup=_keyboard(entry.settings, view)
        )
    except TelegramBadRequest as exc:
        if "not modified" not in str(exc).lower():
            logger.warning("edit_message_reply_markup не удался: %s", exc)
            await callback.answer(TEXT_ERROR, show_alert=True)
            return
    store.put(chat_id, message_id, entry.source, entry.settings, custom_palette=entry.custom_palette, view=view)
    await callback.answer()


@router.callback_query(F.data.startswith("eff:"))
async def cb_effect(callback: CallbackQuery, bot: Bot) -> None:
    """Переключатели эффектов: обводка, CRT, зерно."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    kind = (callback.data or "").split(":", 1)[-1]
    field = {"outline": "outline", "crt": "crt", "grain": "grain"}.get(kind)
    if field is None:
        await callback.answer()
        return
    settings = replace(entry.settings, **{field: not getattr(entry.settings, field)})
    await _apply(callback, bot, chat_id, message_id, entry, settings)


@router.callback_query(F.data.startswith("frame:"))
async def cb_frame(callback: CallbackQuery, bot: Bot) -> None:
    """Кадр и фон: вырезка, объект, порог и отступ кнопками ±."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    action = (callback.data or "").split(":", 1)[-1]
    if action == "cutbg":
        settings = replace(entry.settings, cut_bg=not entry.settings.cut_bg)
        await _apply(callback, bot, chat_id, message_id, entry, settings)
        return
    if action == "crop":
        settings = replace(entry.settings, crop_object=not entry.settings.crop_object)
        await _apply(callback, bot, chat_id, message_id, entry, settings)
        return
    parts = action.split(":")
    if len(parts) == 2 and parts[0] in _FRAME_STEPS:
        field, (lo, hi), step, default, _suffix = _FRAME_STEPS[parts[0]]
        current = getattr(entry.settings, field)
        new_value = int(max(lo, min(hi, current + (step if parts[1] == "p" else -step))))
        if new_value == current:
            await callback.answer("Предел достигнут", show_alert=True)
            return
        settings = replace(entry.settings, **{field: new_value})
        await _apply(callback, bot, chat_id, message_id, entry, settings)
        return
    await callback.answer()


@router.callback_query(F.data.startswith("export:"))
async def cb_export(callback: CallbackQuery, bot: Bot) -> None:
    """Экспорт: стикер (способ доставки) и водяной знак (на результат)."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    action = (callback.data or "").split(":", 1)[-1]
    if action == "sticker":
        settings = replace(entry.settings, sticker=not entry.settings.sticker)
        store.put(chat_id, message_id, entry.source, settings,
                  custom_palette=entry.custom_palette, view=entry.view)
        await bot.edit_message_reply_markup(
            chat_id=chat_id, message_id=message_id,
            reply_markup=_keyboard(settings, entry.view),
        )
        await callback.answer(
            "🧩 Стикер включён: «📎 PNG» отдаст файл 512 px"
            + (" с прозрачным фоном" if settings.cut_bg else "")
        )
        return
    if action == "watermark":
        settings = replace(entry.settings, watermark=not entry.settings.watermark)
        await _apply(callback, bot, chat_id, message_id, entry, settings)
        return
    await callback.answer()


@router.callback_query(F.data == "reset_all")
async def cb_reset_all(callback: CallbackQuery, bot: Bot) -> None:
    """Полный сброс настроек к значениям по умолчанию."""
    loaded = _load(callback)
    if loaded is None:
        await _stale(callback)
        return
    chat_id, message_id, entry = loaded
    fresh = Settings()
    await _apply(callback, bot, chat_id, message_id, entry, fresh, view="main",
                 toast="Все настройки сброшены")


# ---------------------------------------------------------------- ассет-паки


@router.message(Command("pack"))
async def cmd_pack(message: Message) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    if uid in pack_sessions:
        count = len(pack_sessions[uid])
        await message.answer(
            f"🧱 Сессия пака уже идёт: {count}/{MAX_PACK_PHOTOS} фото.\n"
            "/packdone — собрать, /packcancel — отмена, /packset — настройки."
        )
        return
    pack_sessions[uid] = []
    await message.answer(
        "🧱 Режим ассет-пака включён.\n\n"
        "Пришли фото-текстуры — альбомом или по одному (до 20).\n"
        "Каждое станет бесшовным тайлом; индивидуальные настройки пикселизации "
        "в этом режиме не применяются — пак собирается по общим настройкам.\n\n"
        "/packdone — собрать пак (zip + превью)\n"
        "/packset — настройки (размер, бесшовность, палитра, дизеринг)\n"
        "/packcancel — отмена"
    )


@router.message(Command("packdone"))
async def cmd_packdone(message: Message, bot: Bot) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    sources = pop_photos(uid)
    if not sources:
        await message.answer("Сессия пака пуста. /pack — начать заново.")
        return
    settings = pack_settings.get(uid) or PackSettings()
    custom = None
    if settings.palette == "custom":
        custom = custom_palettes.get(uid)
        if custom is None:
            pack_sessions[uid] = sources  # сессия не потерялась
            await message.answer("Своя палитра не задана — /palette. Сессия сохранена.")
            return
    await bot.send_message(message.chat.id, f"🧱 Собираю пак: {len(sources)} тайлов…")
    async with _jobs:
        zip_bytes, preview = await asyncio.to_thread(build_asset_pack, sources, settings, custom)
    await bot.send_document(
        chat_id=message.chat.id,
        document=BufferedInputFile(zip_bytes, filename=f"asset_pack_{settings.grid}x{settings.grid}.zip"),
        caption=(
            f"📦 {len(sources)} тайлов · {settings.grid}×{settings.grid} · "
            f"бесшовность: {SEAM_MODES[settings.seam]} · {settings.palette_label()} · "
            f"дизеринг {'вкл' if settings.dither else 'выкл'}"
        ),
    )
    await bot.send_photo(
        chat_id=message.chat.id,
        photo=BufferedInputFile(preview, filename="pack_preview.png"),
        caption=f"🖼 Превью (×{max(1, 512 // settings.grid)})",
    )


@router.message(Command("packcancel"))
async def cmd_packcancel(message: Message) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    dropped = pop_photos(uid)
    await message.answer(
        f"🧱 Сессия отменена, забыты фото: {len(dropped)} шт." if dropped else "Активной сессии пака нет."
    )


@router.message(Command("packset"))
async def cmd_packset(message: Message) -> None:
    uid = _user_ok(message)
    if uid is None:
        return
    chats.touch(message.chat.id)
    settings = pack_settings.get(uid) or PackSettings()
    await message.answer(_pack_text(settings), reply_markup=_pack_keyboard(settings))


def _pack_text(settings: PackSettings) -> str:
    return (
        "🧱 Настройки ассет-пака\n\n"
        f"Размер тайла: {settings.grid}×{settings.grid}\n"
        f"Бесшовность: {SEAM_MODES[settings.seam]}\n"
        f"Палитра: {settings.palette_label()}\n"
        f"Дизеринг: {'вкл' if settings.dither else 'выкл'}\n\n"
        "Тайлы — квадратные PNG 1:1 натурального размера (без апскейла, "
        "движки масштабируют сами), бесшовные.\n"
        "/pack — начать, /packdone — собрать, /packcancel — отмена."
    )


def _pack_keyboard(settings: PackSettings) -> InlineKeyboardMarkup:
    grid_row = [
        InlineKeyboardButton(
            text=_mark(f"{grid}×{grid}", settings.grid == grid), callback_data=f"pk:grid:{grid}"
        )
        for grid in PACK_GRIDS
    ]
    seam_row = [
        InlineKeyboardButton(
            text=_mark(label, settings.seam == key), callback_data=f"pk:seam:{key}"
        )
        for key, label in SEAM_MODES.items()
    ]
    palette_buttons = []
    for key in ("auto", "custom", *FIXED_PALETTES.keys()):
        if key == "auto":
            label = "Авто (общая)"
        elif key == "custom":
            label = "Свои"
        else:
            label = PALETTE_LABELS[key]
        palette_buttons.append(
            InlineKeyboardButton(
                text=_mark(label, settings.palette == key), callback_data=f"pk:pal:{key}"
            )
        )
    palette_rows = [palette_buttons[i : i + 3] for i in range(0, len(palette_buttons), 3)]
    dither_row = [
        InlineKeyboardButton(
            text=f"Дизеринг: {'вкл' if settings.dither else 'выкл'}", callback_data="pk:dither"
        )
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[grid_row, seam_row, dither_row, *palette_rows]
    )


@router.callback_query(F.data.startswith("pk:"))
async def cb_pack_set(callback: CallbackQuery) -> None:
    """Панель настроек ассет-пака: размер, бесшовность, палитра, дизеринг."""
    uid = callback.from_user.id if callback.from_user else None
    if uid is None:
        await callback.answer()
        return
    parts = (callback.data or "").split(":")
    settings = pack_settings.get(uid) or PackSettings()
    if len(parts) == 3 and parts[1] == "grid" and parts[2].isdigit() and int(parts[2]) in PACK_GRIDS:
        settings.grid = int(parts[2])
    elif len(parts) == 3 and parts[1] == "seam" and parts[2] in SEAM_MODES:
        settings.seam = parts[2]
    elif len(parts) == 3 and parts[1] == "pal" and (parts[2] in ("auto", "custom") or parts[2] in FIXED_PALETTES):
        settings.palette = parts[2]
    elif len(parts) == 2 and parts[1] == "dither":
        settings.dither = not settings.dither
    else:
        await callback.answer()
        return
    pack_settings.put(uid, settings)
    try:
        await callback.message.edit_text(_pack_text(settings), reply_markup=_pack_keyboard(settings))
    except TelegramBadRequest:
        pass  # «message is not modified» и подобное
    await callback.answer()


# ---------------------------------------------------------------- демо-исходник


async def _get_demo_source() -> bytes:
    global _demo_source
    if _demo_source is None:
        async with _jobs:
            if _demo_source is None:
                _demo_source = await asyncio.to_thread(
                    lambda: encode_source(build_demo_image())
                )
    return _demo_source


# ---------------------------------------------------------------- общие помощники


def _load(callback: CallbackQuery):
    """Найти запись состояния по сообщению с кнопками; None, если недоступна."""
    message = callback.message
    if not isinstance(message, Message):
        return None
    chats.touch(message.chat.id)
    entry = store.get(message.chat.id, message.message_id)
    if entry is None:
        return None
    return message.chat.id, message.message_id, entry


async def _stale(callback: CallbackQuery) -> None:
    await callback.answer(TEXT_STALE, show_alert=True)


async def _render_and_edit(
    bot: Bot,
    chat_id: int,
    message_id: int,
    entry,
    settings: Settings,
    custom_palette: tuple | None = None,
    view: str | None = None,
) -> bool:
    """Рендер + edit_message_media + сохранение настроек. False — не удалось."""
    if custom_palette is None:
        custom_palette = entry.custom_palette
    if view is None:
        view = entry.view
    async with _jobs:
        png = await asyncio.to_thread(render, entry.source, settings, custom_palette, flatten=True)
    try:
        await bot.edit_message_media(
            chat_id=chat_id,
            message_id=message_id,
            media=InputMediaPhoto(
                media=BufferedInputFile(png, filename="pixel_art.png"),
                caption=_caption(settings),
            ),
            reply_markup=_keyboard(settings, view),
        )
    except TelegramBadRequest as exc:
        if "not modified" in str(exc).lower():
            pass  # та же картинка и та же клавиатура — не ошибка
        else:
            logger.warning("edit_message_media не удался: %s", exc)
            return False
    store.put(chat_id, message_id, entry.source, settings, custom_palette=custom_palette, view=view)
    _recent_entry[chat_id] = (chat_id, message_id)
    return True


async def _apply(
    callback: CallbackQuery,
    bot: Bot,
    chat_id: int,
    message_id: int,
    entry,
    settings: Settings,
    custom_palette: tuple | None = None,
    toast: str | None = None,
    view: str | None = None,
) -> None:
    """Перерисовать результат по кнопке; ответить тостом или ошибкой."""
    if await _render_and_edit(bot, chat_id, message_id, entry, settings, custom_palette, view):
        await callback.answer(toast)
    else:
        await callback.answer(TEXT_ERROR, show_alert=True)


def _mark(text: str, active: bool) -> str:
    return f"{text} ✓" if active else text


def _keyboard(settings: Settings, view: str = "main") -> InlineKeyboardMarkup:
    """Клавиатура-категории: главный экран и подменю с «⬅️ В меню»."""
    back = [[InlineKeyboardButton(text="⬅️ В меню", callback_data="menu:main")]]
    if view == "palettes":
        labels = []
        for key in [*PALETTE_ORDER, "custom"]:
            if key == "auto":
                label = f"Авто ({settings.n_colors})"
            elif key == "custom":
                label = "Свои"
            else:
                label = PALETTE_LABELS[key]
            labels.append((_mark(label, settings.palette == key), key))
        keyboard = [
            [InlineKeyboardButton(text=text, callback_data=f"pal:{key}") for text, key in labels[i : i + 3]]
            for i in range(0, len(labels), 3)
        ]
        keyboard.append([
            InlineKeyboardButton(text="Больше цветов", callback_data="more"),
            InlineKeyboardButton(text="Меньше цветов", callback_data="less"),
        ])
        return InlineKeyboardMarkup(inline_keyboard=keyboard + back)
    if view == "adjust":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(
                    text=f"Дизеринг: {'вкл' if settings.dither else 'выкл'}", callback_data="dither")],
                [InlineKeyboardButton(text="Контраст −", callback_data="adj:con:m"),
                 InlineKeyboardButton(text="Контраст +", callback_data="adj:con:p")],
                [InlineKeyboardButton(text="Насыщ −", callback_data="adj:sat:m"),
                 InlineKeyboardButton(text="Насыщ +", callback_data="adj:sat:p")],
                [InlineKeyboardButton(text="Резк −", callback_data="adj:shp:m"),
                 InlineKeyboardButton(text="Резк +", callback_data="adj:shp:p")],
                [InlineKeyboardButton(text="↺ Сброс корр.", callback_data="adj_reset")],
                *back,
            ]
        )
    if view == "effects":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=_mark("Обводка", settings.outline), callback_data="eff:outline"),
                 InlineKeyboardButton(text=_mark("CRT", settings.crt), callback_data="eff:crt"),
                 InlineKeyboardButton(text=_mark("Зерно", settings.grain), callback_data="eff:grain")],
                *back,
            ]
        )
    if view == "frame":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=_mark("✂️ Вырезать фон", settings.cut_bg), callback_data="frame:cutbg"),
                 InlineKeyboardButton(text=_mark("🎯 Объект", settings.crop_object), callback_data="frame:crop")],
                [InlineKeyboardButton(text=f"Порог {settings.bg_tolerance} −", callback_data="frame:bgtol:m"),
                 InlineKeyboardButton(text=f"Порог {settings.bg_tolerance} +", callback_data="frame:bgtol:p")],
                [InlineKeyboardButton(text=f"Отступ {settings.padding}% −", callback_data="frame:pad:m"),
                 InlineKeyboardButton(text=f"Отступ {settings.padding}% +", callback_data="frame:pad:p")],
                *back,
            ]
        )
    if view == "export":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔖 Запомнить", callback_data="fav_save"),
                 InlineKeyboardButton(text="📎 PNG", callback_data="download")],
                [InlineKeyboardButton(text=_mark("🧩 Стикер 512", settings.sticker), callback_data="export:sticker"),
                 InlineKeyboardButton(text=_mark("💧 Водяной знак", settings.watermark), callback_data="export:watermark")],
                *back,
            ]
        )
    # main
    palette_done = settings.palette != "auto" or settings.n_colors != 16
    adjust_done = (
        settings.dither
        or settings.contrast != DEFAULT_CONTRAST
        or settings.saturation != DEFAULT_SATURATION
        or settings.sharpness != DEFAULT_SHARPNESS
    )
    effects_done = settings.outline or settings.crt or settings.grain
    frame_done = settings.cut_bg or settings.crop_object or settings.padding != DEFAULT_PADDING
    export_done = settings.sticker or settings.watermark
    size_row = [
        InlineKeyboardButton(
            text=_mark(SIZE_LABELS[key], settings.pixel_size == key),
            callback_data=f"size:{key}",
        )
        for key in SIZE_ORDER
    ]
    size_row.append(
        InlineKeyboardButton(
            text=_mark("📏 Свой", settings.pixel_size == "custom"),
            callback_data="size:custom",
        )
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            size_row,
            [
                InlineKeyboardButton(text=_mark("🎨 Палитры", palette_done), callback_data="menu:palettes"),
                InlineKeyboardButton(text=_mark("🎛 Коррекции", adjust_done), callback_data="menu:adjust"),
                InlineKeyboardButton(text=_mark("✨ Эффекты", effects_done), callback_data="menu:effects"),
            ],
            [
                InlineKeyboardButton(text=_mark("🖼 Кадр", frame_done), callback_data="menu:frame"),
                InlineKeyboardButton(text=_mark("📤 Экспорт", export_done), callback_data="menu:export"),
                InlineKeyboardButton(text="🎲 Другой стиль", callback_data="random"),
            ],
            [
                InlineKeyboardButton(text="🎲 Демо", callback_data="demo"),
                InlineKeyboardButton(text="↺ Сброс", callback_data="reset_all"),
            ],
        ]
    )


def _grid_label(settings: Settings) -> str:
    """Размер сетки для подписи: пресет или свой уровень."""
    if settings.pixel_size == "custom":
        return f"{settings.custom_grid} px (свой)"
    return f"{GRID_BY_SIZE[settings.pixel_size]} px"


def _caption(settings: Settings) -> str:
    palette_label = (
        f"Авто ({settings.n_colors})" if settings.palette == "auto" else (
            "Свои" if settings.palette == "custom" else PALETTE_LABELS[settings.palette]
        )
    )
    text = f"🎨 {_grid_label(settings)} · {palette_label} · дизеринг {'вкл' if settings.dither else 'выкл'}"
    adjustments = []
    if settings.contrast != DEFAULT_CONTRAST:
        adjustments.append(f"контраст {settings.contrast:g}")
    if settings.saturation != DEFAULT_SATURATION:
        adjustments.append(f"насыщ {settings.saturation:g}")
    if settings.sharpness != DEFAULT_SHARPNESS:
        adjustments.append(f"резкость {settings.sharpness:g}")
    if adjustments:
        text += " · " + ", ".join(adjustments)
    return text


def _describe_settings(settings: Settings) -> str:
    return _caption(settings)


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
    from .admin import prune_daily

    while True:
        await asyncio.sleep(300)
        try:
            removed = store.purge_expired()
            if removed:
                logger.info("очищено истёкших записей: %d", removed)
            prune_daily()
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
