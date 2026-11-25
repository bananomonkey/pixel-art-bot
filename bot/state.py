"""Состояние в памяти: настройки сессии и исходники (LRU + TTL).

На диск ничего не пишется; исходники хранятся уменьшенными (<=1024 px)
в сжатом виде (байты JPEG), чтобы экономить RAM на хосте с 1 ГБ.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from .config import (
    DEFAULT_CONTRAST,
    DEFAULT_CUSTOM_GRID,
    DEFAULT_SATURATION,
    DEFAULT_SHARPNESS,
)

# --- уровни размера пикселя: сетка по большей стороне ---
SIZE_ORDER: tuple[str, ...] = ("small", "medium", "large", "huge")
SIZE_LABELS: dict[str, str] = {
    "small": "Мелкий",
    "medium": "Средний",
    "large": "Крупный",
    "huge": "Огромный",
}

# ряд числа цветов для «Авто»: 8 -> 16 -> 32 -> 64
COLOR_STEPS: tuple[int, ...] = (8, 16, 32, 64)


@dataclass
class Settings:
    """Настройки пиксель-арта для одного фото."""

    pixel_size: str = "medium"  # small | medium | large | huge | custom
    custom_grid: int = DEFAULT_CUSTOM_GRID  # свой уровень пикселизации (8–256)
    palette: str = "auto"  # auto | gameboy | nes | pico8 | c64 | bw | bw1 | zx | sweetie | cga | custom
    n_colors: int = 16  # только для «Авто», одно из COLOR_STEPS
    dither: str = "off"  # off | fs | atkinson | bayer
    details: bool = False  # чёткость мелких деталей
    contrast: float = DEFAULT_CONTRAST
    saturation: float = DEFAULT_SATURATION
    sharpness: float = DEFAULT_SHARPNESS
    # кадр и фон (порог подбирается автоматически: bg_tolerance=0)
    cut_bg: bool = False  # вырезать фон (flood fill от краёв)
    crop_mode: str = "off"  # off (весь кадр) | fill (объект крупно) | pad (объект с полем)
    # эффекты
    outline: str = "off"  # off | dark (тёмный контур) | volume (со светом/тенью)
    seamless: str = "off"  # off | blend | mirror — бесшовный тайл
    crt: bool = False  # сканлайны
    grain: bool = False  # зерно
    show_grid: bool = False  # подсветка границ пикселей
    # экспорт
    sticker: bool = False  # «📎 PNG» отдаёт стикер 512 px
    watermark: bool = False  # подпись бота на результате


def with_changes(settings: Settings, **changes) -> Settings:
    """Копия настроек с изменениями (иммутабельный стиль)."""
    return replace(settings, **changes)


@dataclass
class Entry:
    """Запись состояния: подготовленный исходник + текущие настройки.

    custom_palette — снимок своей палитры автора (нужен для перерисовки
    кнопками, если автор выбрал режим «Свои»). view — открытая категория
    клавиатуры (возврат после перерисовки в то же подменю).
    """

    source: bytes  # исходник, уменьшенный и сжатый (JPEG), только в памяти
    settings: Settings
    updated_at: float
    custom_palette: tuple | None = None
    view: str = "main"


class StateStore:
    """Словарь (chat_id, message_id) -> Entry с LRU-вытеснением и TTL.

    TTL скользящий: обращение к записи обновляет её метку времени,
    так что активная сессия не «протухает» посреди настройки стиля.
    """

    def __init__(self, max_entries: int = 200, ttl_seconds: float = 1800.0):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[tuple[int, int], Entry]" = OrderedDict()

    def put(
        self,
        chat_id: int,
        message_id: int,
        source: bytes,
        settings: Settings,
        custom_palette: tuple | None = None,
        view: str = "main",
    ) -> None:
        """Сохранить/обновить запись и поднять её наверх LRU."""
        key = (chat_id, message_id)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = Entry(
            source=source,
            settings=settings,
            updated_at=time.monotonic(),
            custom_palette=custom_palette,
            view=view,
        )
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)  # вытеснить наименее давнюю

    def get(self, chat_id: int, message_id: int) -> Entry | None:
        """Вернуть запись или None, если её нет/истёк TTL; обновляет LRU и TTL."""
        key = (chat_id, message_id)
        entry = self._entries.get(key)
        if entry is None:
            return None
        now = time.monotonic()
        if now - entry.updated_at > self._ttl_seconds:
            del self._entries[key]
            return None
        entry.updated_at = now  # скользящий TTL
        self._entries.move_to_end(key)
        return entry

    def drop(self, chat_id: int, message_id: int) -> None:
        """Удалить запись (например, если сообщение результата удалено)."""
        self._entries.pop((chat_id, message_id), None)

    def purge_expired(self) -> int:
        """Вычистить истёкшие записи; вернуть количество удалённых."""
        now = time.monotonic()
        expired = [k for k, e in self._entries.items() if now - e.updated_at > self._ttl_seconds]
        for key in expired:
            del self._entries[key]
        return len(expired)

    def __len__(self) -> int:
        return len(self._entries)


class ChatsRegistry:
    """LRU-реестр чатов, которые общались с ботом (для рассылок админом).

    Только в памяти процесса: после перезапуска реестр пуст.
    Хранится только chat_id и время последней активности, без содержимого.
    """

    def __init__(self, max_entries: int = 10_000):
        self._max_entries = max_entries
        self._chats: "OrderedDict[int, float]" = OrderedDict()  # chat_id -> время (unix)

    def touch(self, chat_id: int, now: float | None = None) -> None:
        """Отметить чат как активный (поднимает его наверх LRU)."""
        if chat_id in self._chats:
            self._chats.move_to_end(chat_id)
        self._chats[chat_id] = time.time() if now is None else now
        while len(self._chats) > self._max_entries:
            self._chats.popitem(last=False)

    def remove(self, chat_id: int) -> None:
        """Удалить чат (например, бот заблокирован пользователем)."""
        self._chats.pop(chat_id, None)

    def chat_ids(self) -> list[int]:
        """Список активных чатов."""
        return list(self._chats.keys())

    def recent_ids(self, now: float, days: float) -> list[int]:
        """Чаты, активные за последние days суток."""
        cutoff = now - days * 86400.0
        return [chat_id for chat_id, ts in self._chats.items() if ts >= cutoff]

    def __len__(self) -> int:
        return len(self._chats)


class PerUserStore:
    """LRU-словарь user_id -> значение (любимые стили, свои палитры и т.п.)."""

    def __init__(self, max_entries: int = 2000):
        self._max_entries = max_entries
        self._items: "OrderedDict[int, object]" = OrderedDict()

    def put(self, user_id: int, value) -> None:
        if user_id in self._items:
            self._items.move_to_end(user_id)
        self._items[user_id] = value
        while len(self._items) > self._max_entries:
            self._items.popitem(last=False)

    def get(self, user_id: int):
        value = self._items.get(user_id)
        if value is not None:
            self._items.move_to_end(user_id)
        return value

    def drop(self, user_id: int) -> None:
        self._items.pop(user_id, None)

    def __len__(self) -> int:
        return len(self._items)
