"""Состояние в памяти: настройки сессии и исходники (LRU + TTL).

На диск ничего не пишется; исходники хранятся уменьшенными (<=1024 px)
в сжатом виде (байты JPEG), чтобы экономить RAM на хосте с 1 ГБ.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, replace

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

    pixel_size: str = "medium"  # small | medium | large | huge
    palette: str = "auto"  # auto | gameboy | nes | pico8 | c64 | bw | cga
    n_colors: int = 16  # только для «Авто», одно из COLOR_STEPS
    dither: bool = False


def with_changes(settings: Settings, **changes) -> Settings:
    """Копия настроек с изменениями (иммутабельный стиль)."""
    return replace(settings, **changes)


@dataclass
class Entry:
    """Запись состояния: подготовленный исходник + текущие настройки."""

    source: bytes  # исходник, уменьшенный и сжатый (JPEG), только в памяти
    settings: Settings
    updated_at: float


class StateStore:
    """Словарь (chat_id, message_id) -> Entry с LRU-вытеснением и TTL.

    TTL скользящий: обращение к записи обновляет её метку времени,
    так что активная сессия не «протухает» посреди настройки стиля.
    """

    def __init__(self, max_entries: int = 200, ttl_seconds: float = 1800.0):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[tuple[int, int], Entry]" = OrderedDict()

    def put(self, chat_id: int, message_id: int, source: bytes, settings: Settings) -> None:
        """Сохранить/обновить запись и поднять её наверх LRU."""
        key = (chat_id, message_id)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = Entry(source=source, settings=settings, updated_at=time.monotonic())
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
