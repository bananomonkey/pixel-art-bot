"""Рассылки: чистая логика расписания, без зависимости от aiogram.

Интервалы задаются админом текстом вида «30м», «1ч30м», «1д».
Всё хранится в памяти процесса (файловая система хостинга эфемерна).
"""

import re
import time
from dataclasses import dataclass

# русские обозначения единиц: секунды, минуты, часы, дни
UNIT_SECONDS: dict[str, float] = {"с": 1.0, "м": 60.0, "ч": 3600.0, "д": 86400.0}

_INTERVAL_PART = re.compile(r"(\d+(?:[.,]\d+)?)\s*([смчд])")
_INTERVAL_ALLOWED = re.compile(r"^[\d\s.,смчд]+$")


def parse_interval(text: str) -> float | None:
    """«30м» -> 1800.0, «1ч30м» -> 5400.0, «1,5ч» -> 5400.0; None если не удалось."""
    cleaned = (text or "").strip().lower()
    if not cleaned or not _INTERVAL_ALLOWED.match(cleaned):
        return None
    total = 0.0
    matched = False
    for number, unit in _INTERVAL_PART.findall(cleaned):
        total += float(number.replace(",", ".")) * UNIT_SECONDS[unit]
        matched = True
    if not matched:
        return None
    return total


def format_interval(seconds: float) -> str:
    """5400 -> «1ч 30м»; 90000 -> «1д 1ч»; обратный к parse_interval формат."""
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if days:
        parts.append(f"{days}д")
    if hours:
        parts.append(f"{hours}ч")
    if minutes:
        parts.append(f"{minutes}м")
    if secs and not days:
        parts.append(f"{secs}с")
    return " ".join(parts) or "0с"


@dataclass
class Broadcast:
    """Одна рассылка: пост (chat_id+message_id у админа) и расписание."""

    id: int
    chat_id: int  # чат, где лежит исходный пост (для copy_message)
    message_id: int
    mode: str  # "once" — разово | "cycle" — циклично
    interval: float  # секунды: задержка для once, период для cycle
    next_run: float  # время следующей отправки (time.monotonic)
    sent_total: int = 0
    target_days: float | None = None  # «активным за N дней» — фильтр чатов


class BroadcastStore:
    """Активные рассылки в памяти; лимит на количество."""

    def __init__(self, max_active: int = 20):
        self._max_active = max_active
        self._next_id = 1
        self._items: dict[int, Broadcast] = {}

    def add(
        self,
        chat_id: int,
        message_id: int,
        mode: str,
        interval: float,
        now: float,
        target_days: float | None = None,
    ) -> Broadcast | None:
        """Создать рассылку; None, если достигнут лимит активных."""
        if mode not in ("once", "cycle") or len(self._items) >= self._max_active:
            return None
        item = Broadcast(
            id=self._next_id,
            chat_id=chat_id,
            message_id=message_id,
            mode=mode,
            interval=interval,
            next_run=now + interval,
            target_days=target_days,
        )
        self._next_id += 1
        self._items[item.id] = item
        return item

    def remove(self, broadcast_id: int) -> bool:
        return self._items.pop(broadcast_id, None) is not None

    def get_all(self) -> list[Broadcast]:
        """Все активные, в порядке создания."""
        return sorted(self._items.values(), key=lambda b: b.id)

    def due(self, now: float) -> list[Broadcast]:
        """Рассылки, время которых наступило."""
        return [b for b in self.get_all() if b.next_run <= now]

    def reschedule(self, item: Broadcast, now: float) -> None:
        """Для циклической: назначить следующий запуск от момента отправки."""
        item.next_run = now + item.interval

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------- шаблоны постов


@dataclass
class Template:
    """Сохранённый пост админа: можно отправить сейчас одним нажатием."""

    id: int
    chat_id: int
    message_id: int
    created_at: float


class TemplateStore:
    """Шаблоны рассылок в памяти (без содержимого — только ссылка на пост)."""

    def __init__(self, max_items: int = 20):
        self._max_items = max_items
        self._next_id = 1
        self._items: dict[int, Template] = {}

    def add(self, chat_id: int, message_id: int, now: float) -> Template | None:
        if len(self._items) >= self._max_items:
            return None
        item = Template(id=self._next_id, chat_id=chat_id, message_id=message_id, created_at=now)
        self._next_id += 1
        self._items[item.id] = item
        return item

    def remove(self, template_id: int) -> bool:
        return self._items.pop(template_id, None) is not None

    def get(self, template_id: int) -> Template | None:
        return self._items.get(template_id)

    def get_all(self) -> list[Template]:
        return sorted(self._items.values(), key=lambda t: t.id)

    def __len__(self) -> int:
        return len(self._items)


def parse_target_days(text: str) -> float | None:
    """«7» -> 7.0 (дней активности для целевой рассылки); None если неверно."""
    cleaned = (text or "").strip()
    if not cleaned.isdigit() or int(cleaned) < 1:
        return None
    from .config import TARGETED_MAX_DAYS

    if int(cleaned) > TARGETED_MAX_DAYS:
        return None
    return float(int(cleaned))
