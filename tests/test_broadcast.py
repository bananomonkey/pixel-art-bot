"""Тесты рассылок: парсер/форматтер интервалов и хранилище расписания."""

import pytest

from bot.broadcast import BroadcastStore, format_interval, parse_interval


# ------------------------------------------------------------------ parse_interval


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30м", 1800.0),
        ("1ч30м", 5400.0),
        ("1д", 86400.0),
        ("90с", 90.0),
        ("1,5ч", 5400.0),
        ("2Ч 10М", 7800.0),  # регистр и пробелы
        ("1д 2ч", 93600.0),
        ("10м", 600.0),
    ],
)
def test_parse_interval_valid(text, expected):
    assert parse_interval(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "abc", "30", "м30", "5x", "10м5г", "через 5 минут", "-", ".."],
)
def test_parse_interval_invalid(text):
    assert parse_interval(text) is None


# ------------------------------------------------------------------ format_interval


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (5400, "1ч 30м"),
        (86400, "1д"),
        (90000, "1д 1ч"),
        (60, "1м"),
        (45, "45с"),
        (0, "0с"),
    ],
)
def test_format_interval(seconds, expected):
    assert format_interval(seconds) == expected


def test_format_parse_roundtrip():
    for text in ("30м", "1ч30м", "1д 2ч"):
        assert format_interval(parse_interval(text)).replace(" ", "").lower() == text.replace(" ", "").lower()


# ------------------------------------------------------------------ BroadcastStore


def test_add_and_due():
    store = BroadcastStore()
    item = store.add(1, 2, "once", 100.0, now=1000.0)
    assert item is not None and item.id == 1 and item.next_run == 1100.0
    assert store.due(1050.0) == []
    assert store.due(1100.0) == [item]


def test_remove():
    store = BroadcastStore()
    item = store.add(1, 2, "cycle", 60.0, now=0.0)
    assert store.remove(item.id) is True
    assert store.remove(item.id) is False
    assert store.get_all() == []


def test_reschedule_cycle():
    store = BroadcastStore()
    item = store.add(1, 2, "cycle", 300.0, now=0.0)
    store.reschedule(item, now=500.0)
    assert item.next_run == 800.0
    assert store.due(500.0) == []


def test_max_active_limit():
    store = BroadcastStore(max_active=2)
    assert store.add(1, 1, "cycle", 60.0, now=0.0) is not None
    assert store.add(1, 2, "cycle", 60.0, now=0.0) is not None
    assert store.add(1, 3, "cycle", 60.0, now=0.0) is None  # лимит


def test_invalid_mode_rejected():
    store = BroadcastStore()
    assert store.add(1, 1, "weekly", 60.0, now=0.0) is None


def test_ids_increment_and_order():
    store = BroadcastStore()
    a = store.add(1, 1, "once", 60.0, now=0.0)
    b = store.add(1, 2, "once", 60.0, now=0.0)
    assert b.id == a.id + 1
    assert [x.id for x in store.get_all()] == [a.id, b.id]
