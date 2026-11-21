"""Тесты хранилища состояния: LRU-вытеснение и TTL."""

import time

from bot.state import COLOR_STEPS, Settings, StateStore, with_changes


def test_put_get_roundtrip():
    store = StateStore()
    store.put(1, 2, b"source", Settings())
    entry = store.get(1, 2)
    assert entry is not None
    assert entry.source == b"source"
    assert entry.settings == Settings()


def test_missing_entry():
    store = StateStore()
    assert store.get(1, 2) is None


def test_ttl_expiry():
    store = StateStore(ttl_seconds=0.05)
    store.put(1, 2, b"src", Settings())
    assert store.get(1, 2) is not None
    time.sleep(0.12)
    assert store.get(1, 2) is None  # истёк и удалён


def test_sliding_ttl_refreshed_on_get():
    store = StateStore(ttl_seconds=0.1)
    store.put(1, 2, b"src", Settings())
    time.sleep(0.06)
    assert store.get(1, 2) is not None  # обновил TTL
    time.sleep(0.06)
    assert store.get(1, 2) is not None  # всё ещё жива благодаря скользящему TTL
    time.sleep(0.15)
    assert store.get(1, 2) is None


def test_lru_eviction_evicts_least_recently_used():
    store = StateStore(max_entries=3)
    store.put(1, 1, b"a", Settings())
    store.put(1, 2, b"b", Settings())
    store.put(1, 3, b"c", Settings())
    assert store.get(1, 1) is not None  # (1,1) стал самым свежим
    store.put(1, 4, b"d", Settings())  # вытесняется (1,2) — самый давний
    assert store.get(1, 2) is None
    assert store.get(1, 3) is not None
    assert store.get(1, 4) is not None
    assert store.get(1, 1) is not None
    assert len(store) == 3


def test_put_same_key_updates_and_does_not_evict():
    store = StateStore(max_entries=2)
    store.put(1, 1, b"a", Settings())
    store.put(1, 2, b"b", Settings())
    store.put(1, 1, b"a2", Settings())
    assert len(store) == 2
    assert store.get(1, 1).source == b"a2"


def test_purge_expired_removes_only_expired():
    store = StateStore(ttl_seconds=0.05)
    store.put(1, 1, b"a", Settings())
    store.put(1, 2, b"b", Settings())
    time.sleep(0.12)
    removed = store.purge_expired()
    assert removed == 2
    assert len(store) == 0
    assert store.purge_expired() == 0


def test_drop():
    store = StateStore()
    store.put(1, 1, b"a", Settings())
    store.drop(1, 1)
    assert store.get(1, 1) is None
    store.drop(1, 1)  # повторно — без ошибок


def test_settings_defaults_and_changes():
    settings = Settings()
    assert settings.pixel_size == "medium"
    assert settings.palette == "auto"
    assert settings.n_colors == 16
    assert settings.dither is False
    changed = with_changes(settings, palette="gameboy", dither=True)
    assert changed.palette == "gameboy" and changed.dither is True
    assert settings.palette == "auto" and settings.dither is False  # оригинал не тронут


def test_color_steps_row():
    assert COLOR_STEPS == (8, 16, 32, 64)
