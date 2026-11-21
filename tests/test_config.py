"""Тесты конфигурации: разбор списка админов из переменной окружения."""

from bot.config import parse_admin_ids


def test_parse_admin_ids_basic():
    assert parse_admin_ids("123, 456") == (123, 456)


def test_parse_admin_ids_empty():
    assert parse_admin_ids("") == ()
    assert parse_admin_ids(None) == ()


def test_parse_admin_ids_garbage_ignored():
    assert parse_admin_ids("abc,789,x1") == (789,)
    assert parse_admin_ids("0,-5,42") == (42,)


def test_parse_admin_ids_semicolons_and_duplicates():
    assert parse_admin_ids("111; 222;111") == (111, 222)
    assert parse_admin_ids("  7  7 ") == (7,)
