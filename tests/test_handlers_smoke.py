"""Сквозной smoke всех колбэков с FakeBot: ловит ошибки времени выполнения
(несуществующие методы/атрибуты), которые py_compile не видит."""

import asyncio
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from aiogram.types import Chat, Message, PhotoSize, User

from bot import handlers
from bot.admin import stats
from bot.state import Settings

SOURCE = bytes(2000)  # «сжатый исходник» — рендер по нему не пойдёт? пойдёт: PIL откроет?
# нет: для хэндлеров, вызывающих render, нужен настоящий исходник


def _real_source():
    from bot.pixelart import build_demo_image, encode_source

    return encode_source(build_demo_image())


@pytest.fixture()
def env(monkeypatch):
    """Чистый store + запись-исходник; возвращает FakeBot и помощники."""
    handlers.store._entries.clear()
    handlers._last_photo_at.clear()
    chat_id, message_id = 555, 42
    source = _real_source()
    handlers.store.put(chat_id, message_id, source, Settings())
    bot = FakeBot()
    return SimpleNamespace(bot=bot, chat_id=chat_id, message_id=message_id, source=source)


class FakeBot:
    """Фейковый Bot: фиксирует вызовы, возвращает минимальные объекты."""

    def __init__(self):
        self.calls = []

    async def edit_message_media(self, **kwargs):
        self.calls.append(("edit_message_media", kwargs))
        return True

    async def edit_message_reply_markup(self, **kwargs):
        self.calls.append(("edit_message_reply_markup", kwargs))
        return True

    async def send_photo(self, chat_id, photo, **kwargs):
        self.calls.append(("send_photo", chat_id))
        return SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=chat_id),
            photo=[SimpleNamespace(file_id="ph1")],
        )

    async def send_document(self, chat_id, document, **kwargs):
        self.calls.append(("send_document", chat_id))
        return Message(
            message_id=100,
            chat=Chat(id=chat_id, type="private"),
            date=datetime.now(timezone.utc),
        )

    async def send_message(self, chat_id, text, **kwargs):
        self.calls.append(("send_message", text))
        return Message(
            message_id=102,
            chat=Chat(id=chat_id, type="private"),
            date=datetime.now(timezone.utc),
        )

    async def get_file(self, file_id):
        self.calls.append(("get_file", file_id))
        return SimpleNamespace(file_path="photos/file.jpg")

    async def download_file(self, file_path, destination):
        self.calls.append(("download_file",))
        from bot.pixelart import build_demo_image, encode_source

        destination.write(encode_source(build_demo_image()))
        return destination

    async def copy_message(self, **kwargs):
        self.calls.append(("copy_message", kwargs.get("chat_id")))
        return Message(
            message_id=101,
            chat=Chat(id=1, type="private"),
            date=datetime.now(timezone.utc),
        )


def _callback(env, data, uid=1):
    """Фейковый CallbackQuery с настоящим Message из стора."""
    message = Message(
        message_id=env.message_id,
        chat=Chat(id=env.chat_id, type="private"),
        date=datetime.now(timezone.utc),
        from_user=User(id=uid, is_bot=False, first_name="t"),
    )
    toasts = []

    async def answer(text=None, show_alert=False):
        toasts.append((text, show_alert))

    cq = SimpleNamespace(
        data=data, message=message, from_user=User(id=uid, is_bot=False, first_name="t"),
        answer=answer, toasts=toasts,
    )
    return cq


def _run(coro):
    return asyncio.run(coro)


def _assert_edited(env, cq):
    """Категорийная кнопка: медиа перерисовано, тост — не ошибка."""
    assert any(name == "edit_message_media" for name, _ in env.bot.calls), cq.toasts
    assert all("пошло не так" not in (t or "") for t, _ in cq.toasts), cq.toasts


def test_size(env):
    cq = _callback(env, "size:huge")
    _run(handlers.cb_size(cq, env.bot))
    _assert_edited(env, cq)
    assert handlers.store.get(env.chat_id, env.message_id).settings.pixel_size == "huge"


def test_palette(env):
    cq = _callback(env, "pal:gameboy")
    _run(handlers.cb_palette(cq, env.bot))
    _assert_edited(env, cq)
    assert handlers.store.get(env.chat_id, env.message_id).settings.palette == "gameboy"


def test_dither(env):
    cq = _callback(env, "dither")
    _run(handlers.cb_dither(cq, env.bot))
    _assert_edited(env, cq)
    assert handlers.store.get(env.chat_id, env.message_id).settings.dither is True


def test_more_less_colors(env):
    cq = _callback(env, "more")
    _run(handlers.cb_more_colors(cq, env.bot))
    _assert_edited(env, cq)
    assert handlers.store.get(env.chat_id, env.message_id).settings.n_colors == 32
    cq = _callback(env, "less")
    _run(handlers.cb_less_colors(cq, env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.n_colors == 16


def test_adjustments(env):
    cq = _callback(env, "adj:con:p")
    _run(handlers.cb_adjust(cq, env.bot))
    _assert_edited(env, cq)
    assert handlers.store.get(env.chat_id, env.message_id).settings.contrast == pytest.approx(1.3)


def test_adjust_reset(env):
    _run(handlers.cb_adjust(_callback(env, "adj:con:p"), env.bot))
    cq = _callback(env, "adj_reset")
    _run(handlers.cb_adjust_reset(cq, env.bot))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings.dither is False
    assert entry.settings.contrast == handlers.DEFAULT_CONTRAST


def test_menu_opens_submenus_without_render(env):
    for view in ("palettes", "adjust", "effects", "frame", "export", "main"):
        env.bot.calls.clear()
        cq = _callback(env, f"menu:{view}")
        _run(handlers.cb_menu(cq, env.bot))
        assert any(name == "edit_message_reply_markup" for name, _ in env.bot.calls), (view, cq.toasts)
        assert handlers.store.get(env.chat_id, env.message_id).view == view


def test_effects(env):
    for data, field in (("eff:outline", "outline"), ("eff:crt", "crt"), ("eff:grain", "grain")):
        env.bot.calls.clear()
        cq = _callback(env, data)
        _run(handlers.cb_effect(cq, env.bot))
        _assert_edited(env, cq)
        assert getattr(handlers.store.get(env.chat_id, env.message_id).settings, field) is True


def test_frame_toggles_and_steps(env):
    _run(handlers.cb_frame(_callback(env, "frame:cutbg"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.cut_bg is True
    _run(handlers.cb_frame(_callback(env, "frame:crop"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.crop_object is True
    before = handlers.store.get(env.chat_id, env.message_id).settings.bg_tolerance
    _run(handlers.cb_frame(_callback(env, "frame:bgtol:p"), env.bot))
    after = handlers.store.get(env.chat_id, env.message_id).settings.bg_tolerance
    assert after == before + handlers.BG_TOLERANCE_STEP
    _run(handlers.cb_frame(_callback(env, "frame:pad:p"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.padding > handlers.DEFAULT_PADDING


def test_export_toggles(env):
    cq = _callback(env, "export:sticker")
    _run(handlers.cb_export(cq, env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.sticker is True
    assert any(name == "edit_message_reply_markup" for name, _ in env.bot.calls)
    cq = _callback(env, "export:watermark")
    _run(handlers.cb_export(cq, env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.watermark is True


def test_download_sticker_and_png(env):
    _run(handlers.cb_export(_callback(env, "export:sticker"), env.bot))
    cq = _callback(env, "download")
    _run(handlers.cb_download(cq, env.bot))
    assert any(name == "send_document" for name, _ in env.bot.calls), cq.toasts
    handlers.store.get(env.chat_id, env.message_id).settings.sticker = False
    env.bot.calls.clear()
    _run(handlers.cb_download(_callback(env, "download"), env.bot))
    assert any(name == "send_document" for name, _ in env.bot.calls)


def test_random(env):
    cq = _callback(env, "random")
    _run(handlers.cb_random(cq, env.bot))
    _assert_edited(env, cq)


def test_fav_save(env):
    cq = _callback(env, "fav_save")
    _run(handlers.cb_fav_save(cq))
    assert handlers.favorites.get(1) is not None


def test_fav_clear(env):
    handlers.favorites.put(1, Settings())
    _run(handlers.cb_fav_clear(_callback(env, "fav_clear", uid=1)))
    assert handlers.favorites.get(1) is None


def test_demo(env):
    handlers._demo_source = None
    cq = _callback(env, "demo", uid=7)
    _run(handlers.cb_demo(cq, env.bot))
    assert any(name == "send_photo" for name, _ in env.bot.calls), cq.toasts


def test_reset_all(env):
    _run(handlers.cb_frame(_callback(env, "frame:cutbg"), env.bot))
    cq = _callback(env, "reset_all")
    _run(handlers.cb_reset_all(cq, env.bot))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings == Settings()
    assert entry.view == "main"


def test_size_custom_flow(env):
    """📏 Свой: кнопка ставит флоу, число применяет свой уровень к той же картинке."""
    cq = _callback(env, "size:custom")
    _run(handlers.cb_size_custom(cq))
    assert 1 in handlers._grid_flow

    answered = []

    class FakeMessage(SimpleNamespace):
        async def answer(self, text):
            answered.append(text)

    env.bot.calls.clear()
    msg = FakeMessage(from_user=SimpleNamespace(id=1), text="48")
    _run(handlers.grid_capture(msg, env.bot))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings.pixel_size == "custom" and entry.settings.custom_grid == 48
    assert any(name == "edit_message_media" for name, _ in env.bot.calls), answered
    assert 1 not in handlers._grid_flow
    assert answered and "Свой уровень" in answered[-1]


def test_size_custom_flow_invalid_keeps_asking(env):
    _run(handlers.cb_size_custom(_callback(env, "size:custom")))
    answered = []

    class FakeMessage(SimpleNamespace):
        async def answer(self, text):
            answered.append(text)

    msg = FakeMessage(from_user=SimpleNamespace(id=1), text="abc")
    _run(handlers.grid_capture(msg, env.bot))
    assert answered and "Нужно целое число" in answered[0]
    assert 1 in handlers._grid_flow  # флоу не сброшен — ждём корректное число


def test_grid_command_applies_to_recent(env):
    handlers._recent_entry[env.chat_id] = (env.chat_id, env.message_id)
    from aiogram.types import Chat, Message, User

    message = Message(
        message_id=8,
        chat=Chat(id=env.chat_id, type="private"),
        date=datetime.now(timezone.utc),
        from_user=User(id=1, is_bot=False, first_name="t"),
    )
    command = SimpleNamespace(args="64")
    _run(handlers.cmd_grid(message, env.bot, command))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings.pixel_size == "custom" and entry.settings.custom_grid == 64
    assert any(
        name == "send_message" and "Свой уровень" in text
        for name, text in env.bot.calls
    )


def test_all_bot_methods_exist():
    """Регрессия: каждый bot.<метод> из кода хэндлеров существует у aiogram."""
    import inspect
    import re

    from aiogram import Bot

    from bot import admin as admin_module

    source = inspect.getsource(handlers) + inspect.getsource(admin_module)
    used = set(re.findall(r"\bbot\.([a-z_]+)\(", source))
    assert used, "регулярка перестала находить вызовы"
    missing = [m for m in sorted(used) if not hasattr(Bot, m)]
    assert not missing, f"несуществующие методы Bot: {missing}"


def test_photo_flow_end_to_end(env):
    """Полный путь нового фото: скачивание -> рендер -> отправка с клавиатурой."""
    photos = [
        PhotoSize(file_id="small", file_unique_id="s1", width=100, height=100),
        PhotoSize(file_id="big", file_unique_id="s2", width=800, height=600),
    ]
    message = Message(
        message_id=7,
        chat=Chat(id=555, type="private"),
        date=datetime.now(timezone.utc),
        from_user=User(id=9, is_bot=False, first_name="u"),
        photo=photos,
    )
    _run(handlers.on_photo(message, env.bot))
    assert any(name == "send_photo" for name, *_ in env.bot.calls)
    assert stats["photos"] >= 1
