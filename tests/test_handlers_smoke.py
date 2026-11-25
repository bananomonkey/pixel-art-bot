"""Сквозной smoke всех колбэков с FakeBot: ловит ошибки времени выполнения
(несуществующие методы/атрибуты), которые py_compile не видит."""

import asyncio
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from aiogram.types import Chat, Message, PhotoSize, User

from bot import admin as bot_admin
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
    assert handlers.store.get(env.chat_id, env.message_id).settings.dither == "fs"


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
    assert entry.settings.dither == "off"
    assert entry.settings.contrast == handlers.DEFAULT_CONTRAST


def test_menu_opens_submenus_without_render(env):
    for view in ("palettes", "adjust", "effects", "frame", "export", "main"):
        env.bot.calls.clear()
        cq = _callback(env, f"menu:{view}")
        _run(handlers.cb_menu(cq, env.bot))
        assert any(name == "edit_message_reply_markup" for name, _ in env.bot.calls), (view, cq.toasts)
        assert handlers.store.get(env.chat_id, env.message_id).view == view


def test_effects(env):
    for data, field in (("eff:crt", "crt"), ("eff:grain", "grain")):
        env.bot.calls.clear()
        cq = _callback(env, data)
        _run(handlers.cb_effect(cq, env.bot))
        _assert_edited(env, cq)
        assert getattr(handlers.store.get(env.chat_id, env.message_id).settings, field) is True


def test_outline_cycles_modes(env):
    _run(handlers.cb_effect(_callback(env, "eff:outline"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.outline == "dark"
    _run(handlers.cb_effect(_callback(env, "eff:outline"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.outline == "volume"
    _run(handlers.cb_effect(_callback(env, "eff:outline"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.outline == "off"


def test_frame_toggles_and_modes(env):
    _run(handlers.cb_frame(_callback(env, "frame:bg"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.cut_bg is True
    # кадр циклится: off -> fill -> pad -> off
    for expected in ("fill", "pad", "off"):
        _run(handlers.cb_frame(_callback(env, "frame:crop"), env.bot))
        assert handlers.store.get(env.chat_id, env.message_id).settings.crop_mode == expected
    # бесшовность циклится: off -> blend -> mirror -> off
    for expected in ("blend", "mirror", "off"):
        _run(handlers.cb_frame(_callback(env, "frame:seamless"), env.bot))
        assert handlers.store.get(env.chat_id, env.message_id).settings.seamless == expected


def test_grid_plus_minus(env):
    """Сетка −/+ в подменю: сразу перерисовывает картинку в custom-уровень."""
    _run(handlers.cb_size_custom(_callback(env, "size:custom", uid=2), env.bot))
    env.bot.calls.clear()
    _run(handlers.cb_grid(_callback(env, "grid:p", uid=2), env.bot))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings.pixel_size == "custom" and entry.settings.custom_grid == 104
    assert any(name == "edit_message_media" for name, _ in env.bot.calls)
    _run(handlers.cb_grid(_callback(env, "grid:m", uid=2), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).settings.custom_grid == 96


def test_my_grid_applies_to_new_photos(env):
    """«💾 Сделать моим»: новое фото сразу в сохранённом уровне."""
    handlers.my_grids.put(3, 40)
    handlers.pack_sessions.clear()
    photo = PhotoSize(file_id="gg", file_unique_id="u", width=100, height=100)
    message = Message(
        message_id=9,
        chat=Chat(id=env.chat_id, type="private"),
        date=datetime.now(timezone.utc),
        from_user=User(id=3, is_bot=False, first_name="t"),
        photo=[photo],
    )
    _run(handlers.on_photo(message, env.bot))
    assert any(c[0] == "send_photo" for c in env.bot.calls)
    entry = handlers.store.get(env.chat_id, 99)  # FakeBot.send_photo вернул message_id=99
    assert entry is not None
    assert entry.settings.pixel_size == "custom" and entry.settings.custom_grid == 40


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


def test_admin_capture_does_not_block_grid_flow(env):
    """Регрессия: админский перехватчик съедал текст админа до флоу сетки —
    «📏 Свой» не работал для админов. Теперь вне админ-флоу всегда SkipHandler."""
    from aiogram.dispatcher.event.bases import SkipHandler

    admin_uid = 424242  # ADMIN_IDS из conftest
    _open_grid_menu(env, uid=admin_uid)
    assert admin_uid in handlers._grid_flow

    answered = []

    class FakeMessage(SimpleNamespace):
        async def answer(self, text):
            answered.append(text)

    msg = FakeMessage(from_user=SimpleNamespace(id=admin_uid), text="256")
    # порядок как в проде: admin_capture стоит в цепочке раньше grid_capture
    with pytest.raises(SkipHandler):
        _run(bot_admin.admin_capture(msg))
    _run(handlers.grid_capture(msg, env.bot))
    entry = handlers.store.get(env.chat_id, env.message_id)
    assert entry.settings.pixel_size == "custom" and entry.settings.custom_grid == 256
    assert not any("Нет активного действия" in (t or "") for t in answered), answered
    assert any("Свой уровень" in (t or "") for t in answered), answered


def test_admin_capture_does_not_block_palette_flow(env):
    from aiogram.dispatcher.event.bases import SkipHandler

    admin_uid = 424242
    handlers._palette_flow.add(admin_uid)
    answered = []

    class FakeMessage(SimpleNamespace):
        async def answer(self, text):
            answered.append(text)

    msg = FakeMessage(
        from_user=SimpleNamespace(id=admin_uid),
        text="#FF004D 00E436 7E2553",
        media_group_id=None,
        photo=None,
        document=None,
    )
    with pytest.raises(SkipHandler):
        _run(bot_admin.admin_capture(msg))
    _run(handlers.palette_capture(msg, bot=None))
    assert admin_uid not in handlers._palette_flow
    palette = handlers.custom_palettes.get(admin_uid)
    assert palette is not None and len(palette) == 3
    assert not any("Нет активного действия" in (t or "") for t in answered), answered
    handlers.custom_palettes.drop(admin_uid)


def _open_grid_menu(env, uid=1):
    """Открыть подменю «📏 Свой» и запустить ввод числа."""
    toasts = []

    def _cq(data):
        cq = _callback(env, data, uid=uid)
        async def answer(text=None, show_alert=False):
            toasts.append(text)
        cq.answer = answer
        return cq

    _run(handlers.cb_size_custom(_cq("size:custom"), env.bot))
    assert handlers.store.get(env.chat_id, env.message_id).view == "grid"
    env.bot.calls.clear()
    _run(handlers.cb_grid(_cq("grid:ask"), env.bot))
    assert uid in handlers._grid_flow
    return toasts


def test_grid_menu_opens_and_ask(env):
    toasts = _open_grid_menu(env)
    assert any("число" in (t or "") for t in toasts), toasts


def test_grid_flow_applies_number(env):
    """📏 Свой → ввод числа → свой уровень применён к той же картинке."""
    _open_grid_menu(env, uid=1)
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
    _open_grid_menu(env, uid=1)
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


def test_pack_session_flow(env):
    """/pack -> фото идут в буфер без пикселизации -> /packdone отдаёт zip+превью."""
    handlers.pack_sessions.clear()
    uid = 21

    def _msg(mid, file_id="p1"):
        return Message(
            message_id=mid,
            chat=Chat(id=env.chat_id, type="private"),
            date=datetime.now(timezone.utc),
            from_user=User(id=uid, is_bot=False, first_name="t"),
            photo=[PhotoSize(file_id=file_id, file_unique_id="u", width=100, height=100)],
        )

    async def _noop(*a, **kw):
        return None

    start = SimpleNamespace(chat=SimpleNamespace(id=env.chat_id), from_user=SimpleNamespace(id=uid))
    start.answer = _noop
    _run(handlers.cmd_pack(start))
    assert uid in handlers.pack_sessions
    # фото (в т.ч. «альбом») добавляются без рендера
    env.bot.calls.clear()
    for i in range(3):
        _run(handlers.on_photo(_msg(50 + i, f"pf{i}"), env.bot))
    assert len(handlers.pack_sessions[uid]) == 3
    assert not any(name == "send_photo" for name, *_ in env.bot.calls)  # пикселизации не было
    # сборка
    done = SimpleNamespace(chat=SimpleNamespace(id=env.chat_id), from_user=SimpleNamespace(id=uid))
    done.answer = _noop
    env.bot.calls.clear()
    _run(handlers.cmd_packdone(done, env.bot))
    assert any(name == "send_document" for name, _ in env.bot.calls)
    assert any(name == "send_photo" for name, *_ in env.bot.calls)  # превью
    assert uid not in handlers.pack_sessions  # сессия закрыта


def test_pack_session_cancel(env):
    handlers.pack_sessions.clear()
    uid = 22
    handlers.pack_sessions[uid] = [b"x", b"y"]
    cancelled = SimpleNamespace(chat=SimpleNamespace(id=env.chat_id), from_user=SimpleNamespace(id=uid))

    async def _answer(text=None, **kw):
        return None

    cancelled.answer = _answer
    _run(handlers.cmd_packcancel(cancelled))
    assert uid not in handlers.pack_sessions


def test_pack_settings_panel(env):
    handlers.pack_settings.drop(31)

    async def _edit(*a, **kw):
        return None

    cq = SimpleNamespace(
        data="pk:grid:32",
        from_user=SimpleNamespace(id=31),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=env.chat_id),
            edit_text=_edit,
        ),
    )

    async def _answer(text=None, show_alert=False):
        cq.toast = text

    cq.answer = _answer
    _run(handlers.cb_pack_set(cq))
    settings = handlers.pack_settings.get(31)
    assert settings is not None and settings.grid == 32
    assert settings.dither == "off"
    cq.data = "pk:seam:mirror"
    _run(handlers.cb_pack_set(cq))
    assert handlers.pack_settings.get(31).seam == "mirror"
    cq.data = "pk:dither"
    _run(handlers.cb_pack_set(cq))
    assert handlers.pack_settings.get(31).dither == "fs"
    cq.data = "pk:autotile"
    _run(handlers.cb_pack_set(cq))
    assert handlers.pack_settings.get(31).autotile is True


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
