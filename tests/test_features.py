"""Тесты пользовательских функций: hex-палитры, коррекции, кадры, демо, хранилища."""

import io

import numpy as np
import pytest
from PIL import Image

from bot.palettes import parse_hex_colors
from bot.pixelart import (
    build_demo_image,
    encode_source,
    extract_first_frame,
    extract_palette,
    make_pixel_art,
)
from bot.state import ChatsRegistry, PerUserStore
from bot.broadcast import BroadcastStore, TemplateStore, parse_target_days


def _gradient(w=400, h=300):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return Image.fromarray(arr, "RGB")


# ------------------------------------------------------------------ parse_hex_colors


def test_parse_hex_colors_6_and_3_digit():
    assert parse_hex_colors("#FF004D 0f380f,#abc") == ((255, 0, 77), (15, 56, 15), (170, 187, 204))


def test_parse_hex_colors_dedup_and_mixed_separators():
    text = "ff004d\nFF004D; #ff004d,,00E436"
    assert parse_hex_colors(text) == ((255, 0, 77), (0, 228, 54))


def test_parse_hex_colors_garbage():
    assert parse_hex_colors("привет как дела") == ()
    assert parse_hex_colors("") == ()
    assert parse_hex_colors("#12345") == ()  # 5 цифр — не цвет
    assert parse_hex_colors("#abcd") == ()  # 4 цифры — не разбираем как #abc


def test_parse_hex_colors_newlines():
    assert parse_hex_colors("ffffff\n000000") == ((255, 255, 255), (0, 0, 0))


# ------------------------------------------------------------------ extract_palette и custom


def test_extract_palette_from_gradient():
    colors = extract_palette(_gradient(), 16)
    assert 2 <= len(colors) <= 16
    for rgb in colors:
        assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)


def test_custom_palette_render_membership():
    img = _gradient()
    colors = extract_palette(img, 8)
    result = make_pixel_art(img, palette="custom", palette_colors=colors, n_colors=8)
    allowed = set(colors)
    pixels = set(map(tuple, np.asarray(result.convert("RGB")).reshape(-1, 3).tolist()))
    assert pixels <= allowed


def test_custom_palette_without_colors_raises():
    with pytest.raises(ValueError):
        make_pixel_art(_gradient(), palette="custom")


# ------------------------------------------------------------------ коррекции


def test_adjustments_change_result():
    img = _gradient()
    low = np.asarray(make_pixel_art(img, contrast=0.6))
    high = np.asarray(make_pixel_art(img, contrast=2.0))
    assert low.tobytes() != high.tobytes()
    sat_low = np.asarray(make_pixel_art(img, saturation=0.4))
    sat_high = np.asarray(make_pixel_art(img, saturation=2.2))
    assert sat_low.tobytes() != sat_high.tobytes()
    sharp_low = np.asarray(make_pixel_art(img, sharpness=0.5))
    sharp_high = np.asarray(make_pixel_art(img, sharpness=3.0))
    assert sharp_low.tobytes() != sharp_high.tobytes()


def test_demo_image_shape():
    demo = build_demo_image()
    assert demo.size == (1024, 768)
    assert demo.mode == "RGB"
    # кодируется в исходник и обрабатывается всеми путями
    source = encode_source(demo)
    result = make_pixel_art(Image.open(io.BytesIO(source)).convert("RGB"), palette="gameboy")
    assert max(result.size) >= 1000


# ------------------------------------------------------------------ первый кадр анимации


def _animated_gif_bytes():
    bio = io.BytesIO()
    Image.new("RGB", (60, 40), (220, 40, 60)).save(
        bio, "GIF", save_all=True, append_images=[Image.new("RGB", (60, 40), (30, 200, 80))],
        duration=100, loop=0,
    )
    return bio.getvalue()


def test_extract_first_frame_gif():
    frame = extract_first_frame(_animated_gif_bytes())
    img = Image.open(io.BytesIO(frame))
    assert img.format == "PNG" and not getattr(img, "is_animated", False)


def test_extract_first_frame_then_pixelize():
    src = encode_source(Image.open(io.BytesIO(extract_first_frame(_animated_gif_bytes()))).convert("RGB"))
    result = make_pixel_art(Image.open(io.BytesIO(src)), palette="pico8")
    assert max(result.size) >= 1000


def test_extract_first_frame_mp4():
    try:
        import imageio.v2 as imageio  # noqa: F401

        imageio.get_writer  # noqa: B018
    except ImportError:
        pytest.skip("imageio недоступен")
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    try:
        writer = imageio.get_writer(path, format="ffmpeg", mode="I", macro_block_size=1)
        writer.append_data(np.full((64, 64, 3), 200, dtype=np.uint8))
        writer.append_data(np.full((64, 64, 3), 20, dtype=np.uint8))
        writer.close()
        frame = extract_first_frame(open(path, "rb").read())
        img = Image.open(io.BytesIO(frame))
        assert img.format == "PNG"
        arr = np.asarray(img.convert("RGB"))
        assert abs(arr.mean() - 200.0) < 30  # именно ПЕРВЫЙ (светлый) кадр
    finally:
        os.remove(path)


# ------------------------------------------------------------------ PerUserStore


def test_per_user_store_put_get_drop():
    store = PerUserStore()
    store.put(1, "a")
    assert store.get(1) == "a"
    store.drop(1)
    assert store.get(1) is None
    store.drop(1)  # повторно — без ошибок


def test_per_user_store_lru():
    store = PerUserStore(max_entries=2)
    store.put(1, "a")
    store.put(2, "b")
    assert store.get(1) == "a"  # поднимаем наверх
    store.put(3, "c")  # вытесняется (2)
    assert store.get(2) is None
    assert store.get(3) == "c"
    assert len(store) == 2


# ------------------------------------------------------------------ ChatsRegistry.recent_ids


def test_recent_ids_filters_by_recency():
    registry = ChatsRegistry()
    now = 1_000_000.0
    registry.touch(1, now=now - 100.0)      # активен недавно
    registry.touch(2, now=now - 100_000.0)  # активен больше суток назад
    registry.touch(3, now=now - 10.0)
    assert set(registry.recent_ids(now, days=1)) == {1, 3}
    assert set(registry.recent_ids(now, days=30)) == {1, 2, 3}


def test_touch_updates_timestamp():
    registry = ChatsRegistry()
    registry.touch(1, now=10.0)
    registry.touch(1, now=999_000.0)  # позже — метка обновилась
    assert set(registry.recent_ids(1_000_000.0, days=1)) == {1}


# ------------------------------------------------------------------ шаблоны и target_days


def test_template_store_roundtrip():
    store = TemplateStore(max_items=3)
    a = store.add(1, 10, now=100.0)
    b = store.add(1, 11, now=200.0)
    assert b.id == a.id + 1
    assert [t.id for t in store.get_all()] == [a.id, b.id]
    assert store.get(a.id) is not None
    assert store.remove(a.id) is True
    assert store.get(a.id) is None


def test_template_store_limit():
    store = TemplateStore(max_items=2)
    assert store.add(1, 1, now=0.0) is not None
    assert store.add(1, 2, now=0.0) is not None
    assert store.add(1, 3, now=0.0) is None


def test_broadcast_target_days_stored():
    store = BroadcastStore()
    item = store.add(1, 2, "once", 60.0, now=0.0, target_days=7.0)
    assert item.target_days == 7.0
    plain = store.add(1, 3, "cycle", 60.0, now=0.0)
    assert plain.target_days is None


# ------------------------------------------------------------------ регрессия: перехват вне флоу


def test_palette_capture_skips_outside_flow():
    """Фото пользователя вне флоу палитры должно пропускаться дальше (SkipHandler),
    а не падать с NameError — регрессия на потерянный импорт."""
    import asyncio
    from types import SimpleNamespace

    from aiogram.dispatcher.event.bases import SkipHandler

    from bot import handlers

    fake = SimpleNamespace(
        from_user=SimpleNamespace(id=424242),
        media_group_id=None,
        text=None,
        photo=None,
        document=None,
    )
    with pytest.raises(SkipHandler):
        asyncio.run(handlers.palette_capture(fake, bot=None))


def test_palette_capture_consumes_text_in_flow():
    """Внутри флоу текст пользователя перехватывается: мусорный hex — подсказка."""
    import asyncio
    from types import SimpleNamespace

    from bot import handlers

    handlers._palette_flow.add(777)
    answered = []

    class FakeMessage(SimpleNamespace):
        async def answer(self, text):
            answered.append(text)

    fake = FakeMessage(
        from_user=SimpleNamespace(id=777),
        media_group_id=None,
        text="привет как дела",
        photo=None,
        document=None,
    )
    try:
        asyncio.run(handlers.palette_capture(fake, bot=None))
    finally:
        handlers._palette_flow.discard(777)
    assert answered and "Не нашёл цветов" in answered[0]


def test_parse_target_days():
    assert parse_target_days("7") == 7.0
    assert parse_target_days(" 1 ") == 1.0
    assert parse_target_days("0") is None
    assert parse_target_days("abc") is None
    assert parse_target_days("999") is None  # больше TARGETED_MAX_DAYS
