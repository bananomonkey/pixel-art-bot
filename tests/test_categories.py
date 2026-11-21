"""Тесты топ-5 функций: вырезка фона, кадр, эффекты, водяной знак, стикер,
новые палитры и клавиатура-категории."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bot.handlers import _keyboard
from bot.palettes import FIXED_PALETTES
from bot.pixelart import (
    _flood_from_border,
    make_pixel_art,
    post_effects,
    render,
    resize_for_sticker,
)
from bot.state import Settings, StateStore


def _object_on_white():
    img = Image.new("RGB", (300, 240), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((80, 60, 220, 180), fill=(220, 40, 60))
    return img


def _png_bytes(img):
    bio = io.BytesIO()
    img.save(bio, "PNG")
    return bio.getvalue()


# ------------------------------------------------------------------ палитры ZX/Sweetie/1-bit


def test_new_palettes_membership():
    img = _object_on_white()
    for key in ("zx", "sweetie", "bw1"):
        result = make_pixel_art(img, palette=key)
        pixels = set(map(tuple, np.asarray(result.convert("RGB")).reshape(-1, 3).tolist()))
        assert pixels <= set(FIXED_PALETTES[key]), key


# ------------------------------------------------------------------ вырезка фона


def test_flood_from_border_synthetic():
    close = np.ones((10, 10), dtype=bool)
    close[4:7, 4:7] = False  # объект в центре
    flooded = _flood_from_border(close)
    assert flooded[0, 0] and flooded[9, 9]
    assert not flooded[5, 5]  # объект не затронут
    empty = np.zeros((5, 5), dtype=bool)
    assert not _flood_from_border(empty).any()


def test_cut_background_alpha():
    art = make_pixel_art(_object_on_white(), pixel_size="huge", palette="pico8",
                         cut_bg=True, bg_tolerance=48)
    arr = np.asarray(art)
    assert art.mode == "RGBA"
    assert arr[0, 0, 3] == 0           # угол — фон удалён
    h, w = arr.shape[:2]
    assert arr[h // 2, w // 2, 3] == 255  # центр — объект непрозрачный
    assert tuple(arr[h // 2, w // 2, :3]) in set(FIXED_PALETTES["pico8"])


def test_cut_background_tolerance_changes_result():
    # градиентный фон: низкий порог режет только светлую часть, высокий — почти весь
    img = Image.new("RGB", (200, 200))
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[..., :] = np.linspace(245, 150, 200, dtype=np.uint8)[:, None]
    draw = ImageDraw.Draw(img)
    arr_img = Image.fromarray(arr)
    img.paste(arr_img, (0, 0))
    draw.rectangle((80, 80, 120, 120), fill=(200, 30, 50))
    a = np.asarray(make_pixel_art(img, cut_bg=True, bg_tolerance=16))
    b = np.asarray(make_pixel_art(img, cut_bg=True, bg_tolerance=140))
    assert a[..., 3].sum() != b[..., 3].sum()


# ------------------------------------------------------------------ кадр и объект


def test_crop_object_fills_frame():
    art = make_pixel_art(_object_on_white(), pixel_size="medium", palette="pico8",
                         crop_object=True, padding=0)
    arr = np.asarray(art.convert("RGB"))
    h, w = arr.shape[:2]
    edge_pixel = arr[h // 2, 2]  # левый край середины — объект, а не белый фон
    assert edge_pixel.mean() < 240


def test_padding_changes_composition():
    a0 = make_pixel_art(_object_on_white(), crop_object=True, padding=0)
    a40 = make_pixel_art(_object_on_white(), crop_object=True, padding=40)
    assert a0.size != a40.size


# ------------------------------------------------------------------ эффекты и водяной знак


def test_outline_changes_result():
    plain = np.asarray(make_pixel_art(_object_on_white(), palette="gameboy", cut_bg=True))
    edged = np.asarray(make_pixel_art(_object_on_white(), palette="gameboy", cut_bg=True, outline=True))
    assert plain.tobytes() != edged.tobytes()
    # обводка не выводит цвета за пределы палитры (потемнение умножением)
    pixels = set(map(tuple, edged[..., :3].reshape(-1, 3).tolist()))
    # тёмные пиксели обводки — умножение на 0.45, допускаем только округлённые варианты палитры
    allowed = set(FIXED_PALETTES["gameboy"])
    assert pixels <= allowed | {tuple(int(c * 0.45) for c in col) for col in allowed}


def test_post_effects_change_result():
    base = make_pixel_art(_object_on_white(), palette="pico8")
    base_bytes = np.asarray(base).tobytes()
    for kwargs in ({"crt": True}, {"grain": True}, {"watermark_text": "@pixelartphoto_bot"}):
        fx = np.asarray(post_effects(base, **kwargs))
        assert fx.tobytes() != base_bytes, kwargs


def test_render_flatten_removes_alpha():
    src = _png_bytes(_object_on_white())
    png = render(src, Settings(cut_bg=True), flatten=True)
    assert Image.open(io.BytesIO(png)).mode == "RGB"


def test_render_keeps_alpha_for_export():
    src = _png_bytes(_object_on_white())
    png = render(src, Settings(cut_bg=True), flatten=False)
    assert Image.open(io.BytesIO(png)).mode == "RGBA"


def test_watermark_setting_applied():
    src = _png_bytes(_object_on_white())
    off = np.asarray(Image.open(io.BytesIO(render(src, Settings()))))
    on = np.asarray(Image.open(io.BytesIO(render(src, Settings(watermark=True), flatten=True))))
    assert off.tobytes() != on.tobytes()


# ------------------------------------------------------------------ стикер


def test_resize_for_sticker_512():
    src = _png_bytes(_object_on_white())
    png = render(src, Settings(sticker=True, cut_bg=True), flatten=False)
    sticker = Image.open(io.BytesIO(resize_for_sticker(png)))
    assert max(sticker.size) <= 512
    assert sticker.mode == "RGBA"


def test_resize_for_sticker_small_image_untouched():
    img = Image.new("RGB", (100, 50), (10, 10, 10))
    png = _png_bytes(img)
    out = Image.open(io.BytesIO(resize_for_sticker(png)))
    assert out.size == (100, 50)


# ------------------------------------------------------------------ клавиатура-категории и view


def test_main_keyboard_categories():
    kb = _keyboard(Settings(), "main")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    for need in ("🎨 Палитры", "🎛 Коррекции", "✨ Эффекты", "🖼 Кадр", "📤 Экспорт",
                 "🎲 Другой стиль", "🎲 Демо", "↺ Сброс", "Средний ✓"):
        assert any(need in label for label in labels), need
    assert len(kb.inline_keyboard) <= 4  # компактный главный экран


def test_submenus_have_back_button():
    for view in ("palettes", "adjust", "effects", "frame", "export"):
        kb = _keyboard(Settings(), view)
        last_row = kb.inline_keyboard[-1]
        assert last_row[-1].callback_data == "menu:main", view


def test_palettes_submenu_contains_all():
    kb = _keyboard(Settings(), "palettes")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    for need in ("Авто (16)", "Game Boy", "NES", "PICO-8", "C64", "Ч/Б",
                 "1-bit", "ZX", "Sweetie", "CGA", "Свои", "Больше цветов"):
        assert any(need in label for label in labels), need


def test_category_marks_active():
    kb = _keyboard(Settings(cut_bg=True), "main")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any(l == "🖼 Кадр ✓" for l in labels)


def test_store_keeps_view():
    store = StateStore()
    store.put(1, 1, b"src", Settings(), view="palettes")
    assert store.get(1, 1).view == "palettes"
    store.put(1, 1, b"src", Settings(), view="main")
    assert store.get(1, 1).view == "main"


# ------------------------------------------------------------------ свой уровень пикселизации


def test_parse_grid():
    from bot.pixelart import parse_grid

    assert parse_grid("48") == 48
    assert parse_grid(" 8 ") == 8
    assert parse_grid("256") == 256
    assert parse_grid("7") is None      # меньше GRID_MIN
    assert parse_grid("257") is None    # больше GRID_MAX
    assert parse_grid("abc") is None
    assert parse_grid("") is None
    assert parse_grid("12.5") is None
    assert parse_grid("-5") is None


def test_custom_grid_render_size():
    img = _object_on_white()  # 300x240
    result = make_pixel_art(img, pixel_size="custom", custom_grid=50)
    # gw=50, gh=round(240*50/300)=40, scale=round(1024/50)=20
    assert result.size == (1000, 800)


def test_custom_grid_clamped():
    small = make_pixel_art(_object_on_white(), pixel_size="custom", custom_grid=4)
    big = make_pixel_art(_object_on_white(), pixel_size="custom", custom_grid=999)
    # кламп в 8 и 256: обе стороны результата ~1024
    assert max(small.size) == 1024 and max(big.size) == 1024


def test_custom_grid_without_value_raises():
    with pytest.raises(ValueError):
        make_pixel_art(_object_on_white(), pixel_size="custom")


def test_caption_custom_grid():
    from bot.handlers import _caption

    settings = Settings(pixel_size="custom", custom_grid=48)
    assert "48 px (свой)" in _caption(settings)
    assert "96 px" in _caption(Settings())


def test_keyboard_has_custom_size_button():
    kb = _keyboard(Settings(), "main")
    size_row = kb.inline_keyboard[0]
    assert any(b.callback_data == "size:custom" for b in size_row)
    custom = Settings(pixel_size="custom", custom_grid=33)
    kb = _keyboard(custom, "main")
    labels = [b.text for b in kb.inline_keyboard[0]]
    assert "📏 Свой ✓" in labels and "Средний" in labels  # пресет без ✓
