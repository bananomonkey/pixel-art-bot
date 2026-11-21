"""Тесты пикселизации: размеры (кратность сетке), число цветов, палитры,
прозрачность, крайние пропорции, дизеринг, рендер end-to-end."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bot.palettes import FIXED_PALETTES
from bot.pixelart import (
    GRID_BY_SIZE,
    RESULT_MAX_SIDE,
    ImageTooLargeError,
    UnsupportedMediaError,
    _nearest_palette,
    decode_source,
    encode_source,
    make_pixel_art,
    render,
)
from bot.state import Settings


# ------------------------------------------------------------------ фикстуры

def _gradient(w=800, h=600):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    arr[..., 2] = 128
    return Image.fromarray(arr, "RGB")


def _circle(w=640, h=480):
    img = Image.new("RGB", (w, h), (30, 30, 30))
    ImageDraw.Draw(img).ellipse((100, 80, 540, 400), fill=(220, 40, 60))
    return img


def _png_bytes(img):
    bio = io.BytesIO()
    img.save(bio, "PNG")
    return bio.getvalue()


def _expected_grid(img_w, img_h, level):
    """Формула сетки из ТЗ: пропорции сохранены, большая сторона = уровень."""
    grid_big = min(GRID_BY_SIZE[level], max(img_w, img_h))
    if img_w >= img_h:
        gw = grid_big
        gh = max(1, round(img_h * grid_big / img_w))
    else:
        gh = grid_big
        gw = max(1, round(img_w * grid_big / img_h))
    return gw, gh


def _unique_colors(img):
    return np.unique(np.asarray(img.convert("RGB")).reshape(-1, 3), axis=0)


# ------------------------------------------------------------------ размеры


def test_result_size_multiple_of_grid():
    img = _gradient()
    for level in GRID_BY_SIZE:
        result = make_pixel_art(img, pixel_size=level, palette="auto")
        gw, gh = _expected_grid(*img.size, level)
        scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
        assert result.size == (gw * scale, gh * scale), level
        # большая сторона около 1024 (в пределах половины шага сетки)
        assert abs(max(result.size) - RESULT_MAX_SIDE) <= max(gw, gh) / 2, level


def test_grid_capped_by_small_source():
    # крошечный источник: сетка не больше исходной стороны
    img = Image.new("RGB", (20, 20), (10, 20, 30))
    result = make_pixel_art(img, pixel_size="medium")
    gw, gh = _expected_grid(20, 20, "medium")
    scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
    assert result.size == (gw * scale, gh * scale)
    assert max(result.size) >= 1000


def test_result_pixels_are_square_blocks():
    img = _circle()
    result = make_pixel_art(img, pixel_size="large")
    gw, gh = _expected_grid(*img.size, "large")
    scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
    # после NEAREST-увеличения каждый блок сетки — ровный квадрат scale x scale
    assert result.size[0] / gw == result.size[1] / gh == scale


# ------------------------------------------------------------------ цвета


def test_auto_palette_unique_colors_le_n():
    img = _gradient()
    for n in (8, 16, 32, 64):
        result = make_pixel_art(img, palette="auto", n_colors=n)
        assert len(_unique_colors(result)) <= n, n


def test_fixed_palettes_membership():
    img = _circle()
    for key, colors in FIXED_PALETTES.items():
        for dither in (False, True):
            result = make_pixel_art(img, palette=key, dither=dither)
            allowed = set(colors)
            pixels = set(map(tuple, np.asarray(result.convert("RGB")).reshape(-1, 3).tolist()))
            assert pixels <= allowed, (key, dither)


def test_dither_changes_result():
    img = _gradient()
    plain = np.asarray(make_pixel_art(img, palette="gameboy", dither=False))
    dithered = np.asarray(make_pixel_art(img, palette="gameboy", dither=True))
    assert plain.tobytes() != dithered.tobytes()
    # «Авто» с Floyd–Steinberg тоже работает
    auto_dithered = make_pixel_art(img, palette="auto", dither=True)
    assert auto_dithered.size == make_pixel_art(img, palette="auto", dither=False).size


def test_nearest_palette_vectorized_redmean():
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    pixels = np.array([[10, 10, 10], [250, 250, 250], [200, 30, 30], [128, 128, 128]], dtype=np.float64)
    mapped = _nearest_palette(pixels, palette)
    assert mapped[0].tolist() == [0, 0, 0]
    assert mapped[1].tolist() == [255, 255, 255]
    assert mapped[2].tolist() == [255, 0, 0]
    assert mapped[3].tolist() == [255, 255, 255]  # середина — белый по redmean


# ------------------------------------------------------------------ входы


def test_transparent_png_flattened_to_white():
    img = Image.new("RGBA", (100, 80), (10, 200, 30, 0))  # полностью прозрачный
    src = decode_source(_png_bytes(img))
    assert src.mode == "RGB"
    assert np.asarray(src).reshape(-1, 3).min() > 240  # белая заливка
    result = make_pixel_art(src, palette="auto", n_colors=8)
    assert result.mode == "RGB"


def test_partially_transparent_png():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    ImageDraw.Draw(img).rectangle((40, 40, 60, 60), fill=(0, 0, 255, 0))  # дырка прозрачная
    src = decode_source(_png_bytes(img))
    assert src.mode == "RGB"
    corners = [src.getpixel((0, 0)), src.getpixel((99, 99))]
    assert all(px == (255, 0, 0) for px in corners)
    result = make_pixel_art(src, palette="pico8")
    assert result.mode == "RGB"


def test_narrow_image():
    img = _gradient(50, 2000)  # очень высокая
    result = make_pixel_art(img, pixel_size="medium")
    gw, gh = _expected_grid(50, 2000, "medium")
    scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
    assert result.size == (gw * scale, gh * scale)
    assert result.size[1] > result.size[0]


def test_wide_image():
    img = _gradient(2000, 100)  # очень широкая
    result = make_pixel_art(img, pixel_size="medium")
    gw, gh = _expected_grid(2000, 100, "medium")
    scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
    assert result.size == (gw * scale, gh * scale)
    assert result.size[0] > result.size[1]


def test_1x1_image():
    img = Image.new("RGB", (1, 1), (123, 45, 67))
    result = make_pixel_art(img, pixel_size="huge", palette="pico8")
    assert result.size == (1024, 1024)
    unique = _unique_colors(result)
    assert len(unique) == 1
    assert tuple(unique[0].tolist()) in FIXED_PALETTES["pico8"]


def test_source_downscaled_to_1024():
    img = _gradient(3000, 1500)
    src = decode_source(_png_bytes(img))
    assert src.size == (1024, 512)
    assert src.mode == "RGB"


def test_too_large_rejected():
    img = Image.new("RGB", (4001, 10), (0, 0, 0))
    with pytest.raises(ImageTooLargeError):
        decode_source(_png_bytes(img))


def test_exif_rotation_applied():
    # 20x40 с EXIF orientation=6 (поворот на 90): после transpose станет 40x20
    img = _gradient(20, 40)
    exif = Image.Exif()
    exif[274] = 6
    bio = io.BytesIO()
    img.save(bio, "JPEG", exif=exif)
    src = decode_source(bio.getvalue())
    assert src.size == (40, 20)


def test_metadata_not_carried():
    img = _gradient(200, 100)
    exif = Image.Exif()
    exif[274] = 6
    bio = io.BytesIO()
    img.save(bio, "JPEG", exif=exif)
    src = decode_source(bio.getvalue())
    stored = encode_source(src)
    reopened = Image.open(io.BytesIO(stored))
    assert "exif" not in reopened.info
    result = make_pixel_art(src)
    assert "exif" not in result.info


def test_animated_gif_rejected():
    frames = [Image.new("RGB", (10, 10), (i * 100, 0, 0)) for i in range(2)]
    bio = io.BytesIO()
    frames[0].save(bio, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    with pytest.raises(UnsupportedMediaError):
        decode_source(bio.getvalue())


def test_garbage_bytes_rejected():
    with pytest.raises(UnsupportedMediaError):
        decode_source(b"this is definitely not an image")


def test_empty_bytes_rejected():
    with pytest.raises(UnsupportedMediaError):
        decode_source(b"")


# ------------------------------------------------------------------ рендер


def test_render_end_to_end_png():
    source = encode_source(decode_source(_png_bytes(_circle())))
    png = render(source, Settings())
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    gw, gh = _expected_grid(640, 480, "medium")
    scale = max(1, round(RESULT_MAX_SIDE / max(gw, gh)))
    assert img.size == (gw * scale, gh * scale)
    assert len(_unique_colors(img)) <= 16


def test_render_with_all_settings_combinations():
    source = encode_source(decode_source(_png_bytes(_gradient(400, 300))))
    for level in GRID_BY_SIZE:
        for palette in ("auto", *FIXED_PALETTES.keys()):
            for dither in (False, True):
                settings = Settings(
                    pixel_size=level, palette=palette, n_colors=32, dither=dither
                )
                png = render(source, settings)
                img = Image.open(io.BytesIO(png))
                assert img.format == "PNG"
                assert max(img.size) >= 1000


def test_render_is_deterministic():
    source = encode_source(decode_source(_png_bytes(_circle())))
    assert render(source, Settings()) == render(source, Settings())
