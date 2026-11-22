"""Тесты ассет-паков: кроп, бесшовность, общая палитра, сборка zip/превью, сессии."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bot.pack import MAX_PACK_PHOTOS, PackSettings, add_photo, build_asset_pack, pop_photos
from bot.pixelart import (
    contact_sheet,
    make_seamless,
    render_pack_tile,
    shared_palette_from_grids,
    square_crop,
)


def _gradient_grid(seed=1, size=64):
    rng = np.random.RandomState(seed)
    return (rng.rand(size, size, 3) * 255).astype(np.uint8)


def _source_image(seed=3, size=320):
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    base = np.random.RandomState(seed).randint(60, 240, 3)
    for y in range(size):
        draw.line([(0, y), (size, y)], fill=tuple(int(v) for v in base * (1 - y / 400)))
    draw.ellipse((60, 60, 260, 260), fill=(180, 40, 60))
    bio = io.BytesIO()
    img.save(bio, "JPEG", quality=90)
    return bio.getvalue()


def _sources(count=3):
    return [(None, _source_image(seed)) for seed in range(3, 3 + count)]


# ------------------------------------------------------------------ кроп и бесшовность


def test_square_crop():
    img = Image.new("RGB", (400, 300), (10, 20, 30))
    cropped = square_crop(img)
    assert cropped.size == (300, 300)
    assert square_crop(Image.new("RGB", (100, 100))).size == (100, 100)


def test_seamless_blend_edges_match_opposite():
    arr = np.random.RandomState(7).randint(0, 255, (32, 32, 3)).astype(np.float64)
    result = make_seamless(arr, "blend")
    rolled = np.roll(arr, (16, 16), axis=(0, 1))
    assert (result[0] == rolled[0]).all()      # верх = бывший низ
    assert (result[:, -1] == rolled[:, -1]).all()  # право = бывшее лево


def test_seamless_blend_exact_on_odd_grid():
    arr = np.random.RandomState(5).randint(0, 255, (33, 33, 3)).astype(np.float64)
    result = make_seamless(arr, "blend")
    rolled = np.roll(arr, (16, 16), axis=(0, 1))
    assert (result[0] == rolled[0]).all() and (result[:, 0] == rolled[:, 0]).all()


def test_seamless_mirror_symmetry():
    arr = _gradient_grid(seed=2, size=32)
    result = make_seamless(arr, "mirror")
    assert (result[:, 0] == result[:, -1]).all()
    assert (result[0] == result[-1]).all()
    assert result.shape == (32, 32, 3)


def test_seamless_off_identity():
    arr = _gradient_grid(seed=3, size=16)
    assert (make_seamless(arr.astype(np.float64), "off") == arr).all()
    with pytest.raises(ValueError):
        make_seamless(arr, "unknown")


# ------------------------------------------------------------------ общая палитра и тайлы


def test_shared_palette_from_grids():
    palette = shared_palette_from_grids([_gradient_grid(1), _gradient_grid(2)], 32)
    assert 1 <= len(palette) <= 32
    assert all(len(rgb) == 3 and all(0 <= c <= 255 for c in rgb) for rgb in palette)


def test_render_pack_tile_membership():
    palette = shared_palette_from_grids([_gradient_grid(1), _gradient_grid(2)], 16)
    grid = _gradient_grid(3)
    for dither in (False, True):
        tile = render_pack_tile(grid, palette, dither=dither)
        assert tile.shape == (64, 64, 3)
        pixels = set(map(tuple, tile.reshape(-1, 3).tolist()))
        assert pixels <= set(palette), dither


def test_render_pack_tile_dither_changes_result():
    palette = shared_palette_from_grids([_gradient_grid(1)], 8)
    grid = _gradient_grid(4)
    plain = render_pack_tile(grid, palette, dither=False)
    dithered = render_pack_tile(grid, palette, dither=True)
    assert plain.tobytes() != dithered.tobytes()


def test_contact_sheet_size():
    tiles = [_gradient_grid(seed) for seed in range(5)]
    sheet = contact_sheet(tiles, scale=2)
    assert sheet.size == (3 * 64 * 2, 2 * 64 * 2)  # ceil(sqrt(5)) = 3 колонки


# ------------------------------------------------------------------ сборка пака


@pytest.mark.parametrize(
    "settings",
    [
        PackSettings(),
        PackSettings(grid=32, seam="mirror", palette="pico8", dither=True),
        PackSettings(seam="off"),
        PackSettings(grid=16, seam="blend", dither=True),
        PackSettings(grid=128, seam="blend", palette="auto"),
    ],
)
def test_build_asset_pack(settings):
    sources = _sources(3)
    zip_bytes, preview = build_asset_pack(sources, settings)
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = archive.namelist()
    assert len(names) == 3
    assert all(name.endswith(f"_{settings.grid}x{settings.grid}.png") for name in names)
    tile = Image.open(io.BytesIO(archive.read(names[0])))
    assert tile.size == (settings.grid, settings.grid) and tile.format == "PNG"
    preview_img = Image.open(io.BytesIO(preview))
    assert max(preview_img.size) <= 2048


def test_build_asset_pack_final_tile_seamless():
    """Ключевая проверка: у готового квантизованного тайла края сходятся."""
    zip_bytes, _ = build_asset_pack(_sources(2), PackSettings(grid=64, seam="blend", palette="sweetie"))
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    tile = np.asarray(Image.open(io.BytesIO(archive.read("tile_01_64x64.png"))))
    assert (tile[:, 0] == tile[:, -1]).all() and (tile[0] == tile[-1]).all()


def test_build_asset_pack_custom_palette():
    from bot.pixelart import extract_palette

    palette = extract_palette(Image.open(io.BytesIO(_sources(1)[0][1])), 8)
    zip_bytes, _ = build_asset_pack(_sources(2), PackSettings(palette="custom"), palette)
    tile = np.asarray(Image.open(io.BytesIO(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("tile_01_64x64.png"))))
    pixels = set(map(tuple, tile.reshape(-1, 3).tolist()))
    assert pixels <= set(palette)


def test_build_asset_pack_custom_palette_missing_raises():
    with pytest.raises(ValueError):
        build_asset_pack(_sources(1), PackSettings(palette="custom"), None)


# ------------------------------------------------------------------ сессии


def test_add_photo_limit():
    uid = 777001
    for _ in range(MAX_PACK_PHOTOS):
        count = add_photo(uid, b"x")
        assert count is not None
    assert add_photo(uid, b"x") is None  # лимит
    dropped = pop_photos(uid)
    assert len(dropped) == MAX_PACK_PHOTOS
    assert add_photo(uid, b"x") == 1  # после сброса сессия началась заново
    pop_photos(uid)


def test_pack_settings_labels():
    ps = PackSettings()
    assert ps.palette_label().startswith("Авто")
    assert PackSettings(palette="custom").palette_label() == "Свои"
    assert PackSettings(palette="gameboy").palette_label() == "Game Boy"
