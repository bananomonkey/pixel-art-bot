"""Тесты палитр: состав, уникальность, опорные цвета, самосопоставимость."""

import numpy as np

from bot.palettes import (
    FIXED_PALETTES,
    PALETTE_LABELS,
    PALETTE_ORDER,
    as_numpy,
    get_palette,
)
from bot.pixelart import _nearest_palette

EXPECTED_SIZES = {"gameboy": 4, "nes": 54, "pico8": 16, "c64": 16, "bw": 8, "cga": 4}


def test_registry_keys():
    assert set(FIXED_PALETTES) == set(EXPECTED_SIZES)
    assert set(PALETTE_LABELS) == set(PALETTE_ORDER)
    assert set(PALETTE_ORDER) == set(FIXED_PALETTES) | {"auto"}


def test_sizes():
    for key, expected in EXPECTED_SIZES.items():
        assert len(FIXED_PALETTES[key]) == expected, key


def test_valid_rgb_and_no_duplicates():
    for key, colors in FIXED_PALETTES.items():
        assert len(set(colors)) == len(colors), key
        for rgb in colors:
            assert len(rgb) == 3
            assert all(isinstance(c, int) and 0 <= c <= 255 for c in rgb), (key, rgb)


def test_anchor_colors():
    assert (0x0F, 0x38, 0x0F) in FIXED_PALETTES["gameboy"]  # #0f380f
    assert (0x00, 0x00, 0x00) in FIXED_PALETTES["cga"]
    assert (0x55, 0xFF, 0xFF) in FIXED_PALETTES["cga"]
    assert (0xFF, 0x55, 0xFF) in FIXED_PALETTES["cga"]
    assert (0xFF, 0xFF, 0xFF) in FIXED_PALETTES["cga"]
    assert (0xFF, 0x00, 0x4D) in FIXED_PALETTES["pico8"]  # #ff004d
    assert (0x00, 0x00, 0xAA) in FIXED_PALETTES["c64"]
    assert (0x00, 0x00, 0xBC) in FIXED_PALETTES["nes"]  # #0000bc
    assert (0, 0, 0) in FIXED_PALETTES["bw"] and (255, 255, 255) in FIXED_PALETTES["bw"]


def test_get_palette_matches_registry():
    for key in FIXED_PALETTES:
        assert get_palette(key) == FIXED_PALETTES[key]


def test_palette_members_map_to_themselves():
    """Собственное сопоставление: цвет палитры должен отображаться сам в себя."""
    for key in FIXED_PALETTES:
        palette = as_numpy(FIXED_PALETTES[key])
        mapped = _nearest_palette(palette.astype(np.float64), palette)
        assert np.array_equal(mapped, palette), key


def test_as_numpy_dtype_and_shape():
    palette = as_numpy(FIXED_PALETTES["gameboy"])
    assert palette.dtype == np.uint8
    assert palette.shape == (4, 3)
