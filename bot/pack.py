"""Ассет-паки: пакетная пикселизация фото в бесшовные пиксельные текстуры.

Сессия: /pack -> фото собираются в память (вместо обычной пикселизации) ->
/packdone собирает пак. Настройки пака (PackSettings) запоминаются на
пользователя и правятся панелью /packset. Всё в памяти процесса.
"""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .palettes import FIXED_PALETTES, PALETTE_LABELS
from .pixelart import (
    contact_sheet,
    decode_source,
    make_seamless,
    render_pack_tile,
    shared_palette_from_grids,
    square_crop,
)

MAX_PACK_PHOTOS = 20  # фото в одной сессии
PACK_GRIDS: tuple[int, ...] = (16, 32, 64, 128)  # размеры тайлов (степени двойки)
SEAM_MODES: dict[str, str] = {"blend": "Блендинг", "mirror": "Зеркало", "off": "Выкл"}
PACK_COLORS = 32  # цветов в общей палитре режима «Авто»

sessions: dict[int, list[bytes]] = {}  # user_id -> список исходников (в памяти)


@dataclass
class PackSettings:
    """Настройки ассет-пака; сохраняются на пользователя (PerUserStore)."""

    grid: int = 64  # 16 | 32 | 64 | 128
    seam: str = "blend"  # blend | mirror | off
    palette: str = "auto"  # auto (общая) | custom | ключ фиксированной палитры
    dither: bool = False

    def palette_label(self) -> str:
        if self.palette == "auto":
            return f"Авто (общая, {PACK_COLORS} цветов)"
        if self.palette == "custom":
            return "Свои"
        return PALETTE_LABELS.get(self.palette, self.palette)


def add_photo(user_id: int, data: bytes) -> int | None:
    """Добавить фото в сессию; None — достигнут лимит."""
    bucket = sessions.setdefault(user_id, [])
    if len(bucket) >= MAX_PACK_PHOTOS:
        return None
    bucket.append(data)
    return len(bucket)


def pop_photos(user_id: int) -> list[bytes]:
    """Забрать и закрыть сессию."""
    return sessions.pop(user_id, [])


def build_asset_pack(
    sources: list[bytes], settings: PackSettings, custom_palette: tuple | None = None
) -> tuple[bytes, bytes]:
    """Собрать пак: (zip_bytes, preview_png_bytes).

    Пайплайн на тайл: decode -> квадратный кроп 1:1 -> resize до сетки ->
    бесшовность (на сетке, ДО квантизации/дизеринга) -> общая палитра ->
    тайловый дизеринг -> PNG нативного размера (без апскейла — движки
    масштабируют сами).
    """
    if settings.palette == "custom" and not custom_palette:
        raise ValueError("своя палитра не задана")
    grids = []
    for data in sources:
        img = decode_source(data)
        img = square_crop(img).resize((settings.grid, settings.grid), Image.Resampling.BOX)
        arr = np.asarray(img.convert("RGB"), dtype=np.float64)
        grids.append(np.clip(make_seamless(arr, settings.seam), 0, 255).astype(np.uint8))

    if settings.palette == "auto":
        palette = shared_palette_from_grids(grids, PACK_COLORS)
    elif settings.palette == "custom":
        palette = custom_palette
    else:
        palette = FIXED_PALETTES[settings.palette]

    tiles = [render_pack_tile(grid, palette, settings.dither) for grid in grids]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, tile in enumerate(tiles, 1):
            bio = io.BytesIO()
            Image.fromarray(tile, "RGB").save(bio, "PNG", optimize=True)
            archive.writestr(f"tile_{index:02d}_{settings.grid}x{settings.grid}.png", bio.getvalue())

    preview = contact_sheet(tiles, scale=max(1, 512 // settings.grid))
    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, "PNG", optimize=True)
    return buffer.getvalue(), preview_buffer.getvalue()
