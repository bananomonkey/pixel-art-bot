"""Ассет-паки: пакетная пикселизация фото в бесшовные пиксельные текстуры.

Сессия: /pack -> фото собираются в память (вместо обычной пикселизации) ->
/packdone собирает пак. Настройки пака (PackSettings) запоминаются на
пользователя и правятся панелью /packset. Всё в памяти процесса.
"""

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .palettes import FIXED_PALETTES, PALETTE_LABELS
from .pixelart import (
    build_autotile_set,
    contact_sheet,
    decode_source,
    make_seamless,
    normal_map,
    render_pack_tile,
    shared_palette_from_grids,
    square_crop,
)

MAX_PACK_PHOTOS = 20  # фото в одной сессии
PACK_GRIDS: tuple[int, ...] = (16, 32, 64, 128)  # размеры тайлов (степени двойки)
SEAM_MODES: dict[str, str] = {"blend": "Блендинг", "mirror": "Зеркало", "off": "Выкл"}
PACK_COLORS = 32  # цветов в общей палитре режима «Авто»

sessions: dict[int, list[tuple[str | None, bytes]]] = {}  # user_id -> [(имя, байты)]


@dataclass
class PackSettings:
    """Настройки ассет-пака; сохраняются на пользователя (PerUserStore)."""

    grid: int = 64  # 16 | 32 | 64 | 128
    seam: str = "blend"  # blend | mirror | off
    palette: str = "auto"  # auto (общая) | custom | ключ фиксированной палитры
    dither: str = "off"  # off | fs | atkinson | bayer
    autotile: bool = False  # режим авто-тайлов: ровно 2 фото -> 16 переходов

    def palette_label(self) -> str:
        if self.palette == "auto":
            return f"Авто (общая, {PACK_COLORS} цветов)"
        if self.palette == "custom":
            return "Свои"
        return PALETTE_LABELS.get(self.palette, self.palette)

    def dither_label(self) -> str:
        from .pixelart import DITHER_LABELS

        return DITHER_LABELS.get(self.dither, self.dither)


def add_photo(user_id: int, data: bytes, name: str | None = None) -> int | None:
    """Добавить фото в сессию; None — достигнут лимит."""
    bucket = sessions.setdefault(user_id, [])
    if len(bucket) >= MAX_PACK_PHOTOS:
        return None
    bucket.append((name, data))
    return len(bucket)


def pop_photos(user_id: int) -> list[tuple[str | None, bytes]]:
    """Забрать и закрыть сессию."""
    return sessions.pop(user_id, [])


def _safe_stem(name: str | None, index: int) -> str:
    """Имя тайла: из имени файла (если было) или tile_NN; без дублей."""
    stem = ""
    if name:
        stem = re.sub(r"[^a-z0-9_-]+", "_", name.rsplit(".", 1)[0].lower()).strip("_")[:32]
    return stem or f"tile_{index:02d}"


def build_asset_pack(
    sources: list[tuple[str | None, bytes]], settings: PackSettings, custom_palette: tuple | None = None
) -> tuple[bytes, bytes]:
    """Собрать пак: (zip_bytes, preview_png_bytes).

    Пайплайн на тайл: decode (альфа сохраняется) -> квадратный кроп 1:1 ->
    resize до сетки -> бесшовность (на сетке, ДО квантизации/дизеринга) ->
    общая палитра -> тайловый дизеринг -> PNG нативного размера (без
    апскейла — движки масштабируют сами).
    """
    if settings.palette == "custom" and not custom_palette:
        raise ValueError("своя палитра не задана")
    grids = []
    for _name, data in sources:
        img = decode_source(data, keep_alpha=True)
        img = square_crop(img).resize((settings.grid, settings.grid), Image.Resampling.BOX)
        arr = np.asarray(img, dtype=np.float64)
        if arr.shape[-1] == 4 and settings.seam == "blend":
            # блендинг только для RGB: альфу не трогаем, иначе прозрачный фон
            # «закрашивается» содержимым с противоположной стороны тайла
            rgb = np.clip(make_seamless(arr[..., :3], "blend"), 0, 255)
            arr = np.dstack([rgb, arr[..., 3]])
        else:
            arr = make_seamless(arr, settings.seam)
        grids.append(np.clip(arr, 0, 255).astype(np.uint8))

    if settings.palette == "auto":
        palette = shared_palette_from_grids([g[..., :3] for g in grids], PACK_COLORS)
    elif settings.palette == "custom":
        palette = custom_palette
    else:
        palette = FIXED_PALETTES[settings.palette]

    autotile = settings.autotile and len(grids) == 2
    if autotile:
        raw_tiles = build_autotile_set(grids[0][..., :3], grids[1][..., :3])
        stems = [f"blob_{bits:04b}" for bits in range(16)]
        tiles = [render_pack_tile(t, palette, settings.dither) for t in raw_tiles]
    else:
        tiles = [render_pack_tile(grid, palette, settings.dither) for grid in grids]

    counts: dict[str, int] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        stems_out = []
        for index, tile in enumerate(tiles, 1):
            base = stems[index - 1] if autotile else _safe_stem(sources[index - 1][0], index)
            number = counts.get(base, 0) + 1
            counts[base] = number
            stem = base if number == 1 else f"{base}_{number}"
            stems_out.append(stem)
            rgb_tile = tile[..., :3]
            bio = io.BytesIO()
            Image.fromarray(tile).save(bio, "PNG", optimize=True)
            archive.writestr(f"{stem}_{settings.grid}x{settings.grid}.png", bio.getvalue())
            # карта нормалей (для динамического 2D-освещения в движках)
            nrm = io.BytesIO()
            Image.fromarray(normal_map(rgb_tile), "RGB").save(nrm, "PNG", optimize=True)
            archive.writestr(f"{stem}_normal.png", nrm.getvalue())
            # сверхсжатая версия 4×4 — миникарты и дальние LOD
            cmap = io.BytesIO()
            Image.fromarray(rgb_tile, "RGB").resize((4, 4), Image.Resampling.BOX).save(
                cmap, "PNG", optimize=True
            )
            archive.writestr(f"{stem}_4x4.png", cmap.getvalue())
        # палитра пака для редакторов (GIMP/Aseprite/.hex)
        gpl = "GIMP Palette\nName: pixel-art-pack\nColumns: 3\n#\n" + "\n".join(
            f"{r:3d} {g:3d} {b:3d}\t#{r:02X}{g:02X}{b:02X}" for r, g, b in palette
        )
        hexfile = "\n".join(f"{r:02X}{g:02X}{b:02X}" for r, g, b in palette) + "\n"
        archive.writestr("palette.gpl", gpl)
        archive.writestr("palette.hex", hexfile)

    # превью: RGBA-тайлы кладём на белый фон
    rgb_tiles = []
    for tile in tiles:
        if tile.shape[-1] == 4:
            solid = tile[..., 3:4] >= 128
            tile = np.where(solid, tile[..., :3], 255).astype(np.uint8)
        rgb_tiles.append(tile)
    preview = contact_sheet(rgb_tiles, scale=max(1, 512 // settings.grid))
    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, "PNG", optimize=True)
    return buffer.getvalue(), preview_buffer.getvalue()
