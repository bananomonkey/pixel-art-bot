"""Классическая обработка изображений: фото -> пиксель-арт. Без ИИ, без сети.

Чистые функции (вход/выход — bytes/PIL/numpy), без зависимости от aiogram.
Всё выполняется в памяти; на диск ничего не пишется, метаданные не переносятся.
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .config import (
    CRT_STEP,
    DEFAULT_CONTRAST,
    DEFAULT_SATURATION,
    DEFAULT_SHARPNESS,
    GRID_MAX,
    GRID_MIN,
    MAX_IMAGE_SIDE,
    RESULT_MAX_SIDE,
    SOURCE_MAX_SIDE,
    STICKER_MAX_SIDE,
    STORAGE_JPEG_QUALITY,
    UNIFORM_TRIM_TOLERANCE,
    WATERMARK_TEXT,
)
from .palettes import FIXED_PALETTES, as_numpy
from .state import Settings

# размер сетки (px по большей стороне) для каждого уровня кнопки «размер пикселя»
GRID_BY_SIZE: dict[str, int] = {"small": 128, "medium": 96, "large": 64, "huge": 32}


def parse_grid(text: str) -> int | None:
    """«48» -> 48; целое в диапазоне GRID_MIN..GRID_MAX, иначе None."""
    cleaned = (text or "").strip()
    if not cleaned.isdigit():
        return None
    value = int(cleaned)
    if value < GRID_MIN or value > GRID_MAX:
        return None
    return value

# амплитуда упорядоченного дизеринга Bayer (сдвиг яркости, -32..+31)
DITHER_STRENGTH = 64.0

# матрица упорядоченного дизеринга Bayer 4x4
_BAYER_4X4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float64,
)


class PixelArtError(Exception):
    """Базовая понятная ошибка обработки."""


class ImageTooLargeError(PixelArtError):
    """Изображение больше допустимых размеров."""


class UnsupportedMediaError(PixelArtError):
    """Файл не изображение или анимация."""


# ---------------------------------------------------------------- исходник


def decode_source(data: bytes, keep_alpha: bool = False) -> Image.Image:
    """Открыть присланное фото и подготовить исходник.

    Проверяет ограничения (4000x4000), выпрямляет по EXIF, уменьшает до
    SOURCE_MAX_SIDE. По умолчанию прозрачность заливается белым; keep_alpha=True
    (для ассет-паков) сохраняет альфа-канал. Метаданные не переносятся.
    """
    try:
        img = Image.open(io.BytesIO(data))
        if getattr(img, "is_animated", False):
            raise UnsupportedMediaError("анимированные изображения не поддерживаются")
        # проверка по заголовку, до декодирования пикселей
        if max(img.size) > MAX_IMAGE_SIDE:
            raise ImageTooLargeError("изображение больше 4000x4000")
        img = ImageOps.exif_transpose(img)
        img = _flatten_to_rgb(img, keep_alpha=keep_alpha)
        img.thumbnail((SOURCE_MAX_SIDE, SOURCE_MAX_SIDE), Image.Resampling.LANCZOS)
        return img
    except (ImageTooLargeError, UnsupportedMediaError):
        raise
    except Exception as exc:  # битый файл / не изображение / декомпрессия
        raise UnsupportedMediaError(f"не удалось открыть изображение: {type(exc).__name__}") from exc


def _flatten_to_rgb(img: Image.Image, keep_alpha: bool = False) -> Image.Image:
    """Привести к RGB (прозрачность — залить белым) или к RGBA при keep_alpha."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        if keep_alpha:
            return rgba
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if keep_alpha:
        return img.convert("RGBA")
    return img.convert("RGB")


def encode_source(img: Image.Image) -> bytes:
    """Сжать подготовленный исходник для хранения в памяти (JPEG без EXIF)."""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=STORAGE_JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------- пикселизация


def make_pixel_art(
    img: Image.Image,
    *,
    pixel_size: str = "medium",
    custom_grid: int | None = None,
    palette: str = "auto",
    n_colors: int = 16,
    dither: bool = False,
    contrast: float = DEFAULT_CONTRAST,
    saturation: float = DEFAULT_SATURATION,
    sharpness: float = DEFAULT_SHARPNESS,
    palette_colors: tuple | None = None,
    cut_bg: bool = False,
    bg_tolerance: int = 0,
    crop_object: bool = False,
    padding: int = 10,
    outline: bool = False,
    show_grid: bool = False,
) -> Image.Image:
    """Пикселизация RGB-изображения: фон/кадр -> сетка -> квантизация -> квадраты.

    Возвращает RGBA (при вырезке фона) или RGB, большая сторона — целое кратное
    размера сетки и близка к RESULT_MAX_SIDE. pixel_size="custom" использует
    custom_grid (свой уровень пикселизации, GRID_MIN..GRID_MAX). bg_tolerance=0 —
    порог фона подбирается автоматически. show_grid — подсветка границ пикселей.
    """
    if pixel_size in GRID_BY_SIZE:
        grid_level = GRID_BY_SIZE[pixel_size]
    elif pixel_size == "custom" and custom_grid:
        grid_level = max(GRID_MIN, min(GRID_MAX, int(custom_grid)))
    else:
        raise ValueError(f"неизвестный размер пикселя: {pixel_size}")
    if palette == "custom" and not palette_colors:
        raise ValueError("для своей палитры не заданы цвета")
    if palette != "auto" and palette != "custom" and palette not in FIXED_PALETTES:
        raise ValueError(f"неизвестная палитра: {palette}")
    n_colors = max(2, min(64, int(n_colors)))

    # 0) коррекции контраста/насыщенности/резкости
    adjusted = ImageEnhance.Contrast(img).enhance(contrast)
    adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
    adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)

    # 1) фон и кадр (возвращает RGBA)
    framed = _frame_image(adjusted, cut_bg, bg_tolerance, crop_object, padding)

    # 2) размер сетки с сохранением пропорций
    width, height = framed.size
    grid_big = min(grid_level, max(width, height))
    if width >= height:
        grid_w = grid_big
        grid_h = max(1, round(height * grid_big / width))
    else:
        grid_h = grid_big
        grid_w = max(1, round(width * grid_big / height))

    # 3) premultiplied BOX-усреднение до сетки (чтобы прозрачность не «замыливала» цвет)
    src = np.asarray(framed).astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    pre = np.concatenate([src[..., :3] * alpha, alpha * 255.0], axis=2).astype(np.uint8)
    small_rgba = np.asarray(
        Image.fromarray(pre, "RGBA").resize((grid_w, grid_h), Image.Resampling.BOX)
    ).astype(np.float32)
    a_grid = small_rgba[..., 3:4]
    solid = a_grid[..., 0] >= 96.0  # пиксель-арт: жёсткая граница, без полупрозрачных краёв
    rgb_grid = np.clip(small_rgba[..., :3] * (255.0 / np.maximum(a_grid, 1.0)), 0, 255)

    # 4) квантизация (прозрачные пиксели невидимы — цвет в них не важен)
    rgb_u8 = rgb_grid.astype(np.uint8)
    grid_img = Image.fromarray(rgb_u8, "RGB")
    if palette == "auto":
        dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        grid_img = grid_img.quantize(
            colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=dither_mode
        ).convert("RGB")
    elif palette == "custom":
        grid_img = _map_to_fixed_palette(grid_img, palette_colors, dither=dither)
    else:
        grid_img = _map_to_fixed_palette(grid_img, FIXED_PALETTES[palette], dither=dither)
    rgb_out = np.asarray(grid_img)

    # 5) обводка по границам объекта (на сетке, до увеличения)
    if outline:
        rgb_out = _apply_outline(rgb_out, solid)

    # 6) сборка и увеличение NEAREST'ом в целое число раз
    scale = max(1, round(RESULT_MAX_SIDE / max(grid_w, grid_h)))
    if solid.all():  # прозрачности нет — возвращаем обычный RGB
        art = Image.fromarray(rgb_out, "RGB").resize(
            (grid_w * scale, grid_h * scale), Image.Resampling.NEAREST
        )
        return _draw_pixel_grid(art, scale) if show_grid else art
    final = np.dstack([rgb_out, np.where(solid, 255, 0).astype(np.uint8)])
    art = Image.fromarray(final, "RGBA")
    art = art.resize((grid_w * scale, grid_h * scale), Image.Resampling.NEAREST)
    return _draw_pixel_grid(art, scale) if show_grid else art


def _draw_pixel_grid(img: Image.Image, scale: int) -> Image.Image:
    """Подсветить границы пикселей сетки (тонные тёмные линии)."""
    arr = np.asarray(img).copy()
    arr[::scale, :, :3] = (arr[::scale, :, :3].astype(np.float32) * 0.7).astype(np.uint8)
    arr[:, ::scale, :3] = (arr[:, ::scale, :3].astype(np.float32) * 0.7).astype(np.uint8)
    return Image.fromarray(arr, img.mode)


# ---------------------------------------------------------------- фон и кадр


def _flood_from_border(close: np.ndarray) -> np.ndarray:
    """Достижимые из границы области «фонового» цвета (волновой BFS, векторно)."""
    reached = np.zeros_like(close, dtype=bool)
    if not close.any():
        return reached
    frontier = np.zeros_like(close, dtype=bool)
    frontier[0, :] = close[0, :]
    frontier[-1, :] = close[-1, :]
    frontier[:, 0] = close[:, 0]
    frontier[:, -1] = close[:, -1]
    while frontier.any():
        reached |= frontier
        grow = np.zeros_like(close, dtype=bool)
        grow[1:, :] |= frontier[:-1, :]
        grow[:-1, :] |= frontier[1:, :]
        grow[:, 1:] |= frontier[:, :-1]
        grow[:, :-1] |= frontier[:, 1:]
        frontier = grow & close & ~reached
    return reached


def _dilate(mask: np.ndarray) -> np.ndarray:
    grow = np.zeros_like(mask, dtype=bool)
    grow[1:, :] |= mask[:-1, :]
    grow[:-1, :] |= mask[1:, :]
    grow[:, 1:] |= mask[:, :-1]
    grow[:, :-1] |= mask[:, 1:]
    return mask | grow


def _bg_color(arr: np.ndarray) -> np.ndarray:
    """Цвет фона — медиана четырёх углов изображения."""
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]).astype(np.float64)
    return np.median(corners, axis=0)


_AUTO_TOLERANCE_CANDIDATES = (24, 40, 64, 96, 128)


def _auto_bg_tolerance(arr: np.ndarray) -> int:
    """Подобрать порог фона автоматически (для bg_tolerance=0).

    Перебирает кандидатов на уменьшенной копии и берёт первый, при котором
    заливка от краёв убирает правдоподобную долю фона (5–92% кадра).
    """
    factor = max(1, max(arr.shape[:2]) // 96)
    sub = arr[::factor, ::factor]
    bg = _bg_color(sub)
    dist = np.sqrt(((sub.astype(np.float64) - bg) ** 2).sum(axis=2))
    for tol in _AUTO_TOLERANCE_CANDIDATES:
        close = dist <= tol
        flooded = _flood_from_border(close)
        coverage = flooded.sum() / flooded.size
        if 0.05 <= coverage <= 0.92:
            return tol
    return 64


def _frame_image(
    img: Image.Image,
    cut_bg: bool,
    bg_tolerance: int,
    crop_object: bool,
    padding: int,
) -> Image.Image:
    """Вырезать фон (flood fill от краёв) и/или кадрировать до объекта.

    bg_tolerance=0 — порог подбирается автоматически. Возвращает RGBA;
    без вырезки альфа непрозрачная.
    """
    arr = np.asarray(img.convert("RGB"))
    height, width = arr.shape[:2]
    alpha = np.full((height, width), 255, dtype=np.uint8)
    bg = _bg_color(arr)
    dist = np.sqrt(((arr.astype(np.float64) - bg) ** 2).sum(axis=2))

    if cut_bg:
        tolerance = bg_tolerance if bg_tolerance and bg_tolerance > 0 else _auto_bg_tolerance(arr)
        close = dist <= max(1, tolerance)
        # BFS на четверти разрешения — быстро даже на 4000 px
        flooded_q = _flood_from_border(close[::4, ::4])
        flooded = np.repeat(np.repeat(flooded_q, 4, axis=0), 4, axis=1)[:height, :width]
        for _ in range(3):  # догнать границы на полном разрешении
            flooded = _dilate(flooded) & close
        alpha = np.where(flooded, 0, 255).astype(np.uint8)

    if crop_object:
        if cut_bg:
            content = alpha > 0
        else:
            content = dist > UNIFORM_TRIM_TOLERANCE  # «не фон» без вырезки
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        if len(rows) and len(cols):
            box_h = rows[-1] - rows[0] + 1
            box_w = cols[-1] - cols[0] + 1
            pad = int(max(0, padding) / 100 * max(box_h, box_w))
            r0 = max(0, rows[0] - pad)
            r1 = min(height, rows[-1] + 1 + pad)
            c0 = max(0, cols[0] - pad)
            c1 = min(width, cols[-1] + 1 + pad)
            arr = arr[r0:r1, c0:c1]
            alpha = alpha[r0:r1, c0:c1]

    return Image.fromarray(np.dstack([arr, alpha]), "RGBA")


def _apply_outline(rgb: np.ndarray, solid: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Затемнить граничные пиксели объекта (пиксельная обводка на сетке)."""
    out = rgb.copy()
    h, w = solid.shape
    edge = np.zeros_like(solid, dtype=bool)
    ints = rgb.astype(np.int16)
    for dy, dx, a_sl, b_sl in ((0, 1, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                               (0, -1, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               (1, 0, (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                               (-1, 0, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        diff = np.abs(ints[a_sl] - ints[b_sl]).max(axis=2) > threshold
        boundary = solid[a_sl] != solid[b_sl]
        mark = a_sl
        edge[mark] |= (diff | boundary) & solid[mark]
    out[edge] = (out[edge].astype(np.float32) * 0.45).astype(np.uint8)
    return out


# ---------------------------------------------------------------- ассет-паки (текстуры)


def square_crop(img: Image.Image) -> Image.Image:
    """Центральный кроп 1:1 — текстуры в движках квадратные."""
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def make_seamless(arr: np.ndarray, mode: str) -> np.ndarray:
    """Сделать тайл бесшовным на уровне сетки (до квантизации и дизеринга).

    blend: циклический сдвиг на полширины/полвысоты (np.roll) + блендинг
    радиальной маской — края гарантированно совпадают с противоположными.
    mirror: четверть отражается в 2×2 (калейдоскоп — для плитки/узоров).
    off: без обработки.
    """
    height, width = arr.shape[:2]
    if mode == "blend":
        rolled = np.roll(arr, (height // 2, width // 2), axis=(0, 1))
        yy, xx = np.mgrid[0:height, 0:width]
        # ромбовидная маска от краёв: на всех краевых рядах/столбцах ровно 0,
        # поэтому шов гарантированно совпадает с противоположной стороной
        dy = np.minimum(yy, height - 1 - yy) / (height / 2)
        dx = np.minimum(xx, width - 1 - xx) / (width / 2)
        mask = np.clip(np.minimum(dy, dx), 0.0, 1.0)[..., None]
        return arr * mask + rolled * (1.0 - mask)
    if mode == "mirror":
        quarter = arr[: height // 2, : width // 2]
        top = np.concatenate([quarter, quarter[:, ::-1]], axis=1)
        return np.concatenate([top, top[::-1, :]], axis=0)
    if mode == "off":
        return arr
    raise ValueError(f"неизвестный режим бесшовности: {mode}")


def shared_palette_from_grids(grids: list[np.ndarray], n_colors: int) -> tuple:
    """Одна общая палитра mediancut'ом на ВСЕ тайлы пака — чтобы сочетались."""
    stacked = np.concatenate(grids, axis=0)
    quantized = Image.fromarray(stacked, "RGB").quantize(
        colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    palette = quantized.getpalette() or []
    used = sorted(set(np.asarray(quantized).reshape(-1).tolist()))
    colors = [tuple(int(v) for v in palette[i * 3 : i * 3 + 3]) for i in used]
    return tuple(dict.fromkeys(colors))


def render_pack_tile(grid_arr: np.ndarray, palette: tuple, dither: bool) -> np.ndarray:
    """Квантизация тайла в палитру пака; дизеринг — FS по 3×3-размножению
    (wrap-around), чтобы ошибка распространялась и через швы.

    RGBA-вход (пак с прозрачностью): альфа бинаризуется, прозрачные пиксели
    в палитре не участвуют визуально.
    """
    n = grid_arr.shape[0]
    has_alpha = grid_arr.shape[-1] == 4
    rgb = grid_arr[..., :3]
    alpha = grid_arr[..., 3] if has_alpha else np.full((n, n), 255, dtype=np.uint8)
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette([v for rgb_c in palette for v in rgb_c])
    if dither:
        tiled = np.tile(rgb, (3, 3, 1))
        quantized = Image.fromarray(tiled, "RGB").quantize(
            palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG
        )
        rgb_out = np.asarray(quantized.convert("RGB"))[n : 2 * n, n : 2 * n]
    else:
        quantized = Image.fromarray(rgb, "RGB").quantize(
            palette=palette_img, dither=Image.Dither.NONE
        )
        rgb_out = np.asarray(quantized.convert("RGB"))
    if not has_alpha:
        return rgb_out
    solid = alpha >= 128
    if solid.all():  # альфа непрозрачная — RGB меньше весит и совместим со всем
        return rgb_out
    out = np.dstack([rgb_out, np.where(solid, 255, 0).astype(np.uint8)])
    return out


def contact_sheet(tiles: list[np.ndarray], scale: int = 4) -> Image.Image:
    """Превью-простыня: все тайлы пака, увеличенные NEAREST'ом."""
    count = len(tiles)
    size = tiles[0].shape[0]
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    sheet = np.full((rows * size * scale, cols * size * scale, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        up = np.kron(tile, np.ones((scale, scale, 1), dtype=np.uint8))
        sheet[row * size * scale : (row + 1) * size * scale,
              col * size * scale : (col + 1) * size * scale] = up
    return Image.fromarray(sheet, "RGB")


# ---------------------------------------------------------------- эффекты и экспорт


def post_effects(
    img: Image.Image,
    crt: bool = False,
    grain: bool = False,
    watermark_text: str | None = None,
) -> Image.Image:
    """Эффекты поверх увеличенного результата: сканлайны, зерно, водяной знак."""
    if not crt and not grain and not watermark_text:
        return img
    from PIL import ImageDraw, ImageFont

    had_alpha = img.mode == "RGBA"
    if not had_alpha:
        img = img.convert("RGBA")
    arr = np.asarray(img).copy()
    if crt:
        arr[3::CRT_STEP, :, :3] = (arr[3::CRT_STEP, :, :3].astype(np.float32) * 0.72).astype(np.uint8)
    if grain:
        noise = np.random.normal(0.0, 9.0, arr[..., :3].shape)
        arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr, "RGBA")
    if watermark_text:
        font_size = max(12, result.width // 28)
        font = ImageFont.load_default(size=font_size)
        layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        box = draw.textbbox((0, 0), watermark_text, font=font)
        text_w, text_h = box[2] - box[0], box[3] - box[1]
        x = max(0, result.width - text_w - font_size // 2)
        y = max(0, result.height - text_h - font_size // 2)
        draw.text((x + 1, y + 1), watermark_text, font=font, fill=(0, 0, 0, 150))
        draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 200))
        result = Image.alpha_composite(result, layer)
    if not had_alpha:
        return result.convert("RGB")
    return result


def resize_for_sticker(png_bytes: bytes, max_side: int = STICKER_MAX_SIDE) -> bytes:
    """Уменьшить PNG до стикер-размера (большая сторона 512), NEAREST."""
    img = Image.open(io.BytesIO(png_bytes))
    scale = max_side / max(img.size)
    if scale < 1.0:
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.Resampling.NEAREST,
        )
    buffer = io.BytesIO()
    img.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def _map_to_fixed_palette(img: Image.Image, colors: tuple, dither: bool) -> Image.Image:
    """Сопоставить каждому пикселю ближайший цвет палитры (векторно, redmean)."""
    pixels = np.asarray(img, dtype=np.float64)
    if dither:
        pixels = np.clip(pixels + _bayer_offsets(*img.size)[..., None], 0.0, 255.0)
    flat = pixels.reshape(-1, 3)
    palette = as_numpy(colors)
    mapped = _nearest_palette(flat, palette).reshape(img.size[1], img.size[0], 3)
    return Image.fromarray(mapped, "RGB")


def _nearest_palette(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Ближайший цвет палитры для каждого пикселя, redmean в numpy.

    pixels: (N, 3) float64, palette: (K, 3) uint8 -> (N, 3) uint8.
    Взвешенное расстояние учитывает восприятие: вклад R-канала зависит
    от средней красноты пары (2 + rmean/256), G — фиксированно 4, B — 2 + (255-rmean)/256.
    """
    r = pixels[:, 0][:, None]
    g = pixels[:, 1][:, None]
    b = pixels[:, 2][:, None]
    pr = palette[:, 0].astype(np.float64)[None, :]
    pg = palette[:, 1].astype(np.float64)[None, :]
    pb = palette[:, 2].astype(np.float64)[None, :]
    rmean = (r + pr) / 2.0
    dist = (
        (2.0 + rmean / 256.0) * (r - pr) ** 2
        + 4.0 * (g - pg) ** 2
        + (2.0 + (255.0 - rmean) / 256.0) * (b - pb) ** 2
    )
    return palette[np.argmin(dist, axis=1)]


def _bayer_offsets(width: int, height: int) -> np.ndarray:
    """Пороговая матрица Bayer 4x4, размноженная на весь кадр (диапазон -0.5..0.5)."""
    normalized = (_BAYER_4X4 + 0.5) / 16.0 - 0.5
    tiles_y = math.ceil(height / 4)
    tiles_x = math.ceil(width / 4)
    tiled = np.tile(normalized, (tiles_y, tiles_x))[:height, :width]
    return tiled * DITHER_STRENGTH


# ---------------------------------------------------------------- рендер результата


def render(
    source: bytes,
    settings: Settings,
    custom_palette: tuple | None = None,
    flatten: bool = False,
) -> bytes:
    """Собрать PNG результата из сжатого исходника по настройкам.

    flatten=True (для отправки фото) — прозрачность заливается белым,
    иначе PNG сохраняет альфу (для «📎 PNG» и стикера).
    """
    img = Image.open(io.BytesIO(source)).convert("RGB")
    art = make_pixel_art(
        img,
        pixel_size=settings.pixel_size,
        custom_grid=settings.custom_grid,
        palette=settings.palette,
        n_colors=settings.n_colors,
        dither=settings.dither,
        contrast=settings.contrast,
        saturation=settings.saturation,
        sharpness=settings.sharpness,
        palette_colors=custom_palette,
        cut_bg=settings.cut_bg,
        bg_tolerance=settings.bg_tolerance,
        crop_object=settings.crop_mode != "off",
        padding=0 if settings.crop_mode == "fill" else 15,
        outline=settings.outline,
        show_grid=settings.show_grid,
    )
    art = post_effects(
        art,
        crt=settings.crt,
        grain=settings.grain,
        watermark_text=WATERMARK_TEXT if settings.watermark else None,
    )
    if flatten and art.mode == "RGBA":
        background = Image.new("RGB", art.size, (255, 255, 255))
        background.paste(art, mask=art.getchannel("A"))
        art = background
    buffer = io.BytesIO()
    art.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------- свои палитры и анимации


def extract_palette(img: Image.Image, n_colors: int = 16) -> tuple:
    """Основные цвета изображения (mediancut) — для «своей палитры» из фото."""
    n_colors = max(2, min(64, int(n_colors)))
    quantized = img.quantize(
        colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    palette = quantized.getpalette() or []
    used_indices = sorted(set(np.asarray(quantized).reshape(-1).tolist()))
    colors = []
    for index in used_indices:
        rgb = tuple(int(c) for c in palette[index * 3 : index * 3 + 3])
        if len(rgb) == 3:
            colors.append(rgb)
    return tuple(dict.fromkeys(colors))


def extract_first_frame(data: bytes) -> bytes:
    """Первый кадр анимации как PNG: GIF/WebP — через Pillow, MP4 — через imageio+ffmpeg."""
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)  # первый кадр
        out = io.BytesIO()
        img.save(out, "PNG")
        return out.getvalue()
    except Exception:
        pass  # не картинка для Pillow — пробуем как видео
    try:
        import imageio.v3 as iio

        frame = np.asarray(iio.imread(io.BytesIO(data), index=0, extension=".mp4"))
    except ImportError as exc:
        raise UnsupportedMediaError("модуль imageio не установлен") from exc
    except Exception as exc:
        raise UnsupportedMediaError("не удалось извлечь первый кадр") from exc
    if frame.ndim == 2:  # ч/б кадр
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:  # альфа-канал
        frame = frame[..., :3]
    out = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8), "RGB").save(out, "PNG")
    return out.getvalue()


# ---------------------------------------------------------------- демо-картинка


def build_demo_image() -> Image.Image:
    """Тестовая картинка для /demo: небо, солнце, холмы, домик."""
    width, height = 1024, 768
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    # небо: градиент от синего к закатному
    for y in range(height):
        t = y / height
        r = int(110 + (255 - 110) * t)
        g = int(180 + (180 - 180) * t)
        b = int(255 + (120 - 255) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    # солнце и облака
    draw.ellipse((760, 80, 940, 260), fill=(255, 235, 90))
    for cx, cy, cw in ((160, 140, 220), (430, 90, 160), (620, 210, 180)):
        draw.ellipse((cx - cw // 2, cy - 40, cx + cw // 2, cy + 40), fill=(250, 250, 250))
    # земля и холмы
    draw.ellipse((-200, 640, width + 200, 1100), fill=(74, 120, 60))
    draw.ellipse((500, 600, 1400, 1050), fill=(96, 150, 70))
    draw.rectangle((0, 660, width, height), fill=(70, 110, 55))
    # домик
    draw.rectangle((180, 430, 470, 640), fill=(214, 122, 84))
    draw.polygon([(160, 430), (490, 430), (325, 300)], fill=(146, 66, 52))
    draw.rectangle((240, 480, 330, 560), fill=(255, 222, 120))  # окно
    draw.rectangle((370, 500, 440, 640), fill=(110, 70, 45))  # дверь
    return img
