"""Классическая обработка изображений: фото -> пиксель-арт. Без ИИ, без сети.

Чистые функции (вход/выход — bytes/PIL/numpy), без зависимости от aiogram.
Всё выполняется в памяти; на диск ничего не пишется, метаданные не переносятся.
"""

import io
import math
from collections import OrderedDict

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
from .palettes import FIXED_PALETTES
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

# режимы дизеринга и подписи кнопок
DITHER_MODES: tuple[str, ...] = ("off", "fs", "atkinson", "bayer")
DITHER_LABELS: dict[str, str] = {
    "off": "Выкл",
    "fs": "Флойд–Стейнберг",
    "atkinson": "Аткинсон",
    "bayer": "Байер 8×8",
}


def _bayer_matrix(size: int) -> np.ndarray:
    """Матрица Байера (2→4→8) с порогами 0..1."""
    matrix = np.array([[0.0, 2.0], [3.0, 1.0]])
    while matrix.shape[0] < size:
        matrix = np.block([[4 * matrix, 4 * matrix + 2], [4 * matrix + 3, 4 * matrix + 1]])
    return (matrix + 0.5) / (size * size)


_BAYER_8 = _bayer_matrix(8)


def _nearest_index(buf: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Индексы ближайших цветов палитры для буфера (h, w, 3) float."""
    diff = buf[..., None, :] - palette[None, None, :, :]
    return np.argmin((diff ** 2).sum(-1), axis=-1)


def _apply_dither(pixels: np.ndarray, palette: np.ndarray, mode: str) -> np.ndarray:
    """Квантизация в палитру с выбранным дизерингом. pixels — (h, w, 3) uint8.

    off — без дизеринга; bayer — упорядоченный (Байер 8×8, периодический,
    бесшовный для сеток кратных 8); fs / atkinson — последовательное
    распространение ошибки.
    """
    palette = np.asarray(palette, dtype=np.float64)
    if mode == "fs" or mode == "atkinson":
        return _error_diffusion(pixels, palette, atkinson=(mode == "atkinson"))
    if mode == "bayer":
        height, width = pixels.shape[:2]
        tile = np.tile(_BAYER_8, (height // 8 + 1, width // 8 + 1))[:height, :width]
        shifted = np.clip(pixels.astype(np.float64) + (tile - 0.5)[..., None] * DITHER_STRENGTH, 0, 255)
        return palette[_nearest_index(shifted, palette)].astype(np.uint8)
    shifted = np.clip(pixels.astype(np.float64), 0, 255)
    return palette[_nearest_index(shifted, palette)].astype(np.uint8)


def _error_diffusion(pixels: np.ndarray, palette: np.ndarray, atkinson: bool) -> np.ndarray:
    """Последовательное распространение ошибки: Floyd–Steinberg или Atkinson."""
    height, width = pixels.shape[:2]
    buf = pixels.astype(np.float64).copy()
    out = np.zeros_like(buf)
    if atkinson:
        spread = [(1, 0, 1 / 8), (2, 0, 1 / 8), (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8), (0, 2, 1 / 8)]
    else:
        spread = [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)]
    for y in range(height):
        row = buf[y]
        for x in range(width):
            old = row[x]
            idx = int(np.argmin(((palette - old) ** 2).sum(-1)))
            new = palette[idx]
            out[y, x] = new
            err = old - new
            for dx, dy, weight in spread:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    buf[ny, nx] += err * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _kmeans_palette(pixels: np.ndarray, k: int, iters: int = 12) -> tuple:
    """Палитра k-means (детерминированная) — сохраняет редкие акценты лучше mediancut."""
    flat = pixels.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(0)
    if len(flat) > 8000:
        flat = flat[rng.choice(len(flat), 8000, replace=False)]
    k = max(2, min(k, len(np.unique(flat, axis=0)) or k))
    centers = flat[rng.choice(len(flat), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        assign = np.argmin(((flat[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=-1)
        for j in range(k):
            members = flat[assign == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    unique = {tuple(int(v) for v in c) for c in np.clip(centers, 0, 255).round()}
    return tuple(sorted(unique))


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
    dither: str = "off",
    contrast: float = DEFAULT_CONTRAST,
    saturation: float = DEFAULT_SATURATION,
    sharpness: float = DEFAULT_SHARPNESS,
    palette_colors: tuple | None = None,
    cut_bg: bool = False,
    crop_object: bool = False,
    padding: int = 10,
    outline: str = "off",
    seamless: str = "off",
    details: bool = False,
    show_grid: bool = False,
    return_grid: bool = False,
):
    """Пикселизация: фон/кадр -> сетка -> бесшовность -> квантизация -> квадраты.

    Возвращает RGB/RGBA-изображение (большая сторона — целое кратное сетки,
    около RESULT_MAX_SIDE), а при return_grid=True — (rgb, solid) сетки без
    увеличения. outline: off | dark | volume; seamless: off | blend | mirror.
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

    # 1) маска фона — по исходнику, без коррекций: не зависит от регуляторов
    original = np.asarray(img.convert("RGB")).astype(np.float64)
    mask = None
    if cut_bg or crop_object or outline != "off":
        found, _share = _background_mask(original)
        if found.any():
            mask = found

    # 2) чёткость деталей и коррекции
    source = _enhance_details(img) if details else img
    adjusted = ImageEnhance.Contrast(source).enhance(contrast)
    adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
    adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)

    # 3) фон и кадр
    arr, alpha, mask = _frame_image(adjusted, cut_bg, crop_object, padding, mask)
    width, height = arr.shape[1], arr.shape[0]

    # 4) размер сетки с сохранением пропорций
    grid_big = min(grid_level, max(width, height))
    if width >= height:
        grid_w = grid_big
        grid_h = max(1, round(height * grid_big / width))
    else:
        grid_h = grid_big
        grid_w = max(1, round(width * grid_big / height))

    # 5) premultiplied BOX-усреднение до сетки (прозрачность не «замыливает» цвет)
    src = np.dstack([arr, alpha]).astype(np.float32)
    a = src[..., 3:4] / 255.0
    pre = np.concatenate([src[..., :3] * a, a * 255.0], axis=2).astype(np.uint8)
    small = Image.fromarray(pre, "RGBA").resize((grid_w, grid_h), Image.Resampling.BOX)
    small_rgba = np.asarray(small).astype(np.float32)
    a_grid = small_rgba[..., 3:4]
    solid = a_grid[..., 0] >= 96.0  # жёсткая граница, без полупрозрачных краёв
    rgb_grid = np.clip(small_rgba[..., :3] * (255.0 / np.maximum(a_grid, 1.0)), 0, 255)

    # 6) бесшовный тайл — до квантизации и дизеринга
    if seamless != "off":
        fill = rgb_grid[solid].mean(axis=0) if solid.any() else np.array([255.0, 255.0, 255.0])
        rgb_grid = np.where(solid[..., None], rgb_grid, fill)
        rgb_grid = make_seamless(rgb_grid, seamless)
        if not solid.all():
            weight = make_seamless(solid.astype(np.float64)[..., None], seamless)[..., 0]
            solid = weight >= 0.5

    # 7) квантизация (палитру строим только по видимым пикселям)
    visible = rgb_grid if solid.all() else rgb_grid[solid]
    if len(visible) == 0:
        visible = rgb_grid
    if palette == "auto":
        palette_colors = _kmeans_palette(visible[:, None, :].astype(np.uint8), n_colors)
    elif palette != "custom":
        palette_colors = FIXED_PALETTES[palette]
    pal = np.asarray(palette_colors, dtype=np.uint8)
    rgb_out = _apply_dither(rgb_grid.astype(np.uint8), pal, dither if dither in DITHER_MODES else "off")

    # 8) обводка по силуэту объекта
    if outline != "off":
        if cut_bg:
            obj = solid
        elif mask is not None:
            obj = _shrink_mask(~mask, (grid_w, grid_h))
        else:
            obj = np.ones_like(solid)
        rgb_out = _apply_outline(rgb_out, obj, pal, outline)

    if return_grid:
        return rgb_out, solid

    # 9) сборка и увеличение NEAREST'ом в целое число раз
    scale = max(1, round(RESULT_MAX_SIDE / max(grid_w, grid_h)))
    if solid.all():  # прозрачности нет — обычный RGB
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


def _border_color(arr: np.ndarray) -> np.ndarray:
    """Цвет фона — медиана полосы пикселей по периметру кадра."""
    band = max(2, min(arr.shape[:2]) // 50)
    frame = np.concatenate([
        arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3),
    ])
    return np.median(frame, axis=0)


def _sweep_forward(seed: np.ndarray, cand: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Размножить seed вправо (по строкам): только по пикселям cand и только
    там, где в следующий пиксель можно зайти от предыдущего (step)."""
    width = cand.shape[1]
    starts = cand.copy()
    starts[:, 1:] &= ~step  # новый отрезок начинается там, откуда нельзя зайти слева
    columns = np.arange(width, dtype=np.int32)[None, :]
    start_at = np.maximum.accumulate(np.where(starts, columns, 0), axis=1)
    counts = np.cumsum((seed & cand).astype(np.int32), axis=1)
    before = np.take_along_axis(counts, np.maximum(start_at - 1, 0), axis=1)
    before = np.where(start_at == 0, 0, before)
    return seed | (cand & (counts - before > 0))


def _sweep(seed: np.ndarray, cand: np.ndarray, step: np.ndarray, axis: int, forward: bool) -> np.ndarray:
    """Проход по строкам (axis=1) или столбцам (axis=0) в одну сторону."""
    if axis == 0:
        seed, cand, step = seed.T, cand.T, step.T
    if not forward:
        seed, cand, step = seed[:, ::-1], cand[:, ::-1], step[:, ::-1]
    out = _sweep_forward(seed, cand, step)
    if not forward:
        out = out[:, ::-1]
    return out if axis == 1 else out.T


def _flood(cand: np.ndarray, step_h: np.ndarray | None = None, step_v: np.ndarray | None = None,
           max_cycles: int = 8) -> np.ndarray:
    """Заливка от края кадра (4-связность).

    step_h[y, x] / step_v[y, x] — можно ли зайти в пиксель справа / снизу
    от (y, x); без карт шагов годится любой переход между пикселями cand.
    Вместо волны по одному пикселю — проходы целыми отрезками по строкам и
    столбцам: сходится за несколько циклов на любом разрешении.
    """
    if step_h is None:
        step_h = cand[:, :-1] & cand[:, 1:]
        step_v = cand[:-1, :] & cand[1:, :]
    reached = np.zeros_like(cand)
    reached[0, :] = cand[0, :]
    reached[-1, :] = cand[-1, :]
    reached[:, 0] |= cand[:, 0]
    reached[:, -1] |= cand[:, -1]
    for _ in range(max_cycles):
        before = int(reached.sum())
        for axis, step in ((1, step_h), (0, step_v)):
            reached = _sweep(reached, cand, step, axis, True)
            reached = _sweep(reached, cand, step, axis, False)
        if int(reached.sum()) == before:
            break
    return reached


# соседние пиксели «плавные», если отличаются не сильнее: градиент, тень, шум JPEG
# проходят, текстуры (трава, кожа, ткань) и края объекта — нет
_SMOOTH_STEP = 10.0
_LOCAL_CAP = 200.0  # плавный переход не уводит дальше от цвета фона
_TOLERANCES = (12, 20, 28, 36, 44, 52, 64)
_MIN_BG_COVERAGE = 0.04
_bg_mask_cache: "OrderedDict[tuple, tuple[np.ndarray, float]]" = OrderedDict()


def _bg_candidate(arr: np.ndarray, bg: np.ndarray, tolerance: float):
    """Кандидаты в фон и карты переходов для заливки.

    cand — пиксели, которым вообще разрешено быть фоном: близкие к цвету фона
    ИЛИ не слишком от него далёкие. step — в какой соседний пиксель можно зайти:
    если он близок к цвету фона, либо переход к нему плавный (виньетка, мягкая
    тень, шум сжатия). Край объекта — резкий скачок — переход не пропускает.
    """
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    close = dist <= tolerance
    step_h = np.sqrt(((arr[:, :-1] - arr[:, 1:]) ** 2).sum(axis=2))  # (h, w-1)
    step_v = np.sqrt(((arr[:-1, :] - arr[1:, :]) ** 2).sum(axis=2))  # (h-1, w)
    return (
        close | (dist <= _LOCAL_CAP),
        close[:, 1:] | (step_h <= _SMOOTH_STEP),
        close[1:, :] | (step_v <= _SMOOTH_STEP),
    )


def _background_mask(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Маска фона кадра (True — фон) и её доля.

    Фон — то, что достижимо от края кадра по фоновым пикселям. Порог схожести
    с цветом фона подбирается: растёт, пока заливка не начинает съедать объект
    (резкий прирост доли). Фона нет — маска пустая, доля 0. Кэш по кадру,
    чтобы перерисовка кнопками не пересчитывала заливку.
    """
    key = (hash(arr.astype(np.uint8).tobytes()), arr.shape)
    cached = _bg_mask_cache.get(key)
    if cached is not None:
        _bg_mask_cache.move_to_end(key)
        return cached
    band = max(2, min(arr.shape[:2]) // 50)
    sides = [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
             arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)]
    medians = np.array([np.median(side, axis=0) for side in sides])
    spread = max(np.sqrt(((medians[i] - medians[j]) ** 2).sum())
                 for i in range(4) for j in range(i + 1, 4))
    if spread > 100:  # периметр разноцветный — однородного фона в кадре нет
        return np.zeros(arr.shape[:2], dtype=bool), 0.0
    bg = _border_color(arr)
    sub = arr[::8, ::8]
    tolerance = _TOLERANCES[0]
    coverage = _flood(*_bg_candidate(sub, bg, tolerance)).mean()
    for candidate in _TOLERANCES[1:]:
        grown = _flood(*_bg_candidate(sub, bg, candidate)).mean()
        if grown - coverage > max(0.015, 0.08 * (1.0 - coverage)):
            break
        tolerance, coverage = candidate, grown
    flooded = _flood(*_bg_candidate(arr, bg, tolerance))
    share = float(flooded.mean())
    if share < _MIN_BG_COVERAGE or share > 0.985:  # фона нет либо съелось всё
        flooded = np.zeros(arr.shape[:2], dtype=bool)
        share = 0.0
    _bg_mask_cache[key] = (flooded, share)
    while len(_bg_mask_cache) > 64:
        _bg_mask_cache.popitem(last=False)
    return flooded, share


def _frame_image(
    img: Image.Image,
    cut_bg: bool,
    crop_object: bool,
    padding: int,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Вырезать фон по маске и/или кадрировать до объекта.

    Возвращает (RGB, альфа, маска фона) одного размера; без вырезки альфа
    непрозрачна. mask=None — однородного фона в кадре не нашлось.
    """
    arr = np.asarray(img.convert("RGB"))
    height, width = arr.shape[:2]
    if mask is not None and cut_bg:
        alpha = np.where(mask, 0, 255).astype(np.uint8)
    else:
        alpha = np.full((height, width), 255, dtype=np.uint8)
    if crop_object:
        if mask is not None:
            content = ~mask
        else:
            far = np.sqrt(((arr.astype(np.float64) - _border_color(arr)) ** 2).sum(axis=2))
            content = far > UNIFORM_TRIM_TOLERANCE
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
            if mask is not None:
                mask = mask[r0:r1, c0:c1]
    return arr, alpha, mask


def _erode4(mask: np.ndarray) -> np.ndarray:
    """Сжать маску на 1 пиксель (за краем кадра считается «объект»)."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _shrink_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Уменьшить маску до сетки: пиксель покрыт, если занят от ~38%."""
    small = Image.fromarray((mask * 255).astype(np.uint8), "L").resize(size, Image.Resampling.BOX)
    return np.asarray(small) >= 96


_OUTLINE_WEIGHTS = np.array([0.299, 0.587, 0.114])


def _apply_outline(rgb: np.ndarray, obj: np.ndarray, palette, mode: str) -> np.ndarray:
    """Обводка по силуэту объекта, только цветами палитры.

    obj — маска объекта на сетке. dark — тёмный контур в 1 пиксель по краю
    объекта (внутренние границы цветов и край кадра не обводятся).
    volume — тот же контур + второй ряд: свет сверху, тень снизу.
    """
    ring = obj & ~_erode4(obj)
    if not ring.any():
        return rgb
    pal = np.asarray(palette, dtype=np.float64)
    lum = pal @ _OUTLINE_WEIGHTS
    out = rgb.astype(np.float64).copy()
    darkest = int(np.argmin(lum))
    if lum[darkest] <= 80:
        out[ring] = pal[darkest]
    else:  # тёмного в палитре нет — ближайший к затемнённому пикселю
        cur = rgb[ring].astype(np.float64) * 0.45
        out[ring] = pal[((cur[:, None, :] - pal[None, :, :]) ** 2).sum(-1).argmin(1)]
    if mode != "volume":
        return np.clip(out, 0, 255).astype(np.uint8)

    inner = obj & ~ring
    second = inner & ~_erode4(inner)  # ряд объекта вплотную к контуру
    above = np.zeros_like(obj)
    above[1:, :] = ring[:-1, :]
    below = np.zeros_like(obj)
    below[:-1, :] = ring[1:, :]
    top_facing = second & above
    bottom_facing = second & below & ~top_facing
    side = second & ~top_facing & ~bottom_facing
    for selection, factor, lighter in (
        (top_facing, 1.35, True), (bottom_facing, 0.6, False), (side, 0.8, False),
    ):
        if not selection.any():
            continue
        current = rgb[selection].astype(np.float64)
        target = current * factor
        candidate = pal[((target[:, None, :] - pal[None, :, :]) ** 2).sum(-1).argmin(1)]
        keep = (candidate @ _OUTLINE_WEIGHTS >= current @ _OUTLINE_WEIGHTS) if lighter else (
            candidate @ _OUTLINE_WEIGHTS <= current @ _OUTLINE_WEIGHTS
        )
        out[selection] = np.where(keep[:, None], candidate, current)
    return np.clip(out, 0, 255).astype(np.uint8)


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
        # половины берём с запасом: центральная ось отражается сама на себя
        quarter = arr[: math.ceil(height / 2), : math.ceil(width / 2)]
        top = np.concatenate([quarter, quarter[:, ::-1]], axis=1)[:, :width]
        return np.concatenate([top, top[::-1, :]], axis=0)[:height]
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


def render_pack_tile(grid_arr: np.ndarray, palette: tuple, dither: str) -> np.ndarray:
    """Квантизация тайла в палитру пака; последовательный дизеринг (FS/Аткинсон)
    — по 3×3-размножению (wrap-around), чтобы ошибка шла и через швы; Байер —
    периодическая матрица, сама бесшовная.

    RGBA-вход (пак с прозрачностью): альфа бинаризуется.
    """
    n = grid_arr.shape[0]
    has_alpha = grid_arr.shape[-1] == 4
    rgb = grid_arr[..., :3]
    alpha = grid_arr[..., 3] if has_alpha else None
    pal = np.asarray(palette, dtype=np.uint8)
    if dither in ("fs", "atkinson"):
        rgb_out = _apply_dither(np.tile(rgb, (3, 3, 1)), pal, dither)[n : 2 * n, n : 2 * n]
    else:
        rgb_out = _apply_dither(rgb, pal, dither)
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


def mosaic_sheet(tiles: list[np.ndarray], per_side: int = 3,
                 max_side: int = RESULT_MAX_SIDE, gap: int = 8) -> Image.Image:
    """Превью пака: каждый тайл сложен мозаикой per_side×per_side — стыки
    видны сразу, до открытия архива."""
    count = len(tiles)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    size = tiles[0].shape[0]
    cell = max(per_side * size, min(768, (max_side - gap * (cols + 1)) // cols))
    sheet = Image.new("RGB", (cols * cell + gap * (cols + 1), rows * cell + gap * (rows + 1)), (255, 255, 255))
    for index, tile in enumerate(tiles):
        rgb = tile[..., :3]
        if tile.shape[-1] == 4:  # прозрачное кладём на белый
            rgb = np.where(tile[..., 3:4] >= 128, rgb, 255)
        mosaic = np.tile(rgb, (per_side, per_side, 1)).astype(np.uint8)
        piece = Image.fromarray(mosaic, "RGB")
        if cell >= piece.width:  # увеличение целым числом раз — пиксели ровные
            factor = max(1, cell // piece.width)
            piece = piece.resize((piece.width * factor, piece.height * factor), Image.Resampling.NEAREST)
        else:
            piece = piece.resize((cell, cell), Image.Resampling.NEAREST)
        row, col = divmod(index, cols)
        left = gap + col * (cell + gap) + (cell - piece.width) // 2
        top = gap + row * (cell + gap) + (cell - piece.height) // 2
        sheet.paste(piece, (left, top))
    return sheet


# ---------------------------------------------------------------- геймдев-инструменты


_DETAILS_STRENGTH = 0.5  # доля локального контраста в результате (мягкий эффект)


def _enhance_details(img: Image.Image, clip: float = 2.0, tiles: int = 8) -> Image.Image:
    """Чёткость мелких деталей: мягкий локальный контраст по яркости
    (упрощённый CLAHE, билинейные LUT), подмешанный к оригиналу — плоские
    участки почти не меняются, мелкие элементы не сливаются при уменьшении."""
    arr = np.asarray(img.convert("RGB")).astype(np.float64)
    h, w, _ = arr.shape
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    bins = np.clip((lum * 255).astype(np.int64), 0, 255)
    ys = np.linspace(0, h, tiles + 1).astype(int)
    xs = np.linspace(0, w, tiles + 1).astype(int)
    luts = np.zeros((tiles, tiles, 256))
    for i in range(tiles):
        for j in range(tiles):
            block = bins[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel()
            hist = np.bincount(block, minlength=256).astype(np.float64)
            limit = max(1.0, clip * block.size / 256)
            excess = hist[hist > limit].sum() - np.minimum(hist, limit).sum()
            hist = np.minimum(hist, limit) + excess / 256
            cdf = hist.cumsum()
            luts[i, j] = cdf / cdf[-1] if cdf[-1] > 0 else np.arange(256) / 255
    gy = np.clip(np.arange(h) / max(1, h) * tiles - 0.5, 0, tiles - 1)
    gx = np.clip(np.arange(w) / max(1, w) * tiles - 0.5, 0, tiles - 1)
    i0 = np.floor(gy).astype(int)
    i1 = np.minimum(i0 + 1, tiles - 1)
    j0 = np.floor(gx).astype(int)
    j1 = np.minimum(j0 + 1, tiles - 1)
    wy = (gy - i0)[:, None]
    wx = (gx - j0)[None, :]
    v00 = luts[i0[:, None], j0[None, :], bins]
    v01 = luts[i0[:, None], j1[None, :], bins]
    v10 = luts[i1[:, None], j0[None, :], bins]
    v11 = luts[i1[:, None], j1[None, :], bins]
    newlum = (v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx
              + v10 * wy * (1 - wx) + v11 * wy * wx)
    scale = np.where(lum > 1 / 255, newlum / np.maximum(lum, 1 / 255), 1.0)
    out = np.clip(arr * (1.0 - _DETAILS_STRENGTH * (1.0 - scale[..., None])), 0, 255)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def normal_map(tile: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Карта нормалей тайла из яркости (градиенты Собеля через np.gradient,
    OpenGL-ориентация, Y+ вверх). Вход/выход — (n, n, 3) uint8."""
    rgb = tile[..., :3].astype(np.float64) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    d_row, d_col = np.gradient(lum)
    nx = -d_col * strength
    ny = -d_row * strength  # Y+ вверх: рост яркости вниз наклоняет вверх
    nz = np.ones_like(nx)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    encoded = (np.dstack([nx, ny, nz]) / norm[..., None]) * 0.5 + 0.5
    return (np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def build_autotile_set(grid_a: np.ndarray, grid_b: np.ndarray) -> list[np.ndarray]:
    """16 blob-тайлов перехода B->A по битам 4 углов (TL, TR, BL, BR).
    Тайлы смешаны по диагональным ramps, квантизация — на этапе пака."""
    n = grid_a.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    tx = xx / max(1, n - 1)
    ty = yy / max(1, n - 1)
    ramps = {
        0: 1 - np.maximum(tx, ty),            # TL
        1: 1 - np.maximum(1 - tx, ty),        # TR
        2: 1 - np.maximum(tx, 1 - ty),        # BL
        3: 1 - np.maximum(1 - tx, 1 - ty),    # BR
    }
    tiles = []
    for bits in range(16):
        weight = np.zeros((n, n))
        for bit in range(4):
            if bits >> bit & 1:
                weight = np.maximum(weight, ramps[bit])
        mixed = grid_b * weight[..., None] + grid_a * (1 - weight[..., None])
        tiles.append(np.clip(mixed, 0, 255).astype(np.uint8))
    return tiles


def extract_frames(data: bytes, max_frames: int = 12) -> list[Image.Image]:
    """Раскадровать GIF/WebP (Pillow) или MP4 (imageio) — до max_frames
    кадров, выбранных равномерно по всей длительности."""
    try:
        img = Image.open(io.BytesIO(data))
        if getattr(img, "is_animated", False) and img.n_frames > 1:
            indices = np.linspace(0, img.n_frames - 1, min(img.n_frames, max_frames)).astype(int)
            frames = []
            for index in indices:
                img.seek(int(index))
                frames.append(_flatten_to_rgb(img.copy(), keep_alpha=True).convert("RGB"))
            return frames
    except UnsupportedMediaError:
        raise
    except Exception:
        pass  # не анимация для Pillow — пробуем как видео
    try:
        import imageio.v3 as iio

        collected = []
        for frame in iio.imiter(io.BytesIO(data), extension=".mp4"):
            collected.append(np.asarray(frame)[..., :3])
            if len(collected) >= max_frames * 4:
                break
        if not collected:
            raise UnsupportedMediaError("кадры не найдены")
        pick = np.linspace(0, len(collected) - 1, min(max_frames, len(collected))).astype(int)
        return [Image.fromarray(np.asarray(collected[i]), "RGB") for i in pick]
    except ImportError as exc:
        raise UnsupportedMediaError("модуль imageio не установлен") from exc
    except UnsupportedMediaError:
        raise
    except Exception as exc:
        raise UnsupportedMediaError("не удалось раскадровать") from exc


def build_sprite_sheet(
    frames: list[Image.Image],
    grid: int,
    palette: tuple | None,
    dither: str,
    colors: int = 32,
) -> tuple[bytes, bytes]:
    """Спрайт-лист: все кадры в одной общей палитре, горизонтальная лента
    PNG нативного размера + увеличенное превью. Возвращает (png, preview)."""
    grids = []
    for frame in frames:
        small = frame.resize((grid, grid), Image.Resampling.BOX)
        grids.append(np.clip(np.asarray(small.convert("RGB"), dtype=np.float64), 0, 255).astype(np.uint8))
    palette_res = palette or shared_palette_from_grids(grids, colors)
    tiles = [render_pack_tile(g, palette_res, dither) for g in grids]
    strip = np.concatenate(tiles, axis=1)
    png_buffer = io.BytesIO()
    Image.fromarray(strip, "RGB").save(png_buffer, "PNG", optimize=True)
    preview = np.kron(strip, np.ones((4, 4, 1), dtype=np.uint8))
    preview_buffer = io.BytesIO()
    Image.fromarray(preview, "RGB").save(preview_buffer, "PNG", optimize=True)
    return png_buffer.getvalue(), preview_buffer.getvalue()


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


def _render_params(settings: Settings, custom_palette: tuple | None) -> dict:
    """Параметры make_pixel_art по настройкам — общие для результата и мозаики."""
    return dict(
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
        crop_object=settings.crop_mode != "off",
        padding=0 if settings.crop_mode == "fill" else 15,
        outline=settings.outline,
        seamless=settings.seamless,
        details=settings.details,
    )


def render(
    source: bytes,
    settings: Settings,
    custom_palette: tuple | None = None,
    flatten: bool = False,
) -> bytes:
    """Собрать PNG результата из сжатого исходника по настройкам.

    flatten=True (для отправки фото) — прозрачность заливается белым,
    иначе PNG сохраняет альфу (для «PNG» и стикера).
    """
    img = Image.open(io.BytesIO(source)).convert("RGB")
    art = make_pixel_art(img, **_render_params(settings, custom_palette), show_grid=settings.show_grid)
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


def tiled_preview(source: bytes, settings: Settings, custom_palette: tuple | None = None,
                  tiles: int = 3) -> bytes:
    """PNG: результат, сложенный мозаикой tiles×tiles — проверка бесшовности.

    Берёт ту же сетку, что и результат (без эффектов и водяного знака).
    """
    img = Image.open(io.BytesIO(source)).convert("RGB")
    rgb, solid = make_pixel_art(img, **_render_params(settings, custom_palette), return_grid=True)
    mosaic = np.tile(np.where(solid[..., None], rgb, 255).astype(np.uint8), (tiles, tiles, 1))
    scale = max(1, RESULT_MAX_SIDE // max(mosaic.shape[:2]))
    if scale > 1:
        mosaic = np.repeat(np.repeat(mosaic, scale, axis=0), scale, axis=1)
    buffer = io.BytesIO()
    Image.fromarray(mosaic, "RGB").save(buffer, "PNG", optimize=True)
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
