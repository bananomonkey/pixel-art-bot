"""Классическая обработка изображений: фото -> пиксель-арт. Без ИИ, без сети.

Чистые функции (вход/выход — bytes/PIL/numpy), без зависимости от aiogram.
Всё выполняется в памяти; на диск ничего не пишется, метаданные не переносятся.
"""

import io
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .config import MAX_IMAGE_SIDE, RESULT_MAX_SIDE, SOURCE_MAX_SIDE, STORAGE_JPEG_QUALITY
from .palettes import FIXED_PALETTES, as_numpy
from .state import Settings

# размер сетки (px по большей стороне) для каждого уровня кнопки «размер пикселя»
GRID_BY_SIZE: dict[str, int] = {"small": 128, "medium": 96, "large": 64, "huge": 32}

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


def decode_source(data: bytes) -> Image.Image:
    """Открыть присланное фото и подготовить исходник.

    Проверяет ограничения (4000x4000), выпрямляет по EXIF, прозрачность
    заливает белым, уменьшает до SOURCE_MAX_SIDE по большей стороне.
    Метаданные (EXIF и прочее) не переносятся.
    """
    try:
        img = Image.open(io.BytesIO(data))
        if getattr(img, "is_animated", False):
            raise UnsupportedMediaError("анимированные изображения не поддерживаются")
        # проверка по заголовку, до декодирования пикселей
        if max(img.size) > MAX_IMAGE_SIDE:
            raise ImageTooLargeError("изображение больше 4000x4000")
        img = ImageOps.exif_transpose(img)
        img = _flatten_to_rgb(img)
        img.thumbnail((SOURCE_MAX_SIDE, SOURCE_MAX_SIDE), Image.Resampling.LANCZOS)
        return img
    except (ImageTooLargeError, UnsupportedMediaError):
        raise
    except Exception as exc:  # битый файл / не изображение / декомпрессия
        raise UnsupportedMediaError(f"не удалось открыть изображение: {type(exc).__name__}") from exc


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Привести к RGB; прозрачность PNG/LA/P — залить белым."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
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
    palette: str = "auto",
    n_colors: int = 16,
    dither: bool = False,
) -> Image.Image:
    """Пикселизация RGB-изображения: сетка -> квантизация -> ровные квадраты.

    Возвращает RGB-изображение, большая сторона которого — целое кратное
    размера сетки и близка к RESULT_MAX_SIDE.
    """
    if pixel_size not in GRID_BY_SIZE:
        raise ValueError(f"неизвестный размер пикселя: {pixel_size}")
    if palette != "auto" and palette not in FIXED_PALETTES:
        raise ValueError(f"неизвестная палитра: {palette}")
    n_colors = max(2, min(64, int(n_colors)))

    # 1) размер сетки с сохранением пропорций (без апскейла сетки для крошечных фото)
    width, height = img.size
    grid_big = min(GRID_BY_SIZE[pixel_size], max(width, height))
    if width >= height:
        grid_w = grid_big
        grid_h = max(1, round(height * grid_big / width))
    else:
        grid_h = grid_big
        grid_w = max(1, round(width * grid_big / height))

    # 2) лёгкое усиление контраста и насыщенности, затем BOX-усреднение до сетки
    enhanced = ImageEnhance.Contrast(img).enhance(1.1)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.15)
    small = enhanced.resize((grid_w, grid_h), Image.Resampling.BOX)

    # 3) квантизация цветов
    if palette == "auto":
        dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        small = small.quantize(
            colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=dither_mode
        ).convert("RGB")
    else:
        small = _map_to_fixed_palette(small, FIXED_PALETTES[palette], dither=dither)

    # 4) увеличение NEAREST'ом в целое число раз — пиксели остаются ровными квадратами
    scale = max(1, round(RESULT_MAX_SIDE / max(grid_w, grid_h)))
    return small.resize(
        (grid_w * scale, grid_h * scale), Image.Resampling.NEAREST
    )


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


def render(source: bytes, settings: Settings) -> bytes:
    """Собрать PNG результата из сжатого исходника по настройкам."""
    img = Image.open(io.BytesIO(source)).convert("RGB")
    art = make_pixel_art(
        img,
        pixel_size=settings.pixel_size,
        palette=settings.palette,
        n_colors=settings.n_colors,
        dither=settings.dither,
    )
    buffer = io.BytesIO()
    art.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()
