"""Классическая обработка изображений: фото -> пиксель-арт. Без ИИ, без сети.

Чистые функции (вход/выход — bytes/PIL/numpy), без зависимости от aiogram.
Всё выполняется в памяти; на диск ничего не пишется, метаданные не переносятся.
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .config import (
    DEFAULT_CONTRAST,
    DEFAULT_SATURATION,
    DEFAULT_SHARPNESS,
    MAX_IMAGE_SIDE,
    RESULT_MAX_SIDE,
    SOURCE_MAX_SIDE,
    STORAGE_JPEG_QUALITY,
)
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
    contrast: float = DEFAULT_CONTRAST,
    saturation: float = DEFAULT_SATURATION,
    sharpness: float = DEFAULT_SHARPNESS,
    palette_colors: tuple | None = None,
) -> Image.Image:
    """Пикселизация RGB-изображения: сетка -> квантизация -> ровные квадраты.

    Возвращает RGB-изображение, большая сторона которого — целое кратное
    размера сетки и близка к RESULT_MAX_SIDE. Для palette="custom" нужен
    palette_colors (своя палитра пользователя).
    """
    if pixel_size not in GRID_BY_SIZE:
        raise ValueError(f"неизвестный размер пикселя: {pixel_size}")
    if palette == "custom" and not palette_colors:
        raise ValueError("для своей палитры не заданы цвета")
    if palette != "auto" and palette != "custom" and palette not in FIXED_PALETTES:
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

    # 2) коррекции контраста/насыщенности/резкости, затем BOX-усреднение до сетки
    adjusted = ImageEnhance.Contrast(img).enhance(contrast)
    adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
    adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)
    small = adjusted.resize((grid_w, grid_h), Image.Resampling.BOX)

    # 3) квантизация цветов
    if palette == "auto":
        dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        small = small.quantize(
            colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=dither_mode
        ).convert("RGB")
    elif palette == "custom":
        small = _map_to_fixed_palette(small, palette_colors, dither=dither)
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


def render(source: bytes, settings: Settings, custom_palette: tuple | None = None) -> bytes:
    """Собрать PNG результата из сжатого исходника по настройкам."""
    img = Image.open(io.BytesIO(source)).convert("RGB")
    art = make_pixel_art(
        img,
        pixel_size=settings.pixel_size,
        palette=settings.palette,
        n_colors=settings.n_colors,
        dither=settings.dither,
        contrast=settings.contrast,
        saturation=settings.saturation,
        sharpness=settings.sharpness,
        palette_colors=custom_palette,
    )
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
