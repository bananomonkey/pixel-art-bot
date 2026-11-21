"""Конфигурация: переменные окружения и константы.

Обязательная переменная — BOT_TOKEN. Необязательные:
ADMIN_IDS — запятая-разделённые Telegram ID админов (панель и рассылки).
"""

import os

# --- обязательное ---
BOT_TOKEN = os.getenv("BOT_TOKEN")

# --- админы (панель /admin, рассылки) ---
def parse_admin_ids(raw: str) -> tuple[int, ...]:
    """'111, 222' -> (111, 222); мусор и дубли игнорируются."""
    ids: list[int] = []
    for part in (raw or "").replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            ids.append(int(part))
    return tuple(dict.fromkeys(ids))


ADMIN_IDS = parse_admin_ids(os.getenv("ADMIN_IDS", ""))

# --- ограничения входных файлов ---
MAX_FILE_SIZE = 20 * 1024 * 1024  # не более 20 МБ на файл
MAX_IMAGE_SIDE = 4000  # не более 4000 px по большей стороне

# --- память: всё в RAM, на диск ничего не пишем ---
SOURCE_MAX_SIDE = 1024  # исходник хранится уменьшенным до 1024 px по большей стороне
RESULT_MAX_SIDE = 1024  # большая сторона результата ~1024 px (целое кратное сетки)

# --- состояние ---
STATE_MAX_ENTRIES = 200  # LRU на 200 записей
STATE_TTL_SECONDS = 30 * 60  # TTL записи — 30 минут (скользящий, обновляется при нажатии кнопок)

# --- устойчивость ---
RATE_LIMIT_SECONDS = 3.0  # не более 1 фото в 3 секунды на пользователя
MAX_CONCURRENT_JOBS = 2  # глобальный семафор обработки

# --- формат хранения исходника в памяти ---
STORAGE_JPEG_QUALITY = 92

# --- рассылки (админ-панель) ---
CHATS_MAX_ENTRIES = 10_000  # LRU-реестр чатов для рассылок
BROADCAST_MAX_ACTIVE = 20  # максимум одновременных рассылок
BROADCAST_MIN_INTERVAL = 60.0  # секунд (минимум — 1 минута)
BROADCAST_MAX_INTERVAL = 30 * 86400.0  # секунд (максимум — 30 дней)
BROADCAST_SEND_RATE = 20  # сообщений в секунду (лимит Telegram ~30)
BROADCAST_TICK = 5.0  # период опроса расписания воркером
