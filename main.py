"""Точка входа pixel-art-bot: long polling, переменные окружения BOT_TOKEN и ADMIN_IDS."""

import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from bot.admin import broadcast_worker
from bot.handlers import periodic_purge, router

_tasks: list[asyncio.Task] = []


async def _on_startup(bot: Bot) -> None:
    _tasks.append(asyncio.create_task(periodic_purge()))
    _tasks.append(asyncio.create_task(broadcast_worker(bot)))
    logging.info("бот запущен (long polling)")


async def _on_shutdown(bot: Bot) -> None:
    for task in _tasks:
        task.cancel()
    _tasks.clear()
    logging.info("бот остановлен")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    token = os.getenv("BOT_TOKEN")
    if not token:
        logging.critical(
            "Переменная окружения BOT_TOKEN не задана — запуск невозможен. "
            "Пример: BOT_TOKEN=123456:ABC-DEF python main.py"
        )
        sys.exit(1)

    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(_on_startup)
    dp.shutdown.register(_on_shutdown)

    bot = Bot(token, default=DefaultBotProperties(parse_mode=None))
    # корректное завершение по SIGINT/SIGTERM внутри run_polling
    dp.run_polling(bot, allowed_updates=["message", "callback_query", "inline_query"])


if __name__ == "__main__":
    main()
