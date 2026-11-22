"""pytest: добавляет корень проекта в sys.path и включает тестового админа,
чтобы админские перехватчики проверялись во всех тестах (как на проде)."""

import os
import sys

# ДО импорта bot.* — config парсит ADMIN_IDS при импорте
os.environ.setdefault("ADMIN_IDS", "424242")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
