"""pytest: добавляет корень проекта в sys.path, чтобы импортировался пакет bot."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
