# -*- coding: utf-8 -*-
"""
Точка входа для запуска бенчмарка как модуля.

Использование:
    python -m src --model qwen2.5:32b --tasks data/tasks.json
"""

from .cli import main

if __name__ == "__main__":
    main()
