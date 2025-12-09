#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Полный бенчмарк: все модели × все типы промптов.

Поддерживает локальные (Ollama, vLLM) и облачные (OpenRouter) API.

Примеры использования:
    # Локальные модели через Ollama
    python run_full_benchmark.py --provider ollama

    # Облачные модели через OpenRouter
    export OPENROUTER_API_KEY='sk-or-v1-...'
    python run_full_benchmark.py --provider openrouter
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Конфигурации провайдеров
PROVIDERS = {
    "ollama": {
        "url": "http://localhost:11434/v1",
        "api_key": None,
        "models": [
            "qwen2.5:32b",
            "qwen2.5:14b",
        ],
    },
    "vllm": {
        "url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "models": [
            "Qwen/Qwen2.5-32B-Instruct",
        ],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": [
            "deepseek/deepseek-v3.2",
            "qwen/qwen3-235b-a22b-2507",
            "qwen/qwen3-30b-a3b",
            "qwen/qwen3-14b",
        ],
    },
}

# Типы промптов
PROMPTS = ["basic", "few_shot", "expert"]

# Тестовые данные
TASKS_FILE = "data/etalon_tasks.json"
GROUND_TRUTH = "data/etalon_ground_truth.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Полный бенчмарк: все модели × все типы промптов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="ollama",
        help="Провайдер API (по умолчанию: ollama)",
    )
    parser.add_argument(
        "--url",
        help="URL API (переопределяет настройку провайдера)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Список моделей (переопределяет настройку провайдера)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=PROMPTS,
        default=PROMPTS,
        help="Типы промптов для тестирования",
    )
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=50,
        help="Количество задач в запросе (по умолчанию: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Таймаут запроса в секундах (по умолчанию: 600)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Пауза между тестами в секундах (по умолчанию: 5)",
    )
    return parser.parse_args()


def run_benchmark(model: str, prompt_type: str, api_url: str, api_key: str,
                  args, results_dir: Path) -> bool:
    """Запустить бенчмарк для одной комбинации модель/промпт."""
    print(f"\n{'='*60}")
    print(f"МОДЕЛЬ: {model}")
    print(f"ПРОМПТ: {prompt_type}")
    print(f"ВРЕМЯ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "src.cli",
        "-m", model,
        "-u", api_url,
        "-t", TASKS_FILE,
        "-g", GROUND_TRUTH,
        "-o", str(results_dir),
        "-p", prompt_type,
        "-c", str(args.chunk_size),
        "--timeout", str(args.timeout),
        "--temperature", "0.1",
    ]

    if api_key:
        cmd.extend(["-k", api_key])

    try:
        result = subprocess.run(cmd, check=False)
        success = result.returncode == 0
        print(f"\n>>> {'УСПЕХ' if success else 'ОШИБКА'}: {model} + {prompt_type}")
        return success
    except Exception as e:
        print(f"\n>>> ИСКЛЮЧЕНИЕ: {model} + {prompt_type} - {e}")
        return False


def main():
    args = parse_args()
    provider_config = PROVIDERS[args.provider]

    # Определение URL
    api_url = args.url or provider_config["url"]

    # Определение API ключа
    api_key = provider_config.get("api_key")
    if "api_key_env" in provider_config:
        api_key = os.environ.get(provider_config["api_key_env"])
        if not api_key:
            print(f"Ошибка: установите переменную окружения {provider_config['api_key_env']}")
            sys.exit(1)

    # Определение списка моделей
    models = args.models or provider_config["models"]
    prompts = args.prompts

    total_tests = len(models) * len(prompts)

    # Директория результатов
    results_dir = Path(f"./results/full_{args.provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print("="*60)
    print("LLM BENCHMARK - FULL MATRIX")
    print("="*60)
    print(f"Провайдер: {args.provider}")
    print(f"URL: {api_url}")
    print(f"Моделей: {len(models)}")
    print(f"Промптов: {len(prompts)} ({', '.join(prompts)})")
    print(f"Всего тестов: {total_tests}")
    print(f"Результаты: {results_dir}")
    print("="*60)

    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    test_num = 0

    for prompt_type in prompts:
        print(f"\n{'#'*60}")
        print(f"# ПРОМПТ: {prompt_type}")
        print(f"{'#'*60}")

        for model in models:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] {model} + {prompt_type}")

            success = run_benchmark(model, prompt_type, api_url, api_key, args, results_dir)
            results[(model, prompt_type)] = success

            if test_num < total_tests and args.delay > 0:
                print(f"\nПауза {args.delay} сек...")
                time.sleep(args.delay)

    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ")
    print("="*60)

    for prompt_type in prompts:
        print(f"\n--- {prompt_type} ---")
        for model in models:
            status = "OK" if results.get((model, prompt_type)) else "FAIL"
            print(f"  [{status}] {model}")

    success_count = sum(results.values())
    print(f"\nУспешно: {success_count}/{total_tests}")
    print(f"Результаты: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
