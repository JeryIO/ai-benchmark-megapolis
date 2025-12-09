# -*- coding: utf-8 -*-
"""CLI интерфейс для запуска бенчмарка LLM моделей."""

import argparse
import logging
import sys
from pathlib import Path

from .config import ModelConfig, BenchmarkConfig, TestDataConfig
from .runner import BenchmarkRunner


def setup_logging(verbose: bool = False) -> None:
    """Настроить логирование."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Бенчмарк LLM для планирования строительных работ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m src.cli -m qwen2.5:32b -t data/etalon_tasks.json
  python -m src.cli -m qwen2.5:32b -t data/etalon_tasks.json -g data/etalon_ground_truth.json
  python -m src.cli -m Qwen/Qwen2.5-32B -u http://localhost:8000/v1 -t data/tasks.json
        """
    )

    parser.add_argument("-m", "--model", required=True, help="Название модели")
    parser.add_argument("-t", "--tasks", required=True, type=Path, help="JSON файл с задачами")
    parser.add_argument("-u", "--url", default="http://localhost:11434/v1", help="URL API")
    parser.add_argument("-k", "--api-key", default=None, help="API ключ")
    parser.add_argument("-p", "--prompt", choices=["basic", "few_shot", "expert"], default="expert")
    parser.add_argument("-c", "--chunk-size", type=int, default=100, help="Задач в запросе")
    parser.add_argument("-g", "--ground-truth", type=Path, default=None, help="JSON с эталоном")
    parser.add_argument("-o", "--output", type=Path, default=Path("./results"), help="Директория результатов")
    parser.add_argument("--timeout", type=int, default=300, help="Таймаут (сек)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=None, help="Макс. токенов (None = без ограничений)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты")

    return parser.parse_args()


def main() -> int:
    """Главная функция CLI."""
    args = parse_args()
    setup_logging(args.verbose)

    if not args.tasks.exists():
        print(f"Ошибка: файл не найден: {args.tasks}")
        return 1

    model_kwargs = {
        "name": args.model,
        "base_url": args.url,
        "api_key": args.api_key,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        model_kwargs["max_tokens"] = args.max_tokens
    model_config = ModelConfig(**model_kwargs)

    benchmark_config = BenchmarkConfig(
        model=model_config,
        prompt_type=args.prompt,
        chunk_size=args.chunk_size,
        timeout_seconds=args.timeout,
        output_dir=args.output,
    )

    test_data_config = TestDataConfig(
        tasks_file=args.tasks,
        ground_truth_file=args.ground_truth,
    )

    runner = BenchmarkRunner(benchmark_config, test_data_config)

    try:
        result = runner.run()
    except KeyboardInterrupt:
        print("\nПрервано")
        return 130
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        return 1

    if not args.no_save:
        output_path = runner.save_results(result)
        print(f"\nРезультаты: {output_path}")

    # Считаем критическими только ошибки парсинга JSON
    critical_errors = [e for e in result.metrics.errors if "Ошибка парсинга JSON" in e]
    return 1 if critical_errors else 0


if __name__ == "__main__":
    sys.exit(main())
