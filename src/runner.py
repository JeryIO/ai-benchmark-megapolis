# -*- coding: utf-8 -*-
"""Основной runner бенчмарка для тестирования LLM моделей."""

import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .config import BenchmarkConfig, TestDataConfig
from .llm_client import LLMClient, LLMError
from .prompts import build_prompt, get_prompt_template
from .metrics import (
    BenchmarkMetrics,
    calculate_dependency_metrics,
    calculate_graph_metrics,
    parse_llm_response,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Результат обработки одного чанка."""
    chunk_index: int
    tasks_count: int
    success: bool = False
    latency_ms: float = 0.0
    tokens_used: int = 0
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Полный результат бенчмарка."""
    model_name: str
    prompt_type: str
    timestamp: str
    total_tasks: int
    chunks_processed: int
    metrics: BenchmarkMetrics
    chunk_results: List[ChunkResult] = field(default_factory=list)
    all_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runner для запуска бенчмарка."""

    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        test_data_config: TestDataConfig,
    ):
        self.config = benchmark_config
        self.test_data = test_data_config
        self.client = LLMClient(
            config=benchmark_config.model,
            timeout=benchmark_config.timeout_seconds,
            retries=benchmark_config.retry_count,
        )
        self.prompt_template = get_prompt_template(benchmark_config.prompt_type)

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Загрузить задачи из файла."""
        with open(self.test_data.tasks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "tasks" in data:
            return data["tasks"]

        raise ValueError(f"Неверный формат файла задач: {self.test_data.tasks_file}")

    def load_ground_truth(self) -> Optional[Dict[str, List[str]]]:
        """Загрузить эталонные зависимости."""
        if not self.test_data.ground_truth_file:
            return None

        if not self.test_data.ground_truth_file.exists():
            logger.warning(f"Файл ground truth не найден: {self.test_data.ground_truth_file}")
            return None

        with open(self.test_data.ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "dependencies" in data:
            return data["dependencies"]

        return data

    def validate_ground_truth(
        self,
        tasks: List[Dict[str, Any]],
        ground_truth: Dict[str, List[str]]
    ) -> List[str]:
        """Проверить согласованность ground truth с задачами."""
        warnings = []
        task_names = {t.get("name", "") for t in tasks}

        for task, deps in ground_truth.items():
            if task not in task_names:
                warnings.append(f"Задача '{task}' в ground_truth отсутствует в списке задач")
            for dep in deps:
                if dep not in task_names:
                    warnings.append(f"Зависимость '{dep}' не найдена в списке задач")

        return warnings

    def split_into_chunks(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Разбить задачи на чанки."""
        size = self.config.chunk_size
        return [tasks[i:i + size] for i in range(0, len(tasks), size)]

    def process_chunk(self, chunk: List[Dict[str, Any]], chunk_index: int) -> ChunkResult:
        """Обработать один чанк задач."""
        result = ChunkResult(chunk_index=chunk_index, tasks_count=len(chunk))

        prompt = build_prompt(self.prompt_template, chunk)
        logger.info(f"Чанк {chunk_index}: отправка {len(chunk)} задач...")

        response = self.client.send(prompt)

        if isinstance(response, LLMError):
            result.errors.append(response.error)
            logger.error(f"Чанк {chunk_index}: ошибка - {response.error}")
            return result

        result.success = True
        result.latency_ms = response.latency_ms
        result.tokens_used = response.total_tokens

        task_names = [t.get("name", "") for t in chunk]
        dependencies, parse_errors = parse_llm_response(response.content, task_names)

        result.dependencies = dependencies
        result.errors.extend(parse_errors)

        logger.info(f"Чанк {chunk_index}: {len(dependencies)} задач, {result.latency_ms:.0f}мс")
        return result

    def run(self) -> BenchmarkResult:
        """Запустить бенчмарк."""
        timestamp = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info(f"БЕНЧМАРК: {self.config.model.name}")
        logger.info(f"Промпт: {self.config.prompt_type}")
        logger.info("=" * 60)

        if not self.client.check_connection():
            logger.error(f"Нет соединения с {self.config.model.base_url}")
            return BenchmarkResult(
                model_name=self.config.model.name,
                prompt_type=self.config.prompt_type,
                timestamp=timestamp,
                total_tasks=0,
                chunks_processed=0,
                metrics=BenchmarkMetrics(errors=[f"Нет соединения с {self.config.model.base_url}"]),
            )

        tasks = self.load_tasks()
        ground_truth = self.load_ground_truth()

        logger.info(f"Загружено {len(tasks)} задач")
        if ground_truth:
            logger.info(f"Ground truth: {len(ground_truth)} зависимостей")
            validation_warnings = self.validate_ground_truth(tasks, ground_truth)
            for warn in validation_warnings[:5]:
                logger.warning(warn)
            if len(validation_warnings) > 5:
                logger.warning(f"...и ещё {len(validation_warnings) - 5} предупреждений")

        chunks = self.split_into_chunks(tasks)
        logger.info(f"Разбито на {len(chunks)} чанков")

        chunk_results: List[ChunkResult] = []
        all_dependencies: Dict[str, List[str]] = {}
        total_latency = 0.0
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            result = self.process_chunk(chunk, i)
            chunk_results.append(result)
            all_dependencies.update(result.dependencies)

            if result.success:
                total_latency += result.latency_ms
                total_tokens += result.tokens_used

        task_names = [t.get("name", "") for t in tasks]
        graph_metrics = calculate_graph_metrics(all_dependencies, task_names)

        dependency_metrics = None
        if ground_truth:
            dependency_metrics = calculate_dependency_metrics(all_dependencies, ground_truth)

        all_errors = []
        for cr in chunk_results:
            all_errors.extend(cr.errors)

        successful = sum(1 for cr in chunk_results if cr.success)
        avg_latency = total_latency / successful if successful > 0 else 0

        metrics = BenchmarkMetrics(
            dependency_metrics=dependency_metrics,
            graph_metrics=graph_metrics,
            latency_ms=avg_latency,
            total_tokens=total_tokens,
            tasks_processed=len(all_dependencies),
            errors=all_errors,
        )

        result = BenchmarkResult(
            model_name=self.config.model.name,
            prompt_type=self.config.prompt_type,
            timestamp=timestamp,
            total_tasks=len(tasks),
            chunks_processed=len(chunks),
            metrics=metrics,
            chunk_results=chunk_results,
            all_dependencies=all_dependencies,
            config={
                "chunk_size": self.config.chunk_size,
                "temperature": self.config.model.temperature,
                "max_tokens": self.config.model.max_tokens,
                "timeout_seconds": self.config.timeout_seconds,
            },
        )

        self._print_summary(result)
        return result

    def _print_summary(self, result: BenchmarkResult) -> None:
        """Вывести сводку результатов."""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
        print("=" * 60)

        print(f"\nМодель: {result.model_name}")
        print(f"Промпт: {result.prompt_type}")
        print(f"Задач: {result.total_tasks}")
        print(f"Обработано: {result.metrics.tasks_processed}")

        print(f"\nПроизводительность:")
        print(f"  Среднее время: {result.metrics.latency_ms:.0f} мс")
        print(f"  Токенов: {result.metrics.total_tokens:,}")

        gm = result.metrics.graph_metrics
        print(f"\nГраф:")
        print(f"  Покрытие: {gm.coverage * 100:.1f}%")
        print(f"  Ациклический: {'Да' if gm.is_acyclic else 'НЕТ'}")
        if gm.cycles_found:
            print(f"  Циклов: {len(gm.cycles_found)}")

        if result.metrics.dependency_metrics:
            dm = result.metrics.dependency_metrics
            print(f"\nКачество:")
            print(f"  Precision: {dm.precision:.4f}")
            print(f"  Recall: {dm.recall:.4f}")
            print(f"  F1: {dm.f1_score:.4f}")
            print(f"  TP: {dm.true_positives}, FP: {dm.false_positives}, FN: {dm.false_negatives}")

        if result.metrics.errors:
            print(f"\nОшибки: {len(result.metrics.errors)}")
            for err in result.metrics.errors[:3]:
                print(f"  - {err[:80]}")

        print("=" * 60)

    def save_results(self, result: BenchmarkResult) -> Path:
        """Сохранить результаты в файл."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = result.model_name.replace("/", "_").replace(":", "_")
        filename = f"benchmark_{model_safe}_{ts}.json"
        output_path = self.config.output_dir / filename

        def serialize(obj):
            if hasattr(obj, "__dict__"):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, Path):
                return str(obj)
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialize(result), f, ensure_ascii=False, indent=2)

        logger.info(f"Результаты: {output_path}")
        return output_path
