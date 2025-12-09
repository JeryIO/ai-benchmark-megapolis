# -*- coding: utf-8 -*-
"""Метрики качества для оценки LLM моделей."""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


@dataclass
class DependencyMetrics:
    """Метрики качества зависимостей."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class GraphMetrics:
    """Метрики качества графа зависимостей."""
    total_tasks: int = 0
    tasks_in_graph: int = 0
    coverage: float = 0.0
    is_acyclic: bool = True
    cycles_found: List[List[str]] = field(default_factory=list)
    root_tasks: List[str] = field(default_factory=list)
    leaf_tasks: List[str] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    """Полные метрики бенчмарка."""
    dependency_metrics: Optional[DependencyMetrics] = None
    graph_metrics: GraphMetrics = field(default_factory=GraphMetrics)
    latency_ms: float = 0.0
    total_tokens: int = 0
    tasks_processed: int = 0
    errors: List[str] = field(default_factory=list)


def calculate_dependency_metrics(
    predicted: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]]
) -> DependencyMetrics:
    """Рассчитать Precision/Recall/F1 для зависимостей."""
    predicted_pairs: Set[Tuple[str, str]] = set()
    for task, deps in predicted.items():
        for dep in deps:
            predicted_pairs.add((task, dep))

    truth_pairs: Set[Tuple[str, str]] = set()
    for task, deps in ground_truth.items():
        for dep in deps:
            truth_pairs.add((task, dep))

    tp = len(predicted_pairs & truth_pairs)
    fp = len(predicted_pairs - truth_pairs)
    fn = len(truth_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DependencyMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def calculate_graph_metrics(
    dependencies: Dict[str, List[str]],
    all_tasks: Optional[List[str]] = None
) -> GraphMetrics:
    """Рассчитать метрики графа зависимостей."""
    if all_tasks is None:
        all_tasks = list(dependencies.keys())

    all_deps: Set[str] = set()
    for deps in dependencies.values():
        all_deps.update(deps)

    tasks_with_outgoing = set(t for t, deps in dependencies.items() if deps)
    tasks_with_incoming = all_deps
    tasks_in_graph = tasks_with_outgoing | tasks_with_incoming

    total = len(all_tasks)
    coverage = len(tasks_in_graph) / total if total > 0 else 0.0

    root_tasks = [t for t in all_tasks if t not in tasks_with_incoming and t in tasks_with_outgoing]
    leaf_tasks = [t for t in all_tasks if t in tasks_with_incoming and t not in tasks_with_outgoing]

    cycles = detect_cycles(dependencies)

    return GraphMetrics(
        total_tasks=total,
        tasks_in_graph=len(tasks_in_graph),
        coverage=round(coverage, 4),
        is_acyclic=len(cycles) == 0,
        cycles_found=cycles[:10],
        root_tasks=root_tasks[:20],
        leaf_tasks=leaf_tasks[:20],
    )


def detect_cycles(dependencies: Dict[str, List[str]]) -> List[List[str]]:
    """Обнаружить циклы в графе зависимостей используя DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]) -> None:
        if len(cycles) >= 10:
            return

        color[node] = GRAY
        path.append(node)

        for dep in dependencies.get(node, []):
            if dep == node:
                cycles.append([node, node])
            elif color[dep] == GRAY:
                idx = path.index(dep)
                cycles.append(path[idx:] + [dep])
            elif color[dep] == WHITE:
                dfs(dep, path)

        path.pop()
        color[node] = BLACK

    for task in dependencies:
        if color[task] == WHITE:
            dfs(task, [])

    return cycles


def _normalize_task_name(name: str) -> str:
    """Нормализация названия задачи: убрать артефакты LLM."""
    # Убрать /think и подобные артефакты от Qwen3
    name = re.sub(r'\s*/think\s*', '', name)
    name = re.sub(r'<think>.*?</think>', '', name, flags=re.DOTALL)
    # Убрать префикс с номером задачи (#1, #2, 1., 2. и т.д.)
    name = re.sub(r'^#?\d+[\.\):\s]+', '', name)
    # Убрать суффикс с группой [...]
    match = re.match(r'^(.+?)\s*\[[^\]]+\]\s*$', name)
    return match.group(1).strip() if match else name.strip()


def _extract_json_array(content: str) -> Optional[str]:
    """Извлечь JSON массив из ответа LLM."""
    # Убрать thinking блоки Qwen3
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()

    # Извлечь из markdown блока
    if "```" in content:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if match:
            content = match.group(1).strip()

    # Найти JSON массив в любом месте
    if not content.startswith("["):
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            content = match.group(0)

    # Восстановить обрезанный JSON
    if content.startswith("[") and not content.endswith("]"):
        last_brace = content.rfind("}")
        if last_brace != -1:
            content = content[:last_brace + 1] + "]"

    return content if content else None


def parse_llm_response(
    response: str,
    expected_tasks: Optional[List[str]] = None
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Парсинг ответа LLM в словарь зависимостей."""
    errors: List[str] = []
    dependencies: Dict[str, List[str]] = {}

    content = _extract_json_array(response)

    if not content:
        errors.append(f"JSON не найден в ответе (длина: {len(response)})")
        return dependencies, errors

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Ошибка парсинга JSON: {e}")
        # Логируем начало ответа для отладки
        preview = response[:200].replace('\n', ' ')
        errors.append(f"Начало ответа: {preview}...")
        return dependencies, errors

    if not isinstance(data, list):
        errors.append(f"Ожидался массив, получен {type(data).__name__}")
        return dependencies, errors

    for item in data:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")
        if not name:
            continue

        name = _normalize_task_name(name)

        depends_on = item.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on] if depends_on else []
        elif not isinstance(depends_on, list):
            depends_on = []

        depends_on = [_normalize_task_name(d) for d in depends_on if isinstance(d, str) and d]
        dependencies[name] = depends_on

    if expected_tasks:
        missing = set(expected_tasks) - set(dependencies.keys())
        if missing:
            errors.append(f"Пропущено {len(missing)} из {len(expected_tasks)} задач")

    return dependencies, errors
