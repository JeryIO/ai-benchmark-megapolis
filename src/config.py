# -*- coding: utf-8 -*-
"""Конфигурация бенчмарка для тестирования LLM моделей."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Конфигурация модели."""
    name: str
    base_url: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None  # None = без ограничений

    def __post_init__(self):
        if not self.name:
            raise ValueError("Название модели обязательно")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature должен быть 0-2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens должен быть > 0 или None")


@dataclass
class BenchmarkConfig:
    """Конфигурация бенчмарка."""
    model: ModelConfig
    prompt_type: str = "expert"
    chunk_size: int = 100
    timeout_seconds: int = 300
    retry_count: int = 3
    output_dir: Path = field(default_factory=lambda: Path("./results"))

    def __post_init__(self):
        if self.prompt_type not in ("basic", "few_shot", "expert"):
            raise ValueError(f"Неверный prompt_type: {self.prompt_type}")
        if self.chunk_size < 1 or self.chunk_size > 500:
            raise ValueError("chunk_size должен быть 1-500")
        if self.timeout_seconds < 10:
            raise ValueError("timeout_seconds должен быть >= 10")
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class TestDataConfig:
    """Конфигурация тестовых данных."""
    tasks_file: Path
    ground_truth_file: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.tasks_file, str):
            self.tasks_file = Path(self.tasks_file)
        if isinstance(self.ground_truth_file, str):
            self.ground_truth_file = Path(self.ground_truth_file)


def ollama_config(model: str) -> ModelConfig:
    """Конфигурация для Ollama."""
    return ModelConfig(name=model, base_url="http://localhost:11434/v1")


def vllm_config(model: str) -> ModelConfig:
    """Конфигурация для vLLM."""
    return ModelConfig(name=model, base_url="http://localhost:8000/v1", api_key="EMPTY")


def lmstudio_config(model: str) -> ModelConfig:
    """Конфигурация для LM Studio."""
    return ModelConfig(name=model, base_url="http://localhost:1234/v1", api_key="lm-studio")
