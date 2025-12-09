# Construction Planning LLM Benchmark

Бенчмарк для оценки качества LLM моделей на задаче определения технологических зависимостей между строительными работами.

## Описание

Инструмент позволяет оценить способность языковых моделей анализировать перечень строительных работ и определять между ними технологические зависимости для построения сетевого графика производства работ.

**Ключевые возможности:**
- Поддержка локальных моделей (Ollama, vLLM, LM Studio)
- Поддержка облачных API (OpenRouter, OpenAI-совместимые)
- Три типа промптов для A/B тестирования
- Метрики качества: Precision, Recall, F1 Score
- Анализ графа зависимостей (покрытие, ацикличность)
- Пакетный запуск для сравнения моделей

## Установка

```bash
git clone git@github.com:JeryIO/ai-benchmark-megapolis.git
cd ai-benchmark-megapolis

python -m venv venv
source venv/bin/activate  # Linux/macOS
# или: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Локальная проверка через Ollama (пошагово)

### Шаг 1: Установка Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Или через Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Шаг 2: Загрузка модели

Выберите модель в зависимости от доступной GPU памяти:

| Модель | VRAM | Команда |
|--------|------|---------|
| qwen2.5:7b | ~5 GB | `ollama pull qwen2.5:7b` |
| qwen2.5:14b | ~10 GB | `ollama pull qwen2.5:14b` |
| qwen2.5:32b | ~20 GB | `ollama pull qwen2.5:32b` |

### Шаг 3: Проверка работы Ollama

```bash
curl http://localhost:11434/v1/models
```

### Шаг 4: Быстрый тест (59 задач)

```bash
python -m src.cli \
    -m qwen2.5:7b \
    -t data/sample_tasks.json \
    -g data/sample_ground_truth.json
```

### Шаг 5: Полный тест (173 задачи)

```bash
python -m src.cli \
    -m qwen2.5:7b \
    -t data/etalon_tasks.json \
    -g data/etalon_ground_truth.json
```

### Параметры для тюнинга

```bash
# Уменьшить chunk-size если модель не справляется
python -m src.cli \
    -m qwen2.5:7b \
    -t data/sample_tasks.json \
    -g data/sample_ground_truth.json \
    -c 30 \
    --timeout 600 \
    -v
```

### Ожидаемый результат

```
============================================================
РЕЗУЛЬТАТЫ БЕНЧМАРКА
============================================================

Модель: qwen2.5:7b
Промпт: expert
Задач: 59
Обработано: 59

Производительность:
  Среднее время: 15000 мс
  Токенов: 8,500

Граф:
  Покрытие: 90.0%
  Ациклический: Да

Качество:
  Precision: 0.45-0.55
  Recall: 0.35-0.45
  F1: 0.40-0.50
============================================================
```

## Быстрый старт

### Локальная модель (Ollama)

```bash
# Запустить Ollama с моделью
ollama pull qwen2.5:32b

# Запустить бенчмарк
python -m src.cli -m qwen2.5:32b -t data/etalon_tasks.json

# С эталонными данными для расчёта метрик
python -m src.cli \
    -m qwen2.5:32b \
    -t data/etalon_tasks.json \
    -g data/etalon_ground_truth.json
```

### Локальная модель (vLLM)

```bash
# Запустить vLLM сервер
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000

# Запустить бенчмарк
python -m src.cli \
    -m Qwen/Qwen2.5-32B-Instruct \
    -u http://localhost:8000/v1 \
    -t data/etalon_tasks.json \
    -g data/etalon_ground_truth.json
```

### Облачный API (OpenRouter)

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'

python -m src.cli \
    -m deepseek/deepseek-v3.2 \
    -u https://openrouter.ai/api/v1 \
    -k $OPENROUTER_API_KEY \
    -t data/etalon_tasks.json \
    -g data/etalon_ground_truth.json
```

## Пакетный запуск

### Несколько моделей через Ollama

```bash
python run_all_models.py --provider ollama
```

### Несколько моделей через vLLM

```bash
python run_all_models.py --provider vllm --url http://localhost:8000/v1
```

### Несколько моделей через OpenRouter

```bash
export OPENROUTER_API_KEY='sk-or-v1-...'
python run_all_models.py --provider openrouter
```

### Свой список моделей

```bash
python run_all_models.py \
    --provider ollama \
    --models qwen2.5:7b qwen2.5:14b qwen2.5:32b
```

### Полная матрица (модели × промпты)

```bash
python run_full_benchmark.py --provider ollama
python run_full_benchmark.py --provider openrouter
```

## Параметры CLI

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `-m, --model` | Название модели (обязательно) | — |
| `-t, --tasks` | JSON файл с задачами (обязательно) | — |
| `-u, --url` | URL API сервера | `http://localhost:11434/v1` |
| `-k, --api-key` | API ключ | — |
| `-p, --prompt` | Тип промпта: `basic`, `few_shot`, `expert` | `expert` |
| `-c, --chunk-size` | Количество задач в одном запросе | `100` |
| `-g, --ground-truth` | JSON с эталонными зависимостями | — |
| `-o, --output` | Директория для результатов | `./results` |
| `--timeout` | Таймаут запроса (секунды) | `300` |
| `--temperature` | Температура генерации | `0.1` |
| `--max-tokens` | Максимум токенов в ответе | без ограничений |
| `-v, --verbose` | Подробный вывод | — |
| `--no-save` | Не сохранять результаты | — |

## Метрики

### Метрики качества (при наличии ground truth)

| Метрика | Описание |
|---------|----------|
| **Precision** | Доля правильных зависимостей среди всех предсказанных |
| **Recall** | Доля найденных зависимостей от всех эталонных |
| **F1 Score** | Гармоническое среднее Precision и Recall |
| **True Positives** | Правильно предсказанные зависимости |
| **False Positives** | Ложные зависимости (не в эталоне) |
| **False Negatives** | Пропущенные зависимости |

### Метрики графа

| Метрика | Описание |
|---------|----------|
| **Coverage** | Процент задач, включённых в граф зависимостей |
| **Is Acyclic** | Граф без циклов (валидный DAG) |
| **Cycles Found** | Список обнаруженных циклов |
| **Root Tasks** | Задачи без входящих зависимостей |
| **Leaf Tasks** | Задачи без исходящих зависимостей |

### Метрики производительности

| Метрика | Описание |
|---------|----------|
| **Latency** | Среднее время обработки чанка (мс) |
| **Total Tokens** | Общее количество токенов |

## Данные

```
data/
├── etalon_tasks.json        # 173 эталонные задачи из ЛСР
├── etalon_ground_truth.json # Эталонные зависимости (экспертная разметка)
├── sample_tasks.json        # Пример для быстрого тестирования (59 задач)
└── sample_ground_truth.json # Эталон для примера
```

### Формат tasks.json

```json
{
  "tasks": [
    {
      "name": "Разработка котлована экскаватором",
      "level": "1.3",
      "group": "Земляные работы"
    }
  ]
}
```

### Формат ground_truth.json

```json
{
  "dependencies": {
    "Устройство песчаной подготовки": ["Разработка котлована экскаватором"],
    "Монтаж фундаментных блоков": ["Устройство песчаной подготовки"]
  }
}
```

## Структура проекта

```
src/
├── __init__.py     # Экспорт классов
├── __main__.py     # Точка входа
├── cli.py          # CLI интерфейс
├── config.py       # Конфигурация
├── llm_client.py   # HTTP клиент для OpenAI API
├── metrics.py      # Расчёт метрик качества
├── prompts.py      # Промпты для LLM
└── runner.py       # Основной runner бенчмарка

run_all_models.py       # Пакетный запуск моделей
run_full_benchmark.py   # Полный бенчмарк (модели × промпты)
```

## Типы промптов

### basic
Минимальный промпт с базовыми правилами определения зависимостей.

### few_shot
Промпт с тремя примерами определения зависимостей для разных типов работ.

### expert
Детальный промпт с ролью эксперта по строительству, полной технологической последовательностью и правилами для сложных случаев.

## Результаты

Результаты сохраняются в JSON формате:

```json
{
  "model_name": "qwen2.5:32b",
  "prompt_type": "expert",
  "timestamp": "2024-12-08T10:30:00",
  "total_tasks": 173,
  "metrics": {
    "dependency_metrics": {
      "precision": 0.85,
      "recall": 0.78,
      "f1_score": 0.81
    },
    "graph_metrics": {
      "coverage": 0.95,
      "is_acyclic": true
    },
    "latency_ms": 1250.5,
    "total_tokens": 15000
  }
}
```

## Требования

- Python 3.10+
- httpx >= 0.25.0

## Лицензия

MIT License
