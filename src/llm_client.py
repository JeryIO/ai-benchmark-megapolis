# -*- coding: utf-8 -*-
"""LLM клиент для OpenAI-совместимых API."""

import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import httpx

from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Ответ от LLM модели."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0


@dataclass
class LLMError:
    """Ошибка запроса к LLM."""
    error: str
    status_code: Optional[int] = None


class LLMClient:
    """Клиент для работы с OpenAI-совместимыми API."""

    def __init__(
        self,
        config: ModelConfig,
        timeout: float = 300.0,
        retries: int = 3
    ):
        self.config = config
        self.timeout = timeout
        self.retries = retries

        base = config.base_url.rstrip("/")
        self.chat_url = f"{base}/chat/completions"
        self.models_url = f"{base}/models"

        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """Формирование payload для запроса."""
        payload = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        return payload

    def _parse_response(self, data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        """Парсинг ответа от API."""
        content = ""
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Расчёт задержки перед повторной попыткой (exponential backoff)."""
        return min(2 ** attempt, 30)

    def send(self, prompt: str) -> Union[LLMResponse, LLMError]:
        """Отправить запрос к модели."""
        payload = self._build_payload(prompt)
        last_error: Optional[LLMError] = None

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retries):
                if attempt > 0:
                    delay = self._backoff_delay(attempt)
                    logger.info(f"Повтор через {delay}с...")
                    time.sleep(delay)

                try:
                    start = time.perf_counter()
                    response = client.post(self.chat_url, headers=self.headers, json=payload)
                    latency_ms = (time.perf_counter() - start) * 1000

                    if response.status_code != 200:
                        last_error = LLMError(
                            error=f"HTTP {response.status_code}: {response.text[:200]}",
                            status_code=response.status_code
                        )
                        logger.warning(f"Попытка {attempt + 1}/{self.retries}: {last_error.error}")
                        continue

                    return self._parse_response(response.json(), latency_ms)

                except httpx.TimeoutException:
                    last_error = LLMError(error=f"Таймаут {self.timeout}с")
                    logger.warning(f"Попытка {attempt + 1}/{self.retries}: таймаут")

                except httpx.RequestError as e:
                    last_error = LLMError(error=f"Ошибка соединения: {e}")
                    logger.warning(f"Попытка {attempt + 1}/{self.retries}: {e}")

                except Exception as e:
                    last_error = LLMError(error=str(e))
                    logger.error(f"Попытка {attempt + 1}/{self.retries}: {e}")

        return last_error or LLMError(error="Все попытки исчерпаны")

    async def send_async(self, prompt: str) -> Union[LLMResponse, LLMError]:
        """Отправить запрос к модели асинхронно."""
        import asyncio

        payload = self._build_payload(prompt)
        last_error: Optional[LLMError] = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.retries):
                if attempt > 0:
                    delay = self._backoff_delay(attempt)
                    logger.info(f"Повтор через {delay}с...")
                    await asyncio.sleep(delay)

                try:
                    start = time.perf_counter()
                    response = await client.post(self.chat_url, headers=self.headers, json=payload)
                    latency_ms = (time.perf_counter() - start) * 1000

                    if response.status_code != 200:
                        last_error = LLMError(
                            error=f"HTTP {response.status_code}: {response.text[:200]}",
                            status_code=response.status_code
                        )
                        logger.warning(f"Попытка {attempt + 1}/{self.retries}: {last_error.error}")
                        continue

                    return self._parse_response(response.json(), latency_ms)

                except httpx.TimeoutException:
                    last_error = LLMError(error=f"Таймаут {self.timeout}с")
                    logger.warning(f"Попытка {attempt + 1}/{self.retries}: таймаут")

                except httpx.RequestError as e:
                    last_error = LLMError(error=f"Ошибка соединения: {e}")
                    logger.warning(f"Попытка {attempt + 1}/{self.retries}: {e}")

                except Exception as e:
                    last_error = LLMError(error=str(e))
                    logger.error(f"Попытка {attempt + 1}/{self.retries}: {e}")

        return last_error or LLMError(error="Все попытки исчерпаны")

    def check_connection(self) -> bool:
        """Проверить соединение с сервером."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(self.models_url, headers=self.headers)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ошибка проверки соединения: {e}")
            return False
