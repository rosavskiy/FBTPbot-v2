"""
Получение баланса LLM-провайдеров.

Используется и дашбордом /admin-status (форматированная строка с 60-сек кэшем),
и фоновым монитором оповещений (числовое значение для сравнения с порогом).
Только DeepSeek отдаёт баланс; у Yandex API баланса нет.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

_CURRENCY_SYMBOLS = {"USD": "$", "CNY": "¥"}

# Кэш форматированной строки для дашборда (как было в status.py)
_balance_cache: dict[str, Any] = {"value": None, "last_attempt": 0.0}
_BALANCE_TTL = 60.0  # секунд между запросами к API


async def fetch_deepseek_balance_raw(api_key: str) -> tuple[float | None, str]:
    """Запросить баланс DeepSeek без кэша.

    Returns:
        (значение, валюта). Значение None при недоступности/ошибке.
    """
    if not api_key:
        return None, "USD"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                DEEPSEEK_BALANCE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            infos = resp.json().get("balance_infos", [])
            entry = next((i for i in infos if i.get("currency") == "USD"), infos[0] if infos else None)
            if entry:
                return float(entry["total_balance"]), str(entry["currency"])
    except Exception as exc:
        logger.debug("[BALANCE] DeepSeek balance fetch failed: %s", exc)
    return None, "USD"


def format_balance(value: float | None, currency: str) -> str | None:
    """Сформировать строку вида "$5.42" из значения и валюты."""
    if value is None:
        return None
    sym = _CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{sym}{value:.2f}"


async def get_deepseek_balance_str(api_key: str) -> str | None:
    """Форматированный баланс DeepSeek с 60-сек кэшем (для дашборда)."""
    now = time.monotonic()
    if now - _balance_cache["last_attempt"] < _BALANCE_TTL:
        return _balance_cache["value"]
    _balance_cache["last_attempt"] = now
    value, currency = await fetch_deepseek_balance_raw(api_key)
    formatted = format_balance(value, currency)
    if formatted is not None:
        _balance_cache["value"] = formatted
    return _balance_cache["value"]
