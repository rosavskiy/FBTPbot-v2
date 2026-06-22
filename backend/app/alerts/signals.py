"""
Кросс-процессные сигналы для монитора оповещений.

Сейчас: флаг сбоя LLM-ключа. Пишется из rag/engine.py (в любом процессе) при
ответах LLM с кодами 401/403/429; читается монитором оповещений.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import SARATOV_TZ

logger = logging.getLogger(__name__)

LLM_LAST_ERROR_PATH = Path("./data/llm_last_error.json")

# Коды, означающие проблему с ключом/оплатой/лимитом
KEY_FAILURE_CODES = {401, 402, 403, 429}


def record_llm_key_failure(provider: str, status: int) -> None:
    """Записать факт сбоя LLM-ключа (если код входит в KEY_FAILURE_CODES)."""
    if status not in KEY_FAILURE_CODES:
        return
    try:
        LLM_LAST_ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(SARATOV_TZ).isoformat(),
            "provider": provider,
            "status": status,
        }
        LLM_LAST_ERROR_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("[SIGNALS] failed to write llm_last_error: %s", exc)


def read_llm_key_failure() -> dict[str, Any] | None:
    """Прочитать последний сбой LLM-ключа, либо None."""
    if not LLM_LAST_ERROR_PATH.exists():
        return None
    try:
        return json.loads(LLM_LAST_ERROR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
