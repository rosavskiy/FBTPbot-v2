"""
Хранение настроек системы оповещений.

Отдельный JSON-файл ./data/alert_settings.json (не смешиваем с llm_settings.json).
Паттерн загрузки/сохранения — как в app/llm_settings.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ALERT_SETTINGS_PATH = Path("./data/alert_settings.json")

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "recipients": [],  # список строк: "123456789" или "@username"
    "poll_interval_sec": 300,
    # Порог баланса API (DeepSeek)
    "balance_enabled": True,
    "balance_threshold_usd": 5.0,
    # Здоровье сервисов (БД / KB / TG-heartbeat / GSheets)
    "health_enabled": True,
    # Всплеск ошибок приложения
    "errors_enabled": True,
    "error_spike_threshold": 5,
    "errors_cooldown_min": 30,
    # Сбой LLM-ключа (401/403/429)
    "llm_key_failure_enabled": True,
    # Общие
    "cooldown_min": 360,  # повтор алерта по длящемуся условию (6ч)
    "notify_on_recovery": True,
}

# Границы валидации (min, max) для числовых полей
_BOUNDS: dict[str, tuple[float, float]] = {
    "poll_interval_sec": (30, 86400),
    "balance_threshold_usd": (0.0, 100000.0),
    "error_spike_threshold": (1, 10000),
    "errors_cooldown_min": (1, 10080),
    "cooldown_min": (1, 10080),
}

_BOOL_KEYS = (
    "enabled",
    "balance_enabled",
    "health_enabled",
    "errors_enabled",
    "llm_key_failure_enabled",
    "notify_on_recovery",
)
_INT_KEYS = ("poll_interval_sec", "error_spike_threshold", "errors_cooldown_min", "cooldown_min")
_FLOAT_KEYS = ("balance_threshold_usd",)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _clamp(key: str, value: float) -> float:
    lo, hi = _BOUNDS.get(key, (None, None))
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _normalize_recipients(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.replace(",", "\n").split("\n")
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = str(item).strip()
        if not token:
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def normalize_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Применить дефолты, типы и границы к произвольному словарю настроек."""
    result = dict(DEFAULTS)
    result["recipients"] = []

    for key in _BOOL_KEYS:
        if key in payload:
            result[key] = _coerce_bool(payload[key], DEFAULTS[key])

    for key in _INT_KEYS:
        if key in payload and payload[key] is not None:
            try:
                result[key] = int(_clamp(key, int(float(payload[key]))))
            except (ValueError, TypeError):
                pass

    for key in _FLOAT_KEYS:
        if key in payload and payload[key] is not None:
            try:
                result[key] = float(_clamp(key, float(payload[key])))
            except (ValueError, TypeError):
                pass

    if "recipients" in payload:
        result["recipients"] = _normalize_recipients(payload["recipients"])

    return result


def get_alert_settings() -> dict[str, Any]:
    """Прочитать текущие настройки (с дефолтами при отсутствии файла)."""
    if not ALERT_SETTINGS_PATH.exists():
        return dict(DEFAULTS)
    try:
        payload = json.loads(ALERT_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[ALERTS] Failed to read alert_settings.json: %s", exc)
        return dict(DEFAULTS)
    return normalize_settings(payload)


def save_alert_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Сохранить настройки (нормализуя их) и вернуть нормализованную версию."""
    normalized = normalize_settings(payload)
    ALERT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_SETTINGS_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized
