from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.config import settings

LLM_SETTING_KEYS = (
    "llm_provider",
    "show_llm_in_chat",
    "llm_temperature",
    "yandex_api_key",
    "yandex_folder_id",
    "yandex_gpt_model",
    "yandex_embedding_model",
    "deepseek_api_key",
    "deepseek_model",
)

# ── Ключи настроек классификации L1 ──
CLASSIFICATION_SETTING_KEYS = (
    "l1_global_min_score",
    "l1_weight_phrase_mask",
    "l1_weight_numeric_tag",
    "l1_weight_noun",
    "l1_weight_verb",
)

CLASSIFICATION_DEFAULTS: dict[str, float] = {
    "l1_global_min_score": 5.0,
    "l1_weight_phrase_mask": 10.0,
    "l1_weight_numeric_tag": 5.0,
    "l1_weight_noun": 2.0,
    "l1_weight_verb": 1.0,
}


def normalize_llm_provider(provider: str) -> str:
    normalized = (provider or "yandex").strip().lower()
    return normalized if normalized in {"yandex", "deepseek"} else "yandex"


def get_runtime_llm_settings_path() -> Path:
    return Path(settings.runtime_llm_settings_path)


def get_llm_settings_snapshot() -> dict[str, str]:
    snapshot = {
        "llm_provider": settings.llm_provider_normalized,
        "show_llm_in_chat": "true" if settings.show_llm_in_chat else "false",
        "llm_temperature": str(settings.llm_temperature),
        "yandex_api_key": settings.yandex_api_key,
        "yandex_folder_id": settings.yandex_folder_id,
        "yandex_gpt_model": settings.yandex_gpt_model,
        "yandex_embedding_model": settings.yandex_embedding_model,
        "deepseek_api_key": settings.deepseek_api_key,
        "deepseek_model": settings.deepseek_model,
    }

    runtime_path = get_runtime_llm_settings_path()
    if runtime_path.exists():
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in LLM_SETTING_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                snapshot[key] = value

    snapshot["llm_provider"] = normalize_llm_provider(snapshot["llm_provider"])
    snapshot["show_llm_in_chat"] = (
        "true" if str(snapshot["show_llm_in_chat"]).strip().lower() in {"1", "true", "yes", "on"} else "false"
    )
    try:
        snapshot["llm_temperature"] = str(max(0.0, min(1.0, float(snapshot.get("llm_temperature") or "0.1"))))
    except (ValueError, TypeError):
        snapshot["llm_temperature"] = "0.1"
    snapshot["yandex_gpt_model"] = snapshot["yandex_gpt_model"].strip() or "yandexgpt"
    snapshot["yandex_embedding_model"] = snapshot["yandex_embedding_model"].strip() or "text-search-query"
    snapshot["deepseek_model"] = snapshot["deepseek_model"].strip() or "deepseek-chat"
    return snapshot


def apply_llm_settings_snapshot(payload: Mapping[str, str]) -> None:
    settings.llm_provider = normalize_llm_provider(str(payload.get("llm_provider", settings.llm_provider)))
    settings.show_llm_in_chat = str(payload.get("show_llm_in_chat", settings.show_llm_in_chat)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        settings.llm_temperature = max(0.0, min(1.0, float(payload.get("llm_temperature", settings.llm_temperature))))
    except (ValueError, TypeError):
        pass
    settings.yandex_api_key = str(payload.get("yandex_api_key", settings.yandex_api_key)).strip()
    settings.yandex_folder_id = str(payload.get("yandex_folder_id", settings.yandex_folder_id)).strip()
    settings.yandex_gpt_model = str(payload.get("yandex_gpt_model", settings.yandex_gpt_model)).strip() or "yandexgpt"
    settings.yandex_embedding_model = (
        str(payload.get("yandex_embedding_model", settings.yandex_embedding_model)).strip() or "text-search-query"
    )
    settings.deepseek_api_key = str(payload.get("deepseek_api_key", settings.deepseek_api_key)).strip()
    settings.deepseek_model = str(payload.get("deepseek_model", settings.deepseek_model)).strip() or "deepseek-chat"


def save_runtime_llm_settings(payload: Mapping[str, str]) -> Path:
    runtime_path = get_runtime_llm_settings_path()
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: str(payload.get(key, "")) for key in LLM_SETTING_KEYS}
    data["llm_provider"] = normalize_llm_provider(data["llm_provider"])
    data["show_llm_in_chat"] = (
        "true" if data["show_llm_in_chat"].strip().lower() in {"1", "true", "yes", "on"} else "false"
    )
    try:
        data["llm_temperature"] = str(max(0.0, min(1.0, float(data.get("llm_temperature") or "0.1"))))
    except (ValueError, TypeError):
        data["llm_temperature"] = "0.1"
    data["yandex_gpt_model"] = data["yandex_gpt_model"].strip() or "yandexgpt"
    data["yandex_embedding_model"] = data["yandex_embedding_model"].strip() or "text-search-query"
    data["deepseek_model"] = data["deepseek_model"].strip() or "deepseek-chat"
    runtime_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    apply_llm_settings_snapshot(data)
    return runtime_path


def get_active_llm_display() -> dict[str, str | bool]:
    snapshot = get_llm_settings_snapshot()
    provider = snapshot["llm_provider"]
    model = snapshot["deepseek_model"] if provider == "deepseek" else snapshot["yandex_gpt_model"]
    provider_title = "DeepSeek" if provider == "deepseek" else "Yandex"
    return {
        "provider": provider,
        "provider_title": provider_title,
        "model": model,
        "label": f"{provider_title} / {model}",
        "show_in_chat": snapshot["show_llm_in_chat"] == "true",
    }


# ── Classification settings (L1 weights & global threshold) ──


def get_classification_settings() -> dict[str, float]:
    """Получить текущие настройки классификации L1 (веса маркеров + глобальный порог)."""
    result = dict(CLASSIFICATION_DEFAULTS)

    runtime_path = get_runtime_llm_settings_path()
    if runtime_path.exists():
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in CLASSIFICATION_SETTING_KEYS:
            raw = payload.get(key)
            if raw is not None:
                try:
                    result[key] = float(raw)
                except (ValueError, TypeError):
                    pass

    return result


def save_classification_settings(data: dict[str, float]) -> None:
    """Сохранить настройки классификации L1 в runtime JSON."""
    runtime_path = get_runtime_llm_settings_path()
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    # Прочитать существующий JSON, добавить/обновить classification keys
    existing: dict = {}
    if runtime_path.exists():
        try:
            existing = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    for key in CLASSIFICATION_SETTING_KEYS:
        if key in data:
            existing[key] = data[key]

    runtime_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
