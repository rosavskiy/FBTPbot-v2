from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.config import settings

LLM_SETTING_KEYS = (
    "llm_provider",
    "show_llm_in_chat",
    "yandex_api_key",
    "yandex_folder_id",
    "yandex_gpt_model",
    "yandex_embedding_model",
    "deepseek_api_key",
    "deepseek_model",
)


def normalize_llm_provider(provider: str) -> str:
    normalized = (provider or "yandex").strip().lower()
    return normalized if normalized in {"yandex", "deepseek"} else "yandex"


def get_runtime_llm_settings_path() -> Path:
    return Path(settings.runtime_llm_settings_path)


def get_llm_settings_snapshot() -> dict[str, str]:
    snapshot = {
        "llm_provider": settings.llm_provider_normalized,
        "show_llm_in_chat": "true" if settings.show_llm_in_chat else "false",
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
