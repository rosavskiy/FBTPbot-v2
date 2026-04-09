from __future__ import annotations

import json
from datetime import timedelta, timezone
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Саратов UTC+4 (без DST)
SARATOV_TZ = timezone(timedelta(hours=4))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # YandexGPT
    llm_provider: str = "yandex"
    show_llm_in_chat: bool = False
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_gpt_model: str = "yandexgpt"
    yandex_embedding_model: str = "text-search-query"

    llm_temperature: float = 0.1

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/support.db"

    # Telegram
    telegram_bot_token: str = ""
    telegram_support_chat_id: str = ""

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # Operator auth
    operator_username: str = "admin"
    operator_password: str = ""  # MUST be set via env OPERATOR_PASSWORD
    operator_display_name: str = "Администратор ТП"

    # Admin panel auth
    superadmin_username: str = "superadmin"
    superadmin_password: str = ""  # MUST be set via env SUPERADMIN_PASSWORD
    admin_jwt_secret: str = ""  # MUST be set via env ADMIN_JWT_SECRET
    admin_token_expire_hours: int = 12

    # RAG
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_confidence_threshold: float = 0.3
    runtime_llm_settings_path: str = "./data/llm_settings.json"

    # Contact Reasons
    contact_reasons_path: str = "./data/contact_reasons.json"

    # Google Sheets
    google_sheets_credentials_file: str = ""
    google_sheets_spreadsheet_id: str = ""
    google_sheets_worksheet: str = "QnA"

    # CORS
    cors_origins: str = '["http://localhost:3000","http://localhost:5173"]'

    @property
    def cors_origins_list(self) -> list[str]:
        return json.loads(self.cors_origins)

    @property
    def yandex_gpt_model_uri(self) -> str:
        return f"gpt://{self.yandex_folder_id}/{self.yandex_gpt_model}/latest"

    @property
    def yandex_embedding_model_uri(self) -> str:
        return f"emb://{self.yandex_folder_id}/{self.yandex_embedding_model}/latest"

    @property
    def llm_provider_normalized(self) -> str:
        provider = (self.llm_provider or "yandex").strip().lower()
        return provider if provider in {"yandex", "deepseek"} else "yandex"

    @property
    def env_file_path(self) -> Path:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[1]


settings = Settings()
