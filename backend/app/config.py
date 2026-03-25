from __future__ import annotations

import json

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # YandexGPT
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_gpt_model: str = "yandexgpt"
    yandex_embedding_model: str = "text-search-query"

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
    app_debug: bool = True

    # RAG
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_confidence_threshold: float = 0.3

    # Contact Reasons
    contact_reasons_path: str = "./data/contact_reasons.json"

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


settings = Settings()
