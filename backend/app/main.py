"""
Фармбазис ИИ-Техподдержка — главный модуль FastAPI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.bot_config import router as bot_config_router
from app.api.chat import router as chat_router
from app.api.escalation import router as escalation_router
from app.api.kb_admin import router as kb_admin_router
from app.api.operator import router as operator_router
from app.config import settings
from app.database.models import init_db
from app.database.reason_store import load_reasons
from app.models.schemas import HealthResponse
from app.rag.session_store import cleanup_expired_sessions

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: инициализация при старте, очистка при остановке."""
    logger.info("🚀 Инициализация Фармбазис ИИ-Техподдержки...")

    # Создаём директории
    Path("./data").mkdir(exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # Инициализируем БД
    await init_db()
    logger.info("✅ База данных инициализирована")

    # Загружаем причины обращения
    reasons = load_reasons()
    count = len(reasons.reasons) if reasons else 0
    logger.info(f"✅ Загружено причин обращения: {count}")

    if count == 0:
        logger.warning(
            "⚠️ Причины обращения не загружены! "
            "Добавьте их через /bot-config или импортируйте."
        )

    logger.info(f"✅ Сервер готов к работе на {settings.app_host}:{settings.app_port}")
    
    # Запускаем фоновую очистку истёкших контекстов уточнения
    import asyncio
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    yield

    cleanup_task.cancel()
    logger.info("Завершение работы...")


# Создание приложения
app = FastAPI(
    title="Фармбазис ИИ-Техподдержка",
    description=(
        "Модуль интеллектуальной техподдержки для ООО «Фармбазис». "
        "RAG-система на основе руководства пользователя."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(operator_router)
app.include_router(kb_admin_router)
app.include_router(bot_config_router)

# Статические файлы (изображения из инструкций)
images_dir = Path(settings.chroma_persist_dir).parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")

# Статические файлы для админки БЗ
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Проверка состояния системы."""
    from app.database.reason_store import get_cached_or_load

    reasons_data = await get_cached_or_load()
    reasons_count = len(reasons_data.reasons) if reasons_data else 0

    return HealthResponse(
        status="ok",
        version="2.0.0",
        knowledge_base_ready=reasons_count > 0,
        total_articles=reasons_count,
        total_chunks=0,
        support_tickets_count=0,
    )


@app.get("/bot-config", tags=["bot-config"])
async def bot_config_page():
    """Веб-интерфейс управления причинами обращения."""
    html_path = Path(__file__).resolve().parent.parent / "static" / "bot_config.html"
    if not html_path.exists():
        return {"error": "bot_config.html не найден"}
    return FileResponse(html_path, media_type="text/html")


@app.get("/kb-admin", tags=["kb-admin"])
async def kb_admin_page():
    """Веб-интерфейс управления базой знаний (квиз-режим)."""
    html_path = Path(__file__).resolve().parent.parent / "static" / "kb_admin.html"
    if not html_path.exists():
        return {"error": "kb_admin.html не найден"}
    return FileResponse(html_path, media_type="text/html")
