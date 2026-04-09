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

from app.api.admin_auth import ensure_superadmin
from app.api.admin_auth import router as admin_router
from app.api.bot_config import router as bot_config_router
from app.api.chat import router as chat_router
from app.api.escalation import router as escalation_router
from app.api.images import router as images_router
from app.api.kb_admin import router as kb_admin_router
from app.api.operator import router as operator_router
from app.config import settings
from app.database.models import init_db
from app.database.reason_store import load_reasons
from app.models.schemas import HealthResponse
from app.rag.engine import close_rag_engine
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

    # Создаём суперадмина из env (если его нет)
    from app.database.models import async_session

    async with async_session() as db:
        await ensure_superadmin(db)

    # Загружаем причины обращения
    reasons = load_reasons()
    count = len(reasons.reasons) if reasons else 0
    logger.info(f"✅ Загружено причин обращения: {count}")

    if count == 0:
        logger.warning("⚠️ Причины обращения не загружены! " "Добавьте их через /bot-config или импортируйте.")

    logger.info(f"✅ Сервер готов к работе на {settings.app_host}:{settings.app_port}")

    # Запускаем фоновую очистку истёкших контекстов уточнения
    import asyncio

    cleanup_task = asyncio.create_task(cleanup_expired_sessions())

    yield

    cleanup_task.cancel()
    await close_rag_engine()
    logger.info("Завершение работы...")


# Создание приложения
app = FastAPI(
    title="Фармбазис ИИ-Техподдержка",
    description=(
        "Модуль интеллектуальной техподдержки для ООО «Фармбазис». " "RAG-система на основе руководства пользователя."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Роутеры
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(operator_router)
app.include_router(kb_admin_router)
app.include_router(bot_config_router)
app.include_router(images_router)
app.include_router(admin_router)

# Статические файлы (изображения из инструкций)
images_dir = Path(settings.chroma_persist_dir).parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")

# Статические файлы для загруженных изображений (bot images)
bot_images_dir = Path("./data/bot_images")
bot_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/bot_images", StaticFiles(directory=str(bot_images_dir)), name="bot_images")

# Статические файлы для админки БЗ
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Проверка состояния системы."""
    from app.database.reason_store import get_cached_or_load

    reasons_data = get_cached_or_load()
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


@app.get("/admin-login", tags=["admin"])
async def admin_login_page():
    """Страница входа в админ-панель."""
    html_path = Path(__file__).resolve().parent.parent / "static" / "admin_login.html"
    if not html_path.exists():
        return {"error": "admin_login.html не найден"}
    return FileResponse(html_path, media_type="text/html")


@app.get("/admin-users", tags=["admin"])
async def admin_users_page():
    """Страница управления пользователями."""
    html_path = Path(__file__).resolve().parent.parent / "static" / "admin_users.html"
    if not html_path.exists():
        return {"error": "admin_users.html не найден"}
    return FileResponse(html_path, media_type="text/html")


@app.get("/admin-audit", tags=["admin"])
async def admin_audit_page():
    """Страница аудит-лога."""
    html_path = Path(__file__).resolve().parent.parent / "static" / "admin_audit.html"
    if not html_path.exists():
        return {"error": "admin_audit.html не найден"}
    return FileResponse(html_path, media_type="text/html")
