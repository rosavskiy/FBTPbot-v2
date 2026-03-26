"""
API эндпоинты чата v2 — L1→L2→L3 pipeline.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import get_db
from app.database.service import DatabaseService
from app.llm_settings import get_active_llm_display
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    compute_confidence_label,
    compute_confidence_level,
)
from app.rag.engine import get_rag_engine
from app.sheets.gsheet_logger import get_gsheet_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Отправка сообщения в чат техподдержки v2.

    Pipeline:
    1. L1 — определение причины обращения по маркерам
    2. L2 — определение тематического раздела
    3. L3 — генерация ответа через YandexGPT

    Если session_id не указан — создаётся новая сессия.
    """
    db_service = DatabaseService(db)
    rag_engine = get_rag_engine()

    # Получаем или создаём сессию
    session = None
    if request.session_id:
        session = await db_service.get_session(request.session_id)

    if session is None:
        session = await db_service.create_session(
            user_ip=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

    # Сохраняем сообщение пользователя
    await db_service.add_message(
        session_id=session.id,
        role="user",
        content=request.message,
    )

    # Получаем историю чата для контекста
    history_messages = await db_service.get_chat_history(session.id, limit=10)
    chat_history = [{"role": msg.role, "content": msg.content} for msg in history_messages[:-1]]

    # ── Основной pipeline: L1→L2→L3 ──
    rag_response = await rag_engine.ask(
        question=request.message,
        chat_history=chat_history,
    )

    # Сохраняем ответ бота
    await db_service.add_message(
        session_id=session.id,
        role="assistant",
        content=rag_response.answer,
        confidence=rag_response.confidence,
        source_articles=rag_response.source_articles,
    )

    conf_level = compute_confidence_level(rag_response.confidence)
    conf_label = compute_confidence_label(rag_response.confidence)
    llm_display = get_active_llm_display()

    # Логируем в Google Sheets (fire-and-forget)
    asyncio.ensure_future(
        get_gsheet_logger().log(
            question=request.message,
            answer=rag_response.answer,
            session_id=session.id,
            confidence=rag_response.confidence,
            confidence_level=conf_level.value,
            confidence_label=conf_label,
            needs_escalation=rag_response.needs_escalation,
            source_articles=rag_response.source_articles,
            detected_reason=rag_response.detected_reason_name,
            thematic_section=rag_response.thematic_section,
            response_type="answer",
            youtube_links=rag_response.youtube_links,
            has_images=bool(rag_response.images),
        )
    )

    return ChatResponse(
        answer=rag_response.answer,
        session_id=session.id,
        confidence=rag_response.confidence,
        confidence_level=conf_level,
        confidence_label=conf_label,
        needs_escalation=rag_response.needs_escalation,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        has_images=bool(rag_response.images),
        response_type="answer",
        detected_reason=rag_response.detected_reason_name,
        thematic_section=rag_response.thematic_section,
        llm_provider=str(llm_display["provider"]),
        llm_model=str(llm_display["model"]),
        llm_label=str(llm_display["label"]),
        show_llm_in_chat=bool(llm_display["show_in_chat"]),
    )
