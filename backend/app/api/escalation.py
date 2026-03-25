"""
API эндпоинты эскалации — передача запросов оператору ТП.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import get_db
from app.database.service import DatabaseService
from app.models.schemas import (
    EscalationRequest,
    EscalationResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.tg.notifier import get_telegram_notifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/escalation", tags=["escalation"])


@router.post("", response_model=EscalationResponse)
async def create_escalation(
    request: EscalationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Эскалация запроса на оператора техподдержки.

    Создаёт заявку и отправляет уведомление в Telegram.
    """
    db_service = DatabaseService(db)
    notifier = get_telegram_notifier()

    # Получаем историю чата для контекста
    history = await db_service.get_chat_history(request.session_id, limit=10)

    # Находим последний вопрос и ответ бота
    last_question = ""
    last_answer = ""
    for msg in reversed(history):
        if msg.role == "user" and not last_question:
            last_question = msg.content
        elif msg.role == "assistant" and not last_answer:
            last_answer = msg.content
        if last_question and last_answer:
            break

    # Создаём эскалацию в БД
    escalation = await db_service.create_escalation(
        session_id=request.session_id,
        reason=request.reason,
        contact_info=request.contact_info,
    )

    # Формируем краткое содержание диалога
    chat_summary = "\n".join(
        f"{'👤' if m.role == 'user' else '🤖'} {m.content[:100]}"
        for m in history[-6:]
    )

    # Отправляем уведомление в Telegram
    tg_message_id = await notifier.send_escalation_notification(
        escalation_id=escalation.id,
        session_id=request.session_id,
        user_question=last_question,
        bot_answer=last_answer,
        reason=request.reason,
        contact_info=request.contact_info,
        chat_summary=chat_summary,
    )

    if tg_message_id:
        await db_service.set_telegram_message_id(escalation.id, tg_message_id)

    # Считаем позицию в очереди
    pending = await db_service.get_pending_escalations()
    position = len(pending)

    return EscalationResponse(
        escalation_id=escalation.id,
        status=escalation.status,
        message=f"Запрос передан оператору. Вы {position}-й в очереди.",
        position_in_queue=position,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Отправка обратной связи по ответу бота."""
    db_service = DatabaseService(db)

    await db_service.add_feedback(
        session_id=request.session_id,
        rating=request.rating,
        message_index=request.message_index,
        comment=request.comment,
    )

    return FeedbackResponse()
