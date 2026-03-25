"""
API панели оператора техподдержки.

Веб-интерфейс для просмотра и обработки эскалаций.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import get_db
from app.database.service import DatabaseService
from app.models.schemas import (
    ChatMessage,
    EscalationDetail,
    EscalationListResponse,
    EscalationStatus,
    OperatorLoginRequest,
    OperatorLoginResponse,
    OperatorReplyRequest,
)
from app.tg.notifier import get_telegram_notifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/operator", tags=["operator"])

# Простой in-memory токен-стор (в продакшене — JWT или Redis)
_active_tokens: dict[str, dict] = {}

# Дефолтный оператор для демо (в продакшене — из БД)
DEMO_OPERATOR = {
    "username": "admin",
    "password_hash": hashlib.sha256(b"farmbazis2024").hexdigest(),
    "display_name": "Администратор ТП",
}


def _verify_token(authorization: Optional[str] = Header(None)) -> dict:
    """Проверка токена авторизации оператора."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    token = authorization.replace("Bearer ", "")
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="Недействительный токен")

    token_data = _active_tokens[token]
    if datetime.now(timezone.utc) > token_data["expires_at"]:
        del _active_tokens[token]
        raise HTTPException(status_code=401, detail="Токен истёк")

    return token_data


@router.post("/login", response_model=OperatorLoginResponse)
async def operator_login(request: OperatorLoginRequest):
    """Авторизация оператора."""
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()

    if (
        request.username != DEMO_OPERATOR["username"]
        or password_hash != DEMO_OPERATOR["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    token = secrets.token_urlsafe(32)
    _active_tokens[token] = {
        "username": request.username,
        "display_name": DEMO_OPERATOR["display_name"],
        "expires_at": datetime.now(timezone.utc) + timedelta(hours=12),
    }

    return OperatorLoginResponse(
        token=token,
        username=request.username,
    )


@router.get("/escalations", response_model=EscalationListResponse)
async def list_escalations(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    operator: dict = Depends(_verify_token),
):
    """Список эскалаций для оператора."""
    db_service = DatabaseService(db)

    escalations, total, pending_count = await db_service.get_all_escalations(
        status=status, limit=limit, offset=offset
    )

    details = []
    for esc in escalations:
        # Загружаем историю чата для каждой эскалации
        messages = await db_service.get_chat_history(esc.session_id, limit=50)
        chat_history = [
            ChatMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.created_at,
            )
            for msg in messages
        ]

        details.append(
            EscalationDetail(
                escalation_id=esc.id,
                session_id=esc.session_id,
                status=EscalationStatus(esc.status),
                reason=esc.reason,
                contact_info=esc.contact_info,
                chat_history=chat_history,
                created_at=esc.created_at,
                updated_at=esc.updated_at,
                operator_notes=esc.operator_notes,
            )
        )

    return EscalationListResponse(
        escalations=details,
        total=total,
        pending_count=pending_count,
    )


@router.post("/reply")
async def operator_reply(
    request: OperatorReplyRequest,
    db: AsyncSession = Depends(get_db),
    operator: dict = Depends(_verify_token),
):
    """Ответ оператора на эскалацию."""
    db_service = DatabaseService(db)
    notifier = get_telegram_notifier()

    escalation = await db_service.get_escalation(request.escalation_id)
    if not escalation:
        raise HTTPException(status_code=404, detail="Эскалация не найдена")

    # Добавляем ответ оператора в чат
    await db_service.add_message(
        session_id=escalation.session_id,
        role="assistant",
        content=f"[Оператор ТП] {request.message}",
    )

    # Обновляем статус эскалации
    new_status = "resolved" if request.close_ticket else "in_progress"
    await db_service.update_escalation_status(
        escalation_id=request.escalation_id,
        status=new_status,
        operator_notes=request.message,
        operator_id=operator["username"],
    )

    # Уведомляем в Telegram
    await notifier.send_operator_reply(
        escalation_id=request.escalation_id,
        operator_name=operator["display_name"],
        reply_text=request.message,
        reply_to_message_id=escalation.telegram_message_id,
    )

    return {"status": "ok", "new_status": new_status}
