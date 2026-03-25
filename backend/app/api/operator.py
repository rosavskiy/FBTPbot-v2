"""
API панели оператора техподдержки.

Веб-интерфейс для просмотра и обработки эскалаций.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
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
from app.sheets.gsheet_logger import get_gsheet_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/operator", tags=["operator"])

# In-memory токен-стор (в продакшене — JWT или Redis)
_active_tokens: dict[str, dict] = {}

# Хеш пароля оператора (создаётся при первом запуске из env)
_operator_password_hash: bytes | None = None


def _get_operator_password_hash() -> bytes:
    """Получить bcrypt-хеш пароля оператора (lazy init из env)."""
    global _operator_password_hash
    if _operator_password_hash is None:
        password = settings.operator_password
        if not password:
            logger.warning("OPERATOR_PASSWORD не задан! Авторизация оператора невозможна.")
            return b""
        _operator_password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return _operator_password_hash


def _cleanup_expired_tokens() -> None:
    """Удалить просроченные токены из хранилища."""
    now = datetime.now(UTC)
    expired = [t for t, data in _active_tokens.items() if now > data["expires_at"]]
    for t in expired:
        del _active_tokens[t]


def _verify_token(authorization: str | None = Header(None)) -> dict:
    """Проверка токена авторизации оператора."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    token = authorization.replace("Bearer ", "")
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="Недействительный токен")

    token_data = _active_tokens[token]
    if datetime.now(UTC) > token_data["expires_at"]:
        del _active_tokens[token]
        raise HTTPException(status_code=401, detail="Токен истёк")

    return token_data


@router.post("/login", response_model=OperatorLoginResponse)
async def operator_login(request: OperatorLoginRequest):
    """Авторизация оператора."""
    _cleanup_expired_tokens()

    if not settings.operator_password:
        raise HTTPException(status_code=503, detail="Авторизация не настроена (OPERATOR_PASSWORD не задан)")

    if request.username != settings.operator_username:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    if not bcrypt.checkpw(request.password.encode("utf-8"), _get_operator_password_hash()):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    token = secrets.token_urlsafe(32)
    _active_tokens[token] = {
        "username": request.username,
        "display_name": settings.operator_display_name,
        "expires_at": datetime.now(UTC) + timedelta(hours=12),
    }

    return OperatorLoginResponse(
        token=token,
        username=request.username,
    )


@router.get("/escalations", response_model=EscalationListResponse)
async def list_escalations(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    operator: dict = Depends(_verify_token),
):
    """Список эскалаций для оператора."""
    db_service = DatabaseService(db)

    escalations, total, pending_count = await db_service.get_all_escalations(status=status, limit=limit, offset=offset)

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

    # Логируем ответ оператора в Google Sheets
    asyncio.ensure_future(
        get_gsheet_logger().log(
            question=f"[Эскалация {request.escalation_id[:8]}]",
            answer=request.message,
            session_id=escalation.session_id,
            response_type="operator",
            escalation_info=f"Оператор: {operator['display_name']}, статус: {new_status}",
        )
    )

    return {"status": "ok", "new_status": new_status}
