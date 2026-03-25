"""
Сервис для работы с базой данных.

CRUD-операции для сессий, сообщений, эскалаций.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import (
    ChatMessageDB,
    ChatSession,
    Escalation,
    Feedback,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 0.5  # секунд


class DatabaseService:
    """Сервис для работы с БД."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def _commit_with_retry(self) -> None:
        """Commit с retry при \"database is locked\" (SQLite concurrent access)."""
        for attempt in range(MAX_RETRIES):
            try:
                await self.session.commit()
                return
            except OperationalError as e:
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    logger.warning("[DB] database is locked, retry %d", attempt + 1)
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise

    # === Сессии ===

    async def create_session(self, user_ip: str | None = None, user_agent: str | None = None) -> ChatSession:
        chat_session = ChatSession(
            id=str(uuid.uuid4()),
            user_ip=user_ip,
            user_agent=user_agent,
        )
        self.session.add(chat_session)
        await self._commit_with_retry()
        return chat_session

    async def get_session(self, session_id: str) -> ChatSession | None:
        result = await self.session.execute(
            select(ChatSession).options(selectinload(ChatSession.messages)).where(ChatSession.id == session_id)
        )
        return result.scalar_one_or_none()

    # === Сообщения ===

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        confidence: float | None = None,
        source_articles: list[str] | None = None,
    ) -> ChatMessageDB:
        message = ChatMessageDB(
            session_id=session_id,
            role=role,
            content=content,
            confidence=confidence,
            source_articles=json.dumps(source_articles) if source_articles else None,
        )
        self.session.add(message)
        await self._commit_with_retry()
        return message

    async def get_chat_history(self, session_id: str, limit: int = 20) -> list[ChatMessageDB]:
        result = await self.session.execute(
            select(ChatMessageDB)
            .where(ChatMessageDB.session_id == session_id)
            .order_by(ChatMessageDB.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()  # Хронологический порядок
        return messages

    # === Эскалации ===

    async def create_escalation(
        self,
        session_id: str,
        reason: str | None = None,
        contact_info: str | None = None,
    ) -> Escalation:
        escalation = Escalation(
            id=str(uuid.uuid4()),
            session_id=session_id,
            reason=reason,
            contact_info=contact_info,
        )
        self.session.add(escalation)
        await self._commit_with_retry()
        return escalation

    async def get_escalation(self, escalation_id: str) -> Escalation | None:
        result = await self.session.execute(select(Escalation).where(Escalation.id == escalation_id))
        return result.scalar_one_or_none()

    async def get_pending_escalations(self) -> list[Escalation]:
        result = await self.session.execute(
            select(Escalation)
            .where(Escalation.status.in_(["pending", "in_progress"]))
            .order_by(Escalation.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_all_escalations(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[Escalation], int, int]:
        """Получение эскалаций с пагинацией."""
        query = select(Escalation).order_by(Escalation.created_at.desc())
        count_query = select(func.count(Escalation.id))
        pending_query = select(func.count(Escalation.id)).where(Escalation.status == "pending")

        if status:
            query = query.where(Escalation.status == status)
            count_query = count_query.where(Escalation.status == status)

        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        escalations = list(result.scalars().all())

        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        pending_result = await self.session.execute(pending_query)
        pending_count = pending_result.scalar() or 0

        return escalations, total, pending_count

    async def update_escalation_status(
        self,
        escalation_id: str,
        status: str,
        operator_notes: str | None = None,
        operator_id: str | None = None,
    ) -> Escalation | None:
        values = {
            "status": status,
            "updated_at": datetime.now(UTC),
        }
        if operator_notes is not None:
            values["operator_notes"] = operator_notes
        if operator_id is not None:
            values["operator_id"] = operator_id

        await self.session.execute(update(Escalation).where(Escalation.id == escalation_id).values(**values))
        await self._commit_with_retry()
        return await self.get_escalation(escalation_id)

    async def set_telegram_message_id(self, escalation_id: str, message_id: str):
        await self.session.execute(
            update(Escalation).where(Escalation.id == escalation_id).values(telegram_message_id=message_id)
        )
        await self._commit_with_retry()

    # === Обратная связь ===

    async def add_feedback(
        self,
        session_id: str,
        rating: int,
        message_index: int = 0,
        comment: str | None = None,
    ) -> Feedback:
        feedback = Feedback(
            session_id=session_id,
            message_index=message_index,
            rating=rating,
            comment=comment,
        )
        self.session.add(feedback)
        await self._commit_with_retry()
        return feedback
