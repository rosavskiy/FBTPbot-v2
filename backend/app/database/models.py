"""
Модели БД и управление сессиями/эскалациями.

SQLite для хранения:
- Сессий чата
- Истории сообщений
- Эскалаций
- Обратной связи
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    user_ip = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    messages = relationship("ChatMessageDB", back_populates="session", order_by="ChatMessageDB.created_at")
    escalations = relationship("Escalation", back_populates="session")


class ChatMessageDB(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    source_articles = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    session = relationship("ChatSession", back_populates="messages")


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    status = Column(String(20), default="pending")  # pending, in_progress, resolved, closed
    reason = Column(Text, nullable=True)
    contact_info = Column(String(255), nullable=True)
    operator_notes = Column(Text, nullable=True)
    operator_id = Column(String(100), nullable=True)
    telegram_message_id = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    session = relationship("ChatSession", back_populates="escalations")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    message_index = Column(Integer, default=0)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class Operator(Base):
    __tablename__ = "operators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    telegram_user_id = Column(String(50), nullable=True)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class AdminUser(Base):
    """Пользователи админ-панели (настройки бота / БЗ)."""

    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="viewer")  # superadmin | admin | viewer
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    last_login_at = Column(DateTime, nullable=True)

    audit_actions = relationship("AuditLog", back_populates="user")


class AuditLog(Base):
    """Лог действий администраторов."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    user_id = Column(Integer, ForeignKey("admin_users.id"), nullable=True)
    username = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)  # create | update | delete | login | import | reindex | settings_change
    entity_type = Column(String(50), nullable=False)  # reason | kb_item | image | llm_settings | user | ...
    entity_id = Column(String(255), nullable=True)
    entity_name = Column(String(255), nullable=True)
    details = Column(Text, nullable=True)

    user = relationship("AdminUser", back_populates="audit_actions")


# Async engine и session
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    connect_args={"timeout": 30},  # Ожидание до 30 сек при блокировке
    pool_pre_ping=True,
)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Включаем WAL-режим и busy_timeout для конкурентного доступа."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Создание всех таблиц."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency для FastAPI — получение сессии БД."""
    async with async_session() as session:
        yield session
