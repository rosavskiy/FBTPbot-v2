"""
Модели БД и управление сессиями/эскалациями.

SQLite для хранения:
- Сессий чата
- Истории сообщений
- Эскалаций
- Обратной связи
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import SARATOV_TZ, settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))
    updated_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), onupdate=lambda: datetime.now(SARATOV_TZ))
    user_ip = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    messages = relationship("ChatMessageDB", back_populates="session", order_by="ChatMessageDB.created_at")
    escalations = relationship("Escalation", back_populates="session")
    pending_clarification = relationship(
        "PendingClarification",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ChatMessageDB(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    source_articles = Column(Text, nullable=True)  # JSON
    source = Column(String(16), nullable=False, server_default="web", index=True)  # web | tg | operator
    detected_reason = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))

    session = relationship("ChatSession", back_populates="messages")


class PendingClarification(Base):
    __tablename__ = "pending_clarifications"

    session_id = Column(String(36), ForeignKey("chat_sessions.id"), primary_key=True)
    clarification_type = Column(String(50), nullable=False)
    original_query = Column(Text, nullable=False)
    prompt = Column(Text, nullable=True)
    fixed_reason_id = Column(String(255), nullable=True)
    fixed_reason_name = Column(String(255), nullable=True)
    payload_json = Column(Text, nullable=True)
    attempts = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))
    expires_at = Column(DateTime, nullable=False)

    session = relationship("ChatSession", back_populates="pending_clarification")


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
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))
    updated_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), onupdate=lambda: datetime.now(SARATOV_TZ))

    session = relationship("ChatSession", back_populates="escalations")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    message_index = Column(Integer, default=0)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))


class Operator(Base):
    __tablename__ = "operators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    telegram_user_id = Column(String(50), nullable=True)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))


class AdminUser(Base):
    """Пользователи админ-панели (настройки бота / БЗ)."""

    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default="viewer")  # superadmin | admin | viewer
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))
    updated_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), onupdate=lambda: datetime.now(SARATOV_TZ))
    last_login_at = Column(DateTime, nullable=True)

    audit_actions = relationship("AuditLog", back_populates="user")


class AuditLog(Base):
    """Лог действий администраторов."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), index=True)
    user_id = Column(Integer, ForeignKey("admin_users.id"), nullable=True)
    username = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)  # create | update | delete | login | import | reindex | settings_change
    entity_type = Column(String(50), nullable=False)  # reason | kb_item | image | llm_settings | user | ...
    entity_id = Column(String(255), nullable=True)
    entity_name = Column(String(255), nullable=True)
    details = Column(Text, nullable=True)

    user = relationship("AdminUser", back_populates="audit_actions")


class ProgressNote(Base):
    """Ежедневная запись журнала доработок на странице «О программе»."""

    __tablename__ = "progress_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    progress_date = Column(Date, unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False, default="")
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ))
    updated_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), onupdate=lambda: datetime.now(SARATOV_TZ))


class AlertLog(Base):
    """История отправленных оповещений (для блока на /admin-status)."""

    __tablename__ = "alert_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(SARATOV_TZ), index=True)
    alert_type = Column(String(40), nullable=False)  # balance | health | errors | llm_key | test
    severity = Column(String(20), nullable=False, default="warning")  # warning | critical | recovery | info
    message = Column(Text, nullable=False)
    recipients_count = Column(Integer, default=0)
    delivered_count = Column(Integer, default=0)
    delivery_error = Column(Text, nullable=True)  # причина недоставки (если delivered < recipients)
    pending = Column(Integer, default=0)  # 1 — алерт недоставлен, ждёт переотправки
    retry_count = Column(Integer, default=0)  # сколько раз пытались переотправить


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


async def _migrate_add_column_if_missing(conn, table: str, column: str, column_def: str) -> None:
    """Идемпотентно добавляет колонку в таблицу SQLite, если её нет."""
    result = await conn.execute(text(f"PRAGMA table_info({table})"))
    cols = [row[1] for row in result.fetchall()]
    if column not in cols:
        await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}"))
        logger.info("[MIGRATE] Added column %s.%s", table, column)


async def init_db():
    """Создание всех таблиц + идемпотентные миграции колонок."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_add_column_if_missing(conn, "chat_messages", "source", "TEXT NOT NULL DEFAULT 'web'")
        await _migrate_add_column_if_missing(conn, "chat_messages", "detected_reason", "TEXT")
        await _migrate_add_column_if_missing(conn, "alert_log", "delivery_error", "TEXT")
        await _migrate_add_column_if_missing(conn, "alert_log", "pending", "INTEGER DEFAULT 0")
        await _migrate_add_column_if_missing(conn, "alert_log", "retry_count", "INTEGER DEFAULT 0")


async def get_db():
    """Dependency для FastAPI — получение сессии БД."""
    async with async_session() as session:
        yield session
