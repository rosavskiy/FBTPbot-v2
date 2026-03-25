"""
In-memory хранилище состояния сессий для многошагового диалога.

Хранит контекст уточнения (предложенные темы, оригинальный запрос),
чтобы обработать выбор пользователя на следующем шаге.

В production можно заменить на Redis.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# TTL для записей — после этого времени контекст уточнения сбрасывается
_SESSION_TTL = timedelta(minutes=15)

# Внутреннее хранилище: session_id → context dict
_store: Dict[str, Dict[str, Any]] = {}


async def save_clarification_context(
    session_id: str,
    original_query: str,
    topics: list[dict],
) -> None:
    """
    Сохраняет контекст уточнения для сессии.

    Args:
        session_id: ID сессии чата
        original_query: исходный запрос пользователя
        topics: список предложенных тем [{title, article_id, score, snippet}, ...]
    """
    _store[session_id] = {
        "state": "awaiting_clarification",
        "original_query": original_query,
        "topics": topics,
        "created_at": datetime.now(timezone.utc),
    }
    logger.debug(f"[SESSION] Saved clarification context for {session_id}: {len(topics)} topics")


def get_clarification_context(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Получает контекст уточнения, если он активен и не истёк.

    Returns:
        dict с полями state, original_query, topics — или None
    """
    if session_id not in _store:
        return None

    ctx = _store[session_id]
    created = ctx.get("created_at", datetime.now(timezone.utc))

    if datetime.now(timezone.utc) - created > _SESSION_TTL:
        del _store[session_id]
        return None

    if ctx.get("state") != "awaiting_clarification":
        return None

    return ctx


def clear_clarification_context(session_id: str) -> None:
    """Очищает контекст уточнения для сессии."""
    _store.pop(session_id, None)
    logger.debug(f"[SESSION] Cleared clarification context for {session_id}")


def resolve_topic_choice(session_id: str, user_input: str) -> Optional[dict]:
    """
    Проверяет, является ли сообщение пользователя выбором темы.

    Args:
        session_id: ID сессии
        user_input: текст сообщения пользователя

    Returns:
        dict выбранной темы {title, article_id, ...} или None,
        если пользователь ввёл свободный текст (уточнение)
    """
    ctx = get_clarification_context(session_id)
    if ctx is None:
        return None

    topics = ctx.get("topics", [])
    stripped = user_input.strip()

    # Пробуем распарсить как номер
    try:
        choice_idx = int(stripped) - 1
        if 0 <= choice_idx < len(topics):
            topic = topics[choice_idx]
            clear_clarification_context(session_id)
            logger.info(f"[SESSION] User selected topic #{choice_idx + 1}: {topic.get('title', '?')}")
            return topic
    except ValueError:
        pass

    # Не номер — пользователь уточняет текстом, сбрасываем контекст
    clear_clarification_context(session_id)
    return None


async def cleanup_expired_sessions() -> None:
    """Фоновая задача: периодическая очистка просроченных записей."""
    while True:
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, ctx in _store.items()
            if now - ctx.get("created_at", now) > _SESSION_TTL
        ]
        for sid in expired:
            del _store[sid]
        if expired:
            logger.debug(f"[SESSION] Cleaned up {len(expired)} expired clarification contexts")
        await asyncio.sleep(300)
