"""
Реестр соответствия username → chat_id Telegram.

Telegram Bot API не позволяет отправлять сообщение приватному пользователю по
@username — только по числовому chat_id. Бот наполняет этот реестр при каждом
сообщении пользователя; бэкенд (монитор оповещений) использует его, чтобы
разрешить @username получателя в chat_id.

Файл общий между процессами бота и бэкенда: ./data/tg_user_registry.json
(паттерн как у tg_heartbeat.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("./data/tg_user_registry.json")


def _read() -> dict[str, int]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    except Exception as exc:
        logger.debug("[REGISTRY] read failed: %s", exc)
    return {}


def record_user(user_id: int, username: str | None) -> None:
    """Запомнить username → chat_id (вызывается из обработчика бота)."""
    if not username:
        return
    key = username.lstrip("@").strip().lower()
    if not key:
        return
    registry = _read()
    if registry.get(key) == user_id:
        return  # без изменений — не пишем
    registry[key] = user_id
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[REGISTRY] write failed: %s", exc)


def resolve_recipient(token: str) -> str | None:
    """Разрешить получателя в chat_id, пригодный для Telegram sendMessage.

    - числовой (в т.ч. с ведущим '-') → возвращается как есть (chat_id/группа);
    - @username/username → ищем в реестре; найдено → числовой id;
      иначе фоллбэк → "@username" (сработает для публичного канала/группы).

    Returns строку chat_id, либо None если токен пустой.
    """
    token = (token or "").strip()
    if not token:
        return None

    candidate = token.lstrip("-")
    if candidate.isdigit():
        return token  # числовой id или -100... группы

    key = token.lstrip("@").strip().lower()
    if not key:
        return None

    registry = _read()
    if key in registry:
        return str(registry[key])

    # Не знаем приватного юзера — пробуем как публичный канал/группу
    return f"@{key}"
