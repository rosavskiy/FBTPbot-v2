"""
API настроек системы оповещений (alerts).

Все endpoints под /api/alerts. Чтение — любой админ, изменение/тест — editor.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alerts.monitor import _fmt
from app.alerts.settings import get_alert_settings, save_alert_settings
from app.api.admin_auth import log_action, require_role, verify_admin_token
from app.database.models import AdminUser, AlertLog
from app.database.models import get_db as get_admin_db
from app.tg.notifier import get_telegram_notifier
from app.tg.user_registry import resolve_recipient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alerts", tags=["alerts"])

_any_admin = Depends(verify_admin_token)
_editor = Depends(require_role("superadmin", "admin"))


# ── Модели ──


class AlertSettingsPayload(BaseModel):
    enabled: bool = False
    recipients: list[str] = []
    poll_interval_sec: int = 300
    balance_enabled: bool = True
    balance_threshold_usd: float = 5.0
    health_enabled: bool = True
    errors_enabled: bool = True
    error_spike_threshold: int = 5
    errors_cooldown_min: int = 30
    llm_key_failure_enabled: bool = True
    cooldown_min: int = 360
    notify_on_recovery: bool = True


class ResolvePayload(BaseModel):
    recipients: list[str] = []


# ── Endpoints ──


@router.get("/settings")
async def get_settings(user: AdminUser = _any_admin) -> dict[str, Any]:
    """Текущие настройки оповещений."""
    return get_alert_settings()


@router.put("/settings")
async def update_settings(
    payload: AlertSettingsPayload,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
) -> dict[str, Any]:
    """Сохранить настройки оповещений."""
    saved = save_alert_settings(payload.model_dump())
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="settings_change",
        entity_type="alert_settings",
        details=f"enabled={saved['enabled']}, recipients={len(saved['recipients'])}",
    )
    return saved


@router.post("/resolve")
async def resolve_recipients(payload: ResolvePayload, user: AdminUser = _editor) -> dict[str, Any]:
    """Предпросмотр: во что разрешается каждый получатель."""
    items = []
    for token in payload.recipients:
        chat_id = resolve_recipient(token)
        # Числовой/совпавший username даёт цифровой id; иначе фоллбэк "@..."
        is_resolved = bool(chat_id) and not (chat_id.startswith("@"))
        items.append(
            {
                "token": token,
                "chat_id": chat_id,
                "resolved": is_resolved,
                "note": "" if is_resolved else "не найден в реестре — будет отправлен как @канал",
            }
        )
    return {"items": items}


@router.post("/test")
async def send_test_alert(user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)) -> dict[str, Any]:
    """Отправить тестовое оповещение текущим получателям и вернуть статус по каждому."""
    cfg = get_alert_settings()
    recipients: list[str] = cfg.get("recipients", [])
    notifier = get_telegram_notifier()
    text_body = _fmt("Тестовое оповещение", "Если вы это видите — доставка настроена корректно.")

    results = []
    delivered = 0
    for token in recipients:
        chat_id = resolve_recipient(token)
        ok = False
        if chat_id:
            msg_id = await notifier.send_message(chat_id, text_body)
            ok = bool(msg_id)
        if ok:
            delivered += 1
        results.append({"token": token, "chat_id": chat_id, "delivered": ok})

    db.add(
        AlertLog(
            alert_type="test",
            severity="info",
            message=text_body[:2000],
            recipients_count=len(recipients),
            delivered_count=delivered,
        )
    )
    await db.commit()

    return {"total": len(recipients), "delivered": delivered, "results": results}


@router.get("/history")
async def get_history(
    limit: int = Query(20, ge=1, le=200),
    user: AdminUser = _any_admin,
    db: AsyncSession = Depends(get_admin_db),
) -> dict[str, Any]:
    """Последние отправленные оповещения."""
    rows = await db.execute(select(AlertLog).order_by(desc(AlertLog.created_at)).limit(limit))
    items = [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "alert_type": r.alert_type,
            "severity": r.severity,
            "message": r.message,
            "recipients_count": r.recipients_count,
            "delivered_count": r.delivered_count,
        }
        for r in rows.scalars().all()
    ]
    return {"items": items}
