"""
Фоновый монитор оповещений.

Раз в poll_interval_sec проверяет: порог баланса API, здоровье сервисов
(БД / база знаний / TG-бот), всплеск ошибок приложения и сбой LLM-ключа.
Шлёт оповещения в Telegram настроенным получателям с дедупликацией
(edge-trigger + cooldown). Запускается из lifespan бэкенда.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from app.alerts.error_capture import ERROR_EVENTS_PATH
from app.alerts.settings import get_alert_settings
from app.alerts.signals import read_llm_key_failure
from app.config import SARATOV_TZ, settings
from app.database.models import AlertLog, async_session
from app.database.reason_store import get_cached_or_load
from app.llm_balance import fetch_deepseek_balance_raw, format_balance
from app.llm_settings import get_active_llm_display, get_llm_settings_snapshot
from app.tg.notifier import get_telegram_notifier
from app.tg.user_registry import resolve_recipient

logger = logging.getLogger(__name__)

ALERT_STATE_PATH = Path("./data/alert_state.json")
TG_STALE_SEC = 90  # heartbeat старше — бот считается offline


# ── Состояние монитора (дедуп) ───────────────────────────────────────


def _load_state() -> dict[str, Any]:
    if not ALERT_STATE_PATH.exists():
        return {"conditions": {}, "error_offset": 0, "errors_last_fired": None}
    try:
        data = json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("conditions", {})
        data.setdefault("error_offset", 0)
        data.setdefault("errors_last_fired", None)
        return data
    except Exception:
        return {"conditions": {}, "error_offset": 0, "errors_last_fired": None}


def _save_state(state: dict[str, Any]) -> None:
    try:
        ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("[ALERTS] failed to save state: %s", exc)


def _now() -> datetime:
    return datetime.now(SARATOV_TZ)


def _minutes_since(iso: str | None) -> float:
    if not iso:
        return float("inf")
    try:
        ts = datetime.fromisoformat(iso)
        return (_now() - ts).total_seconds() / 60.0
    except Exception:
        return float("inf")


def _row_age_minutes(created_at: datetime | None) -> float:
    """Возраст записи в минутах. Наивный datetime трактуем как SARATOV_TZ."""
    if created_at is None:
        return float("inf")
    try:
        ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=SARATOV_TZ)
        return (_now() - ts).total_seconds() / 60.0
    except Exception:
        return float("inf")


# ── Проверки ─────────────────────────────────────────────────────────


def _tg_alive() -> bool | None:
    """True/False — жив ли бот; None — данных нет (не алертим)."""
    path = Path(settings.tg_heartbeat_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts_str = data.get("ts", "")
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(UTC) - ts).total_seconds()
            return age < TG_STALE_SEC
    except Exception:
        return None
    return None


async def _db_ok() -> bool:
    try:
        async with async_session() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("[ALERTS] DB health check failed: %s", exc)
        return False


def _read_new_errors(state: dict[str, Any]) -> list[str]:
    """Вернуть сообщения новых ERROR-записей с сохранённого offset; обновить offset."""
    path = ERROR_EVENTS_PATH
    if not path.exists():
        state["error_offset"] = 0
        return []
    try:
        size = path.stat().st_size
        offset = state.get("error_offset", 0)
        if offset > size:  # файл усечён/ротация
            offset = 0
        messages: list[str] = []
        # Бинарный режим — корректные байтовые offset и без OSError при tell() во время итерации
        with path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read()
            state["error_offset"] = fh.tell()
        for bline in raw.splitlines():
            line = bline.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                messages.append(f"{entry.get('logger', '?')}: {entry.get('message', '')}")
            except Exception:
                messages.append(line[:200])
        return messages
    except Exception as exc:
        logger.warning("[ALERTS] failed to read error events: %s", exc)
        return []


# ── Edge-trigger + cooldown ──────────────────────────────────────────


def _should_fire(
    state: dict[str, Any],
    key: str,
    is_bad: bool,
    cooldown_min: float,
    notify_on_recovery: bool,
) -> str | None:
    """Решить, слать ли алерт по условию key.

    Возвращает 'alert' (проблема), 'recovery' (восстановление) или None.
    Обновляет состояние условия в state.
    """
    cond = state["conditions"].get(key, {"state": "ok", "last_fired": None})
    prev = cond.get("state", "ok")
    decision: str | None = None

    if is_bad:
        if prev != "bad":
            decision = "alert"
            cond["last_fired"] = _now().isoformat()
        elif _minutes_since(cond.get("last_fired")) >= cooldown_min:
            decision = "alert"
            cond["last_fired"] = _now().isoformat()
        cond["state"] = "bad"
    else:
        if prev == "bad" and notify_on_recovery:
            decision = "recovery"
        cond["state"] = "ok"

    state["conditions"][key] = cond
    return decision


# ── Отправка ─────────────────────────────────────────────────────────


MAX_RESEND_ATTEMPTS = 5  # сколько раз пытаемся переотправить недоставленный алерт
PENDING_MAX_AGE_MIN = 24 * 60  # старше суток — больше не переотправляем


async def _deliver(text_body: str, recipients: list[str], retries: int = 2) -> tuple[int, int, str | None]:
    """Разрезолвить получателей и отправить. Возвращает (delivered, total, error).

    error — причина недоставки (первая встреченная), если delivered < total, иначе None.
    """
    notifier = get_telegram_notifier()
    total = len(recipients)
    delivered = 0
    first_error: str | None = None
    for token in recipients:
        chat_id = resolve_recipient(token)
        if not chat_id:
            if first_error is None:
                first_error = f"получатель не разрешён: {token}"
            continue
        msg_id, err = await notifier.send_message_ex(chat_id, text_body, retries=retries)
        if msg_id:
            delivered += 1
        elif first_error is None:
            first_error = err or "неизвестная ошибка"
    return delivered, total, (first_error if delivered < total else None)


async def dispatch_alert(alert_type: str, severity: str, text_body: str, recipients: list[str]) -> tuple[int, int]:
    """Разрезолвить получателей, отправить и записать историю. Возвращает (delivered, total).

    Если доставлено не всем — запись помечается pending=1 и будет переотправлена
    в начале следующих циклов монитора (когда сеть/Telegram вернутся).
    """
    delivered, total, error = await _deliver(text_body, recipients)
    pending = 1 if (total > 0 and delivered < total) else 0

    try:
        async with async_session() as db:
            db.add(
                AlertLog(
                    alert_type=alert_type,
                    severity=severity,
                    message=text_body[:2000],
                    recipients_count=total,
                    delivered_count=delivered,
                    delivery_error=(error or None),
                    pending=pending,
                )
            )
            await db.commit()
    except Exception as exc:
        logger.warning("[ALERTS] failed to write AlertLog: %s", exc)

    log_fn = logger.info if pending == 0 else logger.warning
    log_fn(
        "[ALERTS] %s/%s sent (type=%s, severity=%s)%s",
        delivered,
        total,
        alert_type,
        severity,
        f" — недоставлено, в очереди: {error}" if pending else "",
    )
    return delivered, total


async def flush_pending_alerts(recipients: list[str]) -> None:
    """Переотправить недоставленные алерты текущим получателям.

    Вызывается в начале каждого цикла монитора. Лучше получить алерт с
    опозданием, чем не получить вовсе. Дубликат для тех, кто уже получил,
    приемлем — приоритет у гарантии доставки. Записи старше суток или
    исчерпавшие лимит попыток снимаются с очереди.
    """
    if not recipients:
        return
    try:
        async with async_session() as db:
            rows = await db.execute(
                select(AlertLog).where(AlertLog.pending == 1).order_by(AlertLog.created_at).limit(20)
            )
            pending_rows = rows.scalars().all()
            for row in pending_rows:
                age_min = _row_age_minutes(row.created_at)
                if (row.retry_count or 0) >= MAX_RESEND_ATTEMPTS or age_min >= PENDING_MAX_AGE_MIN:
                    row.pending = 0
                    row.delivery_error = (row.delivery_error or "") + " | снято с очереди (лимит попыток/срок)"
                    continue
                # retries=0: сам flush — это и есть механизм повтора, не блокируем цикл бэкоффом
                delivered, total, error = await _deliver(row.message, recipients, retries=0)
                row.retry_count = (row.retry_count or 0) + 1
                if delivered >= total:
                    row.pending = 0
                    row.delivered_count = total
                    row.recipients_count = total
                    row.delivery_error = None
                    logger.info("[ALERTS] переотправлен недоставленный алерт #%s (%s)", row.id, row.alert_type)
                else:
                    row.delivery_error = error or row.delivery_error
            await db.commit()
    except Exception as exc:
        logger.warning("[ALERTS] flush pending failed: %s", exc)


def _fmt(title: str, body: str = "") -> str:
    text_msg = f"🔔 <b>{title}</b>"
    if body:
        text_msg += f"\n{body}"
    text_msg += f"\n\n🕒 {_now().strftime('%d.%m.%Y %H:%M:%S')}"
    return text_msg


# ── Главный цикл ─────────────────────────────────────────────────────


async def _run_checks(cfg: dict[str, Any], state: dict[str, Any]) -> None:
    recipients: list[str] = cfg.get("recipients", [])
    cooldown = float(cfg.get("cooldown_min", 360))
    recovery = bool(cfg.get("notify_on_recovery", True))

    # Сначала пытаемся добить недоставленные ранее алерты (сеть могла вернуться)
    await flush_pending_alerts(recipients)

    async def emit(alert_type: str, severity: str, title: str, body: str = "") -> None:
        if not recipients:
            return
        await dispatch_alert(alert_type, severity, _fmt(title, body), recipients)

    # ── Баланс API (DeepSeek) ──
    if cfg.get("balance_enabled", True):
        display = get_active_llm_display()
        # Ключ из активного snapshot (runtime JSON) — тот же источник, что и движок,
        # а не settings/.env (который устаревает после рестарта процесса).
        deepseek_key = get_llm_settings_snapshot()["deepseek_api_key"]
        if display["provider"] == "deepseek" and deepseek_key:
            value, currency = await fetch_deepseek_balance_raw(deepseek_key)
            if value is not None:
                threshold = float(cfg.get("balance_threshold_usd", 5.0))
                decision = _should_fire(state, "balance", value < threshold, cooldown, recovery)
                bal_str = format_balance(value, currency)
                if decision == "alert":
                    await emit(
                        "balance",
                        "critical",
                        "Низкий баланс DeepSeek",
                        f"Текущий баланс: <b>{bal_str}</b> (порог: ${threshold:.2f})",
                    )
                elif decision == "recovery":
                    await emit(
                        "balance", "recovery", "Баланс DeepSeek восстановлен", f"Текущий баланс: <b>{bal_str}</b>"
                    )

    # ── Здоровье сервисов ──
    if cfg.get("health_enabled", True):
        db_ok = await _db_ok()
        d = _should_fire(state, "health:database", not db_ok, cooldown, recovery)
        if d == "alert":
            await emit("health", "critical", "База данных недоступна", "Проверка SELECT 1 не прошла.")
        elif d == "recovery":
            await emit("health", "recovery", "База данных восстановлена")

        reasons = get_cached_or_load()
        kb_empty = not reasons or len(reasons.reasons) == 0
        d = _should_fire(state, "health:kb", kb_empty, cooldown, recovery)
        if d == "alert":
            await emit("health", "warning", "База знаний пуста", "Не загружено ни одной причины обращения.")
        elif d == "recovery":
            await emit("health", "recovery", "База знаний загружена")

        alive = _tg_alive()
        if alive is not None:  # None — нет данных, не алертим
            d = _should_fire(state, "health:tg", not alive, cooldown, recovery)
            if d == "alert":
                await emit("health", "critical", "TG-бот offline", "Heartbeat устарел (> 90 сек).")
            elif d == "recovery":
                await emit("health", "recovery", "TG-бот снова online")

    # ── Всплеск ошибок приложения ──
    if cfg.get("errors_enabled", True):
        new_errors = _read_new_errors(state)
        threshold = int(cfg.get("error_spike_threshold", 5))
        err_cooldown = float(cfg.get("errors_cooldown_min", 30))
        if len(new_errors) >= threshold and _minutes_since(state.get("errors_last_fired")) >= err_cooldown:
            sample = "\n".join(f"• {e}" for e in new_errors[:5])
            await emit(
                "errors",
                "warning",
                f"Всплеск ошибок: {len(new_errors)} за интервал",
                f"Примеры:\n{sample}",
            )
            state["errors_last_fired"] = _now().isoformat()

    # ── Сбой LLM-ключа ──
    if cfg.get("llm_key_failure_enabled", True):
        failure = read_llm_key_failure()
        if failure and failure.get("ts"):
            cond = state["conditions"].get("llm_key", {})
            last_seen = cond.get("last_seen_ts")
            if failure["ts"] != last_seen and _minutes_since(cond.get("last_fired")) >= cooldown:
                await emit(
                    "llm_key",
                    "critical",
                    "Сбой LLM-ключа",
                    f"Провайдер <b>{failure.get('provider', '?')}</b> вернул код "
                    f"<b>{failure.get('status', '?')}</b> (ключ/оплата/лимит).",
                )
                cond["last_fired"] = _now().isoformat()
            cond["last_seen_ts"] = failure["ts"]
            state["conditions"]["llm_key"] = cond


async def alert_monitor_loop() -> None:
    """Бесконечный цикл монитора оповещений (запускается в lifespan бэкенда)."""
    logger.info("[ALERTS] Монитор оповещений запущен")
    while True:
        cfg = get_alert_settings()
        interval = int(cfg.get("poll_interval_sec", 300))
        try:
            if cfg.get("enabled"):
                state = _load_state()
                await _run_checks(cfg, state)
                _save_state(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[ALERTS] monitor cycle error: %s", exc)
        await asyncio.sleep(max(30, interval))
