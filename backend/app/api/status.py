"""
API мониторинга состояния бота в реальном времени.

Все endpoints защищены verify_admin_token (любой авторизованный admin).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin_auth import verify_admin_token
from app.api.operator import get_active_operator_tokens_count
from app.config import SARATOV_TZ, settings
from app.database.models import ChatMessageDB, Escalation, get_db
from app.database.reason_store import get_cached_or_load
from app.llm_settings import get_active_llm_display
from app.rag.session_store import get_active_sessions_count
from app.sheets.gsheet_logger import get_gsheet_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/status", tags=["status"])


# ── Helpers ──────────────────────────────────────────────────────────

_balance_cache: dict[str, Any] = {"value": None, "last_attempt": 0.0}
_BALANCE_TTL = 60.0  # seconds between API calls


async def _fetch_deepseek_balance(api_key: str) -> str | None:
    """Fetch DeepSeek account balance with 60-second in-memory cache."""
    now = time.monotonic()
    if now - _balance_cache["last_attempt"] < _BALANCE_TTL:
        return _balance_cache["value"]
    _balance_cache["last_attempt"] = now
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            infos = resp.json().get("balance_infos", [])
            entry = next((i for i in infos if i.get("currency") == "USD"), infos[0] if infos else None)
            if entry:
                sym = {"USD": "$", "CNY": "¥"}.get(entry["currency"], entry["currency"] + " ")
                _balance_cache["value"] = f"{sym}{float(entry['total_balance']):.2f}"
    except Exception as exc:
        logger.debug("[STATUS] DeepSeek balance fetch failed: %s", exc)
    return _balance_cache["value"]


def _today_start() -> datetime:
    """Начало сегодняшнего дня в саратовском времени (naive, как в БД)."""
    today = datetime.now(SARATOV_TZ).date()
    return datetime.combine(today, datetime.min.time())


def _hours_ago(hours: int = 24) -> datetime:
    """Naive datetime N часов назад в саратовском времени (как в БД)."""
    return datetime.now(SARATOV_TZ).replace(tzinfo=None) - timedelta(hours=hours)


def _date_range(date_str: str | None) -> tuple[datetime, datetime] | None:
    """Returns (start, end) naive datetimes for a given YYYY-MM-DD date, or None."""
    if not date_str:
        return None
    day = datetime.strptime(date_str, "%Y-%m-%d")
    return day, day + timedelta(days=1)


def _read_tg_heartbeat() -> dict[str, Any]:
    path = Path(settings.tg_heartbeat_path)
    if not path.exists():
        return {"alive": False, "age_sec": None, "ts": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts_str = data.get("ts", "")
        if ts_str:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_sec = int((datetime.now(UTC) - ts).total_seconds())
            return {"alive": age_sec < 90, "age_sec": age_sec, "ts": ts_str}
    except Exception:
        pass
    return {"alive": False, "age_sec": None, "ts": None}


# ── Response models ───────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    ok: bool
    detail: str = ""


class TodayStats(BaseModel):
    messages_total: int = 0
    messages_web: int = 0
    messages_tg: int = 0
    messages_operator: int = 0
    sessions: int = 0
    escalations: int = 0
    avg_confidence: float | None = None


class TgBotStatus(BaseModel):
    alive: bool
    age_sec: int | None = None


class GsheetsStatus(BaseModel):
    enabled: bool
    last_row: int | None = None


class LlmStatus(BaseModel):
    provider: str
    model: str
    balance: str | None = None


class OverviewResponse(BaseModel):
    backend: ServiceStatus
    database: ServiceStatus
    kb: ServiceStatus
    llm: LlmStatus
    tg_bot: TgBotStatus
    gsheets: GsheetsStatus
    today: TodayStats
    pending_escalations: int
    active_clarifications: int
    active_operators: int


class TimelineBucket(BaseModel):
    bucket: str  # "2026-06-05T14:00"
    web: int = 0
    tg: int = 0
    operator: int = 0
    total: int = 0


class TimelineResponse(BaseModel):
    buckets: list[TimelineBucket]


class ConfidenceDistribution(BaseModel):
    high: int = 0
    acceptable: int = 0
    partial: int = 0
    escalation: int = 0
    total: int = 0


class TopReasonItem(BaseModel):
    reason: str
    count: int


class TopReasonsResponse(BaseModel):
    items: list[TopReasonItem]


class RecentQAItem(BaseModel):
    session_id: str
    source: str
    question: str
    answer: str
    confidence: float | None
    created_at: str


class RecentQAResponse(BaseModel):
    items: list[RecentQAItem]


class PendingEscalationItem(BaseModel):
    id: str
    session_id: str
    reason: str | None
    contact_info: str | None
    created_at: str
    status: str


class PendingEscalationsResponse(BaseModel):
    items: list[PendingEscalationItem]


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Общий срез состояния системы: сервисы, KPI за сегодня/дату, счётчики."""
    dr = _date_range(date)
    if dr:
        ts_start, ts_end = dr
    else:
        ts_start = _today_start()
        ts_end = None

    ts_start_str = ts_start.strftime("%Y-%m-%d %H:%M:%S")
    ts_end_str = ts_end.strftime("%Y-%m-%d %H:%M:%S") if ts_end else None
    extra_cond = " AND created_at < :ts_end" if ts_end_str else ""
    params_base: dict[str, Any] = {"ts_start": ts_start_str}
    if ts_end_str:
        params_base["ts_end"] = ts_end_str

    # ── DB health ──
    db_ok = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        logger.error("[STATUS] DB health check failed: %s", exc)

    # ── KB health ──
    reasons_data = get_cached_or_load()
    kb_count = len(reasons_data.reasons) if reasons_data else 0
    kb_ok = kb_count > 0

    # ── LLM ──
    try:
        llm_display = get_active_llm_display()
        llm_balance: str | None = None
        if llm_display["provider"] == "deepseek" and settings.deepseek_api_key:
            llm_balance = await _fetch_deepseek_balance(settings.deepseek_api_key)
        llm_status = LlmStatus(
            provider=llm_display["provider"],
            model=str(llm_display["model"]),
            balance=llm_balance,
        )
    except Exception:
        llm_status = LlmStatus(provider="unknown", model="unknown")

    # ── TG bot ──
    hb = _read_tg_heartbeat()
    tg_status = TgBotStatus(alive=hb["alive"], age_sec=hb["age_sec"])

    # ── Google Sheets ──
    try:
        gs_status_data = get_gsheet_logger().get_status()
        gs_status = GsheetsStatus(enabled=gs_status_data["enabled"], last_row=gs_status_data["last_row"])
    except Exception:
        gs_status = GsheetsStatus(enabled=False)

    # ── Period stats ──
    source_rows = await db.execute(
        text(
            "SELECT COALESCE(source, 'web') as src, COUNT(*) as cnt "
            "FROM chat_messages WHERE role = 'user' AND created_at >= :ts_start" + extra_cond + " GROUP BY src"
        ),
        params_base,
    )
    source_counts: dict[str, int] = {}
    for row in source_rows:
        source_counts[row[0]] = row[1]

    sessions_row = await db.execute(
        text("SELECT COUNT(DISTINCT session_id) FROM chat_messages" " WHERE created_at >= :ts_start" + extra_cond),
        params_base,
    )
    sessions_count = sessions_row.scalar_one_or_none() or 0

    esc_today_row = await db.execute(
        text("SELECT COUNT(*) FROM escalations WHERE created_at >= :ts_start" + extra_cond),
        params_base,
    )
    esc_today = esc_today_row.scalar_one_or_none() or 0

    avg_conf_row = await db.execute(
        text(
            "SELECT AVG(confidence) FROM chat_messages "
            "WHERE role = 'assistant' AND confidence IS NOT NULL AND created_at >= :ts_start" + extra_cond
        ),
        params_base,
    )
    avg_conf_raw = avg_conf_row.scalar_one_or_none()
    avg_conf = round(float(avg_conf_raw), 3) if avg_conf_raw is not None else None

    msg_web = source_counts.get("web", 0)
    msg_tg = source_counts.get("tg", 0)
    msg_op = source_counts.get("operator", 0)
    msg_total = msg_web + msg_tg + msg_op

    today_stats = TodayStats(
        messages_total=msg_total,
        messages_web=msg_web,
        messages_tg=msg_tg,
        messages_operator=msg_op,
        sessions=sessions_count,
        escalations=esc_today,
        avg_confidence=avg_conf,
    )

    # ── Pending escalations (live state only, zeroed in history mode) ──
    if not dr:
        pending_row = await db.execute(
            text("SELECT COUNT(*) FROM escalations WHERE status IN ('pending', 'in_progress')")
        )
        pending_count = pending_row.scalar_one_or_none() or 0
        active_clarifications = get_active_sessions_count()
        active_operators = get_active_operator_tokens_count()
    else:
        pending_count = 0
        active_clarifications = 0
        active_operators = 0

    return OverviewResponse(
        backend=ServiceStatus(ok=True),
        database=ServiceStatus(ok=db_ok, detail="" if db_ok else "DB error"),
        kb=ServiceStatus(ok=kb_ok, detail=f"{kb_count} reasons" if kb_ok else "empty"),
        llm=llm_status,
        tg_bot=tg_status,
        gsheets=gs_status,
        today=today_stats,
        pending_escalations=pending_count,
        active_clarifications=active_clarifications,
        active_operators=active_operators,
    )


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    hours: int = 24,
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Активность по часам за последние N часов или за конкретную дату."""
    dr = _date_range(date)

    if dr:
        day_start, day_end = dr
        since_str = day_start.strftime("%Y-%m-%d %H:%M:%S")
        until_str = day_end.strftime("%Y-%m-%d %H:%M:%S")
        rows = await db.execute(
            text(
                "SELECT strftime('%Y-%m-%dT%H:00', created_at) as bucket, "
                "COALESCE(source, 'web') as src, COUNT(*) as cnt "
                "FROM chat_messages "
                "WHERE role = 'user' AND created_at >= :since AND created_at < :until "
                "GROUP BY bucket, src "
                "ORDER BY bucket"
            ),
            {"since": since_str, "until": until_str},
        )
    else:
        hours = max(1, min(hours, 168))
        since_str = _hours_ago(hours).strftime("%Y-%m-%d %H:%M:%S")
        rows = await db.execute(
            text(
                "SELECT strftime('%Y-%m-%dT%H:00', created_at) as bucket, "
                "COALESCE(source, 'web') as src, COUNT(*) as cnt "
                "FROM chat_messages "
                "WHERE role = 'user' AND created_at >= :since "
                "GROUP BY bucket, src "
                "ORDER BY bucket"
            ),
            {"since": since_str},
        )

    # Aggregate by bucket
    data: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket, src, cnt = row[0], row[1], row[2]
        if bucket not in data:
            data[bucket] = {"web": 0, "tg": 0, "operator": 0}
        data[bucket][src] = data[bucket].get(src, 0) + cnt

    # Build full hourly grid
    buckets: list[TimelineBucket] = []
    if dr:
        for h in range(24):
            slot = day_start + timedelta(hours=h)
            key = slot.strftime("%Y-%m-%dT%H:00")
            counts = data.get(key, {})
            web = counts.get("web", 0)
            tg = counts.get("tg", 0)
            op = counts.get("operator", 0)
            buckets.append(TimelineBucket(bucket=key, web=web, tg=tg, operator=op, total=web + tg + op))
    else:
        now_saratov = datetime.now(SARATOV_TZ).replace(tzinfo=None)
        for h in range(hours, 0, -1):
            slot = (now_saratov - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
            key = slot.strftime("%Y-%m-%dT%H:00")
            counts = data.get(key, {})
            web = counts.get("web", 0)
            tg = counts.get("tg", 0)
            op = counts.get("operator", 0)
            buckets.append(TimelineBucket(bucket=key, web=web, tg=tg, operator=op, total=web + tg + op))

    return TimelineResponse(buckets=buckets)


@router.get("/confidence-distribution", response_model=ConfidenceDistribution)
async def get_confidence_distribution(
    hours: int = 24,
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Распределение ответов по уровням уверенности за последние N часов или за дату."""
    dr = _date_range(date)
    if dr:
        ts_start, ts_end = dr
        time_filter = "created_at >= :since AND created_at < :until"
        params: dict[str, Any] = {
            "since": ts_start.strftime("%Y-%m-%d %H:%M:%S"),
            "until": ts_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        hours = max(1, min(hours, 168))
        time_filter = "created_at >= :since"
        params = {"since": _hours_ago(hours).strftime("%Y-%m-%d %H:%M:%S")}

    row = await db.execute(
        text(
            "SELECT "
            "SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high, "
            "SUM(CASE WHEN confidence >= 0.6 AND confidence < 0.8 THEN 1 ELSE 0 END) as acceptable, "
            "SUM(CASE WHEN confidence >= 0.3 AND confidence < 0.6 THEN 1 ELSE 0 END) as partial, "
            "SUM(CASE WHEN confidence < 0.3 THEN 1 ELSE 0 END) as escalation_lvl, "
            "COUNT(*) as total "
            "FROM chat_messages "
            f"WHERE role = 'assistant' AND confidence IS NOT NULL AND {time_filter}"
        ),
        params,
    )
    r = row.fetchone()
    if r is None or r[4] == 0:
        return ConfidenceDistribution()
    return ConfidenceDistribution(
        high=r[0] or 0,
        acceptable=r[1] or 0,
        partial=r[2] or 0,
        escalation=r[3] or 0,
        total=r[4] or 0,
    )


@router.get("/top-reasons", response_model=TopReasonsResponse)
async def get_top_reasons(
    hours: int = 24,
    limit: int = 10,
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """TOP-N причин обращения по сохранённому detected_reason за последние N часов или за дату."""
    limit = max(1, min(limit, 50))
    dr = _date_range(date)
    if dr:
        ts_start, ts_end = dr
        time_filter = "created_at >= :since AND created_at < :until"
        params: dict[str, Any] = {
            "since": ts_start.strftime("%Y-%m-%d %H:%M:%S"),
            "until": ts_end.strftime("%Y-%m-%d %H:%M:%S"),
            "lim": limit,
        }
    else:
        hours = max(1, min(hours, 168))
        time_filter = "created_at >= :since"
        params = {"since": _hours_ago(hours).strftime("%Y-%m-%d %H:%M:%S"), "lim": limit}

    rows = await db.execute(
        text(
            "SELECT detected_reason, COUNT(*) as cnt "
            "FROM chat_messages "
            f"WHERE role = 'assistant' AND detected_reason IS NOT NULL AND detected_reason != '' AND {time_filter} "
            "GROUP BY detected_reason "
            "ORDER BY cnt DESC "
            "LIMIT :lim"
        ),
        params,
    )
    items = [TopReasonItem(reason=row[0], count=row[1]) for row in rows]
    return TopReasonsResponse(items=items)


@router.get("/recent-qa", response_model=RecentQAResponse)
async def get_recent_qa(
    limit: int = 50,
    offset: int = Query(0, ge=0),
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Лента последних N пар вопрос-ответ из всех каналов, или все пары за выбранную дату."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    dr = _date_range(date)
    if dr:
        ts_start, ts_end = dr
        date_cond = " AND a.created_at >= :ts_start AND a.created_at < :ts_end"
        params: dict[str, Any] = {
            "lim": limit,
            "off": offset,
            "ts_start": ts_start.strftime("%Y-%m-%d %H:%M:%S"),
            "ts_end": ts_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        date_cond = ""
        params = {"lim": limit, "off": offset}

    rows = await db.execute(
        text(
            "SELECT a.session_id, COALESCE(a.source, 'web'), "
            "COALESCE(u.content, '') as question, a.content as answer, "
            "a.confidence, a.created_at "
            "FROM chat_messages a "
            "LEFT JOIN chat_messages u ON u.session_id = a.session_id "
            "  AND u.role = 'user' "
            "  AND u.id = ("
            "    SELECT MAX(id) FROM chat_messages "
            "    WHERE session_id = a.session_id AND role = 'user' AND id < a.id"
            "  ) "
            "WHERE a.role = 'assistant'" + date_cond + " ORDER BY a.id DESC LIMIT :lim OFFSET :off"
        ),
        params,
    )
    items = []
    for row in rows:
        session_id, source, question, answer, confidence, created_at = row
        items.append(
            RecentQAItem(
                session_id=str(session_id),
                source=str(source),
                question=str(question)[:300],
                answer=str(answer)[:300],
                confidence=float(confidence) if confidence is not None else None,
                created_at=str(created_at)[:19],
            )
        )
    return RecentQAResponse(items=items)


class SessionMessage(BaseModel):
    role: str
    content: str
    confidence: float | None
    detected_reason: str | None
    created_at: str


class SessionDetailResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


class EscalationDetailResponse(BaseModel):
    id: str
    session_id: str
    status: str
    reason: str | None
    contact_info: str | None
    created_at: str
    messages: list[SessionMessage]


@router.get("/session/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Полная история сообщений сессии."""
    result = await db.execute(
        select(ChatMessageDB).where(ChatMessageDB.session_id == session_id).order_by(ChatMessageDB.id.asc()).limit(100)
    )
    messages = result.scalars().all()
    return SessionDetailResponse(
        session_id=session_id,
        messages=[
            SessionMessage(
                role=m.role,
                content=m.content,
                confidence=m.confidence,
                detected_reason=m.detected_reason,
                created_at=str(m.created_at)[:19],
            )
            for m in messages
        ],
    )


@router.get("/escalation/{escalation_id}", response_model=EscalationDetailResponse)
async def get_escalation_detail(
    escalation_id: str,
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Детали эскалации с полной историей сообщений сессии."""
    esc_result = await db.execute(select(Escalation).where(Escalation.id == escalation_id))
    esc = esc_result.scalar_one_or_none()
    if esc is None:
        raise HTTPException(status_code=404, detail="Escalation not found")

    msg_result = await db.execute(
        select(ChatMessageDB)
        .where(ChatMessageDB.session_id == esc.session_id)
        .order_by(ChatMessageDB.id.asc())
        .limit(100)
    )
    messages = msg_result.scalars().all()
    return EscalationDetailResponse(
        id=str(esc.id),
        session_id=str(esc.session_id),
        status=str(esc.status),
        reason=esc.reason,
        contact_info=esc.contact_info,
        created_at=str(esc.created_at)[:19],
        messages=[
            SessionMessage(
                role=m.role,
                content=m.content,
                confidence=m.confidence,
                detected_reason=m.detected_reason,
                created_at=str(m.created_at)[:19],
            )
            for m in messages
        ],
    )


@router.get("/pending-escalations", response_model=PendingEscalationsResponse)
async def get_pending_escalations(
    limit: int = 50,
    offset: int = Query(0, ge=0),
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    _user=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
):
    """Активные (pending/in_progress) эскалации, или все эскалации за выбранную дату."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    dr = _date_range(date)

    if dr:
        ts_start, ts_end = dr
        result = await db.execute(
            select(Escalation)
            .where(Escalation.created_at >= ts_start)
            .where(Escalation.created_at < ts_end)
            .order_by(Escalation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    else:
        result = await db.execute(
            select(Escalation)
            .where(Escalation.status.in_(["pending", "in_progress"]))
            .order_by(Escalation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

    escalations = result.scalars().all()
    items = [
        PendingEscalationItem(
            id=str(e.id),
            session_id=str(e.session_id),
            reason=e.reason,
            contact_info=e.contact_info,
            created_at=str(e.created_at)[:19],
            status=str(e.status),
        )
        for e in escalations
    ]
    return PendingEscalationsResponse(items=items)
