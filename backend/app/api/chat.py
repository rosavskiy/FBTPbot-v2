"""
API эндпоинты чата 1.0.0 — L1→L2→L3 pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import get_db
from app.database.service import DatabaseService
from app.llm_settings import get_active_llm_display, get_chat_routing_policy_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatRoutingPolicy,
    DebugTrace,
    FileData,
    compute_confidence_label,
    compute_confidence_level,
)
from app.rag.engine import get_rag_engine
from app.sheets.gsheet_logger import get_gsheet_logger
from app.tg.notifier import get_telegram_notifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

RESOLVED_RESPONSE_TEXT = "Рад, что помог! Если появятся новые вопросы, обращайтесь."
RESOLVED_POSITIVE_MARKERS = (
    "спасибо",
    "благодарю",
    "благодарствую",
    "помогли",
    "помогло",
    "все получилось",
    "все вышло",
    "разобрался",
    "разобралась",
    "вопрос решен",
    "проблема решена",
    "можно закрывать",
    "не актуально",
)
RESOLVED_NEGATIVE_MARKERS = (
    "не помогло",
    "не помогли",
    "не получилось",
    "не вышло",
    "не сработало",
    "не сработал",
    "не решено",
    "не решен",
    "не разобрался",
    "не разобралась",
    "еще вопрос",
    "ещё вопрос",
    "еще проблема",
    "ещё проблема",
    "а теперь",
)
RESOLVED_MAX_TOKENS = 12

# Единая фраза передачи живому специалисту и режим ожидания оператора.
ESCALATION_HANDOFF_TEXT = "Передаю ваше обращение специалисту техподдержки — он ответит здесь в ближайшее время."
SILENT_MODE_TEXT = "Ваше обращение уже передано специалисту техподдержки. Пожалуйста, ожидайте ответа здесь."

# Маркеры раздражения/тупика: при их появлении после ответа бота сразу зовём оператора,
# не повторяя инструкции. Список намеренно консервативный, чтобы не эскалировать ложно.
FRUSTRATION_MARKERS = (
    # Явная просьба позвать оператора/человека
    "нужен оператор",
    "нужна оператор",
    "нужен живой",
    "живого человека",
    "живой человек",
    "хочу оператора",
    "позовите оператора",
    "дайте оператора",
    "хочу к оператору",
    "соедините с оператором",
    "соединитесь с оператором",
    "подключите оператора",
    "нужен специалист",
    "позовите специалиста",
    # Ненормативная лексика / грубость
    "заебал",
    "заебали",
    "заеб",
    "сучар",
    "пошел нахуй",
    "иди нахуй",
    "нахрен",
    "блять",
    "ёбан",
    "ёбаный",
    # Раздражение / тупик
    "не обезьян",
    "обезьяна",
    "задолбал",
    "задолбали",
    "вы не понимаете",
    "не понимаете",
    "не понимаешь",
    "вы достали",
    "достали уже",
    "хватит копировать",
    "не надо мне копировать",
    "перестань копировать",
    "не копируйте",
    "одно и тоже",
    "одно и то же",
    "валить друг на друга",
    "валите друг на друга",
    "бесполезно",
    "буду жаловаться",
    "напишу жалобу",
    "через ваше руководство",
    "через руководство",
    "тупой бот",
    "бестолков",
)

# Служебные сообщения, которые внешняя программа (WebClient) шлёт как реплику пользователя.
# На них не нужно звать LLM и засорять историю диалога.
SERVICE_NOISE_MARKERS = (
    "пользователь открыл чат",
    "пользователь закрыл чат",
    "чат открыт",
    "чат закрыт",
)

# Порог схожести (0..1) для детекта повторного ответа бота.
DUPLICATE_SIMILARITY_THRESHOLD = 0.85


@router.get("/routing-policy", response_model=ChatRoutingPolicy)
async def get_chat_routing_policy_defaults():
    return ChatRoutingPolicy(**get_chat_routing_policy_settings())


def _match_service_noise(message: str) -> str | None:
    """Возвращает первый совпавший служебный маркер или None."""
    normalized = _normalize_followup_text(message)
    if not normalized:
        return None
    for marker in SERVICE_NOISE_MARKERS:
        if marker in normalized:
            return marker
    return None


def _is_service_noise(message: str) -> bool:
    """Служебная реплика внешней системы (открытие/закрытие чата) — не для LLM."""
    return _match_service_noise(message) is not None


def _match_frustration_marker(message: str) -> str | None:
    """Возвращает первый совпавший маркер раздражения или None."""
    normalized = _normalize_followup_text(message)
    if not normalized:
        return None
    for marker in FRUSTRATION_MARKERS:
        if marker in normalized:
            return marker
    return None


def _is_frustrated(message: str) -> bool:
    """Признаки раздражения/тупика в сообщении пользователя."""
    return _match_frustration_marker(message) is not None


def _last_assistant_answer(chat_history: list[dict]) -> str:
    """Последний непустой ответ ассистента из истории."""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = str(msg.get("content", "")).strip()
            if content:
                return content
    return ""


def _is_duplicate_answer(new_answer: str, chat_history: list[dict]) -> bool:
    """True, если новый ответ почти дословно повторяет предыдущий ответ бота."""
    previous = _last_assistant_answer(chat_history)
    if not previous or not new_answer.strip():
        return False
    a = _normalize_followup_text(new_answer)
    b = _normalize_followup_text(previous)
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= DUPLICATE_SIMILARITY_THRESHOLD


def _build_suggested_topics(candidates: list[dict]) -> list[dict] | None:
    if not candidates:
        return None
    return [
        {
            "title": item.get("reason_name", ""),
            "article_id": item.get("reason_id", ""),
            "score": item.get("score", 0.0),
            "snippet": "",
        }
        for item in candidates
    ]


def _resolve_response_type(classification_method: str) -> str:
    if classification_method in {"clarification", "marker_clarification", "answer_refinement"}:
        return "clarification"
    return "answer"


def _normalize_followup_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = "".join(char if char.isalnum() or char.isspace() else " " for char in normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _has_recent_assistant_reply(chat_history: list[dict]) -> bool:
    return any(msg.get("role") == "assistant" and str(msg.get("content", "")).strip() for msg in chat_history)


def _match_resolved_marker(message: str, chat_history: list[dict]) -> str | None:
    """Возвращает первый совпавший маркер «вопрос решён» или None."""
    if not _has_recent_assistant_reply(chat_history):
        return None
    normalized = _normalize_followup_text(message)
    if not normalized or len(normalized.split()) > RESOLVED_MAX_TOKENS:
        return None
    if any(m in normalized for m in RESOLVED_NEGATIVE_MARKERS):
        return None
    for marker in RESOLVED_POSITIVE_MARKERS:
        if marker in normalized:
            return marker
    return None


def _is_resolved_followup(message: str, chat_history: list[dict]) -> bool:
    return _match_resolved_marker(message, chat_history) is not None


def _combine_query(original_query: str, followup: str) -> str:
    return f"{original_query}\n\nДополнительная информация от пользователя:\n{followup}".strip()


def _load_pending_payload(raw_payload: str | None) -> dict | list | None:
    if not raw_payload:
        return None
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        logger.warning("Не удалось распарсить payload pending clarification")
        return None


def _resolve_routing_policy(
    request_policy: ChatRoutingPolicy | None,
    pending_payload: dict | list | None,
) -> ChatRoutingPolicy | None:
    if request_policy is not None:
        return request_policy

    if isinstance(pending_payload, dict):
        raw_policy = pending_payload.get("routing_policy")
        if isinstance(raw_policy, dict):
            try:
                return ChatRoutingPolicy.model_validate(raw_policy)
            except ValidationError:
                logger.warning("Не удалось распарсить routing_policy из pending clarification")

    return None


async def _trigger_escalation(
    db_service: DatabaseService,
    session_id: str,
    user_message: str,
    reason_label: str,
    is_debug: bool = False,
    chat_guards: dict | None = None,
) -> ChatResponse:
    """Создать реальную эскалацию на оператора и вернуть ответ с передачей специалисту.

    Используется при детекте раздражения, зацикливании ответов или низкой уверенности.
    Идемпотентна: если по сессии уже есть активная эскалация — новую не создаёт.
    """
    existing = await db_service.get_active_escalation(session_id)

    if existing is None:
        history = await db_service.get_chat_history(session_id, limit=10)
        last_question = ""
        last_answer = ""
        for msg in reversed(history):
            if msg.role == "user" and not last_question:
                last_question = msg.content
            elif msg.role == "assistant" and not last_answer:
                last_answer = msg.content
            if last_question and last_answer:
                break
        if not last_question:
            last_question = user_message

        escalation = await db_service.create_escalation(
            session_id=session_id,
            reason=f"Автоэскалация: {reason_label}",
            contact_info=None,
        )

        chat_summary = "\n".join(f"{'👤' if m.role == 'user' else '🤖'} {m.content[:100]}" for m in history[-6:])

        try:
            notifier = get_telegram_notifier()
            tg_message_id = await notifier.send_escalation_notification(
                escalation_id=escalation.id,
                session_id=session_id,
                user_question=last_question,
                bot_answer=last_answer,
                reason=f"Автоэскалация: {reason_label}",
                contact_info=None,
                chat_summary=chat_summary,
            )
            if tg_message_id:
                await db_service.set_telegram_message_id(escalation.id, tg_message_id)
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось отправить уведомление об автоэскалации в Telegram")

        asyncio.ensure_future(
            get_gsheet_logger().log(
                question=last_question,
                answer=ESCALATION_HANDOFF_TEXT,
                session_id=session_id,
                response_type="escalation",
                escalation_info=f"Автоэскалация: {reason_label}",
                needs_escalation=True,
                is_debug=is_debug,
            )
        )

    # Снимаем зависшее уточнение и фиксируем ответ-передачу специалисту.
    await db_service.clear_pending_clarification(session_id)
    await db_service.add_message(
        session_id=session_id,
        role="assistant",
        content=ESCALATION_HANDOFF_TEXT,
        confidence=0.0,
        source_articles=[],
    )

    conf_level = compute_confidence_level(0.0)
    conf_label = compute_confidence_label(0.0)
    llm_display = get_active_llm_display()

    return ChatResponse(
        answer=ESCALATION_HANDOFF_TEXT,
        session_id=session_id,
        confidence=0.0,
        confidence_level=conf_level,
        confidence_label=conf_label,
        needs_escalation=True,
        source_articles=[],
        youtube_links=[],
        has_files=False,
        files=[],
        response_type="escalation",
        clarification_kind=None,
        suggested_topics=None,
        detected_reason=None,
        thematic_section=None,
        llm_provider=str(llm_display["provider"]),
        llm_model=str(llm_display["model"]),
        llm_label=str(llm_display["label"]),
        show_llm_in_chat=bool(llm_display["show_in_chat"]),
        debug_trace=DebugTrace(chat_guards=chat_guards) if (is_debug and chat_guards) else None,
    )


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Отправка сообщения в чат техподдержки 1.0.0.

    Pipeline:
    1. L1 — определение причины обращения по маркерам
    2. L2 — определение тематического раздела
    3. L3 — генерация ответа через YandexGPT

    Если session_id не указан — создаётся новая сессия.
    """
    db_service = DatabaseService(db)
    rag_engine = get_rag_engine()

    # Собираем результаты guard-проверок (только при debug=True)
    guards: dict | None = {} if request.debug else None

    # Получаем или создаём сессию
    session = None
    if request.session_id:
        session = await db_service.get_session(request.session_id)

    if session is None:
        session = await db_service.create_session(
            user_ip=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

    pending = await db_service.get_pending_clarification(session.id)

    # История нужна без текущего сообщения пользователя, чтобы follow-up детектор
    # и RAG опирались только на предыдущее состояние диалога.
    history_messages = await db_service.get_chat_history(session.id, limit=10)
    chat_history = [{"role": msg.role, "content": msg.content} for msg in history_messages]

    # Служебные реплики внешней системы («Пользователь открыл чат») не должны
    # вызывать LLM и засорять историю диалога.
    _svc_noise_marker = _match_service_noise(request.message)
    if guards is not None:
        guards["service_noise"] = {"triggered": _svc_noise_marker is not None, "matched_marker": _svc_noise_marker}
    if _svc_noise_marker is not None:
        llm_display = get_active_llm_display()
        return ChatResponse(
            answer="",
            session_id=session.id,
            confidence=1.0,
            confidence_level=compute_confidence_level(1.0),
            confidence_label=compute_confidence_label(1.0),
            needs_escalation=False,
            source_articles=[],
            youtube_links=[],
            has_files=False,
            files=[],
            response_type="answer",
            clarification_kind=None,
            suggested_topics=None,
            detected_reason=None,
            thematic_section=None,
            llm_provider=str(llm_display["provider"]),
            llm_model=str(llm_display["model"]),
            llm_label=str(llm_display["label"]),
            show_llm_in_chat=bool(llm_display["show_in_chat"]),
            debug_trace=DebugTrace(chat_guards=guards) if guards else None,
        )

    # Сохраняем сообщение пользователя
    await db_service.add_message(
        session_id=session.id,
        role="user",
        content=request.message,
    )

    # «Тихий» режим: если по сессии уже идёт эскалация на оператора — бот не
    # отвечает поверх специалиста, а лишь подтверждает ожидание.
    active_escalation = await db_service.get_active_escalation(session.id)
    if guards is not None:
        guards["silent_mode"] = {
            "triggered": active_escalation is not None,
            "escalation_id": active_escalation.id if active_escalation else None,
        }
    if active_escalation is not None:
        await db_service.add_message(
            session_id=session.id,
            role="assistant",
            content=SILENT_MODE_TEXT,
            confidence=0.0,
            source_articles=[],
        )
        llm_display = get_active_llm_display()
        return ChatResponse(
            answer=SILENT_MODE_TEXT,
            session_id=session.id,
            confidence=0.0,
            confidence_level=compute_confidence_level(0.0),
            confidence_label=compute_confidence_label(0.0),
            needs_escalation=True,
            source_articles=[],
            youtube_links=[],
            has_files=False,
            files=[],
            response_type="escalation",
            clarification_kind=None,
            suggested_topics=None,
            detected_reason=None,
            thematic_section=None,
            llm_provider=str(llm_display["provider"]),
            llm_model=str(llm_display["model"]),
            llm_label=str(llm_display["label"]),
            show_llm_in_chat=bool(llm_display["show_in_chat"]),
            debug_trace=DebugTrace(chat_guards=guards) if guards else None,
        )

    _resolved_marker = _match_resolved_marker(request.message, chat_history)
    if guards is not None:
        guards["resolved"] = {"triggered": _resolved_marker is not None, "matched_marker": _resolved_marker}
    if _resolved_marker is not None:
        if pending is not None:
            await db_service.clear_pending_clarification(session.id)

        await db_service.add_message(
            session_id=session.id,
            role="assistant",
            content=RESOLVED_RESPONSE_TEXT,
            confidence=1.0,
            source_articles=[],
        )

        conf_level = compute_confidence_level(1.0)
        conf_label = compute_confidence_label(1.0)
        llm_display = get_active_llm_display()

        asyncio.ensure_future(
            get_gsheet_logger().log(
                question=request.message,
                answer=RESOLVED_RESPONSE_TEXT,
                session_id=session.id,
                confidence=1.0,
                confidence_level=conf_level.value,
                confidence_label=conf_label,
                needs_escalation=False,
                source_articles=[],
                response_type="resolved",
                youtube_links=[],
                has_files=False,
                is_debug=request.debug,
            )
        )

        return ChatResponse(
            answer=RESOLVED_RESPONSE_TEXT,
            session_id=session.id,
            confidence=1.0,
            confidence_level=conf_level,
            confidence_label=conf_label,
            needs_escalation=False,
            source_articles=[],
            youtube_links=[],
            has_files=False,
            files=[],
            response_type="resolved",
            clarification_kind=None,
            suggested_topics=None,
            detected_reason=None,
            thematic_section=None,
            llm_provider=str(llm_display["provider"]),
            llm_model=str(llm_display["model"]),
            llm_label=str(llm_display["label"]),
            show_llm_in_chat=bool(llm_display["show_in_chat"]),
            debug_trace=None,
        )

    # Детект раздражения/тупика: если пользователь злится и бот уже что-то отвечал —
    # сразу передаём оператору, не повторяя инструкции.
    _frustration_marker = _match_frustration_marker(request.message)
    _has_prior = _has_recent_assistant_reply(chat_history)
    if guards is not None:
        guards["frustration"] = {
            "triggered": bool(_frustration_marker and _has_prior),
            "matched_marker": _frustration_marker,
            "has_prior_reply": _has_prior,
        }
    if _frustration_marker and _has_prior:
        logger.info("[CHAT] Автоэскалация по раздражению пользователя, session=%s", session.id)
        return await _trigger_escalation(
            db_service=db_service,
            session_id=session.id,
            user_message=request.message,
            reason_label="пользователь выражает недовольство ответами бота",
            is_debug=request.debug,
            chat_guards=guards,
        )

    question_for_engine = request.message
    reason_id_override: str | None = None
    pending_used = False
    routing_policy = request.routing_policy
    refinement_attempt = 0

    if pending is not None:
        pending_payload = _load_pending_payload(pending.payload_json)
        pending_used = True
        routing_policy = _resolve_routing_policy(request.routing_policy, pending_payload)

        if pending.clarification_type == "reason_selection":
            if request.message.strip().isdigit() and isinstance(pending_payload, list):
                choice_idx = int(request.message.strip()) - 1
                if 0 <= choice_idx < len(pending_payload):
                    selected = pending_payload[choice_idx]
                    question_for_engine = pending.original_query
                    reason_id_override = selected.get("reason_id") or None
                else:
                    question_for_engine = _combine_query(pending.original_query, request.message)
            else:
                question_for_engine = _combine_query(pending.original_query, request.message)
        elif pending.clarification_type == "reason_details":
            question_for_engine = _combine_query(pending.original_query, request.message)
            reason_id_override = pending.fixed_reason_id
        elif pending.clarification_type == "answer_refinement":
            question_for_engine = _combine_query(pending.original_query, request.message)
            reason_id_override = pending.fixed_reason_id
            refinement_attempt = pending.attempts

    # ── Основной pipeline: L1→L2→L3 ──
    rag_response = await rag_engine.ask(
        question=question_for_engine,
        chat_history=chat_history,
        reason_id=reason_id_override,
        routing_policy=routing_policy,
        refinement_attempt=refinement_attempt,
        debug=request.debug,
    )

    response_type = _resolve_response_type(rag_response.classification_method)
    suggested_topics = _build_suggested_topics(rag_response.clarification_candidates)

    # Анти-повтор: если бот собирается повторить почти дословно предыдущий ответ —
    # это тупик, передаём оператору вместо зацикливания. Уточняющие вопросы не трогаем.
    _prev_answer = _last_assistant_answer(chat_history)
    _dup_sim: float | None = None
    if response_type == "answer" and _prev_answer and rag_response.answer.strip():
        _a = _normalize_followup_text(rag_response.answer)
        _b = _normalize_followup_text(_prev_answer)
        if _a and _b:
            _dup_sim = round(SequenceMatcher(None, _a, _b).ratio(), 3)
    _is_dup = _dup_sim is not None and _dup_sim >= DUPLICATE_SIMILARITY_THRESHOLD
    if guards is not None:
        guards["duplicate_answer"] = {
            "triggered": _is_dup,
            "similarity": _dup_sim,
            "threshold": DUPLICATE_SIMILARITY_THRESHOLD,
        }
    if _is_dup:
        logger.info("[CHAT] Автоэскалация по повтору ответа, session=%s", session.id)
        return await _trigger_escalation(
            db_service=db_service,
            session_id=session.id,
            user_message=request.message,
            reason_label="бот повторяет один и тот же ответ, вопрос не решается",
            is_debug=request.debug,
            chat_guards=guards,
        )

    # Движок сам определил низкую уверенность — создаём реальную эскалацию на оператора
    # (уведомление в Telegram + «тихий» режим), а не просто текст-заглушку.
    if guards is not None:
        guards["needs_escalation"] = {
            "triggered": rag_response.needs_escalation,
            "reason": rag_response.confidence_reason,
        }
    if rag_response.needs_escalation:
        logger.info("[CHAT] Автоэскалация по низкой уверенности движка, session=%s", session.id)
        return await _trigger_escalation(
            db_service=db_service,
            session_id=session.id,
            user_message=request.message,
            reason_label="недостаточно данных в базе знаний для решения вопроса",
            is_debug=request.debug,
            chat_guards=guards,
        )

    if response_type == "clarification":
        if rag_response.classification_method == "clarification" and rag_response.clarification_candidates:
            await db_service.upsert_pending_clarification(
                session_id=session.id,
                clarification_type="reason_selection",
                original_query=question_for_engine,
                prompt=rag_response.answer,
                payload=rag_response.clarification_candidates,
                attempts=(pending.attempts + 1) if pending else 1,
            )
        elif rag_response.classification_method == "marker_clarification" and rag_response.detected_reason:
            await db_service.upsert_pending_clarification(
                session_id=session.id,
                clarification_type="reason_details",
                original_query=question_for_engine,
                prompt=rag_response.answer,
                fixed_reason_id=rag_response.detected_reason,
                fixed_reason_name=rag_response.detected_reason_name,
                attempts=(pending.attempts + 1) if pending else 1,
            )
        elif rag_response.classification_method == "answer_refinement" and rag_response.detected_reason:
            attempts = 1
            if pending and pending.clarification_type == "answer_refinement":
                attempts = pending.attempts + 1
            await db_service.upsert_pending_clarification(
                session_id=session.id,
                clarification_type="answer_refinement",
                original_query=question_for_engine,
                prompt=rag_response.answer,
                fixed_reason_id=rag_response.detected_reason,
                fixed_reason_name=rag_response.detected_reason_name,
                payload={
                    "routing_policy": routing_policy.model_dump() if routing_policy is not None else None,
                    "previous_confidence": rag_response.confidence,
                    "previous_confidence_reason": rag_response.confidence_reason,
                    "thematic_section": rag_response.thematic_section,
                    "clarification_kind": rag_response.clarification_kind or "answer_refinement",
                },
                attempts=attempts,
            )
        else:
            await db_service.clear_pending_clarification(session.id)
    elif pending_used:
        await db_service.clear_pending_clarification(session.id)

    # Сохраняем ответ бота
    await db_service.add_message(
        session_id=session.id,
        role="assistant",
        content=rag_response.answer,
        confidence=rag_response.confidence,
        source_articles=rag_response.source_articles,
        detected_reason=rag_response.detected_reason_name or None,
    )

    conf_level = compute_confidence_level(rag_response.confidence)
    conf_label = compute_confidence_label(rag_response.confidence)
    llm_display = get_active_llm_display()

    # Логируем в Google Sheets (fire-and-forget)
    asyncio.ensure_future(
        get_gsheet_logger().log(
            question=question_for_engine if pending_used else request.message,
            answer=rag_response.answer,
            session_id=session.id,
            confidence=rag_response.confidence,
            confidence_level=conf_level.value,
            confidence_label=conf_label,
            needs_escalation=rag_response.needs_escalation,
            source_articles=rag_response.source_articles,
            detected_reason=rag_response.detected_reason_name,
            thematic_section=rag_response.thematic_section,
            response_type=response_type,
            youtube_links=rag_response.youtube_links,
            has_files=bool(rag_response.files),
            is_debug=request.debug,
        )
    )

    return ChatResponse(
        answer=rag_response.answer,
        session_id=session.id,
        confidence=rag_response.confidence,
        confidence_level=conf_level,
        confidence_label=conf_label,
        needs_escalation=rag_response.needs_escalation,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        has_files=bool(rag_response.files),
        files=[FileData(code=f["code"], data_uri=f["data_uri"], ext=f.get("ext", "")) for f in rag_response.files],
        response_type=response_type,
        clarification_kind=rag_response.clarification_kind or None,
        suggested_topics=suggested_topics,
        detected_reason=rag_response.detected_reason_name,
        thematic_section=rag_response.thematic_section,
        llm_provider=str(llm_display["provider"]),
        llm_model=str(llm_display["model"]),
        llm_label=str(llm_display["label"]),
        show_llm_in_chat=bool(llm_display["show_in_chat"]),
        debug_trace=DebugTrace(**{**rag_response.debug_trace, "chat_guards": guards})
        if rag_response.debug_trace
        else (DebugTrace(chat_guards=guards) if guards else None),
    )
