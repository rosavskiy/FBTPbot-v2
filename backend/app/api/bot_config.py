"""
API для управления причинами обращения (Bot Config).

CRUD-операции + тестирование классификации.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database.reason_store import (
    delete_reason,
    get_all_reasons,
    get_reason,
    invalidate_cache,
    upsert_reason,
)
from app.models.reason_schemas import ContactReason, ContactReasonsData, Markers

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot-config", tags=["bot-config"])


# ── Request/Response models ──

class ReasonSummary(BaseModel):
    id: str
    name: str
    is_active: bool
    markers_count: int = 0
    sections_count: int = 0
    qa_count: int = 0
    complaints_count: int = 0
    examples_count: int = 0


class ReasonsListResponse(BaseModel):
    total: int
    reasons: list[ReasonSummary]


class TestClassifyRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class TestClassifyResponse(BaseModel):
    query: str
    l1_method: str
    l1_confident: bool
    l1_reason: Optional[str] = None
    l1_reason_id: Optional[str] = None
    candidates: list[dict] = []
    l2_method: Optional[str] = None
    l2_section: Optional[str] = None
    l2_best_qa_score: Optional[float] = None
    l2_best_qa: Optional[str] = None
    l2_best_example_score: Optional[float] = None
    l2_best_example: Optional[str] = None


# ── Endpoints ──

@router.get("/reasons", response_model=ReasonsListResponse)
async def list_reasons(active_only: bool = False):
    """Список всех причин обращения."""
    reasons = get_all_reasons(active_only=active_only)
    summaries = []
    for r in reasons:
        m = r.markers
        markers_count = len(m.verbs) + len(m.nouns) + len(m.numeric_tags) + len(m.phrase_masks)
        qa_count = sum(len(s.qa_pairs) for s in r.thematic_sections)
        summaries.append(ReasonSummary(
            id=r.id,
            name=r.name,
            is_active=r.is_active,
            markers_count=markers_count,
            sections_count=len(r.thematic_sections),
            qa_count=qa_count,
            complaints_count=len(r.typical_complaints),
            examples_count=len(r.example_answers),
        ))
    return ReasonsListResponse(total=len(summaries), reasons=summaries)


@router.get("/reasons/{reason_id}", response_model=ContactReason)
async def get_reason_detail(reason_id: str):
    """Получить полные данные причины обращения."""
    reason = get_reason(reason_id)
    if not reason:
        raise HTTPException(status_code=404, detail="Причина обращения не найдена")
    return reason


@router.post("/reasons", response_model=ContactReason)
async def create_reason(reason: ContactReason):
    """Создать новую причину обращения."""
    existing = get_reason(reason.id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Причина с ID '{reason.id}' уже существует")
    upsert_reason(reason)
    logger.info(f"Created reason: {reason.id} ({reason.name})")
    return reason


@router.put("/reasons/{reason_id}", response_model=ContactReason)
async def update_reason(reason_id: str, reason: ContactReason):
    """Обновить причину обращения."""
    existing = get_reason(reason_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Причина обращения не найдена")
    reason.id = reason_id  # ID из пути имеет приоритет
    upsert_reason(reason)
    logger.info(f"Updated reason: {reason_id}")
    return reason


@router.delete("/reasons/{reason_id}")
async def remove_reason(reason_id: str):
    """Удалить причину обращения."""
    if not delete_reason(reason_id):
        raise HTTPException(status_code=404, detail="Причина обращения не найдена")
    logger.info(f"Deleted reason: {reason_id}")
    return {"status": "deleted", "id": reason_id}


@router.post("/reasons/{reason_id}/duplicate", response_model=ContactReason)
async def duplicate_reason(reason_id: str, new_id: str, new_name: str):
    """Дублировать причину обращения."""
    original = get_reason(reason_id)
    if not original:
        raise HTTPException(status_code=404, detail="Причина обращения не найдена")
    if get_reason(new_id):
        raise HTTPException(status_code=409, detail=f"Причина с ID '{new_id}' уже существует")

    clone = original.model_copy(deep=True)
    clone.id = new_id
    clone.name = new_name
    upsert_reason(clone)
    logger.info(f"Duplicated {reason_id} → {new_id}")
    return clone


@router.post("/test-classify", response_model=TestClassifyResponse)
async def test_classify(req: TestClassifyRequest):
    """Тестирование классификации вопроса (без генерации ответа)."""
    from app.rag.engine import get_rag_engine
    engine = get_rag_engine()
    result = await engine.test_classify(req.question)
    return TestClassifyResponse(**result)


@router.post("/reload")
async def reload_reasons():
    """Сбросить кэш причин (перечитать JSON)."""
    invalidate_cache()
    reasons = get_all_reasons(active_only=False)
    return {"status": "reloaded", "total": len(reasons)}
