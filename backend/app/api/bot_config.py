"""
API для управления причинами обращения (Bot Config).

CRUD-операции + тестирование классификации.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.database.reason_store import (
    delete_reason,
    get_all_reasons,
    get_reason,
    invalidate_cache,
    upsert_reason,
)
from app.models.reason_schemas import ContactReason

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
    l1_reason: str | None = None
    l1_reason_id: str | None = None
    candidates: list[dict] = []
    l2_method: str | None = None
    l2_section: str | None = None
    l2_best_qa_score: float | None = None
    l2_best_qa: str | None = None
    l2_best_example_score: float | None = None
    l2_best_example: str | None = None


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
        summaries.append(
            ReasonSummary(
                id=r.id,
                name=r.name,
                is_active=r.is_active,
                markers_count=markers_count,
                sections_count=len(r.thematic_sections),
                qa_count=qa_count,
                complaints_count=len(r.typical_complaints),
                examples_count=len(r.example_answers),
            )
        )
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


@router.get("/template")
async def download_template():
    """Скачать шаблон .docx для импорта причин обращения."""
    candidates = [
        Path(__file__).resolve().parents[3] / "templates" / "Шаблон причины обращения.docx",
        Path("/app/templates/Шаблон причины обращения.docx"),
    ]
    template_path = next((p for p in candidates if p.exists()), None)
    if not template_path:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    return FileResponse(
        path=str(template_path),
        filename="Шаблон причины обращения.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


class ImportResult(BaseModel):
    imported: int = 0
    skipped: int = 0
    errors: list[str] = []
    message: str = ""


@router.post("/import-docx", response_model=ImportResult)
async def import_docx(file: UploadFile = File(...)):
    """Импорт причин обращения из .docx файла."""
    if not file.filename or not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Допустимы только .docx файлы")

    # Сохраняем во временный файл
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 10 МБ)")

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    # Парсим docx
    try:
        from scripts.import_brains import parse_docx

        reason_dict = parse_docx(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(f"Ошибка парсинга docx: {e}")
        raise HTTPException(status_code=422, detail=f"Ошибка парсинга файла: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Валидируем и сохраняем
    errors = []
    imported = 0
    skipped = 0

    try:
        reason = ContactReason.model_validate(reason_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Невалидные данные: {e}")

    if not reason.id or not reason.name:
        raise HTTPException(status_code=422, detail="Не удалось извлечь название причины из файла")

    existing = get_reason(reason.id)
    if existing:
        # Обновляем существующую
        upsert_reason(reason)
        logger.info(f"Updated reason from docx: {reason.id} ({reason.name})")
    else:
        upsert_reason(reason)
        logger.info(f"Imported reason from docx: {reason.id} ({reason.name})")

    imported = 1
    invalidate_cache()

    m = reason.markers
    markers_count = len(m.verbs) + len(m.nouns) + len(m.numeric_tags) + len(m.phrase_masks)
    qa_count = sum(len(s.qa_pairs) for s in reason.thematic_sections)

    return ImportResult(
        imported=imported,
        skipped=skipped,
        errors=errors,
        message=(
            f"{'Обновлена' if existing else 'Импортирована'} причина «{reason.name}»: "
            f"{markers_count} маркеров, {len(reason.thematic_sections)} разделов, "
            f"{qa_count} Q&A, {len(reason.typical_complaints)} жалоб, "
            f"{len(reason.example_answers)} примеров"
        ),
    )
