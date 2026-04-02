"""
API для управления причинами обращения (Bot Config).

CRUD-операции + тестирование классификации.
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

from docx import Document
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database.reason_store import (
    delete_reason,
    get_all_reasons,
    get_reason,
    invalidate_cache,
    upsert_reason,
)
from app.llm_settings import (
    apply_llm_settings_snapshot,
    get_active_llm_display,
    get_classification_settings,
    get_llm_settings_snapshot,
    save_classification_settings,
    save_runtime_llm_settings,
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


class LLMSettingsPayload(BaseModel):
    llm_provider: str = Field(default="yandex")
    show_llm_in_chat: bool = False
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class LLMSettingsResponse(BaseModel):
    llm_provider: str = "yandex"
    show_llm_in_chat: bool = False
    llm_temperature: float = 0.1
    available_providers: list[str] = Field(default_factory=lambda: ["yandex", "deepseek"])
    active_provider: str = "yandex"
    active_model: str = "yandexgpt"
    active_label: str = "Yandex / yandexgpt"


class ClassificationSettingsPayload(BaseModel):
    l1_global_min_score: float = Field(default=5.0, ge=0.0, le=100.0)
    l1_weight_phrase_mask: float = Field(default=10.0, ge=0.0, le=50.0)
    l1_weight_numeric_tag: float = Field(default=5.0, ge=0.0, le=50.0)
    l1_weight_noun: float = Field(default=2.0, ge=0.0, le=50.0)
    l1_weight_verb: float = Field(default=1.0, ge=0.0, le=50.0)


class ClassificationSettingsResponse(BaseModel):
    l1_global_min_score: float = 5.0
    l1_weight_phrase_mask: float = 10.0
    l1_weight_numeric_tag: float = 5.0
    l1_weight_noun: float = 2.0
    l1_weight_verb: float = 1.0


def _normalize_provider_or_422(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized not in {"yandex", "deepseek"}:
        raise HTTPException(status_code=422, detail="Допустимые провайдеры: yandex, deepseek")
    return normalized


def _reason_to_docx_lines(reason: ContactReason) -> list[str]:
    def sanitize_cell(value: str) -> str:
        return (value or "").replace("|", "/").strip()

    lines = [
        f"## БАЗА ЗНАНИЙ: {reason.name} (ПОЛНАЯ ВЕРСИЯ)",
        "",
        "### Раздел: Маркеры классификации",
        "",
        "#### Глаголы-маркеры",
    ]

    verb_lines = [f"- {item}" for item in reason.markers.verbs] or ["- "]
    noun_lines = [f"- {item}" for item in reason.markers.nouns] or ["- "]
    numeric_lines = [f"- {item}" for item in reason.markers.numeric_tags] or ["- "]
    phrase_lines = [f"- {item}" for item in reason.markers.phrase_masks] or ["- "]
    lines.extend(verb_lines)
    lines.extend(["", "#### Существительные-маркеры"])
    lines.extend(noun_lines)
    lines.extend(["", "#### Числовые теги"])
    lines.extend(numeric_lines)
    lines.extend(["", "#### Фразовые маски (100%-маркеры)"])
    lines.extend(phrase_lines)

    for section_index, section in enumerate(reason.thematic_sections, start=1):
        lines.extend(["", "---", "", f"### Раздел {section_index}. {section.title}"])
        for qa_index, qa in enumerate(section.qa_pairs, start=1):
            answer_lines = [line.rstrip() for line in qa.answer.splitlines()] or [""]
            first_answer_line = answer_lines[0] if answer_lines else ""
            lines.extend(
                [
                    "",
                    f"**Вопрос {qa_index}. {qa.question}**",
                    "",
                    f"**Ответ:** {first_answer_line}",
                ]
            )
            lines.extend(answer_lines[1:])
            lines.extend(["", "---"])

    lines.extend(["", "### Раздел: Эскалация на специалиста"])
    lines.extend(["", "| Ситуация | Признаки | Действие |", "| :--- | :--- | :--- |"])
    for complaint in reason.typical_complaints:
        description = sanitize_cell(complaint.description)
        context = sanitize_cell(complaint.context)
        response_template = sanitize_cell(complaint.response_template)
        lines.append(f"| {description} | {context} | {response_template} |")

    lines.extend(["", "---", "", "### Раздел: Готовые ответы"])
    lines.extend(["", "| Вопрос пользователя | Идеальный ответ |", "| :--- | :--- |"])
    for example in reason.example_answers:
        question = sanitize_cell(example.user_question)
        answer = sanitize_cell(example.ideal_answer)
        lines.append(f"| {question} | {answer} |")

    # ── Escalation rules (L1.5) ──
    esc = reason.escalation_rules
    lines.extend(["", "---", "", "### Раздел: Правила 100%-эскалации (L1.5)"])
    lines.append(f"**Статус:** {'Включено' if esc.enabled else 'Выключено'}")
    lines.append(f"**Порог совпадения:** {esc.metrics.score_threshold}")
    lines.extend(["", "#### Ключевые фразы"])
    kw_lines = [f"- {kw}" for kw in esc.metrics.keyword_patterns] or ["- "]
    lines.extend(kw_lines)
    lines.extend(["", "#### Пары вопрос-ответ для эскалации"])
    lines.extend(["| Вопрос | Ответ |", "| :--- | :--- |"])
    for pair in esc.qa_pairs:
        q = sanitize_cell(pair.question)
        a = sanitize_cell(pair.answer)
        lines.append(f"| {q} | {a} |")

    # ── Classification rules (L1.1) ──
    cls = reason.classification_rules
    lines.extend(["", "---", "", "### Раздел: Правила классификации (L1.1)"])
    lines.append(f"**Статус:** {'Включено' if cls.enabled else 'Выключено'}")
    lines.append(
        f"**Минимальный порог:** {cls.min_score_threshold if cls.min_score_threshold is not None else 'глобальный'}"
    )
    lines.append(f"**Обязательные маркеры:** {', '.join(cls.required_markers) if cls.required_markers else 'нет'}")
    lines.append(f"**Текст уточняющего вопроса:** {cls.clarification_text or 'стандартный'}")

    return lines


def _build_reason_docx(reason: ContactReason) -> io.BytesIO:
    document = Document()
    for line in _reason_to_docx_lines(reason):
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer


def _make_export_filename(reason: ContactReason) -> str:
    base = re.sub(r"[^\w\-]+", "_", reason.id or reason.name, flags=re.UNICODE).strip("_")
    return f"{base or 'contact_reason'}.docx"


def _upsert_env_line(lines: list[str], key: str, value: str) -> list[str]:
    new_line = f"{key}={value}"
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = new_line
            return lines
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(new_line)
    return lines


def _persist_llm_settings(payload: LLMSettingsPayload) -> Path:
    env_path = settings.env_file_path
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    values = {
        "LLM_PROVIDER": _normalize_provider_or_422(payload.llm_provider),
        "SHOW_LLM_IN_CHAT": "true" if payload.show_llm_in_chat else "false",
        "LLM_TEMPERATURE": str(payload.llm_temperature),
    }
    for key, value in values.items():
        lines = _upsert_env_line(lines, key, value)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def _apply_llm_settings(payload: LLMSettingsPayload) -> None:
    apply_llm_settings_snapshot(
        {
            "llm_provider": payload.llm_provider,
            "show_llm_in_chat": str(payload.show_llm_in_chat).lower(),
            "llm_temperature": str(payload.llm_temperature),
        }
    )


def _get_llm_settings_response() -> LLMSettingsResponse:
    snapshot = get_llm_settings_snapshot()
    active = get_active_llm_display()
    return LLMSettingsResponse(
        llm_provider=snapshot["llm_provider"],
        show_llm_in_chat=snapshot["show_llm_in_chat"] == "true",
        llm_temperature=float(snapshot.get("llm_temperature", "0.1")),
        active_provider=str(active["provider"]),
        active_model=str(active["model"]),
        active_label=str(active["label"]),
    )


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


@router.get("/llm-settings", response_model=LLMSettingsResponse)
async def get_llm_settings():
    """Получить текущие настройки LLM-провайдера."""
    return _get_llm_settings_response()


@router.put("/llm-settings", response_model=LLMSettingsResponse)
async def update_llm_settings(payload: LLMSettingsPayload):
    """Сохранить только выбор провайдера и флаг показа модели в чате."""
    _normalize_provider_or_422(payload.llm_provider)

    env_path = _persist_llm_settings(payload)
    snapshot = get_llm_settings_snapshot()
    runtime_path = save_runtime_llm_settings(
        {
            "llm_provider": payload.llm_provider,
            "show_llm_in_chat": str(payload.show_llm_in_chat).lower(),
            "llm_temperature": str(payload.llm_temperature),
            "yandex_api_key": snapshot["yandex_api_key"],
            "yandex_folder_id": snapshot["yandex_folder_id"],
            "yandex_gpt_model": snapshot["yandex_gpt_model"],
            "yandex_embedding_model": snapshot["yandex_embedding_model"],
            "deepseek_api_key": snapshot["deepseek_api_key"],
            "deepseek_model": snapshot["deepseek_model"],
        }
    )
    _apply_llm_settings(payload)

    from app.rag.engine import close_rag_engine

    await close_rag_engine()
    logger.info("LLM settings updated and persisted to %s and %s", env_path, runtime_path)
    return _get_llm_settings_response()


@router.get("/classification-settings", response_model=ClassificationSettingsResponse)
async def get_cls_settings():
    """Получить текущие настройки классификации L1 (веса маркеров + глобальный порог)."""
    data = get_classification_settings()
    return ClassificationSettingsResponse(**data)


@router.put("/classification-settings", response_model=ClassificationSettingsResponse)
async def update_cls_settings(payload: ClassificationSettingsPayload):
    """Сохранить настройки классификации L1."""
    save_classification_settings(payload.model_dump())
    logger.info("Classification settings updated: %s", payload.model_dump())
    return ClassificationSettingsResponse(**get_classification_settings())


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


@router.post("/export-docx")
async def export_docx(reason: ContactReason):
    """Экспорт текущей причины обращения в .docx-формат, совместимый с импортом."""
    file_like = _build_reason_docx(reason)
    filename = _make_export_filename(reason)
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
    return StreamingResponse(
        file_like,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
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
