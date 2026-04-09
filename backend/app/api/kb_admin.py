"""
API-эндпоинты для управления базой знаний Q&A (квиз-режим + импорт).

Позволяет:
- Просматривать Q&A пары из JSON-файла
- Редактировать вопросы/ответы/метаданные
- Одобрять пары (approved)
- Удалять некачественные записи
- Импортировать новые данные
- Синхронизировать с ChromaDB (инкрементально + полный реиндекс)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from langchain_chroma import Chroma
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin_auth import log_action, require_role, verify_admin_token
from app.config import settings
from app.database.models import AdminUser
from app.database.models import get_db as get_admin_db
from app.indexer.knowledge_base import (
    SUPPORT_COLLECTION_NAME,
    get_indexer,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["kb-admin"])

# Auth dependencies
_any_admin = Depends(verify_admin_token)
_editor = Depends(require_role("superadmin", "admin"))

# ─── Путь к JSON-файлу базы знаний ───────────────────────────────────
# На сервере: /app/data/support_kb.json (persistent volume)
# Локально: real_support/processed/support_qa_documents_merged_final.json
_KB_PATHS = [
    Path("/app/data/support_kb.json"),  # Docker production
    Path("/app/real_support/processed/support_qa_documents_merged_final.json"),  # Docker alt
    Path(__file__).resolve().parents[3]
    / "real_support"
    / "processed"
    / "support_qa_documents_merged_final.json",  # Local dev
]
KB_JSON_PATH = next((p for p in _KB_PATHS if p.exists()), _KB_PATHS[0])

KB_BACKUP_DIR = KB_JSON_PATH.parent / "backups"


# ─── Pydantic-модели ─────────────────────────────────────────────────


class KBItemMetadata(BaseModel):
    source: str = "real_support_tickets"
    category: str = "Прочее"
    category_en: str = "general"
    tags: list[str] = []
    quality_score: int = 3
    question: str = ""
    answer: str = ""
    type: str = "qa_pair"


class KBItem(BaseModel):
    id: str
    text: str
    metadata: KBItemMetadata
    reviewed: bool = False
    review_date: str | None = None


class KBItemUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    category: str | None = None
    category_en: str | None = None
    tags: list[str] | None = None
    quality_score: int | None = None


class KBStats(BaseModel):
    total: int = 0
    reviewed: int = 0
    unreviewed: int = 0
    by_category: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    avg_quality: float = 0.0


class KBImportResult(BaseModel):
    added: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    message: str = ""


class KBReindexResult(BaseModel):
    total_documents: int = 0
    duration_seconds: float = 0.0
    message: str = ""


class KBReindexStatus(BaseModel):
    job_id: str | None = None
    status: str = "idle"
    total_documents: int = 0
    processed_documents: int = 0
    progress_percent: float = 0.0
    duration_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None
    message: str = "Ожидание запуска"
    error: str | None = None


def _make_reindex_status() -> dict[str, Any]:
    return KBReindexStatus().model_dump()


_REINDEX_STATUS_LOCK = threading.Lock()
_REINDEX_STATUS: dict[str, Any] = _make_reindex_status()
_REINDEX_TASK: asyncio.Task | None = None


def _get_reindex_status() -> dict[str, Any]:
    with _REINDEX_STATUS_LOCK:
        return dict(_REINDEX_STATUS)


def _update_reindex_status(**updates: Any):
    with _REINDEX_STATUS_LOCK:
        _REINDEX_STATUS.update(updates)


def _set_reindex_phase(
    *,
    status: str,
    message: str,
    processed_documents: int | None = None,
    total_documents: int | None = None,
    progress_percent: float | None = None,
    error: str | None = None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
):
    payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat(),
    }
    if processed_documents is not None:
        payload["processed_documents"] = processed_documents
    if total_documents is not None:
        payload["total_documents"] = total_documents
    if progress_percent is not None:
        payload["progress_percent"] = progress_percent
    if error is not None:
        payload["error"] = error
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    _update_reindex_status(**payload)


async def _run_reindex_job(job_id: str):
    global _REINDEX_TASK

    started_at = datetime.now().isoformat()
    started_ts = time.time()
    total_documents = len(_load_kb())
    _update_reindex_status(
        job_id=job_id,
        status="running",
        total_documents=total_documents,
        processed_documents=0,
        progress_percent=0.0,
        duration_seconds=0.0,
        started_at=started_at,
        finished_at=None,
        updated_at=started_at,
        message="Подготовка к переиндексации",
        error=None,
    )

    try:
        indexer = get_indexer()

        def on_progress(processed: int, total: int, message: str):
            progress = round((processed / max(total, 1)) * 100, 1) if total else 0.0
            _set_reindex_phase(
                status="running",
                message=message,
                processed_documents=processed,
                total_documents=total,
                progress_percent=progress,
                duration_seconds=round(time.time() - started_ts, 2),
            )

        count = await asyncio.to_thread(
            indexer.index_support_tickets,
            KB_JSON_PATH,
            on_progress,
        )

        from app.rag.engine import _engine as _rag_engine

        if _rag_engine is not None and hasattr(_rag_engine, "_support_vector_store"):
            delattr(_rag_engine, "_support_vector_store")
            logger.info("Кеш support_vector_store в RAG-движке сброшен")

        indexer.support_vector_store = None

        finished_at = datetime.now().isoformat()
        duration = round(time.time() - started_ts, 2)
        _set_reindex_phase(
            status="completed",
            message=f"Переиндексация завершена: {count} документов за {duration}с",
            processed_documents=count,
            total_documents=count,
            progress_percent=100.0,
            finished_at=finished_at,
            duration_seconds=duration,
            error=None,
        )
    except Exception as e:
        finished_at = datetime.now().isoformat()
        duration = round(time.time() - started_ts, 2)
        logger.error(f"Ошибка переиндексации: {e}")
        _set_reindex_phase(
            status="failed",
            message="Переиндексация завершилась с ошибкой",
            finished_at=finished_at,
            duration_seconds=duration,
            error=str(e),
        )
    finally:
        _REINDEX_TASK = None


# ─── Утилиты для работы с JSON-файлом ────────────────────────────────


def _load_kb() -> list[dict[str, Any]]:
    """Загрузить базу знаний из JSON."""
    if not KB_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Файл БЗ не найден: {KB_JSON_PATH}")
    try:
        with open(KB_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка парсинга JSON: {e}")


def _save_kb(data: list[dict[str, Any]], backup: bool = True):
    """Сохранить базу знаний в JSON с опциональным бэкапом."""
    if backup:
        KB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = KB_BACKUP_DIR / f"kb_backup_{ts}.json"
        if KB_JSON_PATH.exists():
            shutil.copy2(KB_JSON_PATH, backup_path)
            logger.info(f"Бэкап создан: {backup_path}")
            # Оставляем только последние 20 бэкапов
            backups = sorted(KB_BACKUP_DIR.glob("kb_backup_*.json"))
            for old in backups[:-20]:
                old.unlink()

    with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"БЗ сохранена: {len(data)} записей -> {KB_JSON_PATH}")


def _update_chromadb_document(item: dict[str, Any]):
    """Инкрементально обновить один документ в ChromaDB."""
    try:
        indexer = get_indexer()
        store = indexer.get_support_vector_store()
        if store is None:
            store = Chroma(
                collection_name=SUPPORT_COLLECTION_NAME,
                embedding_function=indexer.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            indexer.support_vector_store = store
            logger.info("support_vector_store был создан для инкрементальной синхронизации")

        collection = store._collection
        doc_id = item["id"]
        metadata = item.get("metadata", {})

        # Подготовка метаданных (ChromaDB не поддерживает списки)
        clean_meta = {
            "source": metadata.get("source", "real_support_tickets"),
            "category": metadata.get("category", "Прочее"),
            "category_en": metadata.get("category_en", "general"),
            "quality_score": metadata.get("quality_score", 0),
            "question": metadata.get("question", "")[:500],
            "doc_type": metadata.get("type", "qa_pair"),
            "article_id": f"tp_{doc_id}",
            "title": metadata.get("question", "Заявка ТП")[:200],
        }
        if metadata.get("tags"):
            clean_meta["tags"] = ", ".join(metadata["tags"])
        if item.get("reviewed"):
            clean_meta["reviewed"] = "true"

        text = item["text"]

        # Пробуем обновить, если не существует — добавляем
        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: обновлён документ {doc_id}")
        else:
            collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: добавлен документ {doc_id}")

    except Exception as e:
        logger.error(f"Ошибка обновления ChromaDB для {item.get('id')}: {e}")


def _delete_chromadb_document(doc_id: str):
    """Удалить документ из ChromaDB."""
    try:
        indexer = get_indexer()
        store = indexer.get_support_vector_store()
        if store is None:
            store = Chroma(
                collection_name=SUPPORT_COLLECTION_NAME,
                embedding_function=indexer.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
            indexer.support_vector_store = store
        collection = store._collection
        collection.delete(ids=[doc_id])
        logger.info(f"ChromaDB: удалён документ {doc_id}")
    except Exception as e:
        logger.error(f"Ошибка удаления из ChromaDB {doc_id}: {e}")


# ─── Эндпоинты ───────────────────────────────────────────────────────


@router.get("/stats", response_model=KBStats)
async def get_kb_stats(user: AdminUser = _any_admin):
    """Статистика базы знаний."""
    data = _load_kb()
    by_category: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    reviewed = 0
    total_quality = 0.0

    for item in data:
        meta = item.get("metadata", {})
        cat = meta.get("category", "Прочее")
        by_category[cat] = by_category.get(cat, 0) + 1

        qs = str(meta.get("quality_score", 0))
        by_quality[qs] = by_quality.get(qs, 0) + 1
        total_quality += meta.get("quality_score", 0)

        if item.get("reviewed"):
            reviewed += 1

    return KBStats(
        total=len(data),
        reviewed=reviewed,
        unreviewed=len(data) - reviewed,
        by_category=by_category,
        by_quality=by_quality,
        avg_quality=round(total_quality / max(len(data), 1), 2),
    )


@router.get("/items")
async def list_kb_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: str | None = None,
    reviewed: bool | None = None,
    quality_min: int | None = None,
    quality_max: int | None = None,
    search: str | None = None,
    user: AdminUser = _any_admin,
):
    """Список Q&A пар с пагинацией и фильтрацией."""
    data = _load_kb()

    # Фильтрация
    if category:
        data = [d for d in data if d.get("metadata", {}).get("category") == category]
    if reviewed is not None:
        data = [d for d in data if d.get("reviewed", False) == reviewed]
    if quality_min is not None:
        data = [d for d in data if d.get("metadata", {}).get("quality_score", 0) >= quality_min]
    if quality_max is not None:
        data = [d for d in data if d.get("metadata", {}).get("quality_score", 0) <= quality_max]
    if search:
        search_lower = search.lower()
        data = [
            d
            for d in data
            if search_lower in d.get("id", "").lower()
            or search_lower in d.get("metadata", {}).get("question", "").lower()
            or search_lower in d.get("metadata", {}).get("answer", "").lower()
            or search_lower in d.get("text", "").lower()
        ]

    total = len(data)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "items": data[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/items/{item_id}")
async def get_kb_item(item_id: str, user: AdminUser = _any_admin):
    """Получить одну Q&A пару."""
    data = _load_kb()
    for item in data:
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")


@router.get("/quiz/next")
async def get_next_quiz_item(
    category: str | None = None,
    skip_reviewed: bool = True,
    user: AdminUser = _any_admin,
):
    """
    Получить следующую неотревьюенную Q&A пару для квиза.
    Возвращает элемент + общий прогресс.
    """
    data = _load_kb()

    candidates = data
    if category:
        candidates = [d for d in candidates if d.get("metadata", {}).get("category") == category]
    if skip_reviewed:
        candidates = [d for d in candidates if not d.get("reviewed", False)]

    total = len(data)
    reviewed_count = sum(1 for d in data if d.get("reviewed", False))

    if not candidates:
        return {
            "item": None,
            "progress": {
                "total": total,
                "reviewed": reviewed_count,
                "remaining": 0,
                "percent": 100.0 if total > 0 else 0.0,
            },
            "message": "Все записи проверены! 🎉",
        }

    item = candidates[0]
    idx = data.index(item)

    return {
        "item": item,
        "index": idx,
        "progress": {
            "total": total,
            "reviewed": reviewed_count,
            "remaining": len(candidates),
            "percent": round(reviewed_count / max(total, 1) * 100, 1),
        },
    }


@router.put("/items/{item_id}")
async def update_kb_item(
    item_id: str, update: KBItemUpdate, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)
):
    """Обновить Q&A пару (вопрос, ответ, категорию, теги и т.д.)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # Применяем обновления
    if update.question is not None:
        meta["question"] = update.question
    if update.answer is not None:
        meta["answer"] = update.answer
    if update.category is not None:
        meta["category"] = update.category
    if update.category_en is not None:
        meta["category_en"] = update.category_en
    if update.tags is not None:
        meta["tags"] = update.tags
    if update.quality_score is not None:
        meta["quality_score"] = update.quality_score

    # Перестраиваем текст
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # Инкрементально обновляем ChromaDB
    _update_chromadb_document(item)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="kb_item",
        entity_id=item_id,
        entity_name=meta.get("question", "")[:80],
    )
    return {"status": "ok", "item": item}


@router.post("/items/{item_id}/approve")
async def approve_kb_item(item_id: str, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """Одобрить Q&A пару (пометить как проверенную, quality_score=5)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    item["reviewed"] = True
    item["review_date"] = datetime.now().isoformat()
    meta = item.get("metadata", {})
    meta["quality_score"] = 5
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # Инкрементально обновляем ChromaDB
    _update_chromadb_document(item)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="kb_item",
        entity_id=item_id,
        details="Одобрена",
    )
    return {"status": "ok", "item": item}


@router.post("/items/{item_id}/save-and-approve")
async def save_and_approve_kb_item(
    item_id: str, update: KBItemUpdate, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)
):
    """Обновить и сразу одобрить Q&A пару."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # Применяем обновления
    if update.question is not None:
        meta["question"] = update.question
    if update.answer is not None:
        meta["answer"] = update.answer
    if update.category is not None:
        meta["category"] = update.category
    if update.category_en is not None:
        meta["category_en"] = update.category_en
    if update.tags is not None:
        meta["tags"] = update.tags
    if update.quality_score is not None:
        meta["quality_score"] = update.quality_score
    else:
        meta["quality_score"] = 5

    # Перестраиваем текст
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"
    item["metadata"] = meta
    item["reviewed"] = True
    item["review_date"] = datetime.now().isoformat()

    data[found_idx] = item
    _save_kb(data)

    _update_chromadb_document(item)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="kb_item",
        entity_id=item_id,
        details="Обновлена и одобрена",
    )
    return {"status": "ok", "item": item}


@router.delete("/items/{item_id}")
async def delete_kb_item(item_id: str, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """Удалить Q&A пару из базы знаний."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    data.pop(found_idx)
    _save_kb(data)

    _delete_chromadb_document(item_id)

    await log_action(
        db, user_id=user.id, username=user.username, action="delete", entity_type="kb_item", entity_id=item_id
    )
    return {"status": "ok", "deleted_id": item_id, "remaining": len(data)}


@router.post("/import", response_model=KBImportResult)
async def import_kb_data(
    file: UploadFile = File(...), user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)
):
    """
    Импорт новых Q&A данных из JSON-файла.
    Дубликаты (по id) пропускаются, новые записи добавляются.
    """
    try:
        content = await file.read()
        new_items = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    if not isinstance(new_items, list):
        raise HTTPException(status_code=400, detail="Ожидается JSON-массив")

    data = _load_kb()
    existing_ids = {item["id"] for item in data}

    added = 0
    duplicates = 0
    errors = 0

    for new_item in new_items:
        try:
            if not isinstance(new_item, dict) or "id" not in new_item:
                errors += 1
                continue

            if new_item["id"] in existing_ids:
                duplicates += 1
                continue

            # Убеждаемся, что есть все нужные поля
            if "text" not in new_item:
                meta = new_item.get("metadata", {})
                q = meta.get("question", "")
                a = meta.get("answer", "")
                new_item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"

            if "metadata" not in new_item:
                new_item["metadata"] = {
                    "source": "real_support_tickets",
                    "category": "Прочее",
                    "category_en": "general",
                    "tags": [],
                    "quality_score": 3,
                    "question": "",
                    "answer": "",
                    "type": "qa_pair",
                }

            new_item["reviewed"] = False
            data.append(new_item)
            existing_ids.add(new_item["id"])
            added += 1

            # Добавляем в ChromaDB
            _update_chromadb_document(new_item)

        except Exception as e:
            logger.error(f"Ошибка импорта элемента: {e}")
            errors += 1

    _save_kb(data)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="import",
        entity_type="kb_item",
        details=f"Импорт: +{added} новых, {duplicates} дубликатов, {errors} ошибок",
    )
    return KBImportResult(
        added=added,
        duplicates_skipped=duplicates,
        errors=errors,
        message=f"Импорт завершён: +{added} новых, {duplicates} дубликатов пропущено, {errors} ошибок",
    )


@router.post("/reindex", response_model=KBReindexStatus)
async def reindex_kb(user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """
    Запустить полную переиндексацию support_tickets в фоне.
    Используйте после массовых правок.
    """
    global _REINDEX_TASK

    if _REINDEX_TASK is not None and not _REINDEX_TASK.done():
        return KBReindexStatus(**_get_reindex_status())

    job_id = str(uuid4())
    now = datetime.now().isoformat()
    total_documents = len(_load_kb())
    _update_reindex_status(
        job_id=job_id,
        status="running",
        total_documents=total_documents,
        processed_documents=0,
        progress_percent=0.0,
        duration_seconds=0.0,
        started_at=now,
        finished_at=None,
        updated_at=now,
        message="Задача переиндексации поставлена в очередь",
        error=None,
    )
    _REINDEX_TASK = asyncio.create_task(_run_reindex_job(job_id))
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="reindex",
        entity_type="kb_item",
        details=f"Переиндексация: {total_documents} документов",
    )
    return KBReindexStatus(**_get_reindex_status())


@router.get("/reindex/status", response_model=KBReindexStatus)
async def get_reindex_status(user: AdminUser = _any_admin):
    """Получить текущий статус фоновой переиндексации."""
    return KBReindexStatus(**_get_reindex_status())


@router.get("/categories")
async def get_categories(user: AdminUser = _any_admin):
    """Получить список всех категорий."""
    data = _load_kb()
    categories = {}
    for item in data:
        cat = item.get("metadata", {}).get("category", "Прочее")
        cat_en = item.get("metadata", {}).get("category_en", "general")
        categories[cat] = cat_en
    return {"categories": categories}
