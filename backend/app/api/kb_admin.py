"""
API-СЌРЅРґРїРѕРёРЅС‚С‹ РґР»СЏ СѓРїСЂР°РІР»РµРЅРёСЏ Р±Р°Р·РѕР№ Р·РЅР°РЅРёР№ Q&A (РєРІРёР·-СЂРµР¶РёРј + РёРјРїРѕСЂС‚).

РџРѕР·РІРѕР»СЏРµС‚:
- РџСЂРѕСЃРјР°С‚СЂРёРІР°С‚СЊ Q&A РїР°СЂС‹ РёР· JSON-С„Р°Р№Р»Р°
- Р РµРґР°РєС‚РёСЂРѕРІР°С‚СЊ РІРѕРїСЂРѕСЃС‹/РѕС‚РІРµС‚С‹/РјРµС‚Р°РґР°РЅРЅС‹Рµ
- РћРґРѕР±СЂСЏС‚СЊ РїР°СЂС‹ (approved)
- РЈРґР°Р»СЏС‚СЊ РЅРµРєР°С‡РµСЃС‚РІРµРЅРЅС‹Рµ Р·Р°РїРёСЃРё
- РРјРїРѕСЂС‚РёСЂРѕРІР°С‚СЊ РЅРѕРІС‹Рµ РґР°РЅРЅС‹Рµ
- РЎРёРЅС…СЂРѕРЅРёР·РёСЂРѕРІР°С‚СЊ СЃ ChromaDB (РёРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕ + РїРѕР»РЅС‹Р№ СЂРµРёРЅРґРµРєСЃ)
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
from app.config import SARATOV_TZ, settings
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

# в”Ђв”Ђв”Ђ РџСѓС‚СЊ Рє JSON-С„Р°Р№Р»Сѓ Р±Р°Р·С‹ Р·РЅР°РЅРёР№ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РќР° СЃРµСЂРІРµСЂРµ: /app/data/support_kb.json (persistent volume)
# Р›РѕРєР°Р»СЊРЅРѕ: real_support/processed/support_qa_documents_merged_final.json
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


# в”Ђв”Ђв”Ђ Pydantic-РјРѕРґРµР»Рё в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


class KBItemMetadata(BaseModel):
    source: str = "real_support_tickets"
    category: str = "РџСЂРѕС‡РµРµ"
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
    message: str = "РћР¶РёРґР°РЅРёРµ Р·Р°РїСѓСЃРєР°"
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
        "updated_at": datetime.now(SARATOV_TZ).isoformat(),
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

    started_at = datetime.now(SARATOV_TZ).isoformat()
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
        message="РџРѕРґРіРѕС‚РѕРІРєР° Рє РїРµСЂРµРёРЅРґРµРєСЃР°С†РёРё",
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
            logger.info("РљРµС€ support_vector_store РІ RAG-РґРІРёР¶РєРµ СЃР±СЂРѕС€РµРЅ")

        indexer.support_vector_store = None

        finished_at = datetime.now(SARATOV_TZ).isoformat()
        duration = round(time.time() - started_ts, 2)
        _set_reindex_phase(
            status="completed",
            message=f"РџРµСЂРµРёРЅРґРµРєСЃР°С†РёСЏ Р·Р°РІРµСЂС€РµРЅР°: {count} РґРѕРєСѓРјРµРЅС‚РѕРІ Р·Р° {duration}СЃ",
            processed_documents=count,
            total_documents=count,
            progress_percent=100.0,
            finished_at=finished_at,
            duration_seconds=duration,
            error=None,
        )
    except Exception as e:
        finished_at = datetime.now(SARATOV_TZ).isoformat()
        duration = round(time.time() - started_ts, 2)
        logger.error(f"РћС€РёР±РєР° РїРµСЂРµРёРЅРґРµРєСЃР°С†РёРё: {e}")
        _set_reindex_phase(
            status="failed",
            message="РџРµСЂРµРёРЅРґРµРєСЃР°С†РёСЏ Р·Р°РІРµСЂС€РёР»Р°СЃСЊ СЃ РѕС€РёР±РєРѕР№",
            finished_at=finished_at,
            duration_seconds=duration,
            error=str(e),
        )
    finally:
        _REINDEX_TASK = None


# в”Ђв”Ђв”Ђ РЈС‚РёР»РёС‚С‹ РґР»СЏ СЂР°Р±РѕС‚С‹ СЃ JSON-С„Р°Р№Р»РѕРј в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


def _load_kb() -> list[dict[str, Any]]:
    """Р—Р°РіСЂСѓР·РёС‚СЊ Р±Р°Р·Сѓ Р·РЅР°РЅРёР№ РёР· JSON."""
    if not KB_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Р¤Р°Р№Р» Р‘Р— РЅРµ РЅР°Р№РґРµРЅ: {KB_JSON_PATH}")
    try:
        with open(KB_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"РћС€РёР±РєР° РїР°СЂСЃРёРЅРіР° JSON: {e}")


def _save_kb(data: list[dict[str, Any]], backup: bool = True):
    """РЎРѕС…СЂР°РЅРёС‚СЊ Р±Р°Р·Сѓ Р·РЅР°РЅРёР№ РІ JSON СЃ РѕРїС†РёРѕРЅР°Р»СЊРЅС‹Рј Р±СЌРєР°РїРѕРј."""
    if backup:
        KB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(SARATOV_TZ).strftime("%Y%m%d_%H%M%S")
        backup_path = KB_BACKUP_DIR / f"kb_backup_{ts}.json"
        if KB_JSON_PATH.exists():
            shutil.copy2(KB_JSON_PATH, backup_path)
            logger.info(f"Р‘СЌРєР°Рї СЃРѕР·РґР°РЅ: {backup_path}")
            # РћСЃС‚Р°РІР»СЏРµРј С‚РѕР»СЊРєРѕ РїРѕСЃР»РµРґРЅРёРµ 20 Р±СЌРєР°РїРѕРІ
            backups = sorted(KB_BACKUP_DIR.glob("kb_backup_*.json"))
            for old in backups[:-20]:
                old.unlink()

    with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Р‘Р— СЃРѕС…СЂР°РЅРµРЅР°: {len(data)} Р·Р°РїРёСЃРµР№ -> {KB_JSON_PATH}")


def _update_chromadb_document(item: dict[str, Any]):
    """РРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕ РѕР±РЅРѕРІРёС‚СЊ РѕРґРёРЅ РґРѕРєСѓРјРµРЅС‚ РІ ChromaDB."""
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
            logger.info(
                "support_vector_store Р±С‹Р» СЃРѕР·РґР°РЅ РґР»СЏ РёРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕР№ СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёРё"
            )

        collection = store._collection
        doc_id = item["id"]
        metadata = item.get("metadata", {})

        # РџРѕРґРіРѕС‚РѕРІРєР° РјРµС‚Р°РґР°РЅРЅС‹С… (ChromaDB РЅРµ РїРѕРґРґРµСЂР¶РёРІР°РµС‚ СЃРїРёСЃРєРё)
        clean_meta = {
            "source": metadata.get("source", "real_support_tickets"),
            "category": metadata.get("category", "РџСЂРѕС‡РµРµ"),
            "category_en": metadata.get("category_en", "general"),
            "quality_score": metadata.get("quality_score", 0),
            "question": metadata.get("question", "")[:500],
            "doc_type": metadata.get("type", "qa_pair"),
            "article_id": f"tp_{doc_id}",
            "title": metadata.get("question", "Р—Р°СЏРІРєР° РўРџ")[:200],
        }
        if metadata.get("tags"):
            clean_meta["tags"] = ", ".join(metadata["tags"])
        if item.get("reviewed"):
            clean_meta["reviewed"] = "true"

        text = item["text"]

        # РџСЂРѕР±СѓРµРј РѕР±РЅРѕРІРёС‚СЊ, РµСЃР»Рё РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ вЂ” РґРѕР±Р°РІР»СЏРµРј
        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: РѕР±РЅРѕРІР»С‘РЅ РґРѕРєСѓРјРµРЅС‚ {doc_id}")
        else:
            collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: РґРѕР±Р°РІР»РµРЅ РґРѕРєСѓРјРµРЅС‚ {doc_id}")

    except Exception as e:
        logger.error(f"РћС€РёР±РєР° РѕР±РЅРѕРІР»РµРЅРёСЏ ChromaDB РґР»СЏ {item.get('id')}: {e}")


def _delete_chromadb_document(doc_id: str):
    """РЈРґР°Р»РёС‚СЊ РґРѕРєСѓРјРµРЅС‚ РёР· ChromaDB."""
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
        logger.info(f"ChromaDB: СѓРґР°Р»С‘РЅ РґРѕРєСѓРјРµРЅС‚ {doc_id}")
    except Exception as e:
        logger.error(f"РћС€РёР±РєР° СѓРґР°Р»РµРЅРёСЏ РёР· ChromaDB {doc_id}: {e}")


# в”Ђв”Ђв”Ђ Р­РЅРґРїРѕРёРЅС‚С‹ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ


@router.get("/stats", response_model=KBStats)
async def get_kb_stats(user: AdminUser = _any_admin):
    """РЎС‚Р°С‚РёСЃС‚РёРєР° Р±Р°Р·С‹ Р·РЅР°РЅРёР№."""
    data = _load_kb()
    by_category: dict[str, int] = {}
    by_quality: dict[str, int] = {}
    reviewed = 0
    total_quality = 0.0

    for item in data:
        meta = item.get("metadata", {})
        cat = meta.get("category", "РџСЂРѕС‡РµРµ")
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
    """РЎРїРёСЃРѕРє Q&A РїР°СЂ СЃ РїР°РіРёРЅР°С†РёРµР№ Рё С„РёР»СЊС‚СЂР°С†РёРµР№."""
    data = _load_kb()

    # Р¤РёР»СЊС‚СЂР°С†РёСЏ
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
    """РџРѕР»СѓС‡РёС‚СЊ РѕРґРЅСѓ Q&A РїР°СЂСѓ."""
    data = _load_kb()
    for item in data:
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Р­Р»РµРјРµРЅС‚ {item_id} РЅРµ РЅР°Р№РґРµРЅ")


@router.get("/quiz/next")
async def get_next_quiz_item(
    category: str | None = None,
    skip_reviewed: bool = True,
    user: AdminUser = _any_admin,
):
    """
    РџРѕР»СѓС‡РёС‚СЊ СЃР»РµРґСѓСЋС‰СѓСЋ РЅРµРѕС‚СЂРµРІСЊСЋРµРЅРЅСѓСЋ Q&A РїР°СЂСѓ РґР»СЏ РєРІРёР·Р°.
    Р’РѕР·РІСЂР°С‰Р°РµС‚ СЌР»РµРјРµРЅС‚ + РѕР±С‰РёР№ РїСЂРѕРіСЂРµСЃСЃ.
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
            "message": "Р’СЃРµ Р·Р°РїРёСЃРё РїСЂРѕРІРµСЂРµРЅС‹! рџЋ‰",
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
    """РћР±РЅРѕРІРёС‚СЊ Q&A РїР°СЂСѓ (РІРѕРїСЂРѕСЃ, РѕС‚РІРµС‚, РєР°С‚РµРіРѕСЂРёСЋ, С‚РµРіРё Рё С‚.Рґ.)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Р­Р»РµРјРµРЅС‚ {item_id} РЅРµ РЅР°Р№РґРµРЅ")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # РџСЂРёРјРµРЅСЏРµРј РѕР±РЅРѕРІР»РµРЅРёСЏ
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

    # РџРµСЂРµСЃС‚СЂР°РёРІР°РµРј С‚РµРєСЃС‚
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Р’РѕРїСЂРѕСЃ: {q}\n\nРћС‚РІРµС‚: {a}"
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # РРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕ РѕР±РЅРѕРІР»СЏРµРј ChromaDB
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
    """РћРґРѕР±СЂРёС‚СЊ Q&A РїР°СЂСѓ (РїРѕРјРµС‚РёС‚СЊ РєР°Рє РїСЂРѕРІРµСЂРµРЅРЅСѓСЋ, quality_score=5)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Р­Р»РµРјРµРЅС‚ {item_id} РЅРµ РЅР°Р№РґРµРЅ")

    item = data[found_idx]
    item["reviewed"] = True
    item["review_date"] = datetime.now(SARATOV_TZ).isoformat()
    meta = item.get("metadata", {})
    meta["quality_score"] = 5
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # РРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕ РѕР±РЅРѕРІР»СЏРµРј ChromaDB
    _update_chromadb_document(item)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="kb_item",
        entity_id=item_id,
        details="РћРґРѕР±СЂРµРЅР°",
    )
    return {"status": "ok", "item": item}


@router.post("/items/{item_id}/save-and-approve")
async def save_and_approve_kb_item(
    item_id: str, update: KBItemUpdate, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)
):
    """РћР±РЅРѕРІРёС‚СЊ Рё СЃСЂР°Р·Сѓ РѕРґРѕР±СЂРёС‚СЊ Q&A РїР°СЂСѓ."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Р­Р»РµРјРµРЅС‚ {item_id} РЅРµ РЅР°Р№РґРµРЅ")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # РџСЂРёРјРµРЅСЏРµРј РѕР±РЅРѕРІР»РµРЅРёСЏ
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

    # РџРµСЂРµСЃС‚СЂР°РёРІР°РµРј С‚РµРєСЃС‚
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Р’РѕРїСЂРѕСЃ: {q}\n\nРћС‚РІРµС‚: {a}"
    item["metadata"] = meta
    item["reviewed"] = True
    item["review_date"] = datetime.now(SARATOV_TZ).isoformat()

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
        details="РћР±РЅРѕРІР»РµРЅР° Рё РѕРґРѕР±СЂРµРЅР°",
    )
    return {"status": "ok", "item": item}


@router.delete("/items/{item_id}")
async def delete_kb_item(item_id: str, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """РЈРґР°Р»РёС‚СЊ Q&A РїР°СЂСѓ РёР· Р±Р°Р·С‹ Р·РЅР°РЅРёР№."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Р­Р»РµРјРµРЅС‚ {item_id} РЅРµ РЅР°Р№РґРµРЅ")

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
    РРјРїРѕСЂС‚ РЅРѕРІС‹С… Q&A РґР°РЅРЅС‹С… РёР· JSON-С„Р°Р№Р»Р°.
    Р”СѓР±Р»РёРєР°С‚С‹ (РїРѕ id) РїСЂРѕРїСѓСЃРєР°СЋС‚СЃСЏ, РЅРѕРІС‹Рµ Р·Р°РїРёСЃРё РґРѕР±Р°РІР»СЏСЋС‚СЃСЏ.
    """
    try:
        content = await file.read()
        new_items = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"РћС€РёР±РєР° С‡С‚РµРЅРёСЏ С„Р°Р№Р»Р°: {e}")

    if not isinstance(new_items, list):
        raise HTTPException(status_code=400, detail="РћР¶РёРґР°РµС‚СЃСЏ JSON-РјР°СЃСЃРёРІ")

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

            # РЈР±РµР¶РґР°РµРјСЃСЏ, С‡С‚Рѕ РµСЃС‚СЊ РІСЃРµ РЅСѓР¶РЅС‹Рµ РїРѕР»СЏ
            if "text" not in new_item:
                meta = new_item.get("metadata", {})
                q = meta.get("question", "")
                a = meta.get("answer", "")
                new_item["text"] = f"Р’РѕРїСЂРѕСЃ: {q}\n\nРћС‚РІРµС‚: {a}"

            if "metadata" not in new_item:
                new_item["metadata"] = {
                    "source": "real_support_tickets",
                    "category": "РџСЂРѕС‡РµРµ",
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

            # Р”РѕР±Р°РІР»СЏРµРј РІ ChromaDB
            _update_chromadb_document(new_item)

        except Exception as e:
            logger.error(f"РћС€РёР±РєР° РёРјРїРѕСЂС‚Р° СЌР»РµРјРµРЅС‚Р°: {e}")
            errors += 1

    _save_kb(data)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="import",
        entity_type="kb_item",
        details=f"РРјРїРѕСЂС‚: +{added} РЅРѕРІС‹С…, {duplicates} РґСѓР±Р»РёРєР°С‚РѕРІ, {errors} РѕС€РёР±РѕРє",
    )
    return KBImportResult(
        added=added,
        duplicates_skipped=duplicates,
        errors=errors,
        message=f"РРјРїРѕСЂС‚ Р·Р°РІРµСЂС€С‘РЅ: +{added} РЅРѕРІС‹С…, {duplicates} РґСѓР±Р»РёРєР°С‚РѕРІ РїСЂРѕРїСѓС‰РµРЅРѕ, {errors} РѕС€РёР±РѕРє",
    )


@router.post("/reindex", response_model=KBReindexStatus)
async def reindex_kb(user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """
    Р—Р°РїСѓСЃС‚РёС‚СЊ РїРѕР»РЅСѓСЋ РїРµСЂРµРёРЅРґРµРєСЃР°С†РёСЋ support_tickets РІ С„РѕРЅРµ.
    РСЃРїРѕР»СЊР·СѓР№С‚Рµ РїРѕСЃР»Рµ РјР°СЃСЃРѕРІС‹С… РїСЂР°РІРѕРє.
    """
    global _REINDEX_TASK

    if _REINDEX_TASK is not None and not _REINDEX_TASK.done():
        return KBReindexStatus(**_get_reindex_status())

    job_id = str(uuid4())
    now = datetime.now(SARATOV_TZ).isoformat()
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
        message="Р—Р°РґР°С‡Р° РїРµСЂРµРёРЅРґРµРєСЃР°С†РёРё РїРѕСЃС‚Р°РІР»РµРЅР° РІ РѕС‡РµСЂРµРґСЊ",
        error=None,
    )
    _REINDEX_TASK = asyncio.create_task(_run_reindex_job(job_id))
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="reindex",
        entity_type="kb_item",
        details=f"РџРµСЂРµРёРЅРґРµРєСЃР°С†РёСЏ: {total_documents} РґРѕРєСѓРјРµРЅС‚РѕРІ",
    )
    return KBReindexStatus(**_get_reindex_status())


@router.get("/reindex/status", response_model=KBReindexStatus)
async def get_reindex_status(user: AdminUser = _any_admin):
    """РџРѕР»СѓС‡РёС‚СЊ С‚РµРєСѓС‰РёР№ СЃС‚Р°С‚СѓСЃ С„РѕРЅРѕРІРѕР№ РїРµСЂРµРёРЅРґРµРєСЃР°С†РёРё."""
    return KBReindexStatus(**_get_reindex_status())


@router.get("/categories")
async def get_categories(user: AdminUser = _any_admin):
    """РџРѕР»СѓС‡РёС‚СЊ СЃРїРёСЃРѕРє РІСЃРµС… РєР°С‚РµРіРѕСЂРёР№."""
    data = _load_kb()
    categories = {}
    for item in data:
        cat = item.get("metadata", {}).get("category", "РџСЂРѕС‡РµРµ")
        cat_en = item.get("metadata", {}).get("category_en", "general")
        categories[cat] = cat_en
    return {"categories": categories}
