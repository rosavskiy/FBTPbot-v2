from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import chat
from app.database.models import Base, PendingClarification
from app.rag.engine import RAGResponse


class DummySheetLogger:
    async def log(self, **kwargs):
        return None


class MarkerClarificationRAGStub:
    def __init__(self):
        self.calls: list[dict] = []

    async def ask(self, question: str, chat_history=None, reason_id: str | None = None, debug: bool = False):
        self.calls.append(
            {
                "question": question,
                "chat_history": chat_history or [],
                "reason_id": reason_id,
                "debug": debug,
            }
        )

        if len(self.calls) == 1:
            return RAGResponse(
                answer="Укажите, пожалуйста, номер накладной.",
                confidence=0.3,
                needs_escalation=False,
                detected_reason="invoice_issue",
                detected_reason_name="Проблема с накладной",
                classification_method="marker_clarification",
            )

        return RAGResponse(
            answer="Проверьте статус накладной 12345 в журнале обмена.",
            confidence=0.92,
            needs_escalation=False,
            detected_reason="invoice_issue",
            detected_reason_name="Проблема с накладной",
            thematic_section="Накладные",
            classification_method="L1:forced/L2:section_match",
        )


class ReasonSelectionRAGStub:
    def __init__(self):
        self.calls: list[dict] = []

    async def ask(self, question: str, chat_history=None, reason_id: str | None = None, debug: bool = False):
        self.calls.append(
            {
                "question": question,
                "chat_history": chat_history or [],
                "reason_id": reason_id,
                "debug": debug,
            }
        )

        if len(self.calls) == 1:
            return RAGResponse(
                answer="Уточните, пожалуйста, тему обращения.",
                confidence=0.3,
                needs_escalation=False,
                classification_method="clarification",
                clarification_candidates=[
                    {"reason_id": "inventory", "reason_name": "Инвентаризация", "score": 4.0},
                    {"reason_id": "invoice_issue", "reason_name": "Проблема с накладной", "score": 3.5},
                ],
            )

        return RAGResponse(
            answer="Откройте журнал накладных и проверьте последнюю синхронизацию.",
            confidence=0.88,
            needs_escalation=False,
            detected_reason="invoice_issue",
            detected_reason_name="Проблема с накладной",
            thematic_section="Накладные",
            classification_method="L1:forced/L2:section_match",
        )


async def create_test_context(tmp_path: Path, rag_stub):
    db_path = tmp_path / "chat_clarification_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app = FastAPI()
    app.include_router(chat.router)
    app.dependency_overrides[chat.get_db] = override_get_db

    chat.get_rag_engine = lambda: rag_stub
    chat.get_gsheet_logger = lambda: DummySheetLogger()
    chat.get_active_llm_display = lambda: {
        "provider": "yandex",
        "model": "test-model",
        "label": "Yandex / test-model",
        "show_in_chat": False,
    }

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    return client, engine, session_factory, rag_stub


@pytest.mark.anyio
async def test_marker_clarification_continues_same_session(tmp_path: Path):
    client, engine, session_factory, rag_stub = await create_test_context(tmp_path, MarkerClarificationRAGStub())
    try:
        first_response = await client.post("/api/chat", json={"message": "Не проводится накладная"})
        assert first_response.status_code == 200

        first_data = first_response.json()
        assert first_data["response_type"] == "clarification"
        session_id = first_data["session_id"]

        async with session_factory() as session:
            pending = await session.scalar(
                select(PendingClarification).where(PendingClarification.session_id == session_id)
            )
            assert pending is not None
            assert pending.clarification_type == "reason_details"
            assert pending.fixed_reason_id == "invoice_issue"

        second_response = await client.post(
            "/api/chat",
            json={"message": "12345", "session_id": session_id},
        )
        assert second_response.status_code == 200

        second_data = second_response.json()
        assert second_data["response_type"] == "answer"
        assert second_data["detected_reason"] == "Проблема с накладной"

        assert len(rag_stub.calls) == 2
        assert rag_stub.calls[1]["reason_id"] == "invoice_issue"
        assert "Не проводится накладная" in rag_stub.calls[1]["question"]
        assert "12345" in rag_stub.calls[1]["question"]

        await asyncio.sleep(0)

        async with session_factory() as session:
            pending = await session.scalar(
                select(PendingClarification).where(PendingClarification.session_id == session_id)
            )
            assert pending is None
    finally:
        await client.aclose()
        await engine.dispose()


@pytest.mark.anyio
async def test_reason_selection_uses_original_query_on_numeric_choice(tmp_path: Path):
    client, engine, session_factory, rag_stub = await create_test_context(tmp_path, ReasonSelectionRAGStub())
    try:
        first_response = await client.post("/api/chat", json={"message": "Проблема с документом"})
        assert first_response.status_code == 200

        first_data = first_response.json()
        assert first_data["response_type"] == "clarification"
        assert len(first_data["suggested_topics"]) == 2
        session_id = first_data["session_id"]

        second_response = await client.post(
            "/api/chat",
            json={"message": "2", "session_id": session_id},
        )
        assert second_response.status_code == 200

        second_data = second_response.json()
        assert second_data["response_type"] == "answer"
        assert second_data["detected_reason"] == "Проблема с накладной"

        assert len(rag_stub.calls) == 2
        assert rag_stub.calls[1]["reason_id"] == "invoice_issue"
        assert rag_stub.calls[1]["question"] == "Проблема с документом"

        await asyncio.sleep(0)

        async with session_factory() as session:
            pending = await session.scalar(
                select(PendingClarification).where(PendingClarification.session_id == session_id)
            )
            assert pending is None
    finally:
        await client.aclose()
        await engine.dispose()
