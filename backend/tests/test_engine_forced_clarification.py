from __future__ import annotations

import pytest

from app.models.reason_schemas import ClassificationRules, ContactReason, Markers
from app.rag.engine import RAGEngine


@pytest.mark.anyio
async def test_forced_reason_marker_clarification_with_debug_does_not_crash(monkeypatch):
    reason = ContactReason(
        id="forced_reason",
        name="Forced Reason",
        markers=Markers(numeric_tags=["53"]),
        classification_rules=ClassificationRules(
            enabled=True,
            required_markers=["numeric_tag"],
            clarification_text="Уточните номер ошибки",
        ),
    )

    monkeypatch.setattr(
        "app.database.reason_store.get_reason", lambda reason_id: reason if reason_id == reason.id else None
    )

    engine = RAGEngine()
    response = await engine.ask(
        question="placeholder",
        reason_id=reason.id,
        debug=True,
    )

    assert response.classification_method == "marker_clarification"
    assert response.answer == "Уточните номер ошибки"
    assert response.debug_trace is not None
    assert response.debug_trace["l1_method"] == "forced"
