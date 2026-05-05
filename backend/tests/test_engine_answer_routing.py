from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.reason_schemas import ContactReason
from app.models.schemas import ChatRoutingPolicy
from app.rag.engine import RAGEngine


def _reason() -> ContactReason:
    return ContactReason(id="invoice_issue", name="Проблема с накладной")


def test_answer_routing_returns_clarification_inside_window():
    engine = RAGEngine()
    policy = ChatRoutingPolicy(enabled=True)

    decision = engine._resolve_answer_routing(
        question="Не проводится накладная",
        reason=_reason(),
        section_title="Накладные",
        confidence=0.72,
        routing_policy=policy,
        refinement_attempt=0,
    )

    assert decision is not None
    assert decision.decision == "clarification"
    assert "номер документа" in decision.prompt.lower()


def test_answer_routing_returns_answer_above_threshold():
    engine = RAGEngine()
    policy = ChatRoutingPolicy(enabled=True)

    decision = engine._resolve_answer_routing(
        question="Не проводится накладная",
        reason=_reason(),
        section_title="Накладные",
        confidence=0.91,
        routing_policy=policy,
        refinement_attempt=0,
    )

    assert decision is not None
    assert decision.decision == "answer"


def test_answer_routing_returns_escalation_after_refinement_attempt():
    engine = RAGEngine()
    policy = ChatRoutingPolicy(enabled=True)

    decision = engine._resolve_answer_routing(
        question="Не проводится накладная",
        reason=_reason(),
        section_title="Накладные",
        confidence=0.72,
        routing_policy=policy,
        refinement_attempt=1,
    )

    assert decision is not None
    assert decision.decision == "escalation"


def test_chat_routing_policy_rejects_overlapping_thresholds():
    with pytest.raises(ValidationError):
        ChatRoutingPolicy(
            enabled=True,
            answer_threshold=0.89,
            clarification_min_confidence=0.55,
            clarification_max_confidence=0.89,
            max_refinement_attempts=1,
        )
