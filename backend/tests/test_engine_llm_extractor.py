from __future__ import annotations

import pytest

from app.classifier.reason_classifier import ClassificationCandidate, L1Result
from app.classifier.section_classifier import L2Result
from app.models.reason_schemas import ContactReason
from app.rag.engine import QuestionExtractionResult, RAGEngine


def _reason() -> ContactReason:
    return ContactReason(id="invoice_issue", name="Изменение реквизитов в приходной накладной")


def test_should_try_llm_extractor_for_noisy_below_threshold_query():
    engine = RAGEngine()
    l1 = L1Result(method="below_threshold", is_confident=False)

    should_use = engine._should_try_llm_question_extractor(
        "День добрый, пришла накладная 58329771в, заказ был на 3 шт, а пришла 1 шт, можно изменить в проведенной накладной?",
        l1,
    )

    assert should_use is True


def test_should_not_try_llm_extractor_for_short_clean_query():
    engine = RAGEngine()
    l1 = L1Result(method="below_threshold", is_confident=False)

    should_use = engine._should_try_llm_question_extractor("Не проводится накладная", l1)

    assert should_use is False


@pytest.mark.anyio
async def test_maybe_normalize_question_for_l1_uses_improved_result(monkeypatch):
    engine = RAGEngine()
    current_l1 = L1Result(method="none", is_confident=False)
    reason = _reason()
    improved_candidate = ClassificationCandidate(reason=reason, score=7.0)
    improved_l1 = L1Result(
        reason=reason,
        candidates=[improved_candidate],
        is_confident=True,
        method="marker_score",
        winning_candidate=improved_candidate,
    )

    async def fake_extract(question: str) -> QuestionExtractionResult | None:
        assert "58329771в" in question
        return QuestionExtractionResult(normalized_question="изменить количество в проведенной накладной 58329771в")

    monkeypatch.setattr(engine, "_llm_extract_question_signal", fake_extract)
    monkeypatch.setattr(
        "app.rag.engine.classify_reason",
        lambda question, **_kw: improved_l1
        if question == "изменить количество в проведенной накладной 58329771в"
        else current_l1,
    )

    normalized_question, normalized_l1, used = await engine._maybe_normalize_question_for_l1(
        "День добрый, пришла накладная 58329771в, заказ был на 3 шт, а пришла 1 шт, можно изменить в проведенной накладной?",
        current_l1,
    )

    assert used is True
    assert normalized_question == "изменить количество в проведенной накладной 58329771в"
    assert normalized_l1.reason is not None
    assert normalized_l1.reason.id == reason.id


@pytest.mark.anyio
async def test_maybe_normalize_question_for_l1_skips_confident_query(monkeypatch):
    engine = RAGEngine()
    reason = _reason()
    candidate = ClassificationCandidate(reason=reason, score=8.0)
    confident_l1 = L1Result(
        reason=reason,
        candidates=[candidate],
        is_confident=True,
        method="marker_score",
        winning_candidate=candidate,
    )

    async def fail_extract(_question: str) -> QuestionExtractionResult | None:
        raise AssertionError("extractor should not be called")

    monkeypatch.setattr(engine, "_llm_extract_question_signal", fail_extract)

    normalized_question, normalized_l1, used = await engine._maybe_normalize_question_for_l1(
        "Не проводится накладная",
        confident_l1,
    )

    assert used is False
    assert normalized_question == "Не проводится накладная"
    assert normalized_l1 is confident_l1


@pytest.mark.anyio
async def test_ask_debug_trace_contains_llm_extractor_metadata(monkeypatch):
    engine = RAGEngine()
    question = (
        "День добрый, пришла накладная 58329771в, заказ был на 3 шт, а пришла 1 шт, "
        "можно изменить в проведенной накладной?"
    )
    normalized_question = "изменить количество в проведенной накладной 58329771в"
    reason = _reason()
    improved_candidate = ClassificationCandidate(reason=reason, score=7.0)
    current_l1 = L1Result(method="none", is_confident=False)
    improved_l1 = L1Result(
        reason=reason,
        candidates=[improved_candidate],
        is_confident=True,
        method="marker_score",
        winning_candidate=improved_candidate,
    )

    async def fake_extract(raw_question: str) -> QuestionExtractionResult | None:
        assert raw_question == question
        return QuestionExtractionResult(normalized_question=normalized_question)

    monkeypatch.setattr(engine, "_check_global_escalation", lambda _question: {"matched": False})
    monkeypatch.setattr(engine, "_check_forced_escalation", lambda _question, _reason: {"matched": False})
    monkeypatch.setattr(engine, "_llm_extract_question_signal", fake_extract)
    monkeypatch.setattr(
        "app.rag.engine.classify_reason",
        lambda current_question, **_kw: improved_l1 if current_question == normalized_question else current_l1,
    )
    monkeypatch.setattr("app.rag.engine.classify_section", lambda _question, _reason, **_kw: L2Result(method="none"))

    response = await engine.ask(question, debug=True)

    assert response.debug_trace is not None
    assert response.debug_trace["llm_extractor_used"] is True
    assert response.debug_trace["llm_extractor_original_question"] == question
    assert response.debug_trace["llm_extractor_normalized_question"] == normalized_question
