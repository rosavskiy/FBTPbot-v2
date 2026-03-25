"""
L2-классификатор: определение тематического раздела внутри причины обращения.

Алгоритм:
1. Сопоставление вопроса с Q&A парами каждого раздела (TF-IDF-like)
2. Проверка типовых жалоб
3. Возврат лучшего раздела + ближайшего Q&A (если найден)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pymorphy3

from app.models.reason_schemas import (
    Complaint,
    ContactReason,
    ExampleQA,
    QAPair,
    ThematicSection,
)

logger = logging.getLogger(__name__)

_morph: pymorphy3.MorphAnalyzer | None = None


def _get_morph() -> pymorphy3.MorphAnalyzer:
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


@dataclass
class L2Result:
    """Результат L2-классификации."""

    section: ThematicSection | None = None
    best_qa: QAPair | None = None
    best_qa_score: float = 0.0
    best_complaint: Complaint | None = None
    best_complaint_score: float = 0.0
    best_example: ExampleQA | None = None
    best_example_score: float = 0.0
    method: str = ""  # qa_match, complaint_match, section_match, none


def _text_to_lemma_set(text: str) -> set[str]:
    """Лемматизировать текст → множество лемм."""
    morph = _get_morph()
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())
    lemmas = set()
    for w in words:
        parsed = morph.parse(w)
        if parsed:
            lemmas.add(parsed[0].normal_form)
        else:
            lemmas.add(w)
    return lemmas


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Коэффициент Жаккара между двумя множествами."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _overlap_score(query_lemmas: set[str], text_lemmas: set[str]) -> float:
    """Доля лемм запроса, найденных в тексте."""
    if not query_lemmas:
        return 0.0
    return len(query_lemmas & text_lemmas) / len(query_lemmas)


def classify_section(query: str, reason: ContactReason) -> L2Result:
    """Определить тематический раздел и ближайший Q&A для запроса.

    Args:
        query: Вопрос пользователя.
        reason: Определённая на L1 причина обращения.

    Returns:
        L2Result с лучшим разделом и ближайшим Q&A / жалобой / примером.
    """
    query_lemmas = _text_to_lemma_set(query)

    logger.info(f"[L2] reason={reason.name} | query_lemmas={query_lemmas}")

    # ── 1. Поиск среди примеров ответов (ExampleQA) — самый высокий приоритет ──
    best_example: ExampleQA | None = None
    best_example_score = 0.0

    for ex in reason.example_answers:
        ex_lemmas = _text_to_lemma_set(ex.user_question)
        score = _overlap_score(query_lemmas, ex_lemmas)
        if score > best_example_score:
            best_example_score = score
            best_example = ex

    # Если очень хорошее совпадение с примером — сразу ответ без LLM
    if best_example and best_example_score >= 0.7:
        logger.info(f"[L2] EXACT_EXAMPLE|score={best_example_score:.2f}|q={best_example.user_question[:60]}")
        return L2Result(
            best_example=best_example,
            best_example_score=best_example_score,
            method="example_match",
        )

    # ── 2. Поиск среди типовых жалоб ──
    best_complaint: Complaint | None = None
    best_complaint_score = 0.0

    for complaint in reason.typical_complaints:
        comp_lemmas = _text_to_lemma_set(complaint.description + " " + complaint.context)
        score = _overlap_score(query_lemmas, comp_lemmas)
        if score > best_complaint_score:
            best_complaint_score = score
            best_complaint = complaint

    if best_complaint and best_complaint_score >= 0.6:
        logger.info(f"[L2] COMPLAINT_MATCH|score={best_complaint_score:.2f}|desc={best_complaint.description[:60]}")
        return L2Result(
            best_complaint=best_complaint,
            best_complaint_score=best_complaint_score,
            method="complaint_match",
        )

    # ── 3. Поиск по Q&A парам внутри тематических разделов ──
    best_section: ThematicSection | None = None
    best_qa: QAPair | None = None
    best_qa_score = 0.0
    best_section_score = 0.0

    for section in reason.thematic_sections:
        section_total_score = 0.0
        section_best_qa: QAPair | None = None
        section_best_qa_score = 0.0

        for qa in section.qa_pairs:
            qa_lemmas = _text_to_lemma_set(qa.question)
            score = _overlap_score(query_lemmas, qa_lemmas)
            section_total_score += score

            if score > section_best_qa_score:
                section_best_qa_score = score
                section_best_qa = qa

        # Средний score по разделу + лучший Q&A score
        avg_score = section_total_score / max(len(section.qa_pairs), 1)
        combined = avg_score * 0.3 + section_best_qa_score * 0.7

        if combined > best_section_score:
            best_section_score = combined
            best_section = section
            best_qa = section_best_qa
            best_qa_score = section_best_qa_score

    if best_section:
        logger.info(
            f"[L2] SECTION_MATCH|section={best_section.title}|score={best_section_score:.2f}"
            f"|best_qa_score={best_qa_score:.2f}"
        )
        return L2Result(
            section=best_section,
            best_qa=best_qa,
            best_qa_score=best_qa_score,
            best_example=best_example,
            best_example_score=best_example_score,
            best_complaint=best_complaint,
            best_complaint_score=best_complaint_score,
            method="section_match",
        )

    logger.info("[L2] RESULT: no section match")
    return L2Result(
        best_example=best_example,
        best_example_score=best_example_score,
        best_complaint=best_complaint,
        best_complaint_score=best_complaint_score,
        method="none",
    )
