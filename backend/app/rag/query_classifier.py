"""
Классификатор полноты запросов пользователя.

Определяет, достаточно ли информации для точного ответа,
и формирует уточняющие вопросы при необходимости.
Интегрируется с RAG-движком для анализа результатов поиска в ChromaDB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Пороги ────────────────────────────────────────────────
AMBIGUITY_SCORE_GAP = 0.08      # Макс. разница score между топ-результатами для "неоднозначности"
MIN_QUERY_WORDS = 4             # Минимум слов для "конкретного" запроса
MAX_TOPICS_TO_SUGGEST = 5       # Максимум тем для предложения пользователю
TOP_RESULTS_WINDOW = 8          # Сколько топ-результатов анализировать

# Размытые паттерны — слова, которые сами по себе не несут конкретики
VAGUE_PATTERNS = [
    "проблема", "не работает", "ошибка", "помогите", "помощь",
    "не могу", "сломалось", "баг", "вопрос по", "вопрос",
    "как быть", "что делать", "не получается", "не открывается",
    "не сохраняется", "не отображается", "не печатается",
    "не загружается", "зависает", "глючит", "беда",
]

# Общие объекты — слова, описывающие типовые сущности без детализации
BROAD_OBJECTS = [
    "накладная", "отчёт", "отчет", "документ", "справочник",
    "печать", "товар", "цена", "остаток", "приход", "расход",
    "рецепт", "лицензия", "обновление", "база", "касса",
    "чек", "скидка", "карта", "поставщик", "контрагент",
]


@dataclass
class SuggestedTopic:
    """Тема, предлагаемая пользователю для уточнения."""
    title: str
    article_id: str
    score: float
    snippet: str = ""


@dataclass
class ClassificationResult:
    """Результат классификации запроса."""
    is_complete: bool                              # Достаточно ли информации для ответа
    suggested_topics: List[SuggestedTopic] = field(default_factory=list)
    clarification_message: Optional[str] = None    # Сообщение для уточнения


def classify_query(
    query: str,
    scored_results: List[Tuple[Document, float]],
) -> ClassificationResult:
    """
    Классифицирует запрос пользователя по полноте.

    Args:
        query: текст запроса пользователя
        scored_results: результаты similarity_search_with_relevance_scores
                        — список (Document, score)

    Returns:
        ClassificationResult с решением и (опционально) предложенными темами
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)

    # 1. Определяем "размытость" запроса
    has_vague = any(pattern in query_lower for pattern in VAGUE_PATTERNS)
    has_broad_object = any(obj in query_lower for obj in BROAD_OBJECTS)
    is_short = word_count < MIN_QUERY_WORDS
    is_vague_query = has_vague and (has_broad_object or is_short)

    logger.info(
        f"[CLARIFY] query='{query}' | words={word_count} | "
        f"vague={has_vague} | broad_obj={has_broad_object} | short={is_short} | "
        f"is_vague={is_vague_query}"
    )

    # 2. Если нет результатов — отвечать нечем
    if not scored_results:
        return ClassificationResult(is_complete=True)  # пусть RAG сам скажет "не найдено"

    # 3. Извлекаем уникальные темы
    topics = _extract_unique_topics(scored_results)
    logger.info(f"[CLARIFY] unique_topics={len(topics)}")

    # 4. Если только 1 тема — однозначно
    if len(topics) <= 1:
        return ClassificationResult(is_complete=True)

    # 5. Если запрос НЕ размытый — отвечаем напрямую
    if not is_vague_query:
        return ClassificationResult(is_complete=True)

    # 6. Размытый запрос — проверяем, есть ли явный лидер
    if len(scored_results) >= 2:
        top_score = scored_results[0][1]
        second_score = scored_results[1][1]
        gap = top_score - second_score

        # Если первый результат сильно лучше остальных — отвечаем им
        if gap > AMBIGUITY_SCORE_GAP:
            logger.info(f"[CLARIFY] clear_leader: gap={gap:.3f} > {AMBIGUITY_SCORE_GAP}")
            return ClassificationResult(is_complete=True)

    # 7. Размытый запрос + несколько близких тем → уточняем
    topics_limited = topics[:MAX_TOPICS_TO_SUGGEST]
    message = _build_clarification_message(topics_limited)
    logger.info(f"[CLARIFY] NEED_CLARIFICATION: {len(topics_limited)} topics proposed")

    return ClassificationResult(
        is_complete=False,
        suggested_topics=topics_limited,
        clarification_message=message,
    )


def _extract_unique_topics(
    scored_results: List[Tuple[Document, float]],
) -> List[SuggestedTopic]:
    """Извлекает уникальные темы из результатов поиска."""
    seen_article_ids = set()
    topics: List[SuggestedTopic] = []

    for doc, score in scored_results[:TOP_RESULTS_WINDOW]:
        meta = doc.metadata
        article_id = str(meta.get("article_id", "unknown"))
        title = meta.get("title", "").strip()

        if not title:
            title = f"Статья {article_id}"

        # Дедупликация по article_id
        if article_id in seen_article_ids:
            continue
        seen_article_ids.add(article_id)

        # Берём начало текста как snippet
        snippet = doc.page_content[:120].replace("\n", " ").strip()

        topics.append(SuggestedTopic(
            title=title,
            article_id=article_id,
            score=score,
            snippet=snippet,
        ))

        if len(topics) >= MAX_TOPICS_TO_SUGGEST:
            break

    return topics


def _build_clarification_message(topics: List[SuggestedTopic]) -> str:
    """Формирует текстовое сообщение с вариантами тем."""
    lines = ["Уточните, какая тема вас интересует:"]

    for i, topic in enumerate(topics, 1):
        lines.append(f"{i}. {topic.title}")

    lines.append("\nВыберите номер или опишите проблему подробнее.")

    return "\n".join(lines)
