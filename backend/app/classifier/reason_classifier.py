"""
L1-классификатор: определение причины обращения по маркерам.

Гибридный алгоритм:
1. Фразовые маски (100%-маркеры) — высший приоритет
2. Числовые теги (11, 52, 552...) — с учётом контекста
3. Существительные — морфоанализ + сопоставление лемм
4. Глаголы — морфоанализ + сопоставление лемм
5. Если неоднозначно — LLM выбирает из топ-кандидатов
6. Если ничего — None (fallback / эскалация)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pymorphy3

from app.database.reason_store import get_all_reasons
from app.llm_settings import get_classification_settings
from app.models.reason_schemas import ContactReason

logger = logging.getLogger(__name__)

# Singleton морфоанализатора
_morph: pymorphy3.MorphAnalyzer | None = None


def _get_morph() -> pymorphy3.MorphAnalyzer:
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


# ── Дефолтные веса маркеров (переопределяются через UI → llm_settings.json) ──
DEFAULT_WEIGHT_PHRASE_MASK = 10.0
DEFAULT_WEIGHT_NUMERIC_TAG = 5.0
DEFAULT_WEIGHT_NOUN = 2.0
DEFAULT_WEIGHT_VERB = 1.0
DEFAULT_GLOBAL_MIN_SCORE = 5.0

# Минимальный gap между 1-м и 2-м кандидатом для уверенного выбора
MIN_SCORE_GAP = 3.0

# Минимальный score для хоть какого-то совпадения
MIN_SCORE_THRESHOLD = 1.0


def _get_weights() -> dict[str, float]:
    """Получить текущие веса маркеров и глобальный порог из настроек."""
    cs = get_classification_settings()
    return {
        "phrase_mask": cs.get("l1_weight_phrase_mask", DEFAULT_WEIGHT_PHRASE_MASK),
        "numeric_tag": cs.get("l1_weight_numeric_tag", DEFAULT_WEIGHT_NUMERIC_TAG),
        "noun": cs.get("l1_weight_noun", DEFAULT_WEIGHT_NOUN),
        "verb": cs.get("l1_weight_verb", DEFAULT_WEIGHT_VERB),
        "global_min_score": cs.get("l1_global_min_score", DEFAULT_GLOBAL_MIN_SCORE),
    }


@dataclass
class ClassificationCandidate:
    """Кандидат на причину обращения с деталями скоринга."""

    reason: ContactReason
    score: float = 0.0
    phrase_matches: list[str] = field(default_factory=list)
    numeric_matches: list[str] = field(default_factory=list)
    noun_matches: list[str] = field(default_factory=list)
    verb_matches: list[str] = field(default_factory=list)


@dataclass
class L1Result:
    """Результат L1-классификации."""

    reason: ContactReason | None = None
    candidates: list[ClassificationCandidate] = field(default_factory=list)
    is_confident: bool = False
    needs_clarification: bool = False
    method: str = ""  # phrase_mask, numeric_tag, marker_score, llm, below_threshold, none
    winning_candidate: ClassificationCandidate | None = None
    marker_weights: dict = field(default_factory=dict)  # текущие веса (для debug trace)


def _normalize_text(text: str) -> str:
    """Приведение текста к нижнему регистру, удаление лишних пробелов."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_lemmas(text: str) -> tuple[set[str], set[str]]:
    """Извлечь леммы существительных и глаголов из текста.

    Returns:
        (noun_lemmas, verb_lemmas)
    """
    morph = _get_morph()
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text)

    nouns = set()
    verbs = set()

    for word in words:
        parsed = morph.parse(word.lower())
        if not parsed:
            continue
        best = parsed[0]
        pos = best.tag.POS

        if pos in ("NOUN", "ADJF", "ADJS"):
            nouns.add(best.normal_form)
        elif pos in ("VERB", "INFN", "PRTF", "PRTS", "GRND"):
            verbs.add(best.normal_form)

    return nouns, verbs


def _check_phrase_masks(text_normalized: str, reason: ContactReason) -> list[str]:
    """Проверить совпадение фразовых масок (100%-маркеры) по границам слов."""
    matches = []
    for mask in reason.markers.phrase_masks:
        mask_lower = mask.lower().strip()
        if not mask_lower:
            continue
        pattern = r"(?<!\w)" + re.escape(mask_lower) + r"(?!\w)"
        if re.search(pattern, text_normalized):
            matches.append(mask)
    return matches


def _check_numeric_tags(text: str, reason: ContactReason) -> list[str]:
    """Проверить числовые теги с учётом границ слов."""
    matches = []
    for tag in reason.markers.numeric_tags:
        # Ищем число как отдельное слово (не часть другого числа)
        pattern = r"(?<!\d)" + re.escape(tag) + r"(?!\d)"
        if re.search(pattern, text):
            matches.append(tag)
    return matches


def _check_nouns(user_nouns: set[str], reason: ContactReason) -> list[str]:
    """Сопоставить леммы существительных пользователя с маркерами."""
    morph = _get_morph()
    matches = []

    for marker_noun in reason.markers.nouns:
        # Лемматизируем каждое слово маркера (маркер может быть фразой)
        marker_words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", marker_noun.lower())
        marker_lemmas = set()
        for w in marker_words:
            parsed = morph.parse(w)
            if parsed:
                marker_lemmas.add(parsed[0].normal_form)

        # Если хотя бы одна лемма маркера совпадает с леммами пользователя
        if marker_lemmas & user_nouns:
            matches.append(marker_noun)

    return matches


def _check_verbs(user_verbs: set[str], reason: ContactReason) -> list[str]:
    """Сопоставить леммы глаголов пользователя с маркерами."""
    morph = _get_morph()
    matches = []

    for marker_verb in reason.markers.verbs:
        marker_words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", marker_verb.lower())
        marker_lemmas = set()
        for w in marker_words:
            parsed = morph.parse(w)
            if parsed:
                marker_lemmas.add(parsed[0].normal_form)

        if marker_lemmas & user_verbs:
            matches.append(marker_verb)

    return matches


def classify_reason(query: str) -> L1Result:
    """Основной метод L1-классификации.

    Определяет причину обращения по маркерам (без LLM).
    Если неоднозначно — возвращает is_confident=False + candidates
    для последующей LLM-классификации или уточнения.
    """
    reasons = get_all_reasons(active_only=True)

    if not reasons:
        logger.warning("Нет активных причин обращения для классификации")
        return L1Result(method="none")

    text_normalized = _normalize_text(query)
    user_nouns, user_verbs = _extract_lemmas(query)
    weights = _get_weights()

    logger.info(f"[L1] query='{query}' | nouns={user_nouns} | verbs={user_verbs} | weights={weights}")

    candidates: list[ClassificationCandidate] = []

    for reason in reasons:
        candidate = ClassificationCandidate(reason=reason)

        # 1. Фразовые маски (высший приоритет)
        candidate.phrase_matches = _check_phrase_masks(text_normalized, reason)
        if candidate.phrase_matches:
            candidate.score += weights["phrase_mask"] * len(candidate.phrase_matches)

        # 2. Числовые теги
        candidate.numeric_matches = _check_numeric_tags(query, reason)
        if candidate.numeric_matches:
            candidate.score += weights["numeric_tag"] * len(candidate.numeric_matches)

        # 3. Существительные
        candidate.noun_matches = _check_nouns(user_nouns, reason)
        if candidate.noun_matches:
            candidate.score += weights["noun"] * len(candidate.noun_matches)

        # 4. Глаголы
        candidate.verb_matches = _check_verbs(user_verbs, reason)
        if candidate.verb_matches:
            candidate.score += weights["verb"] * len(candidate.verb_matches)

        if candidate.score > 0:
            candidates.append(candidate)

    # Сортируем по score
    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        logger.info("[L1] RESULT: no matches found")
        return L1Result(method="none")

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    gap = top.score - second_score

    logger.info(
        f"[L1] top={top.reason.name} score={top.score:.1f} | "
        f"second_score={second_score:.1f} | gap={gap:.1f} | "
        f"phrases={top.phrase_matches} nums={top.numeric_matches} "
        f"nouns={top.noun_matches} verbs={top.verb_matches}"
    )

    global_min = weights["global_min_score"]

    # ── Глобальный порог: score ниже минимума → below_threshold ──
    if top.score < global_min:
        logger.info(f"[L1] RESULT: below_threshold | top_score={top.score:.1f} < global_min={global_min:.1f}")
        return L1Result(
            candidates=candidates,
            winning_candidate=top,
            marker_weights=weights,
            method="below_threshold",
        )

    # Фразовая маска — безусловная уверенность
    if top.phrase_matches:
        return L1Result(
            reason=top.reason,
            candidates=candidates,
            is_confident=True,
            winning_candidate=top,
            marker_weights=weights,
            method="phrase_mask",
        )

    # Достаточный gap между 1-м и 2-м
    if top.score >= MIN_SCORE_THRESHOLD and gap >= MIN_SCORE_GAP:
        return L1Result(
            reason=top.reason,
            candidates=candidates,
            is_confident=True,
            winning_candidate=top,
            marker_weights=weights,
            method="marker_score",
        )

    # Единственный кандидат с достаточным score
    if len(candidates) == 1 and top.score >= MIN_SCORE_THRESHOLD:
        return L1Result(
            reason=top.reason,
            candidates=candidates,
            is_confident=True,
            winning_candidate=top,
            marker_weights=weights,
            method="marker_score",
        )

    # Неоднозначно — нужна LLM-помощь или уточнение
    if top.score >= MIN_SCORE_THRESHOLD:
        return L1Result(
            candidates=candidates,
            is_confident=False,
            needs_clarification=True,
            winning_candidate=top,
            marker_weights=weights,
            method="ambiguous",
        )

    # Слишком низкий score — ничего не нашли
    return L1Result(marker_weights=weights, method="none")
