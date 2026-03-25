"""
RAG-движок v2 для ИИ-техподдержки Фармбазис.

Трёхуровневая система:
  L1: Определение причины обращения (classifier/reason_classifier.py)
  L2: Определение тематического раздела (classifier/section_classifier.py)
  L3: Генерация ответа — этот модуль

LLM: YandexGPT (Yandex Cloud Foundation Models API)
Embeddings: Yandex Embeddings (для ChromaDB fallback)
"""

from __future__ import annotations

import logging
import re
import time as _time
from dataclasses import dataclass, field

import httpx

from app.classifier.reason_classifier import L1Result, classify_reason
from app.classifier.section_classifier import L2Result, classify_section
from app.config import settings
from app.models.reason_schemas import ContactReason

logger = logging.getLogger(__name__)

# ── System prompt ──

SYSTEM_PROMPT = """Ты — ИИ-ассистент техподдержки ООО «Фармбазис» (ПО для аптек).

ГЛАВНОЕ ОГРАНИЧЕНИЕ: Ответ ДОЛЖЕН быть НЕ БОЛЕЕ 2000 символов (кириллица). Пиши КРАТКО, только суть и действия. Без вступлений, без "рад помочь", без повторения вопроса.

ПРАВИЛА:
1. Отвечай ТОЛЬКО по предоставленному контексту.
2. Если вопрос про настройку, проведение документа, исправление ошибки или другую процедуру — отвечай НУМЕРОВАННЫМИ ШАГАМИ.
3. Не сокращай пошаговые инструкции до общего пересказа.
4. Если в контексте есть достаточный ответ — дай полный ответ по существу.
5. Предлагай обратиться к оператору ТОЛЬКО если информации недостаточно.
6. Русский язык, профессиональный тон.
7. Не раскрывай механику бота.

В конце ОБЯЗАТЕЛЬНО добавь:
```confidence
{"confidence": <0.0-1.0>, "reason": "<кратко>"}
```
"""

CONTEXT_TEMPLATE = """
ПРИЧИНА ОБРАЩЕНИЯ: {reason_name}
ТЕМАТИЧЕСКИЙ РАЗДЕЛ: {section_title}

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
"""

LLM_CLASSIFY_PROMPT = """Пользователь обратился с вопросом. Определи, к какой причине обращения он относится.

ВОПРОС: {question}

ВАРИАНТЫ ПРИЧИН:
{candidates}

Ответь ТОЛЬКО номером выбранной причины (1, 2, 3...) и кратким обоснованием в формате:
{{"choice": <номер>, "reason": "<обоснование>"}}
"""

MAX_ANSWER_BYTES = 4096


@dataclass
class RAGResponse:
    """Ответ RAG-системы v2."""

    answer: str
    confidence: float = 0.0
    confidence_reason: str = ""
    needs_escalation: bool = False
    source_articles: list[str] = field(default_factory=list)
    youtube_links: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    detected_reason: str = ""
    detected_reason_name: str = ""
    thematic_section: str = ""
    classification_method: str = ""
    # Для уточнения — список кандидатов [{reason_id, reason_name, score}]
    clarification_candidates: list[dict] = field(default_factory=list)


class YandexGPTClient:
    """Клиент для Yandex Foundation Models API."""

    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=60.0)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Запрос к YandexGPT completion API.

        Args:
            messages: Список сообщений [{"role": "system"/"user"/"assistant", "content": "..."}]
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов

        Returns:
            Текст ответа модели.
        """
        url = f"{self.BASE_URL}/completion"
        headers = {
            "Authorization": f"Api-Key {settings.yandex_api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "modelUri": settings.yandex_gpt_model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }

        _start = _time.time()
        response = await self._client.post(url, headers=headers, json=body)
        _elapsed = _time.time() - _start

        if response.status_code != 200:
            error_text = response.text
            logger.error(f"YandexGPT API error {response.status_code}: {error_text}")
            raise RuntimeError(f"YandexGPT API error: {response.status_code}")

        data = response.json()
        result_text = data["result"]["alternatives"][0]["message"]["text"]
        usage = data["result"].get("usage", {})
        logger.info(
            f"[YAGPT] time={_elapsed:.1f}s | "
            f"input_tokens={usage.get('inputTextTokens', '?')} | "
            f"output_tokens={usage.get('completionTokens', '?')}"
        )
        return result_text

    async def embed(self, text: str) -> list[float]:
        """Получить эмбеддинг текста через Yandex Embeddings API."""
        url = f"{self.BASE_URL}/textEmbedding"
        headers = {
            "Authorization": f"Api-Key {settings.yandex_api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "modelUri": settings.yandex_embedding_model_uri,
            "text": text,
        }

        response = await self._client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"Yandex Embedding error: {response.status_code}")

        return response.json()["embedding"]


class RAGEngine:
    """RAG-движок v2: L1→L2→L3 pipeline + YandexGPT."""

    def __init__(self):
        self.llm = YandexGPTClient()

    def _parse_confidence(self, answer: str) -> tuple[str, float, str]:
        """Извлечение блока confidence из ответа LLM."""
        pattern = r'```confidence\s*\n?\{.*?"confidence"\s*:\s*([\d.]+).*?"reason"\s*:\s*"([^"]*)".*?\}\s*\n?```'
        match = re.search(pattern, answer, re.DOTALL)

        if match:
            confidence = min(max(float(match.group(1)), 0.0), 1.0)
            reason = match.group(2)
            clean_answer = answer[: match.start()].strip()
            return clean_answer, confidence, reason

        json_pattern = r'\{[^{}]*"confidence"\s*:\s*([\d.]+)[^{}]*"reason"\s*:\s*"([^"]*)"[^{}]*\}'
        match = re.search(json_pattern, answer)
        if match:
            confidence = min(max(float(match.group(1)), 0.0), 1.0)
            reason = match.group(2)
            clean_answer = answer[: match.start()].strip()
            return clean_answer, confidence, reason

        return answer, 0.5, "Не удалось извлечь оценку уверенности"

    def _build_reason_context(
        self,
        reason: ContactReason,
        l2: L2Result,
        query: str,
    ) -> str:
        """Формирование контекста из причины обращения и L2-результата."""
        parts = []

        # Если найден точный пример ответа
        if l2.best_example and l2.best_example_score >= 0.5:
            parts.append(
                f'ПРИМЕР ОТВЕТА (похожий вопрос: "{l2.best_example.user_question}"):\n'
                f"{l2.best_example.ideal_answer}"
            )

        # Если найдена типовая жалоба
        if l2.best_complaint and l2.best_complaint_score >= 0.4:
            parts.append(
                f"ТИПОВАЯ ЖАЛОБА: {l2.best_complaint.description}\n"
                f"Контекст: {l2.best_complaint.context}\n"
                f"Шаблон ответа: {l2.best_complaint.response_template}"
            )

        # Q&A из тематического раздела
        if l2.section:
            for qa in l2.section.qa_pairs:
                parts.append(f"Вопрос: {qa.question}\nОтвет: {qa.answer}")

        return "\n\n---\n\n".join(parts) if parts else "Нет дополнительного контекста."

    async def ask(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        reason_id: str | None = None,
    ) -> RAGResponse:
        """Основной метод: полный pipeline L1→L2→L3.

        Args:
            question: Вопрос пользователя.
            chat_history: История чата.
            reason_id: Принудительная причина обращения (пропускает L1).

        Returns:
            RAGResponse с ответом и метаданными.
        """
        _start = _time.time()
        logger.info(f"[ENGINE] query={question}" + (f" forced_reason={reason_id}" if reason_id else ""))

        # ── L1: Определение причины обращения ──
        if reason_id:
            # Принудительная причина — пропускаем L1
            from app.database.reason_store import get_reason as _get_reason

            reason = await _get_reason(reason_id)
            if reason is None:
                return RAGResponse(
                    answer="Причина обращения не найдена.",
                    confidence=0.0,
                    needs_escalation=True,
                    classification_method="forced_invalid",
                )
            l1_method = "forced"
        else:
            l1 = classify_reason(question)

            if l1.method == "none":
                logger.info("[ENGINE] L1=none → escalation")
                return RAGResponse(
                    answer="Не удалось определить тему вашего обращения. Передаю вопрос оператору.",
                    confidence=0.0,
                    confidence_reason="L1: причина обращения не определена",
                    needs_escalation=True,
                    classification_method="none",
                )

            # Если неоднозначно — пробуем LLM-классификацию
            if not l1.is_confident and l1.needs_clarification:
                l1 = await self._llm_classify_reason(question, l1)

            if l1.reason is None:
                # LLM тоже не определил — уточнение
                return self._build_clarification_response(question, l1)

            reason = l1.reason
            l1_method = l1.method
        logger.info(f"[ENGINE] L1={reason.name} method={l1_method}")

        # ── L2: Определение тематического раздела ──
        l2 = classify_section(question, reason)

        # Exact match с примером → ответ без LLM
        if l2.method == "example_match" and l2.best_example:
            _total = _time.time() - _start
            logger.info(f"[ENGINE] L2=example_match → direct answer | time={_total:.1f}s")
            return RAGResponse(
                answer=_truncate_to_bytes(l2.best_example.ideal_answer, MAX_ANSWER_BYTES),
                confidence=0.95,
                confidence_reason="Точное совпадение с примером ответа",
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section=l2.section.title if l2.section else "",
                classification_method=f"L1:{l1_method}/L2:{l2.method}",
            )

        # ── L3: Генерация ответа через YandexGPT ──
        context = self._build_reason_context(reason, l2, question)
        section_title = l2.section.title if l2.section else "Общий"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if chat_history:
            for msg in chat_history[-6:]:
                messages.append(
                    {
                        "role": msg["role"],
                        "text": msg["content"],
                    }
                )

        user_message = CONTEXT_TEMPLATE.format(
            reason_name=reason.name,
            section_title=section_title,
            context=context,
            question=question,
        )
        messages.append({"role": "user", "text": user_message})

        logger.info(f"[ENGINE] L3 YandexGPT call | reason={reason.name} | section={section_title}")

        try:
            raw_answer = await self.llm.complete(messages, temperature=0.1, max_tokens=800)
        except Exception as e:
            logger.error(f"YandexGPT error: {e}")
            return RAGResponse(
                answer="Техническая ошибка. Попробуйте позже или обратитесь к оператору.",
                confidence=0.0,
                confidence_reason=f"Ошибка LLM: {e}",
                needs_escalation=True,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                classification_method=f"L1:{l1_method}/L2:{l2.method}",
            )

        clean_answer, confidence, conf_reason = self._parse_confidence(raw_answer)
        clean_answer = _truncate_to_bytes(clean_answer, MAX_ANSWER_BYTES)

        needs_escalation = confidence < settings.rag_confidence_threshold
        if not needs_escalation:
            clean_answer = _strip_operator_footer(clean_answer)

        _total = _time.time() - _start
        logger.info(
            f"[ENGINE] DONE | conf={confidence:.2f} | escalation={needs_escalation} | "
            f"time={_total:.1f}s | method=L1:{l1_method}/L2:{l2.method}"
        )

        return RAGResponse(
            answer=clean_answer,
            confidence=confidence,
            confidence_reason=conf_reason,
            needs_escalation=needs_escalation,
            detected_reason=reason.id,
            detected_reason_name=reason.name,
            thematic_section=section_title,
            classification_method=f"L1:{l1_method}/L2:{l2.method}",
        )

    async def _llm_classify_reason(self, question: str, l1: L1Result) -> L1Result:
        """LLM-классификация причины обращения при неоднозначности."""
        top_candidates = l1.candidates[:5]
        candidates_text = "\n".join(
            f"{i+1}. {c.reason.name} (маркеры: nouns={c.noun_matches}, verbs={c.verb_matches})"
            for i, c in enumerate(top_candidates)
        )

        prompt = LLM_CLASSIFY_PROMPT.format(
            question=question,
            candidates=candidates_text,
        )

        messages = [
            {"role": "system", "text": "Ты — классификатор обращений в техподдержку."},
            {"role": "user", "text": prompt},
        ]

        try:
            response = await self.llm.complete(messages, temperature=0.0, max_tokens=100)
            # Парсим JSON ответ
            json_match = re.search(r'\{[^{}]*"choice"\s*:\s*(\d+)', response)
            if json_match:
                choice_idx = int(json_match.group(1)) - 1
                if 0 <= choice_idx < len(top_candidates):
                    chosen = top_candidates[choice_idx]
                    logger.info(f"[L1-LLM] chose={chosen.reason.name}")
                    return L1Result(
                        reason=chosen.reason,
                        candidates=l1.candidates,
                        is_confident=True,
                        method="llm",
                    )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return l1  # Не удалось — возвращаем как есть

    def _build_clarification_response(self, question: str, l1: L1Result) -> RAGResponse:
        """Сформировать ответ-уточнение с вариантами причин."""
        top_candidates = l1.candidates[:5]
        options = "\n".join(f"{i+1}. {c.reason.name}" for i, c in enumerate(top_candidates))
        answer = f"Уточните, пожалуйста, к какой теме относится ваш вопрос:\n\n{options}\n\nУкажите номер или опишите подробнее."

        candidates_list = [
            {"reason_id": c.reason.id, "reason_name": c.reason.name, "score": c.score} for c in top_candidates
        ]

        return RAGResponse(
            answer=answer,
            confidence=0.3,
            confidence_reason="L1: неоднозначная классификация, требуется уточнение",
            needs_escalation=False,
            classification_method="clarification",
            clarification_candidates=candidates_list,
        )

    async def test_classify(self, question: str) -> dict:
        """Тестирование классификации без генерации ответа.

        Используется в admin-интерфейсе для проверки маркеров.
        """
        l1 = classify_reason(question)

        result = {
            "query": question,
            "l1_method": l1.method,
            "l1_confident": l1.is_confident,
            "l1_reason": l1.reason.name if l1.reason else None,
            "l1_reason_id": l1.reason.id if l1.reason else None,
            "candidates": [
                {
                    "reason_id": c.reason.id,
                    "reason_name": c.reason.name,
                    "score": c.score,
                    "phrase_matches": c.phrase_matches,
                    "numeric_matches": c.numeric_matches,
                    "noun_matches": c.noun_matches,
                    "verb_matches": c.verb_matches,
                }
                for c in l1.candidates[:5]
            ],
        }

        if l1.reason:
            l2 = classify_section(question, l1.reason)
            result["l2_method"] = l2.method
            result["l2_section"] = l2.section.title if l2.section else None
            result["l2_best_qa_score"] = l2.best_qa_score
            result["l2_best_qa"] = l2.best_qa.question if l2.best_qa else None
            result["l2_best_example_score"] = l2.best_example_score
            result["l2_best_example"] = l2.best_example.user_question if l2.best_example else None

        return result


# ── Утилиты ──


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Обрезать текст до max_bytes (UTF-8)."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    return truncated.decode("utf-8", errors="ignore").rstrip()


def _strip_operator_footer(text: str) -> str:
    """Убрать фразы про оператора из уверенных ответов."""
    patterns = [
        r"(?:если\s+)?(?:проблема\s+)?сохр[аняется]+.*?оператор[уа]?\.?\s*$",
        r"обратитесь\s+к\s+оператору\.?\s*$",
        r"свяжитесь\s+с\s+(?:нашей\s+)?(?:технической\s+)?поддержкой\.?\s*$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE).rstrip()
    return text


# ── Singleton ──

_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
