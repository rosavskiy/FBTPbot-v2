"""
RAG-движок v2 для ИИ-техподдержки Фармбазис.

Трёхуровневая система:
  L1: Определение причины обращения (classifier/reason_classifier.py)
  L2: Определение тематического раздела (classifier/section_classifier.py)
  L3: Генерация ответа — этот модуль

LLM: настраиваемый провайдер (YandexGPT / DeepSeek)
Embeddings: Yandex Embeddings (для ChromaDB fallback)
"""

from __future__ import annotations

import logging
import re
import time as _time
from dataclasses import dataclass, field

import httpx

from app.api.images import resolve_file_codes
from app.classifier.reason_classifier import ClassificationCandidate, L1Result, classify_reason, score_reason_candidate
from app.classifier.section_classifier import L2Result, classify_section
from app.config import settings
from app.llm_settings import get_classification_settings, get_llm_settings_snapshot
from app.models.reason_schemas import ClassificationRules, ContactReason
from app.models.schemas import ChatRoutingPolicy

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

FULL_CONTEXT_SYSTEM_PROMPT = """Ты — ИИ-ассистент техподдержки ООО «Фармбазис» (ПО для аптек).

Тебе предоставлена ПОЛНАЯ база знаний по теме обращения. Найди в ней релевантную информацию и дай исчерпывающий ответ.

ГЛАВНОЕ ОГРАНИЧЕНИЕ: Ответ ДОЛЖЕН быть НЕ БОЛЕЕ 2000 символов (кириллица). Пиши КРАТКО, только суть и действия.

ПРАВИЛА:
1. Отвечай ТОЛЬКО по предоставленной базе знаний.
2. Если вопрос про настройку, проведение документа, исправление ошибки или другую процедуру — отвечай НУМЕРОВАННЫМИ ШАГАМИ с описанием действий.
3. Не сокращай пошаговые инструкции до общего пересказа.
4. Дай полный исчерпывающий ответ по существу.
5. Предлагай обратиться к оператору ТОЛЬКО если информации в базе знаний недостаточно.
6. Русский язык, профессиональный тон.
7. Не раскрывай механику бота.

В конце ОБЯЗАТЕЛЬНО добавь:
```confidence
{"confidence": <0.0-1.0>, "reason": "<кратко>"}
```
"""

FULL_CONTEXT_TEMPLATE = """
ПРИЧИНА ОБРАЩЕНИЯ: {reason_name}

ПОЛНАЯ БАЗА ЗНАНИЙ ПО ТЕМЕ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
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
    files: list[dict] = field(default_factory=list)
    detected_reason: str = ""
    detected_reason_name: str = ""
    thematic_section: str = ""
    classification_method: str = ""
    clarification_kind: str = ""
    # Для уточнения — список кандидатов [{reason_id, reason_name, score}]
    clarification_candidates: list[dict] = field(default_factory=list)
    # Debug trace (заполняется только при debug=True)
    debug_trace: dict | None = None


@dataclass(frozen=True)
class AnswerRoutingDecision:
    decision: str
    prompt: str = ""
    prompt_source: str = ""


class YandexGPTClient:
    """Клиент для Yandex Foundation Models API."""

    BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        """Закрыть HTTP-клиент."""
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Запрос к YandexGPT completion API.

        Args:
            messages: Список сообщений [{"role": "system"/"user"/"assistant", "text": "..."}]
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов

        Returns:
            Текст ответа модели.
        """
        url = f"{self.BASE_URL}/completion"
        headers = {
            "Authorization": f"Api-Key {get_llm_settings_snapshot()['yandex_api_key']}",
            "Content-Type": "application/json",
        }
        llm_settings = get_llm_settings_snapshot()

        body = {
            "modelUri": f"gpt://{llm_settings['yandex_folder_id']}/{llm_settings['yandex_gpt_model']}/latest",
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
        llm_settings = get_llm_settings_snapshot()
        headers = {
            "Authorization": f"Api-Key {llm_settings['yandex_api_key']}",
            "Content-Type": "application/json",
        }

        body = {
            "modelUri": f"emb://{llm_settings['yandex_folder_id']}/{llm_settings['yandex_embedding_model']}/latest",
            "text": text,
        }

        response = await self._client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"Yandex Embedding error: {response.status_code}")

        return response.json()["embedding"]


class DeepSeekClient:
    """Клиент для DeepSeek Chat Completions API."""

    BASE_URL = "https://api.deepseek.com"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {get_llm_settings_snapshot()['deepseek_api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        llm_settings = get_llm_settings_snapshot()
        payload_messages = [
            {
                "role": message["role"],
                "content": message.get("content", message.get("text", "")),
            }
            for message in messages
        ]
        body = {
            "model": llm_settings["deepseek_model"],
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "text"},
        }

        _start = _time.time()
        response = await self._client.post(url, headers=headers, json=body)
        _elapsed = _time.time() - _start

        if response.status_code != 200:
            error_text = response.text
            logger.error(f"DeepSeek API error {response.status_code}: {error_text}")
            raise RuntimeError(f"DeepSeek API error: {response.status_code}")

        data = response.json()
        result_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        logger.info(
            f"[DEEPSEEK] time={_elapsed:.1f}s | "
            f"input_tokens={usage.get('prompt_tokens', '?')} | "
            f"output_tokens={usage.get('completion_tokens', '?')}"
        )
        return result_text


class RAGEngine:
    """RAG-движок v2: L1→L2→L3 pipeline + выбранный LLM-провайдер."""

    def __init__(self):
        self.llm = None
        self._provider = None

    def _create_llm_client(self):
        provider = get_llm_settings_snapshot()["llm_provider"]
        if provider == "deepseek":
            logger.info("[ENGINE] Using DeepSeek provider")
            return DeepSeekClient()
        logger.info("[ENGINE] Using Yandex provider")
        return YandexGPTClient()

    async def _ensure_llm_client(self) -> None:
        provider = get_llm_settings_snapshot()["llm_provider"]
        if self.llm is not None and self._provider == provider:
            return
        if self.llm is not None:
            await self.llm.close()
        self.llm = self._create_llm_client()
        self._provider = provider

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
                f'ПРИМЕР ОТВЕТА (похожий вопрос: "{l2.best_example.user_questions[0] if l2.best_example.user_questions else l2.best_example.user_question}"):\n'
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

    @staticmethod
    def _build_full_context(reason: ContactReason) -> str:
        """Формирование ПОЛНОГО контекста из всей БЗ причины обращения."""
        parts: list[str] = []

        for section in reason.thematic_sections:
            section_lines = [f"## {section.title}"]
            for qa in section.qa_pairs:
                section_lines.append(f"Вопрос: {qa.question}\nОтвет: {qa.answer}")
            parts.append("\n\n".join(section_lines))

        if reason.typical_complaints:
            complaint_lines = ["## Типовые жалобы"]
            for c in reason.typical_complaints:
                complaint_lines.append(
                    f"Жалоба: {c.description}\nКонтекст: {c.context}\nШаблон ответа: {c.response_template}"
                )
            parts.append("\n\n".join(complaint_lines))

        if reason.example_answers:
            example_lines = ["## Примеры ответов"]
            for ex in reason.example_answers:
                questions = ", ".join(ex.user_questions) if ex.user_questions else ex.user_question
                example_lines.append(f"Вопрос: {questions}\nОтвет: {ex.ideal_answer}")
            parts.append("\n\n".join(example_lines))

        return "\n\n---\n\n".join(parts) if parts else "Нет дополнительного контекста."

    @staticmethod
    def _create_provider_client(provider: str) -> YandexGPTClient | DeepSeekClient:
        """Создать LLM-клиент по имени провайдера (для per-reason override)."""
        if provider == "deepseek":
            return DeepSeekClient()
        return YandexGPTClient()

    @staticmethod
    def _check_required_markers(
        candidate: ClassificationCandidate,
        cls_rules: ClassificationRules,
    ) -> dict:
        """Проверка обязательных типов маркеров для причины обращения.

        Returns:
            {"passed": True/False, "required": [...], "found": [...], "missing": [...], "default_text": "..."}
        """
        mapping = {
            "numeric_tag": candidate.numeric_matches,
            "phrase_mask": candidate.phrase_matches,
            "noun": candidate.noun_matches,
            "verb": candidate.verb_matches,
        }

        required = cls_rules.required_markers
        found = [m for m in required if mapping.get(m)]
        missing = [m for m in required if not mapping.get(m)]

        label_map = {
            "numeric_tag": "числовой тег (номер ошибки)",
            "phrase_mask": "фразовая маска",
            "noun": "существительное-маркер",
            "verb": "глагол-маркер",
        }
        missing_labels = [label_map.get(m, m) for m in missing]
        default_text = (
            "Уточните, пожалуйста, детали вашего обращения: "
            + ", ".join(missing_labels)
            + ". Напишите подробнее или укажите номер/код."
        )

        return {
            "passed": len(missing) == 0,
            "required": required,
            "found": found,
            "missing": missing,
            "default_text": default_text,
        }

    def _build_base_debug_trace(
        self,
        *,
        l1_method: str,
        l1_confident: bool,
        reason: ContactReason | None,
        l1_candidates_data: list[dict],
        escalation_check: dict | None,
        confidence_reason: str,
        llm_involvement: str,
        start_time: float,
        marker_weights: dict | None = None,
    ) -> dict:
        """Сформировать базовый debug_trace без L2/L3 полей."""
        return {
            "l1_method": l1_method,
            "l1_confident": l1_confident,
            "l1_reason": reason.name if reason else None,
            "l1_reason_id": reason.id if reason else None,
            "l1_candidates": l1_candidates_data,
            "escalation_check": escalation_check,
            "marker_weights": marker_weights,
            "l2_method": None,
            "l2_section": None,
            "l2_best_qa_score": None,
            "l2_best_example_score": None,
            "l2_best_complaint_score": None,
            "llm_prompt": None,
            "llm_raw_response": None,
            "llm_provider": None,
            "llm_temperature": None,
            "confidence_parsed": 0.0,
            "confidence_reason": confidence_reason,
            "llm_involvement": llm_involvement,
            "processing_time_ms": int((_time.time() - start_time) * 1000),
        }

    @staticmethod
    def _build_answer_refinement_prompt(
        question: str,
        reason: ContactReason,
        section_title: str | None = None,
    ) -> tuple[str, str]:
        text_lower = re.sub(r"\s+", " ", question.lower().strip())

        if any(token in text_lower for token in ("ошиб", "код", "ккм", "касс", "чек", "маркиров", "честн", "чз")):
            return (
                "Уточните, пожалуйста, точный текст или код ошибки, который вы видите на экране.",
                "rule:error_text",
            )

        if any(token in text_lower for token in ("наклад", "документ", "приход", "расход", "поставщик")):
            return (
                "Уточните, пожалуйста, номер документа и что именно с ним происходит.",
                "rule:document_number",
            )

        if any(token in text_lower for token in ("товар", "карточк", "категор", "грлс", "жнв")):
            return (
                "Уточните, пожалуйста, название товара и текст сообщения системы.",
                "rule:item_details",
            )

        target = section_title or reason.name
        return (
            f"Уточните, пожалуйста, одну ключевую деталь по теме «{target}»: код ошибки, номер документа или текст сообщения системы.",
            "fallback:generic",
        )

    def _resolve_answer_routing(
        self,
        *,
        question: str,
        reason: ContactReason,
        section_title: str | None,
        confidence: float,
        routing_policy: ChatRoutingPolicy | None,
        refinement_attempt: int,
    ) -> AnswerRoutingDecision | None:
        if routing_policy is None or not routing_policy.enabled:
            return None

        if confidence >= routing_policy.answer_threshold:
            return AnswerRoutingDecision(decision="answer")

        if (
            routing_policy.clarification_min_confidence <= confidence <= routing_policy.clarification_max_confidence
            and refinement_attempt < routing_policy.max_refinement_attempts
        ):
            prompt, prompt_source = self._build_answer_refinement_prompt(question, reason, section_title)
            return AnswerRoutingDecision(decision="clarification", prompt=prompt, prompt_source=prompt_source)

        return AnswerRoutingDecision(decision="escalation")

    def _check_forced_escalation(self, question: str, reason: ContactReason) -> dict:
        """Проверка правил 100%-эскалации (L1.5).

        Проверяет keyword_patterns (точное фразовое совпадение) и Q&A-пары
        (overlap лемм ≥ score_threshold). При совпадении возвращает matched=True.
        """
        rules = reason.escalation_rules
        if not rules.enabled:
            return {"matched": False}

        text_lower = re.sub(r"\s+", " ", question.lower().strip())

        # 1. Проверка keyword_patterns (фразовые маски)
        for pattern in rules.metrics.keyword_patterns:
            if pattern.lower().strip() in text_lower:
                return {"matched": True, "trigger": "keyword", "pattern": pattern, "answer": ""}

        # 2. Проверка Q&A-пар (overlap score)
        if rules.qa_pairs:
            from app.classifier.section_classifier import _overlap_score, _text_to_lemma_set

            query_lemmas = _text_to_lemma_set(question)
            threshold = rules.metrics.score_threshold

            for qa in rules.qa_pairs:
                qa_lemmas = _text_to_lemma_set(qa.question)
                score = _overlap_score(query_lemmas, qa_lemmas)
                if score >= threshold:
                    return {
                        "matched": True,
                        "trigger": "qa_pair",
                        "question": qa.question,
                        "score": score,
                        "answer": qa.answer,
                    }

        return {"matched": False}

    def _check_global_escalation(self, question: str) -> dict:
        """Проверка глобальных правил эскалации (L0).

        Проверяет keyword_patterns глобально (до L1-классификации).
        При совпадении — немедленная эскалация без определения причины.
        """
        from app.database.reason_store import get_global_escalation

        rules = get_global_escalation()
        if not rules.enabled:
            return {"matched": False}

        text_lower = re.sub(r"\s+", " ", question.lower().strip())

        for pattern in rules.keyword_patterns:
            if pattern.lower().strip() in text_lower:
                return {"matched": True, "trigger": "keyword", "pattern": pattern}

        return {"matched": False}

    async def ask(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        reason_id: str | None = None,
        routing_policy: ChatRoutingPolicy | None = None,
        refinement_attempt: int = 0,
        debug: bool = False,
    ) -> RAGResponse:
        """Основной метод: полный pipeline L1→L2→L3.

        Args:
            question: Вопрос пользователя.
            chat_history: История чата.
            reason_id: Принудительная причина обращения (пропускает L1).
            debug: Собирать полный trace pipeline.

        Returns:
            RAGResponse с ответом и метаданными.
        """
        _start = _time.time()
        logger.info(f"[ENGINE] query={question}" + (f" forced_reason={reason_id}" if reason_id else ""))

        # Debug trace accumulator
        llm_used_for_classify = False
        llm_used_for_generate = False
        l1_marker_weights: dict = {}

        # ── L0: Глобальная эскалация (до классификации) ──
        l0_check = self._check_global_escalation(question)
        if l0_check["matched"]:
            _total = _time.time() - _start
            logger.info(f"[ENGINE] L0=global_escalation | pattern={l0_check['pattern']} | time={_total:.1f}s")
            resp = RAGResponse(
                answer=(
                    "По данному вопросу необходима консультация специалиста техподдержки. "
                    "Передаю ваше обращение оператору."
                ),
                confidence=0.0,
                confidence_reason="L0: глобальная эскалация по ключевой фразе",
                needs_escalation=True,
                classification_method="L0:global_escalation",
            )
            if debug:
                resp.debug_trace = {
                    "l0_check": l0_check,
                    "l1_method": None,
                    "l1_confident": False,
                    "l1_reason": None,
                    "l1_reason_id": None,
                    "l1_candidates": [],
                    "escalation_check": None,
                    "l2_method": None,
                    "l2_section": None,
                    "l2_best_qa_score": None,
                    "l2_best_example_score": None,
                    "l2_best_complaint_score": None,
                    "llm_prompt": None,
                    "llm_raw_response": None,
                    "llm_provider": None,
                    "llm_temperature": None,
                    "confidence_parsed": 0.0,
                    "confidence_reason": "L0: глобальная эскалация по ключевой фразе",
                    "llm_involvement": "none",
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        # ── L1: Определение причины обращения ──
        if reason_id:
            # Принудительная причина — пропускаем L1
            from app.database.reason_store import get_reason as _get_reason

            reason = _get_reason(reason_id)
            if reason is None:
                return RAGResponse(
                    answer="Причина обращения не найдена.",
                    confidence=0.0,
                    needs_escalation=True,
                    classification_method="forced_invalid",
                )
            l1_method = "forced"
            l1_confident = True
            l1_winning = score_reason_candidate(question, reason)
            l1_candidates_data = [
                {
                    "reason_id": l1_winning.reason.id,
                    "reason_name": l1_winning.reason.name,
                    "score": l1_winning.score,
                    "phrase_matches": l1_winning.phrase_matches,
                    "numeric_matches": l1_winning.numeric_matches,
                    "noun_matches": l1_winning.noun_matches,
                    "verb_matches": l1_winning.verb_matches,
                }
            ]
        else:
            l1 = classify_reason(question)
            l1_marker_weights = l1.marker_weights

            l1_candidates_data = [
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
            ]

            if l1.method == "none" or l1.method == "below_threshold":
                is_below = l1.method == "below_threshold"
                log_reason = "below_threshold" if is_below else "none"
                logger.info(f"[ENGINE] L1={log_reason} → escalation")
                top_score = l1.winning_candidate.score if l1.winning_candidate else 0.0
                answer_text = (
                    (
                        f"Не удалось точно определить тему вашего обращения (score={top_score:.1f}). "
                        "Передаю вопрос оператору."
                    )
                    if is_below
                    else "Не удалось определить тему вашего обращения. Передаю вопрос оператору."
                )
                resp = RAGResponse(
                    answer=answer_text,
                    confidence=0.0,
                    confidence_reason=f"L1: {log_reason}",
                    needs_escalation=True,
                    classification_method=l1.method,
                )
                if debug:
                    resp.debug_trace = {
                        "l1_method": l1.method,
                        "l1_confident": False,
                        "l1_reason": None,
                        "l1_reason_id": None,
                        "l1_candidates": l1_candidates_data,
                        "escalation_check": None,
                        "l2_method": None,
                        "l2_section": None,
                        "l2_best_qa_score": None,
                        "l2_best_example_score": None,
                        "l2_best_complaint_score": None,
                        "llm_prompt": None,
                        "llm_raw_response": None,
                        "llm_provider": None,
                        "llm_temperature": None,
                        "confidence_parsed": 0.0,
                        "confidence_reason": f"L1: {log_reason}",
                        "llm_involvement": "none",
                        "processing_time_ms": int((_time.time() - _start) * 1000),
                    }
                return resp

            # Если неоднозначно — пробуем LLM-классификацию
            if not l1.is_confident and l1.needs_clarification:
                l1 = await self._llm_classify_reason(question, l1)
                llm_used_for_classify = True

            if l1.reason is None:
                # LLM тоже не определил — уточнение
                return self._build_clarification_response(question, l1)

            reason = l1.reason
            l1_method = l1.method
            l1_confident = l1.is_confident
            l1_winning = l1.winning_candidate
        logger.info(f"[ENGINE] L1={reason.name} method={l1_method}")

        # ── L1.1: Per-reason порог баллов ──
        cls_rules = reason.classification_rules
        marker_clarification_check: dict | None = None
        if cls_rules.enabled:
            winning_score = l1_winning.score if l1_winning else 0.0

            # Per-reason min_score_threshold (если задан, иначе — глобальный уже проверен в L1)
            if (
                l1_method != "forced"
                and cls_rules.min_score_threshold is not None
                and winning_score < cls_rules.min_score_threshold
            ):
                _total = _time.time() - _start
                logger.info(
                    f"[ENGINE] L1.1=per_reason_threshold | score={winning_score:.1f} "
                    f"< threshold={cls_rules.min_score_threshold:.1f} | time={_total:.1f}s"
                )
                resp = RAGResponse(
                    answer=(f"Не удалось точно определить тему (score={winning_score:.1f}). Передаю вопрос оператору."),
                    confidence=0.0,
                    confidence_reason="L1.1: score ниже per-reason порога",
                    needs_escalation=True,
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    classification_method=f"L1:{l1_method}/L1.1:per_reason_threshold",
                )
                if debug:
                    resp.debug_trace = self._build_base_debug_trace(
                        l1_method=l1_method,
                        l1_confident=l1_confident,
                        reason=reason,
                        l1_candidates_data=l1_candidates_data,
                        escalation_check=None,
                        confidence_reason="L1.1: score ниже per-reason порога",
                        llm_involvement="classification_only" if llm_used_for_classify else "none",
                        start_time=_start,
                        marker_weights=l1_marker_weights,
                    )
                return resp

            # Проверка обязательных маркеров
            if cls_rules.required_markers and l1_winning:
                marker_clarification_check = self._check_required_markers(l1_winning, cls_rules)
                if not marker_clarification_check["passed"]:
                    _total = _time.time() - _start
                    clarification_text = cls_rules.clarification_text or marker_clarification_check["default_text"]
                    logger.info(
                        f"[ENGINE] L1.1=marker_clarification | missing={marker_clarification_check['missing']} | time={_total:.1f}s"
                    )
                    resp = RAGResponse(
                        answer=clarification_text,
                        confidence=0.3,
                        confidence_reason="L1.1: обязательный маркер не найден, уточняющий вопрос",
                        needs_escalation=False,
                        detected_reason=reason.id,
                        detected_reason_name=reason.name,
                        classification_method="marker_clarification",
                    )
                    if debug:
                        trace = self._build_base_debug_trace(
                            l1_method=l1_method,
                            l1_confident=l1_confident,
                            reason=reason,
                            l1_candidates_data=l1_candidates_data,
                            escalation_check=None,
                            confidence_reason="L1.1: обязательный маркер не найден",
                            llm_involvement="classification_only" if llm_used_for_classify else "none",
                            start_time=_start,
                            marker_weights=l1_marker_weights,
                        )
                        trace["marker_clarification_check"] = marker_clarification_check
                        resp.debug_trace = trace
                    return resp

        # ── L1.5: Проверка правил 100%-эскалации ──
        escalation_check = self._check_forced_escalation(question, reason)
        if escalation_check["matched"]:
            _total = _time.time() - _start
            logger.info(f"[ENGINE] L1.5=forced_escalation | trigger={escalation_check['trigger']} | time={_total:.1f}s")
            esc_answer = escalation_check.get("answer") or (
                "По данному вопросу необходима консультация специалиста техподдержки. Передаю ваше обращение оператору."
            )
            resp = RAGResponse(
                answer=esc_answer,
                confidence=0.0,
                confidence_reason="L1.5: 100%-эскалация по правилам причины обращения",
                needs_escalation=True,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                classification_method=f"L1:{l1_method}/L1.5:forced_escalation",
            )
            if debug:
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "l2_method": None,
                    "l2_section": None,
                    "l2_best_qa_score": None,
                    "l2_best_example_score": None,
                    "l2_best_complaint_score": None,
                    "llm_prompt": None,
                    "llm_raw_response": None,
                    "llm_provider": None,
                    "llm_temperature": None,
                    "confidence_parsed": 0.0,
                    "confidence_reason": "L1.5: 100%-эскалация по правилам причины обращения",
                    "llm_involvement": "classification_only" if llm_used_for_classify else "none",
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        # ── Full-Context LLM / Standard L2→L3 branching ──
        if reason.full_context_llm.enabled:
            # ── Full-Context path: вся БЗ причины → LLM ──
            # ExampleQA bypass — всё ещё проверяем (быстро и бесплатно)
            l2 = classify_section(question, reason)

            if l2.method == "example_match" and l2.best_example:
                _total = _time.time() - _start
                logger.info(f"[ENGINE] FC:L2=example_match → direct answer | time={_total:.1f}s")
                resp = RAGResponse(
                    answer=_strip_markdown(_truncate_to_bytes(l2.best_example.ideal_answer, MAX_ANSWER_BYTES)),
                    confidence=0.95,
                    confidence_reason="Точное совпадение с примером ответа (full-context bypass)",
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    thematic_section=l2.section.title if l2.section else "",
                    classification_method=f"L1:{l1_method}/FC:example_bypass",
                    files=resolve_file_codes(l2.best_example.file_codes),
                )
                if debug:
                    resp.debug_trace = {
                        "l1_method": l1_method,
                        "l1_confident": l1_confident,
                        "l1_reason": reason.name,
                        "l1_reason_id": reason.id,
                        "l1_candidates": l1_candidates_data,
                        "escalation_check": escalation_check,
                        "full_context_mode": True,
                        "l2_method": l2.method,
                        "l2_section": l2.section.title if l2.section else None,
                        "l2_best_qa_score": l2.best_qa_score,
                        "l2_best_example_score": l2.best_example_score,
                        "l2_best_complaint_score": l2.best_complaint_score,
                        "llm_prompt": None,
                        "llm_raw_response": None,
                        "llm_provider": None,
                        "llm_temperature": None,
                        "confidence_parsed": 0.95,
                        "confidence_reason": "Точное совпадение с примером ответа (full-context bypass)",
                        "llm_involvement": "classification_only" if llm_used_for_classify else "none",
                        "processing_time_ms": int(_total * 1000),
                    }
                return resp

            # Собираем полный контекст из всей БЗ причины
            full_context = self._build_full_context(reason)
            full_context_chars = len(full_context)

            if full_context == "Нет дополнительного контекста.":
                _total = _time.time() - _start
                logger.info(f"[ENGINE] FC=skip_llm (no KB context) | reason={reason.name} | time={_total:.1f}s")
                resp = RAGResponse(
                    answer="По данному вопросу недостаточно информации в базе знаний. Передаю ваше обращение оператору.",
                    confidence=0.0,
                    confidence_reason="Нет контекста в базе знаний — LLM не вызывался",
                    needs_escalation=True,
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    classification_method=f"L1:{l1_method}/FC:no_context",
                )
                if debug:
                    resp.debug_trace = {
                        "l1_method": l1_method,
                        "l1_confident": l1_confident,
                        "l1_reason": reason.name,
                        "l1_reason_id": reason.id,
                        "l1_candidates": l1_candidates_data,
                        "escalation_check": escalation_check,
                        "full_context_mode": True,
                        "full_context_chars": 0,
                        "l2_method": "skipped",
                        "l2_section": None,
                        "l2_best_qa_score": None,
                        "l2_best_example_score": None,
                        "l2_best_complaint_score": None,
                        "llm_prompt": None,
                        "llm_raw_response": None,
                        "llm_provider": None,
                        "llm_temperature": None,
                        "confidence_parsed": 0.0,
                        "confidence_reason": "Нет контекста в базе знаний — LLM не вызывался",
                        "llm_involvement": "classification_only" if llm_used_for_classify else "none",
                        "processing_time_ms": int(_total * 1000),
                    }
                return resp

            # Определяем провайдер: per-reason override или глобальный
            fc_settings = reason.full_context_llm
            fc_system_prompt = fc_settings.custom_prompt or FULL_CONTEXT_SYSTEM_PROMPT

            messages = [{"role": "system", "text": fc_system_prompt}]
            if chat_history:
                for msg in chat_history[-6:]:
                    messages.append({"role": msg["role"], "text": msg["content"]})

            user_message = FULL_CONTEXT_TEMPLATE.format(
                reason_name=reason.name,
                context=full_context,
                question=question,
            )
            messages.append({"role": "user", "text": user_message})

            llm_snapshot = get_llm_settings_snapshot()
            llm_temp = float(llm_snapshot.get("llm_temperature", "0.1"))
            fc_provider = fc_settings.provider
            if fc_provider == "default":
                fc_provider = llm_snapshot["llm_provider"]
            fc_provider_name = fc_provider

            logger.info(
                f"[ENGINE] FC provider={fc_provider_name} | reason={reason.name} | context_chars={full_context_chars}"
            )

            raw_answer = None
            override_client = None
            try:
                if fc_settings.provider != "default":
                    override_client = self._create_provider_client(fc_provider)
                    raw_answer = await override_client.complete(messages, temperature=llm_temp, max_tokens=1200)
                else:
                    await self._ensure_llm_client()
                    raw_answer = await self.llm.complete(messages, temperature=llm_temp, max_tokens=1200)
                llm_used_for_generate = True
            except Exception as e:
                logger.error(f"FC LLM error ({fc_provider_name}): {e}")
                return RAGResponse(
                    answer="Техническая ошибка. Попробуйте позже или обратитесь к оператору.",
                    confidence=0.0,
                    confidence_reason=f"Ошибка LLM (full-context): {e}",
                    needs_escalation=True,
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    classification_method=f"L1:{l1_method}/FC:error",
                )
            finally:
                if override_client is not None:
                    await override_client.close()

            clean_answer, confidence, conf_reason = self._parse_confidence(raw_answer)
            clean_answer = _strip_markdown(clean_answer)
            clean_answer = _truncate_to_bytes(clean_answer, MAX_ANSWER_BYTES)

            routing_decision = self._resolve_answer_routing(
                question=question,
                reason=reason,
                section_title="Full-Context",
                confidence=confidence,
                routing_policy=routing_policy,
                refinement_attempt=refinement_attempt,
            )

            if routing_decision is not None and routing_decision.decision == "clarification":
                _total = _time.time() - _start
                resp = RAGResponse(
                    answer=routing_decision.prompt,
                    confidence=confidence,
                    confidence_reason=conf_reason,
                    needs_escalation=False,
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    thematic_section="Full-Context",
                    classification_method="answer_refinement",
                    clarification_kind="answer_refinement",
                )
                if debug:
                    prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
                    resp.debug_trace = {
                        "l1_method": l1_method,
                        "l1_confident": l1_confident,
                        "l1_reason": reason.name,
                        "l1_reason_id": reason.id,
                        "l1_candidates": l1_candidates_data,
                        "escalation_check": escalation_check,
                        "full_context_mode": True,
                        "full_context_provider": fc_provider_name,
                        "full_context_chars": full_context_chars,
                        "l2_method": "skipped",
                        "l2_section": None,
                        "l2_best_qa_score": None,
                        "l2_best_example_score": None,
                        "l2_best_complaint_score": None,
                        "llm_prompt": prompt_text,
                        "llm_raw_response": raw_answer,
                        "llm_provider": fc_provider_name,
                        "llm_temperature": llm_temp,
                        "confidence_parsed": confidence,
                        "confidence_reason": conf_reason,
                        "routing_decision": "clarification",
                        "clarification_kind": "answer_refinement",
                        "clarification_attempt": refinement_attempt + 1,
                        "previous_confidence": confidence,
                        "llm_involvement": (
                            "classification+full_context_generation"
                            if llm_used_for_classify
                            else "full_context_generation"
                        ),
                        "processing_time_ms": int(_total * 1000),
                    }
                return resp

            if routing_decision is not None and routing_decision.decision == "escalation":
                _total = _time.time() - _start
                resp = RAGResponse(
                    answer="По данному вопросу недостаточно уверенности для автоматического ответа. Передаю ваше обращение оператору.",
                    confidence=confidence,
                    confidence_reason=conf_reason,
                    needs_escalation=True,
                    detected_reason=reason.id,
                    detected_reason_name=reason.name,
                    thematic_section="Full-Context",
                    classification_method="answer_refinement_escalation",
                )
                if debug:
                    prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
                    resp.debug_trace = {
                        "l1_method": l1_method,
                        "l1_confident": l1_confident,
                        "l1_reason": reason.name,
                        "l1_reason_id": reason.id,
                        "l1_candidates": l1_candidates_data,
                        "escalation_check": escalation_check,
                        "full_context_mode": True,
                        "full_context_provider": fc_provider_name,
                        "full_context_chars": full_context_chars,
                        "l2_method": "skipped",
                        "l2_section": None,
                        "l2_best_qa_score": None,
                        "l2_best_example_score": None,
                        "l2_best_complaint_score": None,
                        "llm_prompt": prompt_text,
                        "llm_raw_response": raw_answer,
                        "llm_provider": fc_provider_name,
                        "llm_temperature": llm_temp,
                        "confidence_parsed": confidence,
                        "confidence_reason": conf_reason,
                        "routing_decision": "escalation",
                        "clarification_kind": "answer_refinement",
                        "clarification_attempt": refinement_attempt,
                        "previous_confidence": confidence,
                        "llm_involvement": (
                            "classification+full_context_generation"
                            if llm_used_for_classify
                            else "full_context_generation"
                        ),
                        "processing_time_ms": int(_total * 1000),
                    }
                return resp

            needs_escalation = confidence < settings.rag_confidence_threshold if routing_decision is None else False
            if not needs_escalation:
                clean_answer = _strip_operator_footer(clean_answer)

            _total = _time.time() - _start
            logger.info(
                f"[ENGINE] FC DONE | conf={confidence:.2f} | escalation={needs_escalation} | "
                f"time={_total:.1f}s | method=L1:{l1_method}/FC"
            )

            llm_involvement = (
                "classification+full_context_generation" if llm_used_for_classify else "full_context_generation"
            )

            resp = RAGResponse(
                answer=clean_answer,
                confidence=confidence,
                confidence_reason=conf_reason,
                needs_escalation=needs_escalation,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section="Full-Context",
                classification_method=f"L1:{l1_method}/FC",
            )

            if debug:
                prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "full_context_mode": True,
                    "full_context_provider": fc_provider_name,
                    "full_context_chars": full_context_chars,
                    "l2_method": "skipped",
                    "l2_section": None,
                    "l2_best_qa_score": None,
                    "l2_best_example_score": None,
                    "l2_best_complaint_score": None,
                    "llm_prompt": prompt_text,
                    "llm_raw_response": raw_answer,
                    "llm_provider": fc_provider_name,
                    "llm_temperature": llm_temp,
                    "confidence_parsed": confidence,
                    "confidence_reason": conf_reason,
                    "routing_decision": routing_decision.decision if routing_decision is not None else None,
                    "clarification_attempt": refinement_attempt if routing_decision is not None else None,
                    "previous_confidence": confidence if routing_decision is not None else None,
                    "llm_involvement": llm_involvement,
                    "processing_time_ms": int(_total * 1000),
                }

            return resp

        # ── L2: Определение тематического раздела ──
        l2 = classify_section(question, reason)

        # Exact match с примером → ответ без LLM
        if l2.method == "example_match" and l2.best_example:
            _total = _time.time() - _start
            logger.info(f"[ENGINE] L2=example_match → direct answer | time={_total:.1f}s")
            resp = RAGResponse(
                answer=_strip_markdown(_truncate_to_bytes(l2.best_example.ideal_answer, MAX_ANSWER_BYTES)),
                confidence=0.95,
                confidence_reason="Точное совпадение с примером ответа",
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section=l2.section.title if l2.section else "",
                classification_method=f"L1:{l1_method}/L2:{l2.method}",
                files=resolve_file_codes(l2.best_example.file_codes),
            )
            if debug:
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "l2_method": l2.method,
                    "l2_section": l2.section.title if l2.section else None,
                    "l2_best_qa_score": l2.best_qa_score,
                    "l2_best_example_score": l2.best_example_score,
                    "l2_best_complaint_score": l2.best_complaint_score,
                    "llm_prompt": None,
                    "llm_raw_response": None,
                    "llm_provider": None,
                    "llm_temperature": None,
                    "confidence_parsed": 0.95,
                    "confidence_reason": "Точное совпадение с примером ответа",
                    "llm_involvement": "classification_only" if llm_used_for_classify else "none",
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        # ── L3: Генерация ответа через выбранный LLM ──
        context = self._build_reason_context(reason, l2, question)
        section_title = l2.section.title if l2.section else "Общий"

        # Если контекст из базы знаний пустой — сразу эскалация без вызова LLM
        if context == "Нет дополнительного контекста.":
            _total = _time.time() - _start
            logger.info(f"[ENGINE] L3=skip_llm (no KB context) | reason={reason.name} | time={_total:.1f}s")
            resp = RAGResponse(
                answer=("По данному вопросу недостаточно информации в базе знаний. Передаю ваше обращение оператору."),
                confidence=0.0,
                confidence_reason="Нет контекста в базе знаний — LLM не вызывался",
                needs_escalation=True,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section=section_title,
                classification_method=f"L1:{l1_method}/L2:{l2.method}/L3:no_context",
            )
            if debug:
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "l2_method": l2.method,
                    "l2_section": l2.section.title if l2.section else None,
                    "l2_best_qa_score": l2.best_qa_score,
                    "l2_best_example_score": l2.best_example_score,
                    "l2_best_complaint_score": l2.best_complaint_score,
                    "llm_prompt": None,
                    "llm_raw_response": None,
                    "llm_provider": None,
                    "llm_temperature": None,
                    "confidence_parsed": 0.0,
                    "confidence_reason": "Нет контекста в базе знаний — LLM не вызывался",
                    "llm_involvement": "classification_only" if llm_used_for_classify else "none",
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        messages = [
            {"role": "system", "text": SYSTEM_PROMPT},
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

        logger.info(
            f"[ENGINE] L3 provider={settings.llm_provider_normalized} | reason={reason.name} | section={section_title}"
        )

        llm_snapshot = get_llm_settings_snapshot()
        llm_temp = float(llm_snapshot.get("llm_temperature", "0.1"))
        llm_provider_name = llm_snapshot["llm_provider"]
        raw_answer = None

        try:
            await self._ensure_llm_client()
            raw_answer = await self.llm.complete(messages, temperature=llm_temp, max_tokens=800)
            llm_used_for_generate = True
        except Exception as e:
            logger.error(f"LLM error ({settings.llm_provider_normalized}): {e}")
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
        clean_answer = _strip_markdown(clean_answer)
        clean_answer = _truncate_to_bytes(clean_answer, MAX_ANSWER_BYTES)

        routing_decision = self._resolve_answer_routing(
            question=question,
            reason=reason,
            section_title=section_title,
            confidence=confidence,
            routing_policy=routing_policy,
            refinement_attempt=refinement_attempt,
        )

        if routing_decision is not None and routing_decision.decision == "clarification":
            _total = _time.time() - _start
            resp = RAGResponse(
                answer=routing_decision.prompt,
                confidence=confidence,
                confidence_reason=conf_reason,
                needs_escalation=False,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section=section_title,
                classification_method="answer_refinement",
                clarification_kind="answer_refinement",
            )
            if debug:
                prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "l2_method": l2.method,
                    "l2_section": l2.section.title if l2.section else None,
                    "l2_best_qa_score": l2.best_qa_score,
                    "l2_best_example_score": l2.best_example_score,
                    "l2_best_complaint_score": l2.best_complaint_score,
                    "llm_prompt": prompt_text,
                    "llm_raw_response": raw_answer,
                    "llm_provider": llm_provider_name,
                    "llm_temperature": llm_temp,
                    "confidence_parsed": confidence,
                    "confidence_reason": conf_reason,
                    "routing_decision": "clarification",
                    "clarification_kind": "answer_refinement",
                    "clarification_attempt": refinement_attempt + 1,
                    "previous_confidence": confidence,
                    "llm_involvement": (
                        "classification+generation" if llm_used_for_classify else "generation"
                    ),
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        if routing_decision is not None and routing_decision.decision == "escalation":
            _total = _time.time() - _start
            resp = RAGResponse(
                answer="По данному вопросу недостаточно уверенности для автоматического ответа. Передаю ваше обращение оператору.",
                confidence=confidence,
                confidence_reason=conf_reason,
                needs_escalation=True,
                detected_reason=reason.id,
                detected_reason_name=reason.name,
                thematic_section=section_title,
                classification_method="answer_refinement_escalation",
            )
            if debug:
                prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
                resp.debug_trace = {
                    "l1_method": l1_method,
                    "l1_confident": l1_confident,
                    "l1_reason": reason.name,
                    "l1_reason_id": reason.id,
                    "l1_candidates": l1_candidates_data,
                    "escalation_check": escalation_check,
                    "l2_method": l2.method,
                    "l2_section": l2.section.title if l2.section else None,
                    "l2_best_qa_score": l2.best_qa_score,
                    "l2_best_example_score": l2.best_example_score,
                    "l2_best_complaint_score": l2.best_complaint_score,
                    "llm_prompt": prompt_text,
                    "llm_raw_response": raw_answer,
                    "llm_provider": llm_provider_name,
                    "llm_temperature": llm_temp,
                    "confidence_parsed": confidence,
                    "confidence_reason": conf_reason,
                    "routing_decision": "escalation",
                    "clarification_kind": "answer_refinement",
                    "clarification_attempt": refinement_attempt,
                    "previous_confidence": confidence,
                    "llm_involvement": (
                        "classification+generation" if llm_used_for_classify else "generation"
                    ),
                    "processing_time_ms": int(_total * 1000),
                }
            return resp

        needs_escalation = confidence < settings.rag_confidence_threshold if routing_decision is None else False
        if not needs_escalation:
            clean_answer = _strip_operator_footer(clean_answer)

        _total = _time.time() - _start
        logger.info(
            f"[ENGINE] DONE | conf={confidence:.2f} | escalation={needs_escalation} | "
            f"time={_total:.1f}s | method=L1:{l1_method}/L2:{l2.method}"
        )

        # Определяем степень участия LLM
        if llm_used_for_classify and llm_used_for_generate:
            llm_involvement = "classification+generation"
        elif llm_used_for_generate:
            llm_involvement = "generation"
        elif llm_used_for_classify:
            llm_involvement = "classification_only"
        else:
            llm_involvement = "none"

        resp = RAGResponse(
            answer=clean_answer,
            confidence=confidence,
            confidence_reason=conf_reason,
            needs_escalation=needs_escalation,
            detected_reason=reason.id,
            detected_reason_name=reason.name,
            thematic_section=section_title,
            classification_method=f"L1:{l1_method}/L2:{l2.method}",
            files=resolve_file_codes(l2.best_example.file_codes)
            if l2.best_example and l2.best_example.file_codes
            else [],
        )

        if debug:
            # Формируем промпт как текст для отображения
            prompt_text = "\n\n".join(f"[{m['role']}]\n{m.get('text', m.get('content', ''))}" for m in messages)
            resp.debug_trace = {
                "l1_method": l1_method,
                "l1_confident": l1_confident,
                "l1_reason": reason.name,
                "l1_reason_id": reason.id,
                "l1_candidates": l1_candidates_data,
                "escalation_check": escalation_check,
                "l2_method": l2.method,
                "l2_section": l2.section.title if l2.section else None,
                "l2_best_qa_score": l2.best_qa_score,
                "l2_best_example_score": l2.best_example_score,
                "l2_best_complaint_score": l2.best_complaint_score,
                "llm_prompt": prompt_text,
                "llm_raw_response": raw_answer,
                "llm_provider": llm_provider_name,
                "llm_temperature": llm_temp,
                "confidence_parsed": confidence,
                "confidence_reason": conf_reason,
                "routing_decision": routing_decision.decision if routing_decision is not None else None,
                "clarification_attempt": refinement_attempt if routing_decision is not None else None,
                "previous_confidence": confidence if routing_decision is not None else None,
                "llm_involvement": llm_involvement,
                "processing_time_ms": int(_total * 1000),
            }

        return resp

    async def _llm_classify_reason(self, question: str, l1: L1Result) -> L1Result:
        """LLM-классификация причины обращения при неоднозначности."""
        top_candidates = l1.candidates[:5]
        candidates_text = "\n".join(
            f"{i + 1}. {c.reason.name} (маркеры: nouns={c.noun_matches}, verbs={c.verb_matches})"
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
            await self._ensure_llm_client()
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
                        winning_candidate=chosen,
                        marker_weights=l1.marker_weights,
                        method="llm",
                    )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return l1  # Не удалось — возвращаем как есть

    def _build_clarification_response(self, question: str, l1: L1Result) -> RAGResponse:
        """Сформировать ответ-уточнение с вариантами причин."""
        top_candidates = l1.candidates[:5]
        options = "\n".join(f"{i + 1}. {c.reason.name}" for i, c in enumerate(top_candidates))
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
        Включает dry-run: показывает пороги, обязательные маркеры и вердикт.
        """
        l1 = classify_reason(question)
        cls_settings = get_classification_settings()

        result = {
            "query": question,
            "l1_method": l1.method,
            "l1_confident": l1.is_confident,
            "l1_reason": l1.reason.name if l1.reason else None,
            "l1_reason_id": l1.reason.id if l1.reason else None,
            "marker_weights": l1.marker_weights,
            "global_min_score": cls_settings.get("l1_global_min_score", 5.0),
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

        # ── Dry-run: проверка порогов и обязательных маркеров ──
        dry_run: dict = {"verdict": "pass"}

        if l1.method == "below_threshold":
            top_score = l1.winning_candidate.score if l1.winning_candidate else 0.0
            dry_run["verdict"] = "escalation"
            dry_run["reason"] = f"score={top_score:.1f} < global_min={cls_settings.get('l1_global_min_score', 5.0):.1f}"
        elif l1.method == "none":
            dry_run["verdict"] = "escalation"
            dry_run["reason"] = "no matches"

        if l1.reason:
            cls_rules = l1.reason.classification_rules
            result["per_reason_min_score"] = cls_rules.min_score_threshold
            result["classification_rules_enabled"] = cls_rules.enabled

            if cls_rules.enabled and l1.winning_candidate:
                # Per-reason threshold check
                if (
                    cls_rules.min_score_threshold is not None
                    and l1.winning_candidate.score < cls_rules.min_score_threshold
                ):
                    dry_run["verdict"] = "escalation"
                    dry_run["reason"] = (
                        f"score={l1.winning_candidate.score:.1f} < per_reason_min={cls_rules.min_score_threshold:.1f}"
                    )

                # Required markers check
                if cls_rules.required_markers and dry_run["verdict"] != "escalation":
                    check = self._check_required_markers(l1.winning_candidate, cls_rules)
                    result["required_markers_check"] = check
                    if not check["passed"]:
                        dry_run["verdict"] = "marker_clarification"
                        dry_run["reason"] = f"missing markers: {check['missing']}"

            l2 = classify_section(question, l1.reason)
            result["l2_method"] = l2.method
            result["l2_section"] = l2.section.title if l2.section else None
            result["l2_best_qa_score"] = l2.best_qa_score
            result["l2_best_qa"] = l2.best_qa.question if l2.best_qa else None
            result["l2_best_example_score"] = l2.best_example_score
            result["l2_best_example"] = (
                l2.best_example.user_questions[0]
                if l2.best_example and l2.best_example.user_questions
                else (l2.best_example.user_question if l2.best_example else None)
            )

        result["dry_run"] = dry_run
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


def _strip_markdown(text: str) -> str:
    """Убрать Markdown-разметку, оставив чистый текст."""
    # Блоки кода ``` ... ```
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # Inline-код
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Заголовки
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Жирный + курсив (***text*** или ___text___)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Жирный (**text** или __text__)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Курсив (*text* или _text_) — только если окружён пробелами/началом/концом
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\1", text)
    # Ссылки [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Изображения ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Горизонтальные линии
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Blockquote
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text


# ── Singleton ──

_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


async def close_rag_engine() -> None:
    """Закрыть HTTP-клиент RAG-движка при завершении."""
    global _engine
    if _engine is not None:
        if _engine.llm is not None:
            await _engine.llm.close()
        _engine = None
