"""Pydantic-модели для API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# === Уровни уверенности ===


class ConfidenceLevel(str, Enum):
    """Категории уверенности ответа (4 уровня)."""

    high = "high"  # >= 0.8 — уверенный ответ
    acceptable = "acceptable"  # >= 0.6 — приемлемый
    partial = "partial"  # >= 0.3 — частичный
    escalation = "escalation"  # <  0.3 — эскалация


CONFIDENCE_LABELS = {
    ConfidenceLevel.high: "Уверенный ответ",
    ConfidenceLevel.acceptable: "Приемлемый ответ",
    ConfidenceLevel.partial: "Частичный ответ",
    ConfidenceLevel.escalation: "Требуется эскалация",
}


def compute_confidence_level(confidence: float) -> ConfidenceLevel:
    """Определяет категорию уверенности по числовому значению."""
    if confidence >= 0.8:
        return ConfidenceLevel.high
    elif confidence >= 0.6:
        return ConfidenceLevel.acceptable
    elif confidence >= 0.3:
        return ConfidenceLevel.partial
    else:
        return ConfidenceLevel.escalation


def compute_confidence_label(confidence: float) -> str:
    """Возвращает человекочитаемое описание уровня уверенности."""
    return CONFIDENCE_LABELS[compute_confidence_level(confidence)]


# === Чат ===


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Сообщение пользователя")
    session_id: str | None = Field(None, description="ID сессии (для продолжения диалога)")
    debug: bool = Field(False, description="Включить debug-трейсинг pipeline")


class SuggestedTopicSchema(BaseModel):
    title: str = Field(..., description="Название темы")
    article_id: str = Field(..., description="ID статьи")
    score: float = Field(0.0, description="Релевантность (0-1)")
    snippet: str = Field("", description="Краткий фрагмент текста")


class DebugTrace(BaseModel):
    """Полный trace pipeline L1→L1.5→L2→L3 для debug-чата."""

    l1_method: str | None = Field(None, description="Метод L1-классификации")
    l1_confident: bool = Field(False, description="Уверенность L1")
    l1_reason: str | None = Field(None, description="Определённая причина")
    l1_reason_id: str | None = Field(None, description="ID причины")
    l1_candidates: list[dict] = Field(default_factory=list, description="Кандидаты L1 с scores")
    escalation_check: dict | None = Field(None, description="Результат проверки L1.5 эскалации")
    l2_method: str | None = Field(None, description="Метод L2-классификации")
    l2_section: str | None = Field(None, description="Найденный раздел")
    l2_best_qa_score: float | None = Field(None, description="Лучший QA score")
    l2_best_example_score: float | None = Field(None, description="Лучший example score")
    l2_best_complaint_score: float | None = Field(None, description="Лучший complaint score")
    llm_prompt: str | None = Field(None, description="Полный промпт отправленный LLM")
    llm_raw_response: str | None = Field(None, description="Сырой ответ LLM")
    llm_provider: str | None = Field(None, description="Использованный провайдер")
    llm_temperature: float | None = Field(None, description="Использованная температура")
    confidence_parsed: float | None = Field(None, description="Распарсенный confidence")
    confidence_reason: str | None = Field(None, description="Причина confidence")
    llm_involvement: str = Field(
        "none", description="Степень участия LLM: none / classification_only / generation / classification+generation"
    )
    processing_time_ms: int = Field(0, description="Время обработки в мс")


class ImageData(BaseModel):
    """Изображение в формате base64 data URI."""

    code: str = Field(..., description="Код изображения")
    data_uri: str = Field(..., description="Base64 data URI (data:image/...;base64,...)")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Ответ бота")
    session_id: str = Field(..., description="ID сессии")
    confidence: float = Field(..., description="Уровень уверенности (0-1)")
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Категория уверенности: high (>=0.8), acceptable (>=0.6), partial (>=0.3), escalation (<0.3)",
    )
    confidence_label: str = Field(
        ...,
        description="Описание уровня уверенности на русском языке",
    )
    needs_escalation: bool = Field(False, description="Требуется ли помощь оператора")
    source_articles: list[str] = Field(default_factory=list, description="ID статей-источников")
    youtube_links: list[str] = Field(default_factory=list, description="YouTube ссылки")
    has_images: bool = Field(False, description="Есть ли скриншоты в источниках")
    images: list[ImageData] = Field(default_factory=list, description="Изображения в формате base64 data URI")
    response_type: str = Field("answer", description="Тип ответа: answer | clarification")
    suggested_topics: list[SuggestedTopicSchema] | None = Field(
        None, description="Предложенные темы для уточнения (при response_type=clarification)"
    )
    detected_reason: str | None = Field(None, description="Определённая причина обращения (L1)")
    thematic_section: str | None = Field(None, description="Тематический раздел (L2)")
    llm_provider: str | None = Field(None, description="Активный LLM-провайдер")
    llm_model: str | None = Field(None, description="Активная LLM-модель")
    llm_label: str | None = Field(None, description="Человекочитаемая метка активной LLM")
    show_llm_in_chat: bool = Field(False, description="Показывать ли активную LLM в интерфейсе чата")
    debug_trace: DebugTrace | None = Field(None, description="Debug-трейс pipeline (только при debug=true)")


# === Эскалация ===


class EscalationStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"


class EscalationRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии чата")
    reason: str | None = Field(None, description="Причина эскалации от пользователя")
    contact_info: str | None = Field(None, description="Контактные данные (email/телефон)")


class EscalationResponse(BaseModel):
    escalation_id: str
    status: str = "pending"
    message: str = "Запрос передан оператору."
    position_in_queue: int = 0


class EscalationDetail(BaseModel):
    escalation_id: str
    session_id: str
    status: EscalationStatus
    reason: str | None = None
    contact_info: str | None = None
    chat_history: list[ChatMessage] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None
    operator_notes: str | None = None


# === Панель оператора ===


class OperatorLoginRequest(BaseModel):
    username: str
    password: str


class OperatorLoginResponse(BaseModel):
    token: str
    username: str


class OperatorReplyRequest(BaseModel):
    escalation_id: str
    message: str
    close_ticket: bool = False


class EscalationListResponse(BaseModel):
    escalations: list[EscalationDetail]
    total: int
    pending_count: int


# === Обратная связь ===


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int = Field(0, description="Индекс сообщения")
    rating: int = Field(..., ge=1, le=5, description="Оценка 1-5")
    comment: str | None = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    success: bool = True
    message: str = "Спасибо за обратную связь!"


# === Система ===


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    knowledge_base_ready: bool = False
    total_articles: int = 0
    total_chunks: int = 0
    support_tickets_count: int = 0
