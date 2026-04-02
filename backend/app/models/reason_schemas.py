"""Pydantic-модели для причин обращения (Contact Reasons)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """Пара вопрос-ответ внутри тематического раздела."""

    question: str = Field(..., description="Вопрос пользователя")
    answer: str = Field(..., description="Ответ техподдержки")


class ThematicSection(BaseModel):
    """Тематический раздел внутри причины обращения (L2)."""

    id: str = Field(..., description="Уникальный ID раздела, напр. 'terminologia'")
    title: str = Field(..., description="Название раздела, напр. 'Терминология и основные понятия'")
    order: int = Field(0, description="Порядок отображения")
    qa_pairs: list[QAPair] = Field(default_factory=list, description="Вопросы-ответы раздела")


class Complaint(BaseModel):
    """Типовая жалоба с шаблоном ответа."""

    description: str = Field(..., description="Описание жалобы")
    context: str = Field("", description="Контекст / условия возникновения")
    response_template: str = Field(..., description="Шаблон ответа техподдержки")


class ExampleQA(BaseModel):
    """Пример ответа на частый вопрос (L3)."""

    user_question: str = Field(..., description="Типичный вопрос пользователя")
    ideal_answer: str = Field(..., description="Идеальный ответ бота")


class EscalationQAPair(BaseModel):
    """Пара вопрос-ответ для 100%-эскалации на оператора."""

    question: str = Field(..., description="Паттерн вопроса пользователя")
    answer: str = Field("", description="Шаблон ответа при эскалации (опционально)")


class EscalationMetrics(BaseModel):
    """Метрики для определения 100%-эскалации."""

    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Порог overlap для срабатывания Q&A-эскалации")
    keyword_patterns: list[str] = Field(
        default_factory=list, description="Фразовые маски для 100%-эскалации (точное совпадение)"
    )


class EscalationRules(BaseModel):
    """Правила 100%-эскалации на сотрудника ТП для причины обращения."""

    enabled: bool = Field(False, description="Включены ли правила принудительной эскалации")
    qa_pairs: list[EscalationQAPair] = Field(default_factory=list, description="Q&A-паттерны для эскалации")
    metrics: EscalationMetrics = Field(default_factory=EscalationMetrics, description="Метрики эскалации")


class ClassificationRules(BaseModel):
    """Правила классификации L1 для причины обращения.

    Позволяют задавать per-reason порог баллов и обязательные типы маркеров.
    Если обязательный маркер не найден — бот задаёт уточняющий вопрос.
    """

    enabled: bool = Field(False, description="Включены ли правила классификации для этой причины")
    min_score_threshold: float | None = Field(
        None, ge=0.0, description="Минимальный порог L1-баллов для этой причины (None = глобальный)"
    )
    required_markers: list[str] = Field(
        default_factory=list,
        description="Обязательные типы маркеров: 'numeric_tag', 'phrase_mask', 'noun', 'verb'",
    )
    clarification_text: str = Field(
        "",
        description="Текст уточняющего вопроса (если пусто — стандартный)",
    )


class Markers(BaseModel):
    """4 типа маркеров для L1-классификации."""

    verbs: list[str] = Field(default_factory=list, description="Глаголы-маркеры: списать, уничтожить, удалить...")
    nouns: list[str] = Field(default_factory=list, description="Существительные-маркеры: акт, списание, SGTIN...")
    numeric_tags: list[str] = Field(default_factory=list, description="Числовые теги: 11, 52, 552, 541...")
    phrase_masks: list[str] = Field(
        default_factory=list, description="Фразовые маски (100% маркеры): 'чек завис на кассе'..."
    )


class ContactReason(BaseModel):
    """Причина обращения — основная сущность классификации.

    Содержит маркеры для L1-классификации, тематические разделы (L2),
    типовые жалобы и примеры ответов (L3).
    """

    id: str = Field(..., description="Уникальный ID, напр. 'akt_na_spisanie'")
    name: str = Field(..., description="Название, напр. 'Акт на списание'")
    is_active: bool = Field(True, description="Активна ли причина")
    markers: Markers = Field(default_factory=Markers, description="Маркеры для классификации")
    thematic_sections: list[ThematicSection] = Field(default_factory=list, description="Тематические разделы (L2)")
    typical_complaints: list[Complaint] = Field(default_factory=list, description="Типовые жалобы с шаблонами ответов")
    example_answers: list[ExampleQA] = Field(default_factory=list, description="Примеры готовых ответов")
    escalation_rules: EscalationRules = Field(
        default_factory=EscalationRules, description="Правила 100%-эскалации на ТП"
    )
    classification_rules: ClassificationRules = Field(
        default_factory=ClassificationRules, description="Правила классификации L1 (пороги, обязательные маркеры)"
    )


class ContactReasonsData(BaseModel):
    """Корневая модель хранилища причин обращения."""

    version: str = Field("1.0", description="Версия формата данных")
    reasons: list[ContactReason] = Field(default_factory=list)
