"""Pydantic-модели для причин обращения (Contact Reasons)."""

from __future__ import annotations

from typing import List, Optional

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
    qa_pairs: List[QAPair] = Field(default_factory=list, description="Вопросы-ответы раздела")


class Complaint(BaseModel):
    """Типовая жалоба с шаблоном ответа."""
    description: str = Field(..., description="Описание жалобы")
    context: str = Field("", description="Контекст / условия возникновения")
    response_template: str = Field(..., description="Шаблон ответа техподдержки")


class ExampleQA(BaseModel):
    """Пример ответа на частый вопрос (L3)."""
    user_question: str = Field(..., description="Типичный вопрос пользователя")
    ideal_answer: str = Field(..., description="Идеальный ответ бота")


class Markers(BaseModel):
    """4 типа маркеров для L1-классификации."""
    verbs: List[str] = Field(
        default_factory=list,
        description="Глаголы-маркеры: списать, уничтожить, удалить..."
    )
    nouns: List[str] = Field(
        default_factory=list,
        description="Существительные-маркеры: акт, списание, SGTIN..."
    )
    numeric_tags: List[str] = Field(
        default_factory=list,
        description="Числовые теги: 11, 52, 552, 541..."
    )
    phrase_masks: List[str] = Field(
        default_factory=list,
        description="Фразовые маски (100% маркеры): 'чек завис на кассе'..."
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
    thematic_sections: List[ThematicSection] = Field(
        default_factory=list, description="Тематические разделы (L2)"
    )
    typical_complaints: List[Complaint] = Field(
        default_factory=list, description="Типовые жалобы с шаблонами ответов"
    )
    example_answers: List[ExampleQA] = Field(
        default_factory=list, description="Примеры готовых ответов"
    )


class ContactReasonsData(BaseModel):
    """Корневая модель хранилища причин обращения."""
    version: str = Field("1.0", description="Версия формата данных")
    reasons: List[ContactReason] = Field(default_factory=list)
