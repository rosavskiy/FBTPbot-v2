"""Pydantic-модели справочника клиентов (интеграция с Фармбазисом).

CustomerID контрагента приходит в POST /api/chat. На его основе ведётся
справочник клиентов, группы и ограничение доступа к причинам (L1) и
тематическим разделам (L2).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.config import SARATOV_TZ


def _now_iso() -> str:
    return datetime.now(SARATOV_TZ).isoformat()


class ClientRestrictions(BaseModel):
    """Ограничения доступа: запрет причин (L1) и разделов (L2)."""

    denied_reasons: list[str] = Field(default_factory=list, description="Запрещённые причины целиком (reason_id)")
    denied_sections: list[str] = Field(
        default_factory=list, description="Запрещённые разделы, ключ '<reason_id>::<section_id>'"
    )


class Client(BaseModel):
    """Контрагент из Фармбазиса."""

    customer_id: str = Field(..., description="CustomerID контрагента")
    name: str = Field("", description="Имя контрагента")
    group_id: str | None = Field(None, description="ID группы, к которой привязан клиент")
    auto_added: bool = Field(False, description="Добавлен автоматически из входящего запроса")
    created_at: str = Field(default_factory=_now_iso, description="Время появления в справочнике (ISO)")
    restrictions: ClientRestrictions = Field(default_factory=ClientRestrictions)


class ClientGroup(BaseModel):
    """Группа клиентов с общими ограничениями."""

    id: str = Field(..., description="Уникальный ID группы")
    name: str = Field(..., description="Название группы")
    restrictions: ClientRestrictions = Field(default_factory=ClientRestrictions)


class ClientsDirectory(BaseModel):
    """Корневая структура хранилища справочника клиентов."""

    version: str = Field("1.0", description="Версия формата")
    clients: list[Client] = Field(default_factory=list)
    groups: list[ClientGroup] = Field(default_factory=list)
