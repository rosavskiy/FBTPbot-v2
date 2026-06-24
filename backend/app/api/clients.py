"""
API справочника клиентов и ограничений доступа (интеграция с Фармбазисом).

CRUD клиентов и групп + управление ограничениями (denied_reasons / denied_sections).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin_auth import log_action, require_role, verify_admin_token
from app.clients.store import (
    delete_client,
    delete_group,
    get_all_clients,
    get_all_groups,
    get_client,
    get_group,
    upsert_client,
    upsert_group,
)
from app.database.models import AdminUser
from app.database.models import get_db as get_admin_db
from app.database.reason_store import get_all_reasons
from app.models.client_schemas import Client, ClientGroup, ClientRestrictions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clients", tags=["clients"])

# Auth dependencies
_any_admin = Depends(verify_admin_token)
_editor = Depends(require_role("superadmin", "admin"))


# ── Request/Response models ──


class ClientItem(BaseModel):
    customer_id: str
    name: str = ""
    group_id: str | None = None
    group_name: str | None = None
    auto_added: bool = False
    created_at: str = ""
    denied_reasons: list[str] = Field(default_factory=list)
    denied_sections: list[str] = Field(default_factory=list)
    denied_reasons_count: int = 0
    denied_sections_count: int = 0


class ClientsListResponse(BaseModel):
    total: int
    clients: list[ClientItem]


class GroupsListResponse(BaseModel):
    total: int
    groups: list[ClientGroup]


class ClientUpsertPayload(BaseModel):
    name: str = Field("", max_length=300)
    group_id: str | None = None


class GroupUpsertPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)


class SectionNode(BaseModel):
    section_id: str
    title: str


class ReasonNode(BaseModel):
    reason_id: str
    reason_name: str
    sections: list[SectionNode]


class SectionsTreeResponse(BaseModel):
    reasons: list[ReasonNode]


def _client_to_item(c: Client, groups_by_id: dict[str, ClientGroup]) -> ClientItem:
    group = groups_by_id.get(c.group_id) if c.group_id else None
    return ClientItem(
        customer_id=c.customer_id,
        name=c.name,
        group_id=c.group_id,
        group_name=group.name if group else None,
        auto_added=c.auto_added,
        created_at=c.created_at,
        denied_reasons=list(c.restrictions.denied_reasons),
        denied_sections=list(c.restrictions.denied_sections),
        denied_reasons_count=len(c.restrictions.denied_reasons),
        denied_sections_count=len(c.restrictions.denied_sections),
    )


# ── Клиенты ──


@router.get("", response_model=ClientsListResponse)
async def list_clients(user: AdminUser = _any_admin):
    """Список клиентов справочника."""
    groups_by_id = {g.id: g for g in get_all_groups()}
    items = [_client_to_item(c, groups_by_id) for c in get_all_clients()]
    return ClientsListResponse(total=len(items), clients=items)


@router.post("/clients/{customer_id}", response_model=Client)
async def create_client(
    customer_id: str,
    payload: ClientUpsertPayload,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Создать клиента вручную."""
    if get_client(customer_id):
        raise HTTPException(status_code=409, detail=f"Клиент '{customer_id}' уже существует")
    if payload.group_id and not get_group(payload.group_id):
        raise HTTPException(status_code=404, detail="Группа не найдена")
    client = Client(
        customer_id=customer_id,
        name=payload.name,
        group_id=payload.group_id,
        auto_added=False,
    )
    upsert_client(client)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="create",
        entity_type="client",
        entity_id=customer_id,
        entity_name=payload.name,
    )
    return client


@router.put("/clients/{customer_id}", response_model=Client)
async def update_client(
    customer_id: str,
    payload: ClientUpsertPayload,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Изменить имя/группу клиента."""
    client = get_client(customer_id)
    if not client:
        raise HTTPException(status_code=404, detail="Клиент не найден")
    if payload.group_id and not get_group(payload.group_id):
        raise HTTPException(status_code=404, detail="Группа не найдена")
    client.name = payload.name
    client.group_id = payload.group_id
    upsert_client(client)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="client",
        entity_id=customer_id,
        entity_name=payload.name,
    )
    return client


@router.delete("/clients/{customer_id}")
async def remove_client(
    customer_id: str,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Удалить клиента."""
    if not delete_client(customer_id):
        raise HTTPException(status_code=404, detail="Клиент не найден")
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="delete",
        entity_type="client",
        entity_id=customer_id,
    )
    return {"status": "deleted", "id": customer_id}


@router.put("/clients/{customer_id}/restrictions", response_model=Client)
async def update_client_restrictions(
    customer_id: str,
    restrictions: ClientRestrictions,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Сохранить ограничения клиента (denied_reasons / denied_sections)."""
    client = get_client(customer_id)
    if not client:
        raise HTTPException(status_code=404, detail="Клиент не найден")
    client.restrictions = restrictions
    upsert_client(client)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="client_restrictions",
        entity_id=customer_id,
        entity_name=client.name,
        details=f"reasons={len(restrictions.denied_reasons)}, sections={len(restrictions.denied_sections)}",
    )
    return client


# ── Группы ──


@router.get("/groups", response_model=GroupsListResponse)
async def list_groups(user: AdminUser = _any_admin):
    """Список групп клиентов."""
    groups = get_all_groups()
    return GroupsListResponse(total=len(groups), groups=groups)


@router.post("/groups/{group_id}", response_model=ClientGroup)
async def create_group(
    group_id: str,
    payload: GroupUpsertPayload,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Создать группу клиентов."""
    if get_group(group_id):
        raise HTTPException(status_code=409, detail=f"Группа '{group_id}' уже существует")
    group = ClientGroup(id=group_id, name=payload.name)
    upsert_group(group)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="create",
        entity_type="client_group",
        entity_id=group_id,
        entity_name=payload.name,
    )
    return group


@router.put("/groups/{group_id}", response_model=ClientGroup)
async def update_group(
    group_id: str,
    payload: GroupUpsertPayload,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Переименовать группу."""
    group = get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Группа не найдена")
    group.name = payload.name
    upsert_group(group)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="client_group",
        entity_id=group_id,
        entity_name=payload.name,
    )
    return group


@router.delete("/groups/{group_id}")
async def remove_group(
    group_id: str,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Удалить группу (привязанные клиенты остаются, group_id сбрасывается)."""
    if not delete_group(group_id):
        raise HTTPException(status_code=404, detail="Группа не найдена")
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="delete",
        entity_type="client_group",
        entity_id=group_id,
    )
    return {"status": "deleted", "id": group_id}


@router.put("/groups/{group_id}/restrictions", response_model=ClientGroup)
async def update_group_restrictions(
    group_id: str,
    restrictions: ClientRestrictions,
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Сохранить ограничения группы."""
    group = get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Группа не найдена")
    group.restrictions = restrictions
    upsert_group(group)
    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="update",
        entity_type="group_restrictions",
        entity_id=group_id,
        entity_name=group.name,
        details=f"reasons={len(restrictions.denied_reasons)}, sections={len(restrictions.denied_sections)}",
    )
    return group


# ── Дерево причин → разделов для UI ──


@router.get("/sections-tree", response_model=SectionsTreeResponse)
async def sections_tree(user: AdminUser = _any_admin):
    """Дерево «причина → разделы» для модалки ограничений."""
    reasons = []
    for r in get_all_reasons(active_only=False):
        sections = [SectionNode(section_id=s.id, title=s.title) for s in r.thematic_sections]
        reasons.append(ReasonNode(reason_id=r.id, reason_name=r.name, sections=sections))
    return SectionsTreeResponse(reasons=reasons)
