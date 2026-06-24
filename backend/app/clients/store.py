"""JSON-хранилище справочника клиентов (Clients Directory).

Паттерн — по образцу reason_store.py: кэш в памяти + Lock + бэкапы при сохранении.
Данные: data/clients_directory.json.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock

from app.config import settings
from app.models.client_schemas import Client, ClientGroup, ClientRestrictions, ClientsDirectory

logger = logging.getLogger(__name__)

_STORE_LOCK = Lock()
_MAX_BACKUPS = 20

# Кэш в памяти
_cached_data: ClientsDirectory | None = None


def _get_path() -> Path:
    return Path(settings.clients_directory_path)


def load_clients() -> ClientsDirectory:
    """Загрузить справочник клиентов из JSON-файла."""
    global _cached_data
    path = _get_path()

    if not path.exists():
        _cached_data = ClientsDirectory()
        return _cached_data

    with _STORE_LOCK:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _cached_data = ClientsDirectory.model_validate(raw)

    return _cached_data


def save_clients(data: ClientsDirectory, backup: bool = True) -> None:
    """Сохранить справочник клиентов в JSON-файл с бэкапом."""
    global _cached_data
    path = _get_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _STORE_LOCK:
        if backup and path.exists():
            _make_backup(path)

        path.write_text(
            json.dumps(data.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _cached_data = data

    logger.info(f"Сохранено {len(data.clients)} клиентов / {len(data.groups)} групп → {path}")


def _make_backup(path: Path) -> None:
    """Создать бэкап JSON-файла (макс. _MAX_BACKUPS)."""
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    from app.config import SARATOV_TZ

    ts = datetime.now(SARATOV_TZ).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{ts}{path.suffix}"
    shutil.copy2(path, backup_path)

    # Ротация: удаляем старые бэкапы
    backups = sorted(backup_dir.glob(f"{path.stem}_*{path.suffix}"))
    while len(backups) > _MAX_BACKUPS:
        backups.pop(0).unlink()


def get_cached_or_load() -> ClientsDirectory:
    """Получить данные из кэша или загрузить."""
    global _cached_data
    if _cached_data is None:
        return load_clients()
    return _cached_data


def invalidate_cache() -> None:
    """Сбросить кэш (для reload после внешних изменений)."""
    global _cached_data
    _cached_data = None


# ── CRUD: клиенты ──


def get_client(customer_id: str) -> Client | None:
    data = get_cached_or_load()
    for c in data.clients:
        if c.customer_id == customer_id:
            return c
    return None


def get_all_clients() -> list[Client]:
    return list(get_cached_or_load().clients)


def upsert_client(client: Client) -> None:
    data = get_cached_or_load()
    for i, c in enumerate(data.clients):
        if c.customer_id == client.customer_id:
            data.clients[i] = client
            save_clients(data)
            return
    data.clients.append(client)
    save_clients(data)


def delete_client(customer_id: str) -> bool:
    data = get_cached_or_load()
    original_len = len(data.clients)
    data.clients = [c for c in data.clients if c.customer_id != customer_id]
    if len(data.clients) < original_len:
        save_clients(data)
        return True
    return False


# ── CRUD: группы ──


def get_group(group_id: str) -> ClientGroup | None:
    data = get_cached_or_load()
    for g in data.groups:
        if g.id == group_id:
            return g
    return None


def get_all_groups() -> list[ClientGroup]:
    return list(get_cached_or_load().groups)


def upsert_group(group: ClientGroup) -> None:
    data = get_cached_or_load()
    for i, g in enumerate(data.groups):
        if g.id == group.id:
            data.groups[i] = group
            save_clients(data)
            return
    data.groups.append(group)
    save_clients(data)


def delete_group(group_id: str) -> bool:
    """Удалить группу. Привязанные клиенты остаются, но group_id сбрасывается."""
    data = get_cached_or_load()
    original_len = len(data.groups)
    data.groups = [g for g in data.groups if g.id != group_id]
    if len(data.groups) == original_len:
        return False
    for c in data.clients:
        if c.group_id == group_id:
            c.group_id = None
    save_clients(data)
    return True


# ── Авторегистрация и резолв ограничений ──


def register_customer(customer_id: str, customer_name: str | None = None) -> Client:
    """Авторегистрация клиента из входящего запроса.

    Новый customer_id попадает в справочник с auto_added=True. Если клиент уже
    есть, но без имени, а в запросе пришло customer_name — дозаполняем имя.
    """
    data = get_cached_or_load()
    for c in data.clients:
        if c.customer_id == customer_id:
            if customer_name and not c.name:
                c.name = customer_name
                save_clients(data)
            return c

    client = Client(
        customer_id=customer_id,
        name=customer_name or "",
        auto_added=True,
    )
    data.clients.append(client)
    save_clients(data)
    logger.info(f"[CLIENTS] Авторегистрация нового клиента customer_id={customer_id}")
    return client


def resolve_denied(customer_id: str, customer_name: str | None = None) -> tuple[set[str], set[str]]:
    """Вернуть объединённые ограничения клиента и его группы.

    Попутно авторегистрирует клиента. Возвращает (denied_reason_ids, denied_section_keys).
    Личные ограничения объединяются (union) с ограничениями группы.
    """
    client = register_customer(customer_id, customer_name)

    denied_reasons: set[str] = set(client.restrictions.denied_reasons)
    denied_sections: set[str] = set(client.restrictions.denied_sections)

    if client.group_id:
        group = get_group(client.group_id)
        if group:
            denied_reasons |= set(group.restrictions.denied_reasons)
            denied_sections |= set(group.restrictions.denied_sections)

    return denied_reasons, denied_sections


__all__ = [
    "Client",
    "ClientGroup",
    "ClientRestrictions",
    "ClientsDirectory",
    "load_clients",
    "save_clients",
    "get_cached_or_load",
    "invalidate_cache",
    "get_client",
    "get_all_clients",
    "upsert_client",
    "delete_client",
    "get_group",
    "get_all_groups",
    "upsert_group",
    "delete_group",
    "register_customer",
    "resolve_denied",
]
