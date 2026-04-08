"""JSON-хранилище причин обращения (Contact Reasons).

Паттерн: аналогично _load_kb()/_save_kb() из kb_admin.py.
Поддерживает автоматические бэкапы при сохранении.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock

from app.config import settings
from app.models.reason_schemas import ContactReason, ContactReasonsData, GlobalEscalationRules

logger = logging.getLogger(__name__)

_STORE_LOCK = Lock()
_MAX_BACKUPS = 20

# Кэш в памяти
_cached_data: ContactReasonsData | None = None


def _get_path() -> Path:
    return Path(settings.contact_reasons_path)


def load_reasons() -> ContactReasonsData:
    """Загрузить причины обращения из JSON-файла."""
    global _cached_data
    path = _get_path()

    if not path.exists():
        _cached_data = ContactReasonsData()
        return _cached_data

    with _STORE_LOCK:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _cached_data = ContactReasonsData.model_validate(raw)

    return _cached_data


def save_reasons(data: ContactReasonsData, backup: bool = True) -> None:
    """Сохранить причины обращения в JSON-файл с бэкапом."""
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

    logger.info(f"Сохранено {len(data.reasons)} причин обращения → {path}")


def _make_backup(path: Path) -> None:
    """Создать бэкап JSON-файла (макс. _MAX_BACKUPS)."""
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{ts}{path.suffix}"
    shutil.copy2(path, backup_path)

    # Ротация: удаляем старые бэкапы
    backups = sorted(backup_dir.glob(f"{path.stem}_*{path.suffix}"))
    while len(backups) > _MAX_BACKUPS:
        backups.pop(0).unlink()


def get_reason(reason_id: str) -> ContactReason | None:
    """Получить причину по ID."""
    data = get_cached_or_load()
    for r in data.reasons:
        if r.id == reason_id:
            return r
    return None


def get_all_reasons(active_only: bool = True) -> list[ContactReason]:
    """Список всех причин (опционально — только активных)."""
    data = get_cached_or_load()
    if active_only:
        return [r for r in data.reasons if r.is_active]
    return list(data.reasons)


def upsert_reason(reason: ContactReason) -> None:
    """Создать или обновить причину."""
    data = get_cached_or_load()
    for i, r in enumerate(data.reasons):
        if r.id == reason.id:
            data.reasons[i] = reason
            save_reasons(data)
            return
    data.reasons.append(reason)
    save_reasons(data)


def delete_reason(reason_id: str) -> bool:
    """Удалить причину по ID. Возвращает True если удалена."""
    data = get_cached_or_load()
    original_len = len(data.reasons)
    data.reasons = [r for r in data.reasons if r.id != reason_id]
    if len(data.reasons) < original_len:
        save_reasons(data)
        return True
    return False


def get_cached_or_load() -> ContactReasonsData:
    """Получить данные из кэша или загрузить."""
    global _cached_data
    if _cached_data is None:
        return load_reasons()
    return _cached_data


def invalidate_cache() -> None:
    """Сбросить кэш (для reload после внешних изменений)."""
    global _cached_data
    _cached_data = None


def get_global_escalation() -> GlobalEscalationRules:
    """Получить глобальные правила эскалации (L0)."""
    data = get_cached_or_load()
    return data.global_escalation


def save_global_escalation(rules: GlobalEscalationRules) -> None:
    """Сохранить глобальные правила эскалации (L0)."""
    data = get_cached_or_load()
    data.global_escalation = rules
    save_reasons(data)
