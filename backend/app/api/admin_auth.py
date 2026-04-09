"""
Авторизация администраторов: JWT, управление пользователями, аудит-лог.

Роли: superadmin | admin | viewer
- superadmin: всё + управление пользователями + аудит-лог
- admin: CRUD настроек бота / БЗ
- viewer: только просмотр настроек
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import AdminUser, AuditLog, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

VALID_ROLES = {"superadmin", "admin", "viewer"}


# ── Pydantic-модели ──────────────────────────────────────────────────


class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=255)


class AdminLoginResponse(BaseModel):
    token: str
    user: AdminUserInfo


class AdminUserInfo(BaseModel):
    id: int
    username: str
    display_name: str | None
    role: str
    is_active: bool


class AdminUserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6, max_length=255)
    display_name: str | None = None
    role: str = Field("viewer", pattern=r"^(superadmin|admin|viewer)$")


class AdminUserUpdate(BaseModel):
    display_name: str | None = None
    role: str | None = Field(None, pattern=r"^(superadmin|admin|viewer)$")
    is_active: bool | None = None
    new_password: str | None = Field(None, min_length=6, max_length=255)


class AdminUserListItem(BaseModel):
    id: int
    username: str
    display_name: str | None
    role: str
    is_active: bool
    created_at: str | None
    last_login_at: str | None


class AuditLogEntry(BaseModel):
    id: int
    timestamp: str
    username: str
    action: str
    entity_type: str
    entity_id: str | None
    entity_name: str | None
    details: str | None


class AuditLogResponse(BaseModel):
    items: list[AuditLogEntry]
    total: int
    page: int
    page_size: int


# ── Утилиты ──────────────────────────────────────────────────────────


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _create_jwt(user: AdminUser) -> str:
    secret = settings.admin_jwt_secret
    if not secret:
        raise RuntimeError("ADMIN_JWT_SECRET is not set")
    payload = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "exp": datetime.now(UTC) + timedelta(hours=settings.admin_token_expire_hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _decode_jwt(token: str) -> dict:
    secret = settings.admin_jwt_secret
    if not secret:
        raise RuntimeError("ADMIN_JWT_SECRET is not set")
    return jwt.decode(token, secret, algorithms=["HS256"])


# ── Audit logging ────────────────────────────────────────────────────


async def log_action(
    db: AsyncSession,
    *,
    user_id: int | None,
    username: str,
    action: str,
    entity_type: str,
    entity_id: str | None = None,
    entity_name: str | None = None,
    details: str | None = None,
) -> None:
    entry = AuditLog(
        user_id=user_id,
        username=username,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        entity_name=entity_name,
        details=details,
    )
    db.add(entry)
    await db.commit()


# ── Dependencies ─────────────────────────────────────────────────────


async def verify_admin_token(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> AdminUser:
    """FastAPI Dependency: проверяет JWT, возвращает AdminUser из БД."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    token = authorization.replace("Bearer ", "", 1) if authorization.startswith("Bearer ") else authorization
    try:
        payload = _decode_jwt(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Токен истёк")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Недействительный токен")

    user_id = payload.get("user_id")
    result = await db.execute(select(AdminUser).where(AdminUser.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Учётная запись деактивирована")

    return user


def require_role(*roles: str):
    """Фабрика dependency: проверяет роль пользователя."""

    async def _check(user: AdminUser = Depends(verify_admin_token)) -> AdminUser:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Недостаточно прав")
        return user

    return _check


# ── Инициализация суперадмина ────────────────────────────────────────


async def ensure_superadmin(db: AsyncSession) -> None:
    """Создать суперадмина из env при первом запуске (если его нет)."""
    username = settings.superadmin_username
    password = settings.superadmin_password

    if not username or not password:
        logger.warning("⚠️ SUPERADMIN_USERNAME / SUPERADMIN_PASSWORD не заданы — пропуск создания суперадмина")
        return

    result = await db.execute(select(AdminUser).where(AdminUser.username == username))
    existing = result.scalar_one_or_none()

    if existing:
        logger.info(f"✅ Суперадмин '{username}' уже существует (id={existing.id})")
        return

    user = AdminUser(
        username=username,
        password_hash=_hash_password(password),
        display_name="Суперадминистратор",
        role="superadmin",
        is_active=1,
    )
    db.add(user)
    await db.commit()
    logger.info(f"✅ Суперадмин '{username}' создан автоматически")


# ── Эндпоинты: Логин ────────────────────────────────────────────────


@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(request: AdminLoginRequest, db: AsyncSession = Depends(get_db)):
    """Авторизация администратора."""
    result = await db.execute(select(AdminUser).where(AdminUser.username == request.username))
    user = result.scalar_one_or_none()

    if not user or not _check_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Учётная запись деактивирована")

    # Обновляем время последнего входа
    from app.config import SARATOV_TZ

    user.last_login_at = datetime.now(SARATOV_TZ)
    await db.commit()

    token = _create_jwt(user)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="login",
        entity_type="session",
        details="Вход в админ-панель",
    )

    return AdminLoginResponse(
        token=token,
        user=AdminUserInfo(
            id=user.id,
            username=user.username,
            display_name=user.display_name,
            role=user.role,
            is_active=bool(user.is_active),
        ),
    )


@router.get("/me", response_model=AdminUserInfo)
async def get_current_user(user: AdminUser = Depends(verify_admin_token)):
    """Получить данные текущего пользователя."""
    return AdminUserInfo(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        is_active=bool(user.is_active),
    )


# ── Эндпоинты: Управление пользователями ────────────────────────────


@router.get("/users", response_model=list[AdminUserListItem])
async def list_users(
    user: AdminUser = Depends(require_role("superadmin")),
    db: AsyncSession = Depends(get_db),
):
    """Список всех администраторов (superadmin only)."""
    result = await db.execute(select(AdminUser).order_by(AdminUser.id))
    users = result.scalars().all()

    return [
        AdminUserListItem(
            id=u.id,
            username=u.username,
            display_name=u.display_name,
            role=u.role,
            is_active=bool(u.is_active),
            created_at=u.created_at.isoformat() if u.created_at else None,
            last_login_at=u.last_login_at.isoformat() if u.last_login_at else None,
        )
        for u in users
    ]


@router.post("/users", response_model=AdminUserListItem, status_code=201)
async def create_user(
    payload: AdminUserCreate,
    user: AdminUser = Depends(require_role("superadmin")),
    db: AsyncSession = Depends(get_db),
):
    """Создать нового администратора (superadmin only)."""
    result = await db.execute(select(AdminUser).where(AdminUser.username == payload.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Пользователь '{payload.username}' уже существует")

    new_user = AdminUser(
        username=payload.username,
        password_hash=_hash_password(payload.password),
        display_name=payload.display_name,
        role=payload.role,
        is_active=1,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="create",
        entity_type="user",
        entity_id=str(new_user.id),
        entity_name=new_user.username,
        details=f"Роль: {new_user.role}",
    )

    return AdminUserListItem(
        id=new_user.id,
        username=new_user.username,
        display_name=new_user.display_name,
        role=new_user.role,
        is_active=bool(new_user.is_active),
        created_at=new_user.created_at.isoformat() if new_user.created_at else None,
        last_login_at=None,
    )


@router.put("/users/{user_id}", response_model=AdminUserListItem)
async def update_user(
    user_id: int,
    payload: AdminUserUpdate,
    user: AdminUser = Depends(require_role("superadmin")),
    db: AsyncSession = Depends(get_db),
):
    """Обновить администратора (superadmin only)."""
    result = await db.execute(select(AdminUser).where(AdminUser.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    changes = []

    if payload.display_name is not None:
        target.display_name = payload.display_name
        changes.append(f"display_name → {payload.display_name}")

    if payload.role is not None:
        # Нельзя понизить последнего superadmin
        if target.role == "superadmin" and payload.role != "superadmin":
            count_result = await db.execute(
                select(func.count()).where(AdminUser.role == "superadmin", AdminUser.is_active == 1)
            )
            if count_result.scalar() <= 1:
                raise HTTPException(status_code=400, detail="Нельзя понизить единственного суперадмина")
        target.role = payload.role
        changes.append(f"role → {payload.role}")

    if payload.is_active is not None:
        # Нельзя деактивировать последнего superadmin
        if not payload.is_active and target.role == "superadmin":
            count_result = await db.execute(
                select(func.count()).where(AdminUser.role == "superadmin", AdminUser.is_active == 1)
            )
            if count_result.scalar() <= 1:
                raise HTTPException(status_code=400, detail="Нельзя деактивировать единственного суперадмина")
        target.is_active = 1 if payload.is_active else 0
        changes.append(f"is_active → {payload.is_active}")

    if payload.new_password is not None:
        target.password_hash = _hash_password(payload.new_password)
        changes.append("пароль изменён")

    await db.commit()
    await db.refresh(target)

    if changes:
        await log_action(
            db,
            user_id=user.id,
            username=user.username,
            action="update",
            entity_type="user",
            entity_id=str(target.id),
            entity_name=target.username,
            details="; ".join(changes),
        )

    return AdminUserListItem(
        id=target.id,
        username=target.username,
        display_name=target.display_name,
        role=target.role,
        is_active=bool(target.is_active),
        created_at=target.created_at.isoformat() if target.created_at else None,
        last_login_at=target.last_login_at.isoformat() if target.last_login_at else None,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    user: AdminUser = Depends(require_role("superadmin")),
    db: AsyncSession = Depends(get_db),
):
    """Удалить администратора (superadmin only)."""
    result = await db.execute(select(AdminUser).where(AdminUser.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    if target.id == user.id:
        raise HTTPException(status_code=400, detail="Нельзя удалить самого себя")

    if target.role == "superadmin":
        count_result = await db.execute(
            select(func.count()).where(AdminUser.role == "superadmin", AdminUser.is_active == 1)
        )
        if count_result.scalar() <= 1:
            raise HTTPException(status_code=400, detail="Нельзя удалить единственного суперадмина")

    target_name = target.username
    await db.delete(target)
    await db.commit()

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="delete",
        entity_type="user",
        entity_id=str(user_id),
        entity_name=target_name,
    )

    return {"status": "deleted", "id": user_id}


# ── Эндпоинты: Аудит-лог ────────────────────────────────────────────


@router.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    username: str | None = None,
    action: str | None = None,
    entity_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    user: AdminUser = Depends(require_role("superadmin")),
    db: AsyncSession = Depends(get_db),
):
    """Аудит-лог действий (superadmin only)."""
    query = select(AuditLog)
    count_query = select(func.count()).select_from(AuditLog)

    if username:
        query = query.where(AuditLog.username == username)
        count_query = count_query.where(AuditLog.username == username)
    if action:
        query = query.where(AuditLog.action == action)
        count_query = count_query.where(AuditLog.action == action)
    if entity_type:
        query = query.where(AuditLog.entity_type == entity_type)
        count_query = count_query.where(AuditLog.entity_type == entity_type)
    if date_from:
        try:
            dt = datetime.fromisoformat(date_from)
            query = query.where(AuditLog.timestamp >= dt)
            count_query = count_query.where(AuditLog.timestamp >= dt)
        except ValueError:
            pass
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to)
            query = query.where(AuditLog.timestamp <= dt)
            count_query = count_query.where(AuditLog.timestamp <= dt)
        except ValueError:
            pass

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.order_by(AuditLog.timestamp.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    entries = result.scalars().all()

    return AuditLogResponse(
        items=[
            AuditLogEntry(
                id=e.id,
                timestamp=e.timestamp.isoformat() if e.timestamp else "",
                username=e.username,
                action=e.action,
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                entity_name=e.entity_name,
                details=e.details,
            )
            for e in entries
        ],
        total=total,
        page=page,
        page_size=page_size,
    )
