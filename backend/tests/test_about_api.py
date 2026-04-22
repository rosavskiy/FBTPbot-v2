from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import about
from app.database.models import AdminUser, AuditLog, Base


def make_admin_user(role: str = "admin") -> AdminUser:
    return AdminUser(
        id=1,
        username=role,
        display_name=role.title(),
        role=role,
        is_active=1,
        password_hash="stub",
    )


async def create_test_context(tmp_path: Path, auth_override):
    db_path = tmp_path / "about_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app = FastAPI()
    app.include_router(about.router)
    app.dependency_overrides[about.get_admin_db] = override_get_db
    app.dependency_overrides[about._editor.dependency] = auth_override

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    return client, engine, session_factory


@pytest.mark.anyio
async def test_about_api_requires_auth(tmp_path: Path):
    def auth_override():
        raise HTTPException(status_code=401, detail="Требуется авторизация")

    client, engine, _ = await create_test_context(tmp_path, auth_override)
    try:
        response = await client.get("/api/about")
        assert response.status_code == 401
    finally:
        await client.aclose()
        await engine.dispose()


@pytest.mark.anyio
async def test_about_api_blocks_viewer(tmp_path: Path):
    def auth_override():
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    client, engine, _ = await create_test_context(tmp_path, auth_override)
    try:
        response = await client.get("/api/about")
        assert response.status_code == 403
    finally:
        await client.aclose()
        await engine.dispose()


@pytest.mark.anyio
async def test_about_api_crud_and_export(tmp_path: Path):
    client, engine, session_factory = await create_test_context(tmp_path, lambda: make_admin_user("admin"))
    try:
        create_response = await client.put(
            "/api/about",
            json={
                "progress_date": "2026-04-22",
                "title": "Доработки AI-консоли",
                "content": "- Добавили страницу О программе\n- Добавили экспорт в DOCX",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["progress_date"] == "2026-04-22"
        assert created["title"] == "Доработки AI-консоли"
        assert "экспорт" in created["content"].lower()

        list_response = await client.get("/api/about")
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["total"] == 1
        assert list_data["items"][0]["id"] == created["id"]

        by_date_response = await client.get("/api/about/by-date/2026-04-22")
        assert by_date_response.status_code == 200
        assert by_date_response.json()["id"] == created["id"]

        update_response = await client.put(
            "/api/about",
            json={
                "progress_date": "2026-04-22",
                "title": "Обновлённый лог",
                "content": "- Перезаписали запись за ту же дату",
            },
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["id"] == created["id"]
        assert updated["title"] == "Обновлённый лог"

        export_response = await client.get(f"/api/about/{created['id']}/export")
        assert export_response.status_code == 200
        assert export_response.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert "attachment;" in export_response.headers["content-disposition"]
        assert len(export_response.content) > 0

        async with session_factory() as session:
            audit_entries = await session.execute(select(AuditLog).where(AuditLog.entity_type == "progress_note"))
            assert len(audit_entries.scalars().all()) == 3

        delete_response = await client.delete(f"/api/about/{created['id']}")
        assert delete_response.status_code == 200

        final_list_response = await client.get("/api/about")
        assert final_list_response.status_code == 200
        assert final_list_response.json()["total"] == 0
    finally:
        await client.aclose()
        await engine.dispose()


@pytest.mark.anyio
async def test_about_export_range(tmp_path):
    client, engine, _ = await create_test_context(tmp_path, lambda: make_admin_user("admin"))
    try:
        # Создаём три записи за разные даты
        for iso_date in ("2026-04-01", "2026-04-15", "2026-04-22"):
            r = await client.put(
                "/api/about",
                json={"progress_date": iso_date, "title": f"Запись {iso_date}", "content": f"- Строка за {iso_date}"},
            )
            assert r.status_code == 200

        # Экспорт всех записей
        all_resp = await client.get("/api/about/export-range")
        assert all_resp.status_code == 200
        assert all_resp.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(all_resp.content) > 0

        # Экспорт за подпериод — должны попасть две записи
        sub_resp = await client.get("/api/about/export-range?date_from=2026-04-10&date_to=2026-04-22")
        assert sub_resp.status_code == 200
        assert len(sub_resp.content) > 0

        # Период без записей — 404
        empty_resp = await client.get("/api/about/export-range?date_from=2025-01-01&date_to=2025-01-31")
        assert empty_resp.status_code == 404
    finally:
        await client.aclose()
        await engine.dispose()
