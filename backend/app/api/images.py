"""API для управления файлами (File Management).

Загрузка, список, удаление файлов с уникальными кодами.
Используется для привязки файлов (изображений, документов) к ExampleQA через file_codes.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin_auth import log_action, require_role, verify_admin_token
from app.database.models import AdminUser
from app.database.models import get_db as get_admin_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot-config/images", tags=["images"])

# Auth dependencies
_any_admin = Depends(verify_admin_token)
_editor = Depends(require_role("superadmin", "admin"))

# Configurable paths
IMAGES_DIR = Path("./data/bot_images")
METADATA_FILE = IMAGES_DIR / "metadata.json"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


class FileInfo(BaseModel):
    code: str
    original_name: str
    stored_as: str
    url: str
    ext: str = ""


class FilesListResponse(BaseModel):
    files: list[FileInfo] = Field(default_factory=list)


def _ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> list[dict]:
    _ensure_dirs()
    if not METADATA_FILE.exists():
        return []
    try:
        data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        return data.get("images", [])
    except Exception:
        return []


def _save_metadata(images: list[dict]):
    _ensure_dirs()
    METADATA_FILE.write_text(
        json.dumps({"images": images}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _next_code() -> str:
    """Generate a unique GUID code."""
    return str(uuid.uuid4())


def resolve_file_codes(codes: list[str]) -> list[dict]:
    """Resolve file codes to base64 data URIs, file paths and extensions.

    Returns list of dicts: {"code", "data_uri", "file_path", "ext"}.
    Skips codes that don't exist or can't be read.
    """
    if not codes:
        return []

    files_meta = _load_metadata()
    code_to_meta = {f["code"]: f for f in files_meta}
    result: list[dict] = []

    for code in codes:
        meta = code_to_meta.get(code)
        if not meta:
            logger.warning(f"File code '{code}' not found in metadata")
            continue

        stored_as = meta.get("stored_as", "")
        file_path = IMAGES_DIR / stored_as
        if not file_path.is_file():
            logger.warning(f"File not found: {file_path}")
            continue

        try:
            raw = file_path.read_bytes()
            ext = file_path.suffix.lower().lstrip(".")
            mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            b64 = base64.b64encode(raw).decode("ascii")
            result.append(
                {
                    "code": code,
                    "data_uri": f"data:{mime};base64,{b64}",
                    "file_path": str(file_path.resolve()),
                    "ext": ext,
                }
            )
        except Exception as exc:
            logger.warning(f"Failed to read file '{code}': {exc}")

    return result


@router.get("", response_model=FilesListResponse)
async def list_files(user: AdminUser = _any_admin):
    """Список всех загруженных файлов."""
    files = _load_metadata()
    result = []
    for f in files:
        ext = Path(f.get("stored_as", "")).suffix.lower().lstrip(".")
        result.append(
            FileInfo(
                code=f["code"],
                original_name=f.get("original_name", ""),
                stored_as=f.get("stored_as", ""),
                url=f"/static/bot_images/{f['stored_as']}",
                ext=ext,
            )
        )
    return FilesListResponse(files=result)


@router.post("", response_model=FileInfo)
async def upload_file(
    file: UploadFile = File(...),
    user: AdminUser = _editor,
    db: AsyncSession = Depends(get_admin_db),
):
    """Загрузить файл и присвоить уникальный GUID-код."""
    _ensure_dirs()

    # Validate file name
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла отсутствует")
    ext = Path(file.filename).suffix.lower()

    # Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"Файл слишком большой (макс. {MAX_FILE_SIZE // 1024 // 1024} МБ)")

    images = _load_metadata()

    # Always generate a unique GUID code
    code = _next_code()

    # Sanitize filename — use code as filename to avoid path traversal
    stored_as = f"{code}{ext}"
    dest = IMAGES_DIR / stored_as

    dest.write_bytes(content)

    images.append(
        {
            "code": code,
            "original_name": file.filename,
            "stored_as": stored_as,
        }
    )
    _save_metadata(images)

    logger.info(f"File uploaded: code={code}, file={file.filename}, stored={stored_as}")

    await log_action(
        db,
        user_id=user.id,
        username=user.username,
        action="create",
        entity_type="file",
        entity_id=code,
        entity_name=file.filename,
    )

    return FileInfo(
        code=code,
        original_name=file.filename,
        stored_as=stored_as,
        url=f"/static/bot_images/{stored_as}",
        ext=ext.lstrip("."),
    )


@router.delete("/{image_code}")
async def delete_file(image_code: str, user: AdminUser = _editor, db: AsyncSession = Depends(get_admin_db)):
    """Удалить файл по коду."""
    images = _load_metadata()
    found = None
    for i, img in enumerate(images):
        if img["code"] == image_code:
            found = i
            break
    if found is None:
        raise HTTPException(status_code=404, detail="Файл не найден")

    img = images.pop(found)
    file_path = IMAGES_DIR / img["stored_as"]
    if file_path.exists():
        file_path.unlink()

    _save_metadata(images)
    logger.info(f"File deleted: code={image_code}")

    await log_action(
        db, user_id=user.id, username=user.username, action="delete", entity_type="file", entity_id=image_code
    )

    return {"status": "deleted", "code": image_code}
