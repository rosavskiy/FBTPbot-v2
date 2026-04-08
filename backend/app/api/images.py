"""API для управления изображениями (Image Management).

Загрузка, список, удаление изображений с уникальными кодами.
Используется для привязки изображений к ExampleQA через image_codes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot-config/images", tags=["images"])

# Configurable paths
IMAGES_DIR = Path("./data/bot_images")
METADATA_FILE = IMAGES_DIR / "metadata.json"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB


class ImageInfo(BaseModel):
    code: str
    original_name: str
    stored_as: str
    url: str


class ImagesListResponse(BaseModel):
    images: list[ImageInfo] = Field(default_factory=list)


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


def _next_code(images: list[dict]) -> str:
    """Generate next sequential code."""
    existing_codes = set()
    for img in images:
        try:
            existing_codes.add(int(img["code"]))
        except (ValueError, KeyError):
            pass
    code = 1
    while code in existing_codes:
        code += 1
    return str(code)


@router.get("", response_model=ImagesListResponse)
async def list_images():
    """Список всех загруженных изображений."""
    images = _load_metadata()
    result = []
    for img in images:
        result.append(
            ImageInfo(
                code=img["code"],
                original_name=img.get("original_name", ""),
                stored_as=img.get("stored_as", ""),
                url=f"/static/bot_images/{img['stored_as']}",
            )
        )
    return ImagesListResponse(images=result)


@router.post("", response_model=ImageInfo)
async def upload_image(file: UploadFile = File(...), code: str | None = None):
    """Загрузить изображение и присвоить код."""
    _ensure_dirs()

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла отсутствует")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый формат. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"Файл слишком большой (макс. {MAX_FILE_SIZE // 1024 // 1024} МБ)")

    images = _load_metadata()

    # Determine code
    if code:
        # Check for duplicate
        if any(img["code"] == code for img in images):
            raise HTTPException(status_code=409, detail=f"Изображение с кодом '{code}' уже существует")
    else:
        code = _next_code(images)

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

    logger.info(f"Image uploaded: code={code}, file={file.filename}, stored={stored_as}")

    return ImageInfo(
        code=code,
        original_name=file.filename,
        stored_as=stored_as,
        url=f"/static/bot_images/{stored_as}",
    )


@router.delete("/{image_code}")
async def delete_image(image_code: str):
    """Удалить изображение по коду."""
    images = _load_metadata()
    found = None
    for i, img in enumerate(images):
        if img["code"] == image_code:
            found = i
            break
    if found is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    img = images.pop(found)
    file_path = IMAGES_DIR / img["stored_as"]
    if file_path.exists():
        file_path.unlink()

    _save_metadata(images)
    logger.info(f"Image deleted: code={image_code}")

    return {"status": "deleted", "code": image_code}
