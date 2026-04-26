"""
Логирование вопросов и ответов в Google Sheets.

Каждая строка — одна пара вопрос-ответ.
Работает асинхронно через asyncio.to_thread (gspread — синхронная библиотека).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from app.config import settings

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

HEADER_ROW = [
    "№",
    "Дата",
    "Вопрос",
    "Ответ",
    "session_id",
    "confidence",
    "confidence_level",
    "confidence_label",
    "needs_escalation",
    "Эскалация",
    "source_articles",
    "detected_reason",
    "thematic_section",
    "response_type",
    "youtube_links",
    "has_files",
    "is_debug",
]

SARATOV_TZ = timezone(timedelta(hours=4))
WRITE_TIMEOUT_SECONDS = 20.0


class GoogleSheetLogger:
    """Синглтон-логгер для записи Q&A в Google Sheets."""

    def __init__(self) -> None:
        self._sheet: gspread.Worksheet | None = None
        self._enabled = False
        self._lock = asyncio.Lock()
        self._next_row_number: int | None = None
        self._write_timeout_seconds = WRITE_TIMEOUT_SECONDS
        self._init_sync()

    # ── Инициализация (синхронная, вызывается один раз) ──

    def _init_sync(self) -> None:
        creds_path = settings.google_sheets_credentials_file
        spreadsheet_id = settings.google_sheets_spreadsheet_id

        if not creds_path or not spreadsheet_id:
            logger.warning(
                "Google Sheets логирование отключено: "
                "не заданы GOOGLE_SHEETS_CREDENTIALS_FILE и/или GOOGLE_SHEETS_SPREADSHEET_ID"
            )
            return

        path = Path(creds_path)
        if not path.exists():
            logger.error(f"Файл учётных данных Google не найден: {creds_path}")
            return

        try:
            creds = Credentials.from_service_account_file(str(path), scopes=SCOPES)
            gc = gspread.authorize(creds)
            spreadsheet = gc.open_by_key(spreadsheet_id)
            sheet_name = settings.google_sheets_worksheet
            try:
                self._sheet = spreadsheet.worksheet(sheet_name)
            except gspread.exceptions.WorksheetNotFound:
                logger.warning(f"Лист '{sheet_name}' не найден, используем первый лист")
                self._sheet = spreadsheet.sheet1
            self._ensure_header()
            self._next_row_number = self._get_next_row_number_sync()
            self._enabled = True
            logger.info(f"✅ Google Sheets логгер подключён: {spreadsheet.title}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Google Sheets: {e}", exc_info=True)

    def _ensure_header(self) -> None:
        """Записывает заголовки, если первая строка пуста."""
        if self._sheet is None:
            return
        first_row = self._sheet.row_values(1)
        if not first_row or first_row[0] != HEADER_ROW[0]:
            self._sheet.update("A1", [HEADER_ROW])
            logger.info("Google Sheets: заголовки записаны")

    def _get_next_row_number_sync(self) -> int:
        """Возвращает следующий номер строки по максимальному значению в колонке №."""
        if self._sheet is None:
            return 1

        max_number = 0
        for row in self._sheet.get_all_values()[1:]:
            if row and row[0].strip().isdigit():
                max_number = max(max_number, int(row[0].strip()))
        return max_number + 1

    async def _reserve_row_number(self) -> int:
        async with self._lock:
            if self._next_row_number is None:
                self._next_row_number = await asyncio.to_thread(self._get_next_row_number_sync)
            row_number = self._next_row_number
            self._next_row_number += 1
            return row_number

    def _build_row(
        self,
        *,
        row_number: int,
        question: str,
        answer: str,
        session_id: str,
        confidence: float,
        confidence_level: str,
        confidence_label: str,
        needs_escalation: bool,
        escalation_info: str,
        source_articles: list[str],
        detected_reason: str,
        thematic_section: str,
        response_type: str,
        youtube_links: list[str],
        has_files: bool,
        is_debug: bool,
        created_at: datetime | None = None,
    ) -> list[str | int | float]:
        created_at = created_at or datetime.now(SARATOV_TZ)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=SARATOV_TZ)
        else:
            created_at = created_at.astimezone(SARATOV_TZ)

        return [
            row_number,
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer,
            session_id,
            round(confidence, 4),
            confidence_level,
            confidence_label,
            "Да" if needs_escalation else "Нет",
            escalation_info,
            ", ".join(source_articles),
            detected_reason,
            thematic_section,
            response_type,
            ", ".join(youtube_links),
            "Да" if has_files else "Нет",
            "debug" if is_debug else "",
        ]

    # ── Публичный асинхронный метод ──

    async def log(
        self,
        question: str,
        answer: str,
        session_id: str = "",
        confidence: float = 0.0,
        confidence_level: str = "",
        confidence_label: str = "",
        needs_escalation: bool = False,
        escalation_info: str = "",
        source_articles: list[str] | None = None,
        detected_reason: str | None = None,
        thematic_section: str | None = None,
        response_type: str = "answer",
        youtube_links: list[str] | None = None,
        has_files: bool = False,
        is_debug: bool = False,
    ) -> None:
        """Добавить строку в таблицу (не блокирует event loop)."""
        if not self._enabled:
            return

        row_number = await self._reserve_row_number()
        row = self._build_row(
            row_number=row_number,
            question=question,
            answer=answer,
            session_id=session_id,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_label=confidence_label,
            needs_escalation=needs_escalation,
            escalation_info=escalation_info,
            source_articles=source_articles or [],
            detected_reason=detected_reason or "",
            thematic_section=thematic_section or "",
            response_type=response_type,
            youtube_links=youtube_links or [],
            has_files=has_files,
            is_debug=is_debug,
        )

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._append_row, row),
                timeout=self._write_timeout_seconds,
            )
        except TimeoutError:
            logger.error(
                "Google Sheets: таймаут записи строки %s для session_id=%s",
                row_number,
                session_id,
            )
        except Exception as e:
            logger.error(f"Ошибка записи в Google Sheets: {e}")

    # ── Синхронная запись (выполняется в потоке) ──

    def _append_row(self, row: list[str | int | float]) -> None:
        if self._sheet is None:
            return
        self._sheet.append_row(row, value_input_option="USER_ENTERED")


# ── Синглтон ──

_logger: GoogleSheetLogger | None = None


def get_gsheet_logger() -> GoogleSheetLogger:
    global _logger
    if _logger is None:
        _logger = GoogleSheetLogger()
    return _logger
