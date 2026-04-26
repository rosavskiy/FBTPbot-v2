#!/usr/bin/env python3
"""
Backfill Google Sheets из SQLite только по достоверным полям.

Скрипт восстанавливает пары вопрос-ответ из chat_messages и заполняет только те
колонки, которые можно надёжно восстановить из БД.

Недоступные в SQLite поля остаются пустыми:
- escalation_info
- detected_reason
- thematic_section
- response_type
- youtube_links
- has_files
- is_debug

Использование:
    python -m scripts.backfill_gsheets --start-after "2026-04-24 09:57:06" --dry-run
    python -m scripts.backfill_gsheets --start-after "2026-04-24 09:57:06" --end-before "2026-04-27 00:06:43.198000" --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from app.config import settings
from app.models.schemas import compute_confidence_label, compute_confidence_level
from app.sheets.gsheet_logger import HEADER_ROW, SARATOV_TZ, SCOPES


@dataclass(slots=True)
class ChatMessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    confidence: float | None
    source_articles: list[str]
    created_at: datetime


@dataclass(slots=True)
class BackfillPair:
    session_id: str
    question: str
    answer: str
    answered_at: datetime
    confidence: float | None
    source_articles: list[str]


def parse_cli_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SARATOV_TZ)
    return dt.astimezone(SARATOV_TZ)


def parse_db_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SARATOV_TZ)
    return dt.astimezone(SARATOV_TZ)


def parse_source_articles(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    return []


def load_messages(db_path: Path, start_after: str, end_before: str | None) -> list[ChatMessageRecord]:
    query = [
        "SELECT id, session_id, role, content, confidence, source_articles, created_at",
        "FROM chat_messages",
        "WHERE created_at > ?",
    ]
    params: list[str] = [start_after]

    if end_before:
        query.append("AND created_at < ?")
        params.append(end_before)

    query.append("ORDER BY created_at ASC, id ASC")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("\n".join(query), params).fetchall()
    finally:
        conn.close()

    return [
        ChatMessageRecord(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            source_articles=parse_source_articles(row["source_articles"]),
            created_at=parse_db_datetime(str(row["created_at"])),
        )
        for row in rows
    ]


def pair_messages(messages: list[ChatMessageRecord]) -> tuple[list[BackfillPair], int, int]:
    pending_users: dict[str, deque[ChatMessageRecord]] = defaultdict(deque)
    pairs: list[BackfillPair] = []
    unmatched_assistants = 0

    for message in messages:
        if message.role == "user":
            pending_users[message.session_id].append(message)
            continue

        if message.role != "assistant":
            continue

        queue = pending_users[message.session_id]
        if not queue:
            unmatched_assistants += 1
            continue

        question = queue.popleft()
        pairs.append(
            BackfillPair(
                session_id=message.session_id,
                question=question.content,
                answer=message.content,
                answered_at=message.created_at,
                confidence=message.confidence,
                source_articles=message.source_articles,
            )
        )

    unmatched_users = sum(len(queue) for queue in pending_users.values())
    return pairs, unmatched_users, unmatched_assistants


def open_sheet() -> gspread.Worksheet:
    creds_path = Path(settings.google_sheets_credentials_file)
    if not creds_path.exists():
        raise FileNotFoundError(f"Файл учётных данных Google не найден: {creds_path}")
    if not settings.google_sheets_spreadsheet_id:
        raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID не задан")

    creds = Credentials.from_service_account_file(str(creds_path), scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(settings.google_sheets_spreadsheet_id)

    try:
        sheet = spreadsheet.worksheet(settings.google_sheets_worksheet)
    except gspread.exceptions.WorksheetNotFound:
        sheet = spreadsheet.sheet1

    first_row = sheet.row_values(1)
    if not first_row or first_row[0] != HEADER_ROW[0]:
        sheet.update("A1", [HEADER_ROW])
    return sheet


def get_existing_key_counter(sheet: gspread.Worksheet) -> Counter[tuple[str, str, str]]:
    keys: Counter[tuple[str, str, str]] = Counter()
    for row in sheet.get_all_values()[1:]:
        if len(row) < 5:
            continue
        key = (row[4].strip(), row[2].strip(), row[3].strip())
        keys[key] += 1
    return keys


def get_next_number(sheet: gspread.Worksheet) -> int:
    max_number = 0
    for row in sheet.get_all_values()[1:]:
        if row and row[0].strip().isdigit():
            max_number = max(max_number, int(row[0].strip()))
    return max_number + 1


def build_sheet_rows(
    pairs: list[BackfillPair],
    existing_keys: Counter[tuple[str, str, str]],
    start_number: int,
) -> tuple[list[list[str | int | float]], int]:
    rows: list[list[str | int | float]] = []
    skipped_duplicates = 0
    row_number = start_number

    for pair in pairs:
        key = (pair.session_id, pair.question.strip(), pair.answer.strip())
        if existing_keys[key] > 0:
            existing_keys[key] -= 1
            skipped_duplicates += 1
            continue

        if pair.confidence is None:
            confidence_cell: str | float = ""
            confidence_level = ""
            confidence_label = ""
            needs_escalation = ""
        else:
            confidence_cell = round(pair.confidence, 4)
            confidence_level = compute_confidence_level(pair.confidence).value
            confidence_label = compute_confidence_label(pair.confidence)
            needs_escalation = "Да" if confidence_level == "escalation" else "Нет"

        rows.append(
            [
                row_number,
                pair.answered_at.astimezone(SARATOV_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                pair.question,
                pair.answer,
                pair.session_id,
                confidence_cell,
                confidence_level,
                confidence_label,
                needs_escalation,
                "",
                ", ".join(pair.source_articles),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        row_number += 1

    return rows, skipped_duplicates


def append_rows(sheet: gspread.Worksheet, rows: list[list[str | int | float]], batch_size: int) -> None:
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        sheet.append_rows(batch, value_input_option="USER_ENTERED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill Google Sheets из SQLite")
    parser.add_argument("--db-path", default="./data/support.db", help="Путь к SQLite базе")
    parser.add_argument("--start-after", required=True, help="Включать сообщения строго позже этого времени")
    parser.add_argument("--end-before", help="Включать сообщения строго раньше этого времени")
    parser.add_argument("--batch-size", type=int, default=200, help="Размер батча для append_rows")
    parser.add_argument("--apply", action="store_true", help="Записать строки в Google Sheets")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite база не найдена: {db_path}")

    start_after = parse_cli_datetime(args.start_after).replace(tzinfo=None).isoformat(sep=" ")
    end_before = None
    if args.end_before:
        end_before = parse_cli_datetime(args.end_before).replace(tzinfo=None).isoformat(sep=" ")

    messages = load_messages(db_path=db_path, start_after=start_after, end_before=end_before)
    pairs, unmatched_users, unmatched_assistants = pair_messages(messages)

    sheet = open_sheet()
    existing_keys = get_existing_key_counter(sheet)
    next_number = get_next_number(sheet)
    rows, skipped_duplicates = build_sheet_rows(pairs, existing_keys, next_number)

    print(f"messages_loaded={len(messages)}")
    print(f"pairs_built={len(pairs)}")
    print(f"rows_to_append={len(rows)}")
    print(f"duplicates_skipped={skipped_duplicates}")
    print(f"unmatched_users={unmatched_users}")
    print(f"unmatched_assistants={unmatched_assistants}")

    if rows:
        print(f"first_row_preview={rows[0][:8]}")
        print(f"last_row_preview={rows[-1][:8]}")

    if not args.apply:
        print("mode=dry-run")
        return 0

    append_rows(sheet, rows, args.batch_size)
    print(f"appended={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())