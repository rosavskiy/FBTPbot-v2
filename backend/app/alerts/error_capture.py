"""
Перехват ошибок приложения для оповещений.

logging.Handler уровня ERROR пишет события строками JSON в общий файл
./data/error_events.jsonl (append-only). Ставится в обоих процессах (бэкенд и бот).
Монитор оповещений дочитывает файл с сохранённого byte-offset и считает новые ошибки.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.config import SARATOV_TZ

ERROR_EVENTS_PATH = Path("./data/error_events.jsonl")

# Чтобы не зациклить логирование (ошибка записи -> новый ERROR -> ...)
_LOGGER_BLOCKLIST = {"app.alerts.error_capture", "app.alerts.monitor"}


class JsonlErrorHandler(logging.Handler):
    """Пишет ERROR-записи (и выше) строкой JSON в общий файл."""

    def __init__(self, path: Path = ERROR_EVENTS_PATH) -> None:
        super().__init__(level=logging.ERROR)
        self.path = path
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR:
            return
        if record.name in _LOGGER_BLOCKLIST:
            return
        try:
            entry = {
                "ts": datetime.now(SARATOV_TZ).isoformat(),
                "logger": record.name,
                "message": record.getMessage()[:500],
            }
            line = json.dumps(entry, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            # Никогда не роняем приложение из-за логирования
            pass


_installed = False


def install_error_capture() -> None:
    """Подключить обработчик к корневому логгеру (идемпотентно)."""
    global _installed
    if _installed:
        return
    root = logging.getLogger()
    root.addHandler(JsonlErrorHandler())
    _installed = True
