"""CLI-скрипт для индексации базы знаний."""

import logging
import sys
from pathlib import Path

# Добавляем корень backend в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexer.knowledge_base import get_indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Индексация базы знаний Фармбазис ===")

    indexer = get_indexer()
    total_chunks = indexer.index_instructions()

    logger.info(f"=== Готово! Проиндексировано {total_chunks} чанков ===")


if __name__ == "__main__":
    main()
