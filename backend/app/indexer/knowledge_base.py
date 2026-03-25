"""
Индексатор базы знаний Фармбазис.

Парсит HTML-инструкции, разбивает на чанки,
создаёт эмбеддинги и записывает в ChromaDB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.parser.html_parser import InstructionParser, ParsedInstruction

logger = logging.getLogger(__name__)

# Названия коллекций в ChromaDB
COLLECTION_NAME = "farmbazis_instructions"
SUPPORT_COLLECTION_NAME = "support_tickets"


class KnowledgeBaseIndexer:
    """Индексатор базы знаний на основе ChromaDB."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )
        self.parser = InstructionParser(
            images_dir=Path(settings.chroma_persist_dir).parent / "images"
        )
        self.vector_store: Optional[Chroma] = None
        self.support_vector_store: Optional[Chroma] = None

    def _instruction_to_documents(
        self, instruction: ParsedInstruction
    ) -> List[Document]:
        """Конвертация ParsedInstruction в список LangChain Document."""
        documents = []

        # Формируем метаданные
        metadata_base: Dict = {
            "article_id": instruction.article_id,
            "title": instruction.title,
            "source_file": instruction.source_file,
            "has_images": len(instruction.images) > 0,
            "image_count": len(instruction.images),
        }

        # Добавляем YouTube ссылки
        if instruction.youtube_links:
            metadata_base["youtube_links"] = json.dumps(instruction.youtube_links)

        # Добавляем информацию об изображениях
        if instruction.images:
            image_info = [
                {"filename": img.filename, "alt": img.alt_text}
                for img in instruction.images
            ]
            metadata_base["images_info"] = json.dumps(image_info, ensure_ascii=False)

        # Формируем обогащённый текст для индексации
        enriched_text = f"# {instruction.title}\n\n"
        enriched_text += instruction.text_content

        if instruction.youtube_links:
            enriched_text += "\n\nВидео-инструкции:\n"
            for link in instruction.youtube_links:
                enriched_text += f"- {link}\n"

        # Разбиваем на чанки
        chunks = self.text_splitter.split_text(enriched_text)

        for chunk_idx, chunk in enumerate(chunks):
            metadata = {
                **metadata_base,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            }
            documents.append(
                Document(page_content=chunk, metadata=metadata)
            )

        return documents

    def index_instructions(self, instructions_dir: Optional[Path] = None) -> int:
        """
        Полная индексация всех инструкций.

        Returns:
            Количество проиндексированных документов (чанков).
        """
        instructions_dir = instructions_dir or settings.instructions_path
        logger.info(f"Начало индексации из {instructions_dir}")

        # Парсим все HTML-файлы
        instructions = self.parser.parse_directory(instructions_dir)
        logger.info(f"Распарсено {len(instructions)} инструкций")

        # Конвертируем в документы
        all_documents: List[Document] = []
        for instruction in instructions:
            docs = self._instruction_to_documents(instruction)
            all_documents.extend(docs)

        logger.info(f"Создано {len(all_documents)} чанков для индексации")

        # Создаём/пересоздаём ChromaDB коллекцию
        persist_dir = settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
        )

        logger.info(
            f"Индексация завершена. "
            f"Записано {len(all_documents)} чанков в ChromaDB ({persist_dir})"
        )

        # Сохраняем статистику
        stats = {
            "total_instructions": len(instructions),
            "total_chunks": len(all_documents),
            "instructions_with_images": sum(
                1 for i in instructions if i.images
            ),
            "instructions_with_youtube": sum(
                1 for i in instructions if i.youtube_links
            ),
        }
        stats_path = Path(persist_dir).parent / "indexing_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        logger.info(f"Статистика сохранена: {stats}")

        return len(all_documents)

    def index_support_tickets(
        self,
        json_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        batch_size: int = 200,
    ) -> int:
        """
        Индексация реальных заявок техподдержки из JSON.

        Args:
            json_path: Путь к support_qa_documents.json

        Returns:
            Количество проиндексированных документов.
        """
        import json as _json

        def report_progress(processed: int, total: int, message: str):
            if progress_callback is None:
                return
            progress_callback(processed, total, message)

        logger.info(f"Начало индексации заявок ТП из {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            qa_docs = _json.load(f)

        logger.info(f"Загружено {len(qa_docs)} Q&A документов")
        report_progress(0, len(qa_docs), "JSON загружен, подготовка документов")

        # Конвертируем в LangChain Document
        documents: List[Document] = []
        document_ids: List[str] = []
        for item in qa_docs:
            metadata = item.get('metadata', {})
            # ChromaDB не поддерживает списки в metadata — сериализуем
            clean_meta = {
                'source': metadata.get('source', 'real_support_tickets'),
                'category': metadata.get('category', 'Прочее'),
                'category_en': metadata.get('category_en', 'general'),
                'quality_score': metadata.get('quality_score', 0),
                'question': metadata.get('question', '')[:500],
                'doc_type': metadata.get('type', 'qa_pair'),
                'article_id': f"tp_{item.get('id', 'unknown')}",
                'title': metadata.get('question', 'Заявка ТП')[:200],
            }
            if metadata.get('tags'):
                clean_meta['tags'] = ', '.join(metadata['tags'])

            documents.append(
                Document(page_content=item['text'], metadata=clean_meta)
            )
            document_ids.append(item['id'])

        report_progress(0, len(documents), "Документы подготовлены, очистка коллекции")

        # Полная переиндексация должна пересоздавать коллекцию, а не дописывать дубли.
        persist_dir = settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        try:
            existing_store = Chroma(
                collection_name=SUPPORT_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
            existing_store.delete_collection()
            logger.info(
                f"Существующая коллекция '{SUPPORT_COLLECTION_NAME}' удалена перед полной переиндексацией"
            )
        except Exception as e:
            logger.info(
                f"Коллекция '{SUPPORT_COLLECTION_NAME}' ещё не существовала или уже очищена: {e}"
            )

        self.support_vector_store = Chroma(
            collection_name=SUPPORT_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        total_documents = len(documents)
        report_progress(0, total_documents, "Индексация в ChromaDB запущена")

        for start_idx in range(0, total_documents, batch_size):
            batch_docs = documents[start_idx:start_idx + batch_size]
            batch_ids = document_ids[start_idx:start_idx + batch_size]
            self.support_vector_store.add_documents(batch_docs, ids=batch_ids)
            processed = min(start_idx + len(batch_docs), total_documents)
            report_progress(
                processed,
                total_documents,
                f"Проиндексировано {processed} из {total_documents}",
            )

        logger.info(
            f"Индексация заявок ТП завершена. "
            f"Записано {len(documents)} документов в коллекцию '{SUPPORT_COLLECTION_NAME}'"
        )

        # Сохраняем статистику
        cats = {}
        for item in qa_docs:
            cat = item.get('metadata', {}).get('category', 'Прочее')
            cats[cat] = cats.get(cat, 0) + 1

        stats = {
            'total_documents': len(documents),
            'categories': cats,
            'collection_name': SUPPORT_COLLECTION_NAME,
        }
        stats_path = Path(persist_dir).parent / "support_indexing_stats.json"
        stats_path.write_text(_json.dumps(stats, indent=2, ensure_ascii=False))
        logger.info(f"Статистика заявок ТП: {stats}")
        report_progress(total_documents, total_documents, "Переиндексация завершена")

        return len(documents)

    def get_vector_store(self) -> Chroma:
        """Получение векторного хранилища основной коллекции (инструкции)."""
        if self.vector_store is None:
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
        return self.vector_store


    def get_support_vector_store(self) -> Optional[Chroma]:
        """Получение векторного хранилища коллекции заявок ТП."""
        if not hasattr(self, 'support_vector_store') or self.support_vector_store is None:
            try:
                store = Chroma(
                    collection_name=SUPPORT_COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=settings.chroma_persist_dir,
                )
                # Проверяем, что коллекция реально существует и не пуста
                count = store._collection.count()
                if count > 0:
                    self.support_vector_store = store
                    logger.info(f"Загружена коллекция '{SUPPORT_COLLECTION_NAME}': {count} документов")
                else:
                    self.support_vector_store = None
                    logger.info(f"Коллекция '{SUPPORT_COLLECTION_NAME}' пуста — пропускаем")
            except Exception as e:
                logger.warning(f"Коллекция '{SUPPORT_COLLECTION_NAME}' не найдена: {e}")
                self.support_vector_store = None
        return self.support_vector_store


# Singleton-экземпляр индексатора
_indexer: Optional[KnowledgeBaseIndexer] = None


def get_indexer() -> KnowledgeBaseIndexer:
    global _indexer
    if _indexer is None:
        _indexer = KnowledgeBaseIndexer()
    return _indexer
