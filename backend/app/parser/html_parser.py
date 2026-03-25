"""
Парсер HTML-инструкций Фармбазис.

Извлекает из HTML-файлов:
- Заголовок статьи
- Структурированный текст (пошаговые инструкции)
- Изображения (base64 → файлы)
- YouTube-ссылки
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


@dataclass
class ParsedImage:
    """Изображение, извлечённое из инструкции."""
    filename: str          # имя файла после сохранения
    original_index: int    # порядковый номер в статье
    alt_text: str = ""
    file_path: str = ""    # путь к сохранённому файлу


@dataclass
class ParsedInstruction:
    """Распарсенная инструкция."""
    article_id: str                       # ID статьи (имя файла без расширения)
    title: str                            # Заголовок
    text_content: str                     # Полный текст без HTML
    sections: List[str] = field(default_factory=list)       # Разделы/этапы
    images: List[ParsedImage] = field(default_factory=list) # Изображения
    youtube_links: List[str] = field(default_factory=list)  # YouTube ссылки
    source_file: str = ""                 # Путь к исходному HTML


class InstructionParser:
    """Парсер HTML-инструкций Фармбазис."""

    # Паттерн для поиска YouTube ссылок
    YT_PATTERN = re.compile(
        r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]+'
    )

    # Паттерн для base64 изображений
    BASE64_IMG_PATTERN = re.compile(
        r'data:image/(\w+);base64,([A-Za-z0-9+/=\s]+)'
    )

    def __init__(self, images_dir: Optional[Path] = None):
        self.images_dir = images_dir or Path("./data/images")
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def parse_file(self, file_path: Path) -> ParsedInstruction:
        """Парсинг одного HTML-файла инструкции."""
        article_id = file_path.stem
        logger.info(f"Парсинг статьи {article_id}...")

        html_content = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "lxml")

        # Извлекаем заголовок
        title = self._extract_title(soup, article_id)

        # Извлекаем изображения (и заменяем на плейсхолдеры)
        images = self._extract_images(soup, article_id)

        # Извлекаем YouTube ссылки
        youtube_links = self._extract_youtube_links(html_content)

        # Извлекаем текстовое содержимое
        text_content = self._extract_text(soup)

        # Извлекаем секции/этапы
        sections = self._extract_sections(text_content)

        return ParsedInstruction(
            article_id=article_id,
            title=title,
            text_content=text_content,
            sections=sections,
            images=images,
            youtube_links=youtube_links,
            source_file=str(file_path),
        )

    def parse_directory(self, directory: Path) -> List[ParsedInstruction]:
        """Парсинг всех HTML-файлов в директории."""
        instructions = []
        html_files = sorted(directory.glob("*.html"))
        total = len(html_files)

        logger.info(f"Найдено {total} HTML-файлов для парсинга")

        for i, file_path in enumerate(html_files, 1):
            try:
                instruction = self.parse_file(file_path)
                instructions.append(instruction)
                if i % 50 == 0:
                    logger.info(f"Обработано {i}/{total} файлов...")
            except Exception as e:
                logger.error(f"Ошибка парсинга {file_path.name}: {e}")

        logger.info(f"Парсинг завершён. Обработано {len(instructions)}/{total} файлов")
        return instructions

    def _extract_title(self, soup: BeautifulSoup, article_id: str) -> str:
        """Извлечение заголовка статьи."""
        # Ищём заголовок в элементе с классом af9 (основной стиль заголовков)
        title_el = soup.find("p", class_=re.compile(r"af9"))
        if title_el:
            title = title_el.get_text(strip=True)
            if title:
                return title

        # Пробуем стиль X1 (подзаголовок)
        title_el = soup.find("p", class_=re.compile(r"X1"))
        if title_el:
            title = title_el.get_text(strip=True)
            if title:
                return title

        # Ищём первый span с крупным шрифтом
        for span in soup.find_all("span", style=re.compile(r"font-size:\s*2[0-9]")):
            text = span.get_text(strip=True)
            if text:
                return text

        # Ищём первый непустой параграф
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 3:
                return text[:100]

        return f"Инструкция #{article_id}"

    def _extract_images(self, soup: BeautifulSoup, article_id: str) -> List[ParsedImage]:
        """Извлечение и сохранение base64-изображений."""
        images = []
        img_tags = soup.find_all("img")

        for idx, img_tag in enumerate(img_tags):
            src = img_tag.get("src", "")
            alt = img_tag.get("alt", "")

            if not src:
                continue

            # Base64 изображение
            match = self.BASE64_IMG_PATTERN.match(src)
            if match:
                img_format = match.group(1)
                img_data = match.group(2)

                try:
                    raw_data = base64.b64decode(img_data)
                    # Создаём хэш для уникального имени
                    img_hash = hashlib.md5(raw_data).hexdigest()[:10]
                    filename = f"{article_id}_img{idx}_{img_hash}.{img_format}"
                    file_path = self.images_dir / filename

                    if not file_path.exists():
                        file_path.write_bytes(raw_data)

                    images.append(ParsedImage(
                        filename=filename,
                        original_index=idx,
                        alt_text=alt,
                        file_path=str(file_path),
                    ))
                except Exception as e:
                    logger.warning(f"Ошибка сохранения изображения {article_id}/img{idx}: {e}")

            # Удаляем тег img из HTML чтобы не мешал извлечению текста
            img_tag.decompose()

        return images

    def _extract_youtube_links(self, html_content: str) -> List[str]:
        """Извлечение уникальных YouTube-ссылок."""
        links = self.YT_PATTERN.findall(html_content)
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique.append(link)
        return unique

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Извлечение чистого текста из HTML."""
        # Убираем style и script теги
        for tag in soup.find_all(["style", "script", "head"]):
            tag.decompose()

        # Получаем текст с сохранением структуры
        lines = []
        body = soup.find("body") or soup

        for element in body.descendants:
            if isinstance(element, NavigableString):
                text = element.strip()
                if text:
                    lines.append(text)
            elif isinstance(element, Tag) and element.name in ("p", "br", "div", "li", "tr"):
                lines.append("\n")

        # Объединяем и чистим
        raw_text = " ".join(lines)
        # Убираем множественные пробелы и переносы
        text = re.sub(r'\s+', ' ', raw_text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    def _extract_sections(self, text: str) -> List[str]:
        """Разбиение текста на секции по этапам/пунктам."""
        sections = []

        # Ищем паттерны типа "ЭТАП 1.", "1.", "Шаг 1" и т.д.
        patterns = [
            r'(?:ЭТАП\s+\d+[\.\:])',
            r'(?:^|\n)\s*\d+[\.\)]\s+',
            r'(?:Шаг\s+\d+)',
        ]

        combined_pattern = '|'.join(patterns)
        parts = re.split(f'({combined_pattern})', text)

        current_section = ""
        for part in parts:
            if re.match(combined_pattern, part.strip()):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part
            else:
                current_section += part

        if current_section.strip():
            sections.append(current_section.strip())

        return sections if len(sections) > 1 else [text]
