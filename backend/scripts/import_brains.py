#!/usr/bin/env python3
"""
Импорт причин обращения из docx-файлов (каталог brains/).

Парсит файлы формата «Модель ИИ (N).docx» и создаёт data/contact_reasons.json.

Использование:
    python -m scripts.import_brains                     # из backend/
    python backend/scripts/import_brains.py             # из корня проекта
    python backend/scripts/import_brains.py --dir ../brains --output ./data/contact_reasons.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from docx import Document


# ── Utilities ──

def slugify(text: str) -> str:
    """Превращает название причины в id (латиница + _)."""
    table = str.maketrans(
        "абвгдежзийклмнопрстуфхцчшщъыьэюя ",
        "abvgdezziiklmnoprstufhccss_y_eua_",
    )
    slug = text.lower().strip().translate(table)
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def extract_reason_name(title_text: str) -> str:
    """Извлекает название причины из заголовка '## БАЗА ЗНАНИЙ: XXX (ПОЛНАЯ ...'."""
    title_text = title_text.strip().lstrip("#").strip()
    title_text = title_text.strip("*").strip()
    # Убираем "БАЗА ЗНАНИЙ:" или "БАЗА ЗНАНИЙ :"
    m = re.search(r"БАЗА\s+ЗНАНИЙ\s*:\s*(.+?)(?:\(|$)", title_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return title_text


def clean_bold(text: str) -> str:
    """Убирает Markdown-жирность ** из текста."""
    return text.replace("**", "").strip()


# ── Paragraph classifier ──

class ParaType:
    TITLE = "title"              # ## БАЗА ЗНАНИЙ: ...
    SECTION = "section"          # ### Раздел N. ...
    SUBSECTION = "subsection"    # #### ...
    QUESTION = "question"        # **Вопрос N. ...**
    ANSWER_START = "answer"      # **Ответ:** ...
    SEPARATOR = "separator"      # ---
    LIST_ITEM = "list"           # - item
    TABLE_ROW = "table_row"     # | col | col |
    OTHER = "other"


def classify_para(text: str) -> str:
    """Определяет тип параграфа по формату."""
    t = text.strip()
    if t.startswith("## ") and "БАЗА ЗНАНИЙ" in t.upper():
        return ParaType.TITLE
    if t.startswith("### Раздел") or (t.startswith("### ") and "раздел" in t.lower()):
        return ParaType.SECTION
    if t.startswith("#### "):
        return ParaType.SUBSECTION
    if re.match(r"\*\*Вопрос\s+\d+", t):
        return ParaType.QUESTION
    if t.startswith("**Ответ:**") or t.startswith("**Ответ**:") or t == "**Ответ:**":
        return ParaType.ANSWER_START
    if t == "---" or t == "—" * 3:
        return ParaType.SEPARATOR
    if t.startswith("- "):
        return ParaType.LIST_ITEM
    if t.startswith("|"):
        return ParaType.TABLE_ROW
    return ParaType.OTHER


# ── Parser ──

def parse_docx(path: Path) -> dict:
    """Парсит один docx-файл → dict, пригодный для ContactReason."""
    doc = Document(str(path))
    paragraphs = [(p.text, classify_para(p.text)) for p in doc.paragraphs]

    reason_name = ""
    sections: list[dict] = []
    current_section: Optional[dict] = None
    current_qa: Optional[dict] = None
    collecting_answer = False

    # Markers
    markers = {"verbs": [], "nouns": [], "numeric_tags": [], "phrase_masks": []}
    in_markers_section = False
    current_marker_type: Optional[str] = None

    # Escalation
    escalation_rows: list[dict] = []
    in_escalation_section = False

    # Examples
    example_rows: list[dict] = []
    in_examples_section = False

    # Escalation rules (L1.5)
    escalation_rules: dict = {
        "enabled": False,
        "qa_pairs": [],
        "metrics": {"score_threshold": 0.7, "keyword_patterns": []},
    }
    in_escalation_rules_section = False
    current_esc_rules_sub: Optional[str] = None  # "keywords" | "qa"

    def _flush_qa():
        nonlocal current_qa, collecting_answer
        if current_qa and current_section is not None:
            # Trim answer
            if current_qa.get("answer"):
                current_qa["answer"] = current_qa["answer"].strip()
            if current_qa["question"] and current_qa.get("answer"):
                current_section["qa_pairs"].append(current_qa)
        current_qa = None
        collecting_answer = False

    def _flush_section():
        nonlocal current_section
        _flush_qa()
        if current_section and current_section.get("qa_pairs"):
            sections.append(current_section)
        current_section = None

    def _detect_marker_type(text: str) -> Optional[str]:
        """Detect marker type from heading or bold line."""
        t = text.lower()
        if "глагол" in t:
            return "verbs"
        if "существительн" in t:
            return "nouns"
        if "числов" in t or "системн" in t or "код" in t:
            return "numeric_tags"
        if "фраз" in t or "ключев" in t or "ситуаци" in t or "товарн" in t or "контекстн" in t:
            return "phrase_masks"
        if "сообщени" in t and "маркер" in t:
            return "phrase_masks"
        return None

    def _extract_marker_items(text: str) -> list[str]:
        """Extract marker items from a list line or table row."""
        text = text.strip()
        # List item: - item1, item2
        if text.startswith("- "):
            item = clean_bold(text[2:])
            # Split on comma if multiple
            parts = [p.strip() for p in re.split(r"[,;]", item) if p.strip()]
            # Clean up parenthetical context
            result = []
            for p in parts:
                # Remove trailing parenthetical
                p = re.sub(r"\s*\(.*?\)\s*$", "", p).strip()
                if p:
                    result.append(p)
            return result if result else [item]

        # Table row: | **word** | context |
        if text.startswith("|"):
            cells = [c.strip() for c in text.split("|")]
            cells = [c for c in cells if c and not c.startswith(":---")]
            if cells:
                item = clean_bold(cells[0])
                if item and not any(header in item.lower() for header in ["глагол", "существ", "ошибка", "ситуация", "товар", "вопрос"]):
                    return [item]
        return []

    def _extract_example_from_table_row(text: str) -> Optional[dict]:
        """Extract example Q&A from table row: | question | answer | images? |"""
        if not text.startswith("|"):
            return None
        cells = [c.strip() for c in text.split("|")]
        cells = [c for c in cells if c and not c.startswith(":---")]
        if len(cells) >= 2:
            q = clean_bold(cells[0]).strip("«»\"'")
            a = clean_bold(cells[1])
            # Skip header rows
            if q.lower() in ("вопрос пользователя", "вопрос", ""):
                return None
            if q and a:
                # Support multiple questions separated by " ;; "
                questions = [s.strip() for s in q.split(" ;; ") if s.strip()]
                image_codes = []
                if len(cells) >= 3 and cells[2].strip():
                    image_codes = [c.strip() for c in cells[2].split(",") if c.strip()]
                return {
                    "user_question": questions[0] if questions else q,
                    "user_questions": questions if questions else [q],
                    "ideal_answer": a,
                    "image_codes": image_codes,
                }
        return None

    def _extract_escalation_from_table_row(text: str) -> Optional[dict]:
        """Extract escalation row: | situation | signs | action |"""
        if not text.startswith("|"):
            return None
        cells = [c.strip() for c in text.split("|")]
        cells = [c for c in cells if c and not c.startswith(":---")]
        if len(cells) >= 3:
            situation = clean_bold(cells[0])
            signs = clean_bold(cells[1])
            action = clean_bold(cells[2])
            if situation.lower() in ("ситуация", ""):
                return None
            if situation:
                return {"description": situation, "context": signs, "response_template": action}
        return None

    for idx, (text, ptype) in enumerate(paragraphs):
        text_stripped = text.strip()
        if not text_stripped:
            continue

        # ── Title ──
        if ptype == ParaType.TITLE:
            reason_name = extract_reason_name(text_stripped)
            continue

        # ── Section header ──
        if ptype == ParaType.SECTION:
            section_title = text_stripped.lstrip("#").strip()
            section_title = re.sub(r"^Раздел\s+\d+\.\s*", "", section_title).strip()
            upper = section_title.upper()

            # Detect special sections
            if "МАРКЕР" in upper:
                _flush_section()
                in_markers_section = True
                in_escalation_section = False
                in_examples_section = False
                in_escalation_rules_section = False
                current_marker_type = None
                continue
            elif "100%" in upper or "L1.5" in upper or ("ПРАВИЛА" in upper and "ЭСКАЛАЦИ" in upper):
                _flush_section()
                in_markers_section = False
                in_escalation_section = False
                in_examples_section = False
                in_escalation_rules_section = True
                current_esc_rules_sub = None
                continue
            elif "ЭСКАЛАЦИ" in upper or "СПЕЦИАЛИСТ" in upper:
                _flush_section()
                in_markers_section = False
                in_escalation_section = True
                in_examples_section = False
                in_escalation_rules_section = False
                continue
            elif "ГОТОВЫЕ ОТВЕТЫ" in upper:
                _flush_section()
                in_markers_section = False
                in_escalation_section = False
                in_examples_section = True
                in_escalation_rules_section = False
                continue
            else:
                # Normal Q&A section
                _flush_section()
                in_markers_section = False
                in_escalation_section = False
                in_examples_section = False
                in_escalation_rules_section = False
                current_section = {
                    "id": slugify(section_title),
                    "title": section_title,
                    "order": len(sections) + 1,
                    "qa_pairs": [],
                }
                continue

        # ── Inside markers section ──
        if in_markers_section:
            # Subsection heading or bold heading → detect marker type
            if ptype == ParaType.SUBSECTION or (text_stripped.startswith("**") and text_stripped.endswith("**")):
                mt = _detect_marker_type(text_stripped)
                if mt:
                    current_marker_type = mt
                continue

            # Bold-colon heading: **Глаголы-маркеры:**
            if text_stripped.startswith("**") and ":" in text_stripped:
                mt = _detect_marker_type(text_stripped)
                if mt:
                    current_marker_type = mt
                    # Check if there's content after the colon on same line
                    after_colon = text_stripped.split(":", 1)[-1].strip().rstrip("*").strip()
                    if after_colon and after_colon != "**":
                        items = [i.strip() for i in after_colon.split(",") if i.strip()]
                        if current_marker_type and items:
                            markers[current_marker_type].extend(items)
                continue

            # List items or table rows
            if current_marker_type:
                items = _extract_marker_items(text_stripped)
                for item in items:
                    if item not in markers[current_marker_type]:
                        markers[current_marker_type].append(item)
                continue

            if ptype == ParaType.SEPARATOR:
                in_markers_section = False
                continue

        # ── Inside escalation section ──
        if in_escalation_section:
            if ptype == ParaType.TABLE_ROW:
                row = _extract_escalation_from_table_row(text_stripped)
                if row:
                    escalation_rows.append(row)
                continue
            if ptype == ParaType.SEPARATOR:
                in_escalation_section = False
                continue

        # ── Inside escalation rules (L1.5) section ──
        if in_escalation_rules_section:
            low = text_stripped.lower()
            # Status line: **Статус:** Включено / Выключено
            if "статус" in low:
                escalation_rules["enabled"] = "включен" in low
                continue
            # Threshold line: **Порог совпадения:** 0.7
            if "порог" in low:
                m_thr = re.search(r"(\d+(?:[.,]\d+)?)", text_stripped)
                if m_thr:
                    escalation_rules["metrics"]["score_threshold"] = float(
                        m_thr.group(1).replace(",", ".")
                    )
                continue
            # Subsection headings
            if ptype == ParaType.SUBSECTION:
                sub_lower = text_stripped.lower()
                if "ключев" in sub_lower or "фраз" in sub_lower:
                    current_esc_rules_sub = "keywords"
                elif "пар" in sub_lower or "вопрос" in sub_lower:
                    current_esc_rules_sub = "qa"
                continue
            # Keyword list items
            if current_esc_rules_sub == "keywords" and ptype == ParaType.LIST_ITEM:
                item = text_stripped[2:].strip()
                if item and item != "(пусто)":
                    escalation_rules["metrics"]["keyword_patterns"].append(item)
                continue
            # QA table rows
            if current_esc_rules_sub == "qa" and ptype == ParaType.TABLE_ROW:
                cells = [c.strip() for c in text_stripped.split("|")]
                cells = [c for c in cells if c and not c.startswith(":---")]
                if len(cells) >= 2:
                    q = clean_bold(cells[0])
                    a = clean_bold(cells[1])
                    if q and a and q.lower() not in ("вопрос", ""):
                        escalation_rules["qa_pairs"].append(
                            {"question": q, "answer": a}
                        )
                continue
            if ptype == ParaType.SEPARATOR:
                in_escalation_rules_section = False
                continue

        # ── Inside examples section ──
        if in_examples_section:
            if ptype == ParaType.TABLE_ROW:
                ex = _extract_example_from_table_row(text_stripped)
                if ex:
                    example_rows.append(ex)
                continue
            if ptype == ParaType.SEPARATOR:
                in_examples_section = False
                continue

        # ── Q&A in normal sections ──
        if current_section is not None:
            if ptype == ParaType.QUESTION:
                _flush_qa()
                q_text = re.sub(r"^\*\*Вопрос\s+\d+\.\s*", "", text_stripped)
                q_text = q_text.rstrip("*").strip()
                current_qa = {"question": q_text, "answer": ""}
                collecting_answer = False
                continue

            if ptype == ParaType.ANSWER_START:
                collecting_answer = True
                # Inline answer after "**Ответ:**"
                answer_part = re.sub(r"^\*\*Ответ\*?\*?:?\*?\*?\s*", "", text_stripped).strip()
                if answer_part and current_qa is not None:
                    current_qa["answer"] = answer_part
                continue

            if collecting_answer and current_qa is not None:
                if ptype == ParaType.SEPARATOR:
                    _flush_qa()
                    continue
                # Append to current answer
                if current_qa["answer"]:
                    current_qa["answer"] += "\n" + text_stripped
                else:
                    current_qa["answer"] = text_stripped
                continue

    # Flush remaining
    _flush_section()

    # Build result
    reason_id = slugify(reason_name)
    return {
        "id": reason_id,
        "name": reason_name,
        "is_active": True,
        "markers": markers,
        "thematic_sections": sections,
        "typical_complaints": escalation_rows,
        "example_answers": example_rows,
        "escalation_rules": escalation_rules,
    }


def import_all(brains_dir: Path, output_path: Path) -> None:
    """Импортирует все docx из каталога brains/ → contact_reasons.json."""
    docx_files = sorted(brains_dir.glob("Модель ИИ*.docx"))
    if not docx_files:
        docx_files = sorted(brains_dir.glob("*.docx"))

    if not docx_files:
        print(f"❌ Нет .docx файлов в {brains_dir}")
        sys.exit(1)

    reasons = []
    for fp in docx_files:
        print(f"📄 Парсим: {fp.name}")
        reason = parse_docx(fp)
        sections_count = len(reason["thematic_sections"])
        qa_count = sum(len(s["qa_pairs"]) for s in reason["thematic_sections"])
        markers_count = sum(len(v) for v in reason["markers"].values())
        examples_count = len(reason["example_answers"])
        escalation_count = len(reason["typical_complaints"])

        print(
            f"   → {reason['name']} (id={reason['id']})\n"
            f"     Разделов: {sections_count}, Q&A: {qa_count}, "
            f"Маркеров: {markers_count}, Примеров: {examples_count}, "
            f"Эскалация: {escalation_count}"
        )
        reasons.append(reason)

    result = {"reasons": reasons}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Записано {len(reasons)} причин обращения → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Импорт причин обращения из docx")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Путь к каталогу с docx (default: brains/ от корня проекта)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь для вывода JSON (default: data/contact_reasons.json)",
    )
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # backend/scripts/ → root

    brains_dir = args.dir or (project_root / "brains")
    output_path = args.output or (project_root / "backend" / "data" / "contact_reasons.json")

    if not brains_dir.exists():
        # Try relative to backend/
        alt = script_dir.parent.parent / "brains"
        if alt.exists():
            brains_dir = alt
        else:
            print(f"❌ Каталог не найден: {brains_dir}")
            sys.exit(1)

    print(f"📁 Каталог: {brains_dir}")
    print(f"📝 Выход: {output_path}")
    print()

    import_all(brains_dir, output_path)


if __name__ == "__main__":
    main()
