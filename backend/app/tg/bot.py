from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import settings
from app.database.models import async_session
from app.database.service import DatabaseService
from app.rag.engine import get_rag_engine
from app.sheets.gsheet_logger import get_gsheet_logger
from app.tg.user_registry import record_user

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tg_bot")

# ── Константы ──
MAX_MESSAGE_LENGTH = 4096
WELCOME_TEXT = (
    "👋 Здравствуйте! Я бот техподдержки.\n\n"
    "Задайте вопрос по работе с программой.\n"
    "Если не смогу ответить — переведу на оператора."
)
HELP_TEXT = (
    "📖 <b>Справка:</b>\n\n"
    "1. Напишите вопрос текстом\n"
    "2. Если вопрос широкий — выберите тему кнопкой\n\n"
    "/start — начать заново\n"
    "/reset — сбросить контекст"
)

# Хранилище chat_history и clarification context для Telegram (по user_id)
_chat_histories: dict[int, list] = {}
_clarification_ctx: dict[int, dict] = {}
MAX_HISTORY = 10


def _get_history(user_id: int) -> list:
    return _chat_histories.get(user_id, [])


def _add_to_history(user_id: int, role: str, content: str):
    if user_id not in _chat_histories:
        _chat_histories[user_id] = []
    _chat_histories[user_id].append({"role": role, "content": content})
    if len(_chat_histories[user_id]) > MAX_HISTORY * 2:
        _chat_histories[user_id] = _chat_histories[user_id][-MAX_HISTORY * 2 :]


def _clear_history(user_id: int):
    _chat_histories.pop(user_id, None)
    _clarification_ctx.pop(user_id, None)


# ═══════════════════════════════════════════════════
#  Мониторинг: heartbeat + запись в БД
# ═══════════════════════════════════════════════════


async def _heartbeat_loop() -> None:
    """Записывает timestamp раз в 30 сек — признак жизни бота для дашборда."""
    heartbeat_path = Path(settings.tg_heartbeat_path)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            heartbeat_path.write_text(
                json.dumps({"ts": datetime.now(UTC).isoformat(), "alive": True}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[TG] Heartbeat write failed: %s", exc)
        await asyncio.sleep(30)


async def _on_post_init(app: Application) -> None:  # noqa: ARG001
    """Запускает фоновые задачи после инициализации PTB Application."""
    asyncio.create_task(_heartbeat_loop())


async def _db_write_tg(
    user_id: int,
    question: str,
    answer: str,
    confidence: float | None = None,
    detected_reason: str | None = None,
) -> None:
    """Асинхронная fire-and-forget запись TG Q&A в БД для мониторинга."""
    try:
        session_id = f"tg_{user_id}"
        async with async_session() as db:
            svc = DatabaseService(db)
            await svc.get_or_create_tg_session(session_id)
            await svc.add_message(session_id=session_id, role="user", content=question, source="tg")
            await svc.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
                confidence=confidence,
                source="tg",
                detected_reason=detected_reason or None,
            )
    except Exception as exc:
        logger.warning("[TG] DB write failed: %s", exc)


def _escape(text: str) -> str:
    return html.escape(text)


def _format_answer(
    answer: str,
    confidence: float = 0.0,
    needs_escalation: bool = False,
    detected_reason_name: str = "",
) -> str:
    """Форматирование ответа для Telegram."""
    parts = [_escape(answer)]

    if needs_escalation:
        parts.append("")
        parts.append("\n❗ Рекомендую обратиться к оператору.")

    result = "\n".join(parts)

    if len(result) > MAX_MESSAGE_LENGTH:
        result = result[: MAX_MESSAGE_LENGTH - 20] + "\n\n<i>…(обрезано)</i>"

    return result


def _build_reason_keyboard(candidates: list[dict]) -> InlineKeyboardMarkup:
    """Создаёт inline-клавиатуру из кандидатов причин обращения."""
    buttons = []
    for i, c in enumerate(candidates):
        name = c.get("reason_name", f"Тема {i + 1}")
        callback = f"reason:{i}"
        label = name if len(name) <= 60 else name[:57] + "..."
        buttons.append([InlineKeyboardButton(text=label, callback_data=callback)])
    return InlineKeyboardMarkup(buttons)


def _combine_query(original_query: str, followup: str) -> str:
    return f"{original_query}\n\nДополнительная информация от пользователя:\n{followup}".strip()


def _save_reason_selection_ctx(user_id: int, original_query: str, candidates: list[dict]) -> None:
    _clarification_ctx[user_id] = {
        "type": "reason_selection",
        "original_query": original_query,
        "candidates": candidates,
    }


def _save_reason_details_ctx(user_id: int, original_query: str, reason_id: str, reason_name: str) -> None:
    _clarification_ctx[user_id] = {
        "type": "reason_details",
        "original_query": original_query,
        "reason_id": reason_id,
        "reason_name": reason_name,
    }


_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}


async def _send_files(message, files: list[dict]):
    """Send each file from rag_response.files as a photo or document message."""
    for f in files:
        file_path = f.get("file_path")
        if not file_path:
            continue
        p = Path(file_path)
        if not p.is_file():
            logger.warning(f"[TG] File not found: {file_path}")
            continue
        try:
            ext = f.get("ext", p.suffix.lower().lstrip("."))
            with open(p, "rb") as fh:
                if ext in _IMAGE_EXTS:
                    await message.reply_photo(photo=fh)
                else:
                    await message.reply_document(document=fh, filename=p.name)
        except Exception as exc:
            logger.warning(f"[TG] Failed to send file {file_path}: {exc}")


# ═══════════════════════════════════════════════════
#  Обработчики команд
# ═══════════════════════════════════════════════════


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    record_user(user_id, update.effective_user.username)
    _clear_history(user_id)
    await update.message.reply_text(WELCOME_TEXT, parse_mode=ParseMode.HTML)
    logger.info(f"User {user_id} started bot")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _clear_history(user_id)
    await update.message.reply_text(
        "🔄 Контекст сброшен. Задайте новый вопрос.",
        parse_mode=ParseMode.HTML,
    )


# ═══════════════════════════════════════════════════
#  Обработка текстовых сообщений
# ═══════════════════════════════════════════════════


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстового сообщения пользователя."""
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        return

    user = update.effective_user
    username = user.username or user.first_name or str(user_id)
    logger.info(f"[TG] question={text[:120]}|user={username}")

    # Запоминаем username → chat_id для адресации оповещений
    record_user(user_id, user.username)

    await update.message.chat.send_action(ChatAction.TYPING)

    _add_to_history(user_id, "user", text)
    chat_history = _get_history(user_id)[:-1]

    rag = get_rag_engine()
    ctx = _clarification_ctx.get(user_id)

    if ctx:
        question_for_rag = text
        reason_id: str | None = None

        if ctx.get("type") == "reason_selection":
            candidates = ctx.get("candidates", [])
            if text.strip().isdigit():
                idx = int(text.strip()) - 1
                if 0 <= idx < len(candidates):
                    chosen = candidates[idx]
                    question_for_rag = ctx.get("original_query", text)
                    reason_id = chosen.get("reason_id")
                else:
                    question_for_rag = _combine_query(ctx.get("original_query", text), text)
            else:
                question_for_rag = _combine_query(ctx.get("original_query", text), text)
        elif ctx.get("type") == "reason_details":
            question_for_rag = _combine_query(ctx.get("original_query", text), text)
            reason_id = ctx.get("reason_id")

        _clarification_ctx.pop(user_id, None)

        try:
            rag_response = await rag.ask(
                question=question_for_rag,
                chat_history=chat_history,
                reason_id=reason_id,
            )
        except Exception as e:
            logger.error(f"[TG] RAG error: {e}", exc_info=True)
            await update.message.reply_text("😔 Ошибка обработки.", parse_mode=ParseMode.HTML)
            return

        if rag_response.classification_method == "clarification" and rag_response.clarification_candidates:
            candidates = rag_response.clarification_candidates
            _save_reason_selection_ctx(user_id, question_for_rag, candidates)
            _add_to_history(user_id, "assistant", rag_response.answer)

            keyboard = _build_reason_keyboard(candidates)
            await update.message.reply_text(
                f"🔍 {_escape(rag_response.answer)}",
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            return

        if rag_response.classification_method == "marker_clarification" and rag_response.detected_reason:
            _save_reason_details_ctx(
                user_id,
                question_for_rag,
                rag_response.detected_reason,
                rag_response.detected_reason_name,
            )
            _add_to_history(user_id, "assistant", rag_response.answer)
            await update.message.reply_text(
                f"🔍 {_escape(rag_response.answer)}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return

        reply = _format_answer(
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            needs_escalation=rag_response.needs_escalation,
            detected_reason_name=rag_response.detected_reason_name,
        )
        _add_to_history(user_id, "assistant", rag_response.answer)
        await update.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

        if rag_response.files:
            await _send_files(update.message, rag_response.files)

        asyncio.ensure_future(
            _db_write_tg(
                user_id,
                question_for_rag,
                rag_response.answer,
                rag_response.confidence,
                rag_response.detected_reason_name,
            )
        )
        await get_gsheet_logger().log(
            question=question_for_rag,
            answer=rag_response.answer,
            session_id=f"tg_{user_id}",
            confidence=rag_response.confidence,
            needs_escalation=rag_response.needs_escalation,
            detected_reason=rag_response.detected_reason_name,
            thematic_section=rag_response.thematic_section,
            source_articles=rag_response.source_articles,
            youtube_links=rag_response.youtube_links,
            has_files=bool(rag_response.files),
            response_type="tg_clarification",
        )
        return

    # ── Стандартный путь: ask → L1→L2→L3 ──
    try:
        rag_response = await rag.ask(question=text, chat_history=chat_history)
    except Exception as e:
        logger.error(f"[TG] RAG error: {e}", exc_info=True)
        await update.message.reply_text("😔 Ошибка обработки. Попробуйте позже.", parse_mode=ParseMode.HTML)
        return

    # ── Режим уточнения ──
    if rag_response.classification_method == "clarification" and rag_response.clarification_candidates:
        candidates = rag_response.clarification_candidates
        _save_reason_selection_ctx(user_id, text, candidates)

        logger.info(f"[TG] CLARIFICATION|candidates={len(candidates)}|user={username}")
        _add_to_history(user_id, "assistant", rag_response.answer)

        keyboard = _build_reason_keyboard(candidates)
        await update.message.reply_text(
            f"🔍 {_escape(rag_response.answer)}",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
        return

    if rag_response.classification_method == "marker_clarification" and rag_response.detected_reason:
        _save_reason_details_ctx(
            user_id,
            text,
            rag_response.detected_reason,
            rag_response.detected_reason_name,
        )
        _add_to_history(user_id, "assistant", rag_response.answer)
        await update.message.reply_text(
            f"🔍 {_escape(rag_response.answer)}",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    # ── Обычный ответ ──
    reply = _format_answer(
        answer=rag_response.answer,
        confidence=rag_response.confidence,
        needs_escalation=rag_response.needs_escalation,
        detected_reason_name=rag_response.detected_reason_name,
    )
    _add_to_history(user_id, "assistant", rag_response.answer)

    logger.info(f"[TG] DONE|answer_len={len(rag_response.answer)}|user={username}")
    await update.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    # Отправка файлов отдельными сообщениями
    if rag_response.files:
        await _send_files(update.message, rag_response.files)

    # Запись в БД (для мониторинга) и в Google Sheets
    asyncio.ensure_future(
        _db_write_tg(
            user_id,
            text,
            rag_response.answer,
            rag_response.confidence,
            rag_response.detected_reason_name,
        )
    )
    await get_gsheet_logger().log(
        question=text,
        answer=rag_response.answer,
        session_id=f"tg_{user_id}",
        confidence=rag_response.confidence,
        needs_escalation=rag_response.needs_escalation,
        detected_reason=rag_response.detected_reason_name,
        thematic_section=rag_response.thematic_section,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        has_files=bool(rag_response.files),
        response_type="tg",
    )


# ═══════════════════════════════════════════════════
#  Обработка нажатий на inline-кнопки
# ═══════════════════════════════════════════════════


async def handle_reason_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатия кнопки выбора причины обращения."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data  # "reason:0", "reason:1", ...

    if not data.startswith("reason:"):
        return

    try:
        idx = int(data.split(":")[1])
    except (ValueError, IndexError):
        return

    ctx = _clarification_ctx.pop(user_id, None)
    if ctx is None:
        await query.edit_message_text(
            "⏰ Время выбора истекло. Задайте вопрос заново.",
            parse_mode=ParseMode.HTML,
        )
        return

    candidates = ctx.get("candidates", [])
    if idx < 0 or idx >= len(candidates):
        return

    chosen = candidates[idx]
    original_query = ctx.get("original_query", "")
    chat_history = _get_history(user_id)

    try:
        await query.edit_message_text(
            f"🔍 Выбрана тема: <b>{_escape(chosen['reason_name'])}</b>\n\n⏳ Формирую ответ...",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    logger.info(f"[TG] REASON_SELECTED|reason={chosen['reason_id']}|user_id={user_id}")

    rag = get_rag_engine()
    try:
        rag_response = await rag.ask(
            question=original_query,
            chat_history=chat_history,
            reason_id=chosen["reason_id"],
        )
    except Exception as e:
        logger.error(f"[TG] RAG error on reason: {e}", exc_info=True)
        await query.edit_message_text("😔 Ошибка. Задайте вопрос заново.", parse_mode=ParseMode.HTML)
        return

    reply = _format_answer(
        answer=rag_response.answer,
        confidence=rag_response.confidence,
        needs_escalation=rag_response.needs_escalation,
        detected_reason_name=rag_response.detected_reason_name,
    )

    if rag_response.classification_method == "marker_clarification" and rag_response.detected_reason:
        _save_reason_details_ctx(
            user_id,
            original_query,
            rag_response.detected_reason,
            rag_response.detected_reason_name,
        )
        _add_to_history(user_id, "assistant", rag_response.answer)
        try:
            await query.edit_message_text(
                f"🔍 {_escape(rag_response.answer)}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception:
            await query.message.reply_text(
                f"🔍 {_escape(rag_response.answer)}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        return

    _add_to_history(user_id, "assistant", rag_response.answer)

    try:
        await query.edit_message_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        await query.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    # Отправка файлов отдельными сообщениями
    if rag_response.files:
        await _send_files(query.message, rag_response.files)

    # Запись в БД (для мониторинга) и в Google Sheets
    asyncio.ensure_future(
        _db_write_tg(
            user_id,
            original_query,
            rag_response.answer,
            rag_response.confidence,
            rag_response.detected_reason_name,
        )
    )
    await get_gsheet_logger().log(
        question=original_query,
        answer=rag_response.answer,
        session_id=f"tg_{user_id}",
        confidence=rag_response.confidence,
        needs_escalation=rag_response.needs_escalation,
        detected_reason=rag_response.detected_reason_name,
        thematic_section=rag_response.thematic_section,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        has_files=bool(rag_response.files),
        response_type="tg_callback",
    )


# ═══════════════════════════════════════════════════
#  Запуск бота
# ═══════════════════════════════════════════════════


def main():
    """Точка входа для Telegram-бота."""
    token = settings.telegram_bot_token
    if not token:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не задан!")
        sys.exit(1)

    logger.info("🤖 Запуск Telegram-бота технической поддержки 1.0.0...")

    # Перехват ERROR-логов для оповещений (общий файл с бэкендом)
    from app.alerts.error_capture import install_error_capture

    install_error_capture()

    app = Application.builder().token(token).post_init(_on_post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CallbackQueryHandler(handle_reason_callback, pattern=r"^reason:\d+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("✅ Бот запущен, ожидаю сообщения...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
