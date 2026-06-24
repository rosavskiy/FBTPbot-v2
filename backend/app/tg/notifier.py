"""
Telegram-бот для уведомления операторов техподдержки.

Отправляет уведомления в групповой чат ТП при эскалации.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Отправка уведомлений в Telegram."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_support_chat_id
        self.enabled = bool(self.token and self.chat_id)
        self._client = httpx.AsyncClient(timeout=10.0)

        if not self.enabled:
            logger.warning(
                "Telegram-уведомления отключены: " "не заданы TELEGRAM_BOT_TOKEN и/или TELEGRAM_SUPPORT_CHAT_ID"
            )

    @property
    def api_url(self) -> str:
        return self.BASE_URL.format(token=self.token)

    async def send_escalation_notification(
        self,
        escalation_id: str,
        session_id: str,
        user_question: str,
        bot_answer: str,
        reason: str | None = None,
        contact_info: str | None = None,
        chat_summary: str | None = None,
    ) -> str | None:
        """
        Отправка уведомления об эскалации в Telegram.

        Returns:
            ID отправленного сообщения или None.
        """
        if not self.enabled:
            logger.info("Telegram отключён, пропуск уведомления")
            return None

        # Формируем красивое сообщение
        text = f"🆘 <b>Новая заявка в техподдержку</b>\n\n" f"📋 <b>ID:</b> <code>{escalation_id[:8]}...</code>\n"

        if contact_info:
            text += f"📞 <b>Контакт:</b> {self._escape_html(contact_info)}\n"

        if reason:
            text += f"❓ <b>Причина:</b> {self._escape_html(reason[:200])}\n"

        text += f"\n💬 <b>Последний вопрос:</b>\n" f"{self._escape_html(user_question[:300])}\n"

        if bot_answer:
            text += f"\n🤖 <b>Ответ бота:</b>\n" f"{self._escape_html(bot_answer[:300])}\n"

        if chat_summary:
            text += f"\n📝 <b>Краткое содержание диалога:</b>\n{self._escape_html(chat_summary[:500])}\n"

        text += f"\n🔗 <b>Панель оператора:</b>\n" f"/escalation_{escalation_id[:8]}"

        try:
            response = await self._client.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
            )
            data = response.json()

            if data.get("ok"):
                message_id = str(data["result"]["message_id"])
                logger.info(f"Telegram-уведомление отправлено: escalation={escalation_id}")
                return message_id
            else:
                logger.error(f"Telegram API error: {data}")
                return None

        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
            return None

    async def send_message_ex(
        self,
        chat_id: str,
        text: str,
        parse_mode: str | None = "HTML",
        retries: int = 2,
        backoff_sec: float = 2.0,
    ) -> tuple[str | None, str | None]:
        """Отправить сообщение с ретраями; вернуть (message_id, error).

        На успехе error=None. На неудаче message_id=None, а error — краткое
        описание причины (для записи в историю оповещений). Ретраи делаются
        только на сетевых сбоях/5xx; на ошибку самого Telegram API (ok=false)
        ретраить бессмысленно — возвращаем сразу.

        Returns:
            (message_id, None) при успехе; (None, error_str) при неудаче.
        """
        if not self.token:
            return None, "Telegram-токен не задан"

        payload: dict = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        last_error = "неизвестная ошибка"
        for attempt in range(retries + 1):
            try:
                response = await self._client.post(f"{self.api_url}/sendMessage", json=payload)
                data = response.json()
                if data.get("ok"):
                    return str(data["result"]["message_id"]), None
                # Ошибка уровня Telegram API (неверный chat_id, бот заблокирован и т.п.) — не ретраим
                last_error = f"Telegram API: {data.get('error_code', '?')} {data.get('description', '')}".strip()
                logger.warning("Telegram sendMessage error for %s: %s", chat_id, data)
                return None, last_error
            except Exception as e:
                # Сетевой сбой / таймаут / DNS — стоит повторить
                last_error = f"{type(e).__name__}: {e}"
                logger.error(
                    "Ошибка отправки в Telegram (%s), попытка %s/%s: %s",
                    chat_id,
                    attempt + 1,
                    retries + 1,
                    e,
                )
                if attempt < retries:
                    await asyncio.sleep(backoff_sec * (attempt + 1))
        return None, last_error

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str | None = "HTML",
    ) -> str | None:
        """Отправить произвольное сообщение указанному chat_id.

        Используется системой оповещений для рассылки по настроенным получателям.

        Returns:
            ID отправленного сообщения, либо None при ошибке/отключённом боте.
        """
        msg_id, _ = await self.send_message_ex(chat_id, text, parse_mode)
        return msg_id

    async def send_operator_reply(
        self,
        escalation_id: str,
        operator_name: str,
        reply_text: str,
        reply_to_message_id: str | None = None,
    ) -> str | None:
        """Отправка уведомления об ответе оператора."""
        if not self.enabled:
            return None

        text = (
            f"✅ <b>Оператор ответил</b>\n\n"
            f"👤 <b>Оператор:</b> {self._escape_html(operator_name)}\n"
            f"📋 <b>Заявка:</b> <code>{escalation_id[:8]}...</code>\n\n"
            f"💬 {self._escape_html(reply_text[:500])}"
        )

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        if reply_to_message_id:
            payload["reply_to_message_id"] = int(reply_to_message_id)

        try:
            response = await self._client.post(
                f"{self.api_url}/sendMessage",
                json=payload,
            )
            data = response.json()
            if data.get("ok"):
                return str(data["result"]["message_id"])
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")

        return None

    @staticmethod
    def _escape_html(text: str) -> str:
        """Экранирование HTML для Telegram."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Singleton
_notifier: TelegramNotifier | None = None


def get_telegram_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
