"""
Диалоговый Telegram-бот с векторной памятью (Pinecone + OpenAI).

Логика обработки вынесена в handle_user_text(user, text), чтобы её можно было
импортировать (emulate_validation.py) без запуска поллинга.

Если нет TELEGRAM_BOT_TOKEN или MOCK_MODE=true — запускается консольный эмулятор.
"""

import logging
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from pinecone_manager import PineconeManager, env_flag, is_real_value, setup_logging

load_dotenv("local.env")
setup_logging()
logger = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MOCK_MODE = env_flag("MOCK_MODE")

MEMORY_TOP_K = 5
MAX_TELEGRAM_LEN = 4000

_manager: Optional[PineconeManager] = None
_llm_client = None


# ---------------------------------------------------------------------------
# Ленивая инициализация зависимостей
# ---------------------------------------------------------------------------

def get_manager() -> PineconeManager:
    global _manager
    if _manager is None:
        _manager = PineconeManager()
    return _manager


def set_manager(manager: PineconeManager) -> None:
    """Позволяет подставить готовый менеджер (например, из emulate_validation.py)."""
    global _manager
    _manager = manager


def llm_is_mock() -> bool:
    return MOCK_MODE or not is_real_value(OPENAI_API_KEY)


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        if OPENAI_BASE_URL:
            _llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        else:
            _llm_client = OpenAI(api_key=OPENAI_API_KEY)
    return _llm_client


# ---------------------------------------------------------------------------
# Метаданные
# ---------------------------------------------------------------------------

def _to_iso(unix_ts: Any) -> str:
    try:
        return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_metadata(message: Any) -> Dict[str, Any]:
    """
    Безопасно собирает метаданные сообщения.

    Любое поле может отсутствовать (например, у пользователя нет first_name/last_name),
    поэтому всё читается через getattr(..., None). None и пустые строки в metadata
    не попадают — Pinecone их не принимает.
    """
    user = getattr(message, "from_user", None)
    chat = getattr(message, "chat", None)

    user_id = getattr(user, "id", None)
    if user_id is None:
        user_id = getattr(chat, "id", None) or "unknown"
    user_id = str(user_id)

    username = getattr(user, "username", None)
    first_name = getattr(user, "first_name", None)
    last_name = getattr(user, "last_name", None)

    candidates = {
        "user_id": user_id,
        "username": username,
        "first_name": first_name,
        "last_name": last_name,
        "display_name": first_name or username or f"user_{user_id}",
        "language_code": getattr(user, "language_code", None),
        "is_premium": getattr(user, "is_premium", None),
        "chat_id": str(getattr(chat, "id", "")) if getattr(chat, "id", None) is not None else None,
        "chat_type": getattr(chat, "type", None),
        "message_id": getattr(message, "message_id", None),
        "text": getattr(message, "text", None) or "",
        "timestamp": _to_iso(getattr(message, "date", None)),
        "source": "telegram",
    }
    return {k: v for k, v in candidates.items() if v is not None and v != ""}


# ---------------------------------------------------------------------------
# Генерация ответа
# ---------------------------------------------------------------------------

def _facts_from_matches(matches: List[Dict[str, Any]]) -> List[str]:
    facts = []
    for m in matches:
        text = (m.get("metadata") or {}).get("text")
        if text and text not in facts:
            facts.append(text)
    return facts


def build_system_prompt(facts: List[str]) -> str:
    known = "\n".join(f"- {f}" for f in facts) if facts else "пока ничего"
    return (
        "Ты дружелюбный ассистент. Вот что ты знаешь о пользователе:\n"
        f"{known}\n"
        "Используй эти факты, если они относятся к вопросу, и отвечай по-русски."
    )


def generate_reply(text: str, facts: List[str]) -> str:
    if llm_is_mock():
        reply = f"[эмуляция ответа LLM] Я запомнил: {text}"
        if facts:
            reply += "\nНайденные факты из памяти:\n" + "\n".join(f"• {f}" for f in facts)
        return reply

    try:
        response = get_llm_client().chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(facts)},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )
        return (response.choices[0].message.content or "").strip() or "…"
    except Exception as e:
        logger.error("Ошибка запроса к LLM: %s", e)
        return "Извини, сейчас не получается ответить — языковая модель недоступна. Но я тебя запомнил."


# ---------------------------------------------------------------------------
# Основная обработка (импортируется в emulate_validation.py)
# ---------------------------------------------------------------------------

def handle_user_text(user: Any, text: str, message: Any = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Обрабатывает одно текстовое сообщение пользователя.

    1) Ищет релевантную память пользователя.
    2) Генерирует ответ LLM с этими фактами в system prompt.
    3) Сохраняет в память ТОЛЬКО оригинальный текст пользователя (не ответ бота).

    Возвращает (reply, save_result). Для команд save_result = None.
    """
    if message is None:
        message = SimpleNamespace(
            from_user=user,
            chat=None,
            text=text,
            message_id=int(time.time() * 1000),
            date=int(time.time()),
        )

    user_id = str(getattr(user, "id", "unknown"))
    text = (text or "").strip()
    if not text:
        return "Пустое сообщение — нечего запоминать.", None
    if text.startswith("/"):
        # Команды не сохраняем
        return "Это команда — её я не запоминаю. Список команд: /help", None

    manager = get_manager()
    user_filter = {"user_id": user_id}

    # 1. Релевантная память
    matches = manager.query_by_text(text, top_k=MEMORY_TOP_K, filter=user_filter)
    facts = _facts_from_matches(matches)
    logger.info("user_id: %s | найдено фактов в памяти: %d", user_id, len(facts))

    # 2. Ответ LLM
    reply = generate_reply(text, facts)

    # 3. Сохраняем только исходный текст пользователя
    save_result = manager.upsert_document(
        doc_id=f"{user_id}_{getattr(message, 'message_id', int(time.time() * 1000))}",
        text=text,
        metadata=build_metadata(message),
    )

    score = save_result.get("similarity_score")
    logger.info(
        "user_id: %s | action: %s | score: %s",
        user_id, save_result.get("action"),
        f"{score:.3f}" if isinstance(score, (int, float)) else "n/a",
    )
    return reply, save_result


# ---------------------------------------------------------------------------
# Команды (общие для Telegram и консоли)
# ---------------------------------------------------------------------------

HELP_TEXT = (
    "Я бот с векторной памятью. Пиши мне что угодно — я запомню важное о тебе.\n\n"
    "/memory — показать, что я о тебе помню\n"
    "/forget — забыть всё о тебе\n"
    "/help — эта справка"
)


def cmd_start(user: Any) -> str:
    name = getattr(user, "first_name", None) or getattr(user, "username", None) or f"user_{getattr(user, 'id', '')}"
    return f"Привет, {name}! " + HELP_TEXT


def cmd_memory(user: Any) -> str:
    user_id = str(getattr(user, "id", "unknown"))
    matches = get_manager().query_by_text("что пользователь рассказывал о себе", top_k=10,
                                          filter={"user_id": user_id})
    if not matches:
        return "Я пока ничего о тебе не помню."
    lines = []
    for i, m in enumerate(matches, start=1):
        meta = m.get("metadata") or {}
        mentions = meta.get("mentions", 1)
        suffix = f" (упоминаний: {int(mentions)})" if mentions and int(mentions) > 1 else ""
        lines.append(f"{i}. {meta.get('text', '—')}{suffix}")
    return "Вот что я о тебе помню:\n" + "\n".join(lines)


def cmd_forget(user: Any) -> str:
    user_id = str(getattr(user, "id", "unknown"))
    ok = get_manager().delete_by_filter({"user_id": user_id})
    logger.info("user_id: %s | action: forget | ok: %s", user_id, ok)
    return "Готово, я всё забыл." if ok else "Не получилось очистить память, попробуй позже."


# ---------------------------------------------------------------------------
# Запуск: Telegram
# ---------------------------------------------------------------------------

def _safe_reply(bot, message, text: str) -> None:
    try:
        bot.reply_to(message, text[:MAX_TELEGRAM_LEN])
    except Exception as e:
        logger.error("Не удалось отправить ответ в Telegram: %s", e)


def run_telegram_bot(token: str) -> None:
    import telebot

    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def on_start(message):
        _safe_reply(bot, message, cmd_start(message.from_user))

    @bot.message_handler(commands=["help"])
    def on_help(message):
        _safe_reply(bot, message, HELP_TEXT)

    @bot.message_handler(commands=["memory"])
    def on_memory(message):
        try:
            _safe_reply(bot, message, cmd_memory(message.from_user))
        except Exception as e:
            logger.exception("Ошибка /memory: %s", e)
            _safe_reply(bot, message, "Не получилось прочитать память.")

    @bot.message_handler(commands=["forget"])
    def on_forget(message):
        try:
            _safe_reply(bot, message, cmd_forget(message.from_user))
        except Exception as e:
            logger.exception("Ошибка /forget: %s", e)
            _safe_reply(bot, message, "Не получилось очистить память.")

    @bot.message_handler(content_types=["text"])
    def on_text(message):
        try:
            reply, _ = handle_user_text(message.from_user, message.text, message=message)
        except Exception as e:
            # Одно сломанное сообщение не должно ронять бота
            logger.exception("Ошибка обработки сообщения: %s", e)
            reply = "Что-то пошло не так, попробуй ещё раз."
        _safe_reply(bot, message, reply)

    logger.info("Telegram-бот запущен, жду сообщений…")
    bot.infinity_polling(timeout=30, long_polling_timeout=30, logger_level=logging.WARNING)


# ---------------------------------------------------------------------------
# Запуск: консольный эмулятор
# ---------------------------------------------------------------------------

def run_console_emulator() -> None:
    # Фейковый пользователь БЕЗ first_name и last_name — проверяем фоллбэки build_metadata
    fake_user = SimpleNamespace(id=100500, is_bot=False)
    logger.warning("Telegram не используется (нет токена или MOCK_MODE=true) — консольный эмулятор")
    print("Консольный эмулятор бота. Команды: /start /memory /forget /help /exit\n")

    message_id = 0
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break

        message_id += 1
        try:
            if text == "/start":
                reply = cmd_start(fake_user)
            elif text == "/help":
                reply = HELP_TEXT
            elif text == "/memory":
                reply = cmd_memory(fake_user)
            elif text == "/forget":
                reply = cmd_forget(fake_user)
            else:
                msg = SimpleNamespace(from_user=fake_user, chat=None, text=text,
                                      message_id=message_id, date=int(time.time()))
                reply, _ = handle_user_text(fake_user, text, message=msg)
        except Exception as e:
            logger.exception("Ошибка обработки сообщения: %s", e)
            reply = "Что-то пошло не так, попробуй ещё раз."
        print(f"Бот: {reply}\n")

    print("Пока!")


if __name__ == "__main__":
    if MOCK_MODE or not is_real_value(TELEGRAM_BOT_TOKEN):
        run_console_emulator()
    else:
        run_telegram_bot(TELEGRAM_BOT_TOKEN)
