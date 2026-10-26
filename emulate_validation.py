"""
Эмуляция финальной проверки задания VPg05 без Telegram.

Прогоняет фразы через ту же функцию handle_user_text, что использует бот,
от имени пользователя без first_name/last_name, и проверяет критерий успеха:
фраза «Я полечу на Марс» должна быть распознана как дубликат (updated/skipped).
"""

import logging
import os
import time
from types import SimpleNamespace

from pinecone_manager import LOG_DIR, PineconeManager, setup_logging

import bot

setup_logging()
logger = logging.getLogger("emulate_validation")

REPORT_FILE = os.path.join(LOG_DIR, "validation_report.txt")

PHRASES = [
    "Привет",
    "Зачем люди хотят лететь на Марс?",
    "Хочу на Марс",
    "Я полечу на Марс",
    "Я люблю пиццу",
]
KEY_PHRASE = "Я полечу на Марс"


def main() -> None:
    manager = PineconeManager()  # реальный, если есть ключи, иначе эмуляция
    bot.set_manager(manager)

    logger.info("Удаляем тестовый индекс (delete_all)")
    manager.delete_all()
    if not manager.mock:
        time.sleep(2)

    # Пользователь только с id: проверяем фоллбэки build_metadata
    user = SimpleNamespace(id=777001, is_bot=False)

    rows = []
    for i, phrase in enumerate(PHRASES, start=1):
        message = SimpleNamespace(from_user=user, chat=None, text=phrase,
                                  message_id=i, date=int(time.time()))
        reply, result = bot.handle_user_text(user, phrase, message=message)
        rows.append((phrase, result))
        if not manager.mock:
            time.sleep(1)  # Pinecone индексирует запись не мгновенно

    mode = "эмуляция (mock)" if manager.mock else "реальный (OpenAI + Pinecone)"
    lines = [
        "ОТЧЁТ ЭМУЛЯЦИИ ПРОВЕРКИ VPg05",
        f"Режим: {mode}",
        f"Порог SIMILARITY_THRESHOLD: {manager.similarity_threshold:.3f}",
        f"Пользователь: id={user.id}, display_name=user_{user.id} (без first_name/last_name)",
        "",
        f"{'Фраза':<34} | {'action':<9} | {'score':<6} | {'id':<10} | existing_id",
        "-" * 82,
    ]
    for phrase, res in rows:
        score = res.get("similarity_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        lines.append(
            f"{phrase:<34} | {res['action']:<9} | {score_str:<6} | {res['id']:<10} | {res.get('existing_id') or '—'}"
        )

    key_action = next(res["action"] for phrase, res in rows if phrase == KEY_PHRASE)
    lines.append("")
    if key_action in ("updated", "skipped"):
        lines.append(f"КРИТЕРИЙ УСПЕХА ВЫПОЛНЕН: для фразы «{KEY_PHRASE}» action = {key_action}")
    else:
        lines.append(
            f"КРИТЕРИЙ УСПЕХА НЕ ВЫПОЛНЕН: для фразы «{KEY_PHRASE}» action = {key_action}. "
            "Попробуйте уменьшить SIMILARITY_THRESHOLD в local.env (например, до 0.4) и запустить снова."
        )

    lines.append("")
    lines.append(f"Итог в индексе: {manager.describe_index_stats().get('total_vector_count', '?')} векторов")
    lines.append("Сохранённые факты:")
    for m in manager.query_by_text("что пользователь рассказывал", top_k=10, filter={"user_id": str(user.id)}):
        meta = m["metadata"]
        lines.append(f"  {m['id']:<10} | mentions={int(meta.get('mentions', 1))} | {meta.get('text')}")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info("Отчёт сохранён в %s", REPORT_FILE)


if __name__ == "__main__":
    main()
