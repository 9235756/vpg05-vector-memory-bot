# VPg05 — Диалоговый бот с векторной памятью

Telegram-бот, который запоминает, что говорит пользователь, в векторной базе Pinecone. Перед каждой записью менеджер ищет похожую информацию: если косинусное сходство не ниже `SIMILARITY_THRESHOLD`, это дубликат и существующий слот обновляется, иначе вставляется новый вектор.

## Файлы

| Файл | Что внутри |
|---|---|
| `pinecone_manager.py` | `PineconeManager`: эмбеддинги, проверка сходства, upsert/query/delete, режим эмуляции |
| `bot.py` | Telegram-бот, `handle_user_text()`, `build_metadata()`, консольный эмулятор |
| `emulate_validation.py` | Эмуляция финальной проверки, отчёт в `logs/validation_report.txt` |
| `local.env.example` | Шаблон настроек |

## Установка

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp local.env.example local.env   # и вписать ключи
```

Windows:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy local.env.example local.env
```

Настройки читаются из `local.env` (не из `.env`).

## Запуск

```bash
python pinecone_manager.py    # ручной тест менеджера
python emulate_validation.py  # эмуляция проверки, работает без ключей
python bot.py                 # запуск бота
```

## Индекс Pinecone

Если индекса с именем `PINECONE_INDEX_NAME` нет, он создаётся автоматически: serverless, `cloud=aws`, `region=PINECONE_ENVIRONMENT`, **dimension=1536**, **metric=cosine** (под `text-embedding-3-small`).

## Режим эмуляции

Включается, если `MOCK_MODE=true` или в `local.env` нет ключей Pinecone/OpenAI (плейсхолдеры `your_...` считаются отсутствующими):

- вместо Pinecone — `InMemoryIndex` в памяти, ответы той же структуры;
- вместо OpenAI-эмбеддингов — детерминированный `mock_embedding` (слова, символьные триграммы и небольшой словарь смысловых концептов, L2-нормализация);
- вместо LLM — ответ `[эмуляция ответа LLM] Я запомнил: ...`;
- без `TELEGRAM_BOT_TOKEN` бот запускается как консольный эмулятор от пользователя без имени.

## Порог сходства

`SIMILARITY_THRESHOLD` (по умолчанию 0.5): score ≥ порога — дубликат, score < порога — новая информация. Если реальные эмбеддинги не склеивают «Хочу на Марс» и «Я полечу на Марс», уменьшите порог, например до 0.4.

## Логи

Консоль и `logs/bot.log`. Каждая запись логируется строкой `action: <action> | score: <x.xxx> | id: <id>`.
