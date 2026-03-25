# FBTPbot-v2

Бот техподдержки с RAG (Retrieval-Augmented Generation) на базе YandexGPT. Автоматически отвечает на вопросы пользователей по базе знаний, классифицирует обращения и эскалирует сложные случаи оператору.

## Архитектура

- **Backend** — FastAPI + LangChain + ChromaDB + YandexGPT
- **Frontend** — React (Vite) — панель оператора, чат, встраиваемый виджет
- **Telegram-бот** — python-telegram-bot, работает как отдельный сервис
- **БД** — SQLite (aiosqlite + SQLAlchemy async)

## Возможности

- RAG-ответы по базе знаний (HTML/TXT → чанки → ChromaDB → YandexGPT)
- Классификация причин обращения и секций
- Эскалация на оператора (через Telegram и веб-панель)
- Администрирование базы знаний через веб-интерфейс
- Настройка бота через интерфейс конфигурации
- Встраиваемый чат-виджет для сайтов

## Быстрый старт

### 1. Клонировать

```bash
git clone https://github.com/rosavskiy/FBTPbot-v2.git
cd FBTPbot-v2
```

### 2. Настроить окружение

```bash
cp .env.example backend/.env
# Заполнить YANDEX_API_KEY, YANDEX_FOLDER_ID, TELEGRAM_BOT_TOKEN, TELEGRAM_SUPPORT_CHAT_ID
```

### 3. Запуск через Docker Compose

```bash
docker compose up -d --build
```

Сервисы:
- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:80`

### 4. Локальная разработка (без Docker)

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Структура проекта

```
backend/
  app/
    api/          — HTTP-эндпоинты (чат, оператор, KB-админ, конфиг бота)
    classifier/   — Классификация причин и секций обращений
    database/     — Модели и сервис SQLAlchemy
    indexer/      — Индексация базы знаний в ChromaDB
    models/       — Pydantic-схемы
    parser/       — HTML-парсер документации
    rag/          — RAG-движок (запросы, сессии, классификация)
    tg/           — Telegram-бот и нотификации
  data/           — Рантайм-данные (БД, ChromaDB)
  static/         — HTML-админки
frontend/
  src/
    api/          — HTTP-клиент
    chat/         — Страница чата
    operator/     — Панель оператора
    widget/       — Встраиваемый виджет
  scripts/        — Скрипт импорта причин обращения из .docx
brains/           — Исходные документы для базы знаний
templates/        — Шаблоны .docx для импорта причин обращения
```
