# RAG-ассистент для ювелирного магазина (FAISS)

Демо-проект: отвечает на вопросы по базе знаний и мини-каталогу изделий.  
Локальный векторный поиск на **FAISS** + эмбеддинги **HuggingFace**.  
LLM для краткого ответа — **ChatOpenAI**.

## Зачем это
- Быстрый поиск по описаниям товаров и FAQ.
- Точность по ценам и назначению изделий (данные берутся из каталога).
- Эмбеддинги локально, без внешних сервисов для векторизации.

## Стек
`Python`, `LangChain`, `FAISS`, `sentence-transformers/all-MiniLM-L6-v2`, `ChatOpenAI`.

## Установка и запуск (локально)
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# создать .env из .env.example и вставить ключ:
# OPENAI_API_KEY=sk-...
python faiss_demo.py
