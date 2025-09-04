# RAG-ассистент для ювелирного магазина: FAISS + HuggingFace embeddings + ChatOpenAI
# Безопасность: ключ берётся из .env или интерактивного ввода (в коде не хранится).

import os, json
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------- 0) Ключ ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        from getpass import getpass
        os.environ["OPENAI_API_KEY"] = getpass("Вставь OPENAI_API_KEY: ")
    except Exception:
        raise RuntimeError("Не найден OPENAI_API_KEY (создай .env из .env.example).")

# ---------- 1) Данные (мини-демо) ----------
knowledge = [
    {"text": "Бриллиантовые кольца подходят для особых случаев, таких как помолвка, свадьба или юбилей.", "source": "https://example.com/product/1"},
    {"text": "Изделия с сапфирами и бриллиантами лучше носить на торжественные мероприятия.", "source": "https://example.com/product/10"},
    {"text": "Золотые цепочки с кулонами — универсальный подарок на день рождения.", "source": "https://example.com/product/5"},
    {"text": "Серьги с рубинами подойдут для романтического ужина или вечернего выхода.", "source": "https://example.com/product/6"},
    {"text": "Чистить золото зубной пастой нельзя: абразивы повреждают поверхность и камни.", "source": "https://example.com/care/1"}
]

catalog = [
  {"name": "Кольцо с бриллиантом", "description": "Бриллиант 0.5 карат.", "usage": "Помолвка/свадьба.", "price": "15000 руб.", "url": "https://example.com/product/1"},
  {"name": "Серебряные серьги с аметистами", "description": "Серебро 925 с аметистами.", "usage": "Ежедневно/торжественно.", "price": "8000 руб.", "url": "https://example.com/product/2"},
  {"name": "Золотая подвеска с изумрудом", "description": "14K золото, натуральный изумруд.", "usage": "Повседневно/вечер.", "price": "22000 руб.", "url": "https://example.com/product/3"},
  {"name": "Браслет с цирконами", "description": "Тонкий браслет с цирконами.", "usage": "Праздники/аксессуар.", "price": "3500 руб.", "url": "https://example.com/product/4"},
  {"name": "Серьги с жемчугом", "description": "Классические серьги с жемчугом.", "usage": "Повседневно/элегантно.", "price": "10000 руб.", "url": "https://example.com/product/5"},
  {"name": "Часы с бриллиантами", "description": "Часы с инкрустацией бриллиантами.", "usage": "Изысканный стиль.", "price": "50000 руб.", "url": "https://example.com/product/6"},
  {"name": "Кулон с топазом", "description": "Кулон с топазом, серебро.", "usage": "Праздники/ежедневно.", "price": "7000 руб.", "url": "https://example.com/product/7"},
  {"name": "Золотые обручальные кольца", "description": "Пара обручальных колец, матовая отделка.", "usage": "Свадьба.", "price": "25000 руб.", "url": "https://example.com/product/8"},
  {"name": "Печатка с ониксом", "description": "Мужская печатка с ониксом.", "usage": "Выразительный стиль.", "price": "12000 руб.", "url": "https://example.com/product/9"},
  {"name": "Колье с сапфирами", "description": "Колье с сапфирами и бриллиантами.", "usage": "Особые случаи/вечер.", "price": "45000 руб.", "url": "https://example.com/product/10"}
]

# ---------- 2) Документы ----------
knowledge_docs = [Document(page_content=i["text"], metadata={"source": i.get("source","")}) for i in knowledge]
catalog_docs   = [Document(
    page_content=f"{c['name']}. {c['description']} {c['usage']}",
    metadata={"name": c["name"], "price": c["price"], "url": c["url"], "usage": c["usage"]}
) for c in catalog]

# ---------- 3) Эмбеддинги + FAISS ----------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
knowledge_db = FAISS.from_documents(knowledge_docs, embedding)
catalog_db   = FAISS.from_documents(catalog_docs, embedding)

knowledge_retriever = knowledge_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.35}
)
catalog_retriever = catalog_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.30}
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs) if docs else ""

def format_products(docs):
    if not docs:
        return ""
    lines = []
    for d in docs:
        name  = d.metadata.get("name","")
        price = d.metadata.get("price","")
        usage = d.metadata.get("usage","")
        lines.append(f"- {name} — {price}. {usage}")
    return "\n".join(lines)

# ---------- 4) Промпт + LLM + цепочка ----------
prompt = PromptTemplate(
    input_variables=["context", "products", "question"],
    template=(
        "Роль: эксперт по ювелирным изделиям.\n"
        "Правила:\n"
        "- Используй ТОЛЬКО факты из контекста.\n"
        "- Если ответа нет в контексте — напиши: «Нет данных в базе».\n"
        "- Если вопрос про цену — дай ТОЧНУЮ цену из блока «Товары».\n"
        "- 1–2 коротких предложения, без воды и ссылок.\n\n"
        "Контекст:\n{context}\n\n"
        "Товары:\n{products}\n\n"
        "Вопрос: {question}\n"
        "Краткий ответ:"
    ),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=160)

chain = (
    RunnableMap({
        "context":  lambda x: format_docs(knowledge_retriever.invoke(x["question"])),
        "products": lambda x: format_products(catalog_retriever.invoke(x["question"])),
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

def ask(q: str) -> str:
    return chain.invoke({"question": q})

if __name__ == "__main__":
    tests = [
        "Можно ли чистить золото зубной пастой?",
        "Что подарить женщине на юбилей 50 лет?",
        "Как ухаживать за золотыми украшениями с камнями?",
        "Сколько стоят серьги с жемчугом?",
        "Сколько стоит колье с сапфирами?"
    ]
    for q in tests:
        print("Q:", q)
        print("A:", ask(q))
        print("-"*60)
