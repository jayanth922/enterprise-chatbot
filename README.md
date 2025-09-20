# Enterprise Support Bot — Docs-Only, Citation-First

**A minimal, production-style support chatbot that answers _only_ when grounded in official documentation.**  
When a user asks a question, the bot:

1) infers the domain/subtopic with an LLM,  
2) fetches **official docs** on demand,  
3) indexes & retrieves with **FAISS (E5 embeddings) + BGE reranker**, and  
4) streams a response **with citations**.  
If it can’t ground confidently, it asks **one clarifying question**—no hallucinations.

---

## Core Features

- **Docs-Only Grounding**: Pulls from official sources (e.g., `kubernetes.io`, `docs.python.org`, `redis.io`, `react.dev`, `postgresql.org`, `ubuntu.com`, `elastic.co`).
- **On-Demand Ingestion**: Downloads relevant pages for the detected subtopics, extracts clean text, and reuses it across similar queries (DocPacks).
- **Strong Retrieval**: E5 embeddings + FAISS ANN → BGE cross-encoder rerank for high-precision snippets.
- **Streaming UX**: End-to-end streaming (Server-Sent Events) for snappy responses.
- **Fail-Safe Behavior**: If confidence is low or sources are missing, the bot asks **one** crisp clarifying question.
- **Debug Visibility**: `/debug/packs` (what’s indexed) and `/debug/search` (top-k results with scores).

---

## Architecture (High Level)

User → FastAPI /chat (SSE)
→ LLM Classifier (domain, subtopics, official sources, confidence, optional clarifier)
→ If low confidence/no sources: CLARIFY (one question)
→ Else: Ensure DocPack
- Ingest official docs (httpx + BeautifulSoup)
- Extract text → chunk → E5 embeddings → FAISS upsert
→ Query: E5 embed → FAISS ANN → BGE rerank → top-k context
→ Answer strictly from context + cite URLs




**Key Components**
- `orchestrator/topic_agent.py` — LLM returns strict JSON (domain, subtopics, official sources, confidence, `clarify`, `needs_grounding`).
- `rag/ingest.py` — Fetches & cleans text from official docs (same-domain, subtopic-relevant links).
- `rag/retriever.py` — E5 embeddings + FAISS (cosine via normalized IP) + BGE reranker → precise context.
- `orchestrator/graph.py` — Two routes: **clarify** or **grounded**; streams tokens.
- `api/main.py` — FastAPI app exposing `/chat` (SSE) and debug endpoints.
- `ui/app.py` — Streamlit chat client (receives SSE chunks and renders streaming text).

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **LLM**: Groq Cloud via `langchain-groq` (e.g., `llama-3.1-8b-instant`)
- **RAG**: Sentence Transformers `intfloat/e5-large-v2` + FAISS-CPU + `BAAI/bge-reranker-large`
- **Ingestion**: `httpx`, `beautifulsoup4`, `lxml`
- **Frontend**: Streamlit
- **Tooling**: `uv` (package & runner), Ruff (lint/format)

---

## Getting Started

### Prerequisites
- Python **3.12+**
- [`uv`](https://github.com/astral-sh/uv) (`pipx install uv` or `pip install uv`)
- A **Groq API key** (`GROQ_API_KEY`)

### Setup
```bash
git clone https://github.com/<you>/enterprise-support-bot.git
cd enterprise-support-bot

cp .env.example .env        # add GROQ_API_KEY=...
uv sync                     # install dependencies

Run the API

uv run uvicorn api.main:app --reload --port 8000
# Health: http://localhost:8000/health  → {"ok": true}


Run the UI

uv run streamlit run ui/app.py
# Open the URL shown in the console (usually http://localhost:8501)


