from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List

import requests
import streamlit as st

API_URL = os.getenv("BOT_API_URL", "http://localhost:8000")


@contextmanager
def sse_stream(url: str, payload: Dict) -> Iterator[Iterable[str]]:
    headers = {"content-type": "application/json"}
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()

        def gen() -> Iterable[str]:
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    yield raw[6:]

        yield gen()


def send_and_stream(question: str) -> Iterator[str]:
    body = {"messages": [{"role": "user", "content": question}]}
    chat_url = f"{API_URL}/chat"
    with sse_stream(chat_url, body) as chunks:
        for data in chunks:
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("role") == "assistant":
                piece = obj.get("content", "")
                if piece and piece not in ("(stream start)", "(done)"):
                    yield piece


# ---------- UI ----------
st.set_page_config(page_title="Enterprise Support Bot", page_icon="🤖", layout="centered")

st.sidebar.markdown("### Settings")
api_url_input = st.sidebar.text_input("API URL", API_URL)
if api_url_input != API_URL:
    os.environ["BOT_API_URL"] = api_url_input

st.title("🤖 Enterprise Support Bot (Docs-Only)")
st.caption("Backed by FastAPI · LangChain · Groq · FAISS/E5 (+ reranker)")

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# Debug panel (optional)
st.sidebar.markdown("---")
if st.sidebar.button("Refresh DocPacks"):
    try:
        r = requests.get(f"{api_url_input}/debug/packs", timeout=30)
        r.raise_for_status()
        st.session_state["debug_packs"] = r.json().get("packs", [])
    except Exception as e:
        st.sidebar.error(f"Debug error: {e}")

packs = st.session_state.get("debug_packs", [])
if packs:
    with st.expander("DocPacks"):
        for p in packs:
            st.write(f"**Key:** `{p['key']}`")
            st.write(
                f"Domain: {p.get('tech')} · Version: {p.get('version')} · "
                f"Vectors: {p.get('ntotal')} · Completeness: {p.get('completeness')}"
            )
            st.write("Sources:")
            for s in p.get("sources", []):
                st.write(f"- {s}")
            if p.get("recent_urls"):
                st.write("Recent URLs:")
                for u in p["recent_urls"]:
                    st.write(f"- {u}")
            st.markdown("---")

# Chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything (e.g., 'Kubernetes Ingress basics')")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            for piece in send_and_stream(prompt):
                acc += piece
                placeholder.markdown(acc)
        except requests.HTTPError as e:
            placeholder.error(f"Server error: {e.response.status_code} {e.response.reason}")
        except requests.RequestException as e:
            placeholder.error(f"Network error: {e}")
        except Exception as e:  # noqa: BLE001
            placeholder.error(f"Unexpected error: {e}")

        st.session_state.history.append({"role": "assistant", "content": acc or "(no output)"})
