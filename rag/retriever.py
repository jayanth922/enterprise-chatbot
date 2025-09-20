from __future__ import annotations

from typing import Any, Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

_pack_indexes: dict[str, dict[str, Any]] = {}
_embed_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def _get_embed() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("intfloat/e5-large-v2")
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-large")
    return _reranker


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def _ensure_index(pack_key: str, dim: int = 1024) -> None:
    if pack_key not in _pack_indexes:
        index = faiss.IndexFlatIP(dim)
        _pack_indexes[pack_key] = {"index": index, "meta": []}


def upsert_documents(pack_key: str, docs: List[Dict[str, str]]) -> None:
    _ensure_index(pack_key)
    model = _get_embed()
    texts = [f"passage: {d['text']}" for d in docs]
    vecs = model.encode(texts, convert_to_numpy=True)
    vecs = _normalize(vecs.astype("float32"))
    _pack_indexes[pack_key]["index"].add(vecs)
    _pack_indexes[pack_key]["meta"].extend(docs)


async def retrieve_from_pack(
    pack_key: str, query: str, k: int = 20
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _ensure_index(pack_key)
    index = _pack_indexes[pack_key]["index"]
    meta = _pack_indexes[pack_key]["meta"]
    if index.ntotal == 0:
        return [], []

    embed = _get_embed()
    q_vec = embed.encode([f"query: {query}"], convert_to_numpy=True)
    q_vec = _normalize(q_vec.astype("float32"))

    pre_k = min(max(k * 3, 30), index.ntotal)
    scores, idxs = index.search(q_vec, pre_k)
    idxs = [i for i in idxs[0].tolist() if i != -1]
    cands = [meta[i] for i in idxs]

    pairs = [(query, c["text"]) for c in cands]
    re_scores = _get_reranker().predict(pairs)

    ranked = sorted(
        zip(cands, re_scores), key=lambda x: float(x[1]), reverse=True
    )[:k]
    hits = [
        {"text": c["text"], "url": c["url"], "title": c["title"], "score": float(s)}
        for c, s in ranked
    ]

    context = [{"text": h["text"], "url": h["url"], "title": h["title"]} for h in hits]
    citations = [{"title": h["title"], "url": h["url"], "score": h["score"]} for h in hits[:4]]
    return context, citations


def get_index_stats(pack_key: str) -> dict:
    _ensure_index(pack_key)
    idx = _pack_indexes[pack_key]["index"]
    meta = _pack_indexes[pack_key]["meta"]
    return {"ntotal": idx.ntotal, "meta_count": len(meta)}
