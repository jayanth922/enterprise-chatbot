from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

from orchestrator.topic_agent import TopicDecision
from rag.ingest import ingest_stage_a


@dataclass
class Manifest:
    tech: str | None
    version: str | None
    lang: str
    sources: List[str]
    completeness: float = 0.0
    ttl_days: int = 14


_packs: dict[str, Manifest] = {}
_ingest_log: dict[str, List[str]] = {}  # pack_key -> recent urls (last 50)


def _key_from_sources(
    tech: str | None, version: str | None, lang: str, sources: List[str]
) -> str:
    raw = f"{tech or 'generic'}|{version or 'latest'}|{lang}|{','.join(sorted(sources))}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_manifest(key: str) -> Manifest | None:
    return _packs.get(key)


def _log_ingest(key: str, urls: List[str]) -> None:
    if not urls:
        return
    prev = _ingest_log.get(key, [])
    merged = prev + urls
    seen = set()
    out: List[str] = []
    for u in merged[-200:]:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    _ingest_log[key] = out[-50:]


async def _bg_ingest(key: str, sources: List[str], subs: List[str], limit: int) -> None:
    count, urls = await ingest_stage_a(
        key, sources=sources, subtopics=subs, max_pages_per_source=limit
    )
    _log_ingest(key, urls)
    if count:
        _packs[key].completeness = min(1.0, max(_packs[key].completeness, 0.6))


async def ensure_pack_for_decision(
    decision: TopicDecision, lang: str = "en"
) -> Tuple[str, str]:
    tech = decision.tech
    version = decision.version or "latest"
    sources = decision.candidate_sources or []
    key = _key_from_sources(tech, version, lang, sources)

    if key in _packs:
        return key, "ready"

    _packs[key] = Manifest(tech=tech, version=version, lang=lang, sources=sources)
    count, urls = await ingest_stage_a(
        key, sources=sources, subtopics=decision.subtopics, max_pages_per_source=15
    )
    _log_ingest(key, urls)
    _packs[key].completeness = 0.3 if count else 0.0

    asyncio.create_task(_bg_ingest(key, sources, decision.subtopics, limit=30))
    return key, ("ready" if count else "building")


def pack_summaries() -> List[dict]:
    from rag.retriever import get_index_stats  # lazy import

    out: List[dict] = []
    for key, man in _packs.items():
        stats = get_index_stats(key)
        out.append(
            {
                "key": key,
                "tech": man.tech,
                "version": man.version,
                "lang": man.lang,
                "sources": man.sources,
                "completeness": man.completeness,
                "ntotal": stats.get("ntotal", 0),
                "recent_urls": _ingest_log.get(key, [])[-10:],
            }
        )
    return out
