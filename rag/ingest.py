from __future__ import annotations

from typing import List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from rag.retriever import upsert_documents


def _same_domain(url: str, candidate: str) -> bool:
    u, c = urlparse(url), urlparse(candidate)
    return (u.netloc == c.netloc) and c.scheme in ("http", "https")


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup.find("article") or soup
    parts: List[str] = []
    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "code", "pre"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            parts.append(txt)
    return "\n".join(parts)[:5000]


async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=20)
    r.raise_for_status()
    return r.text


async def _candidate_links(
    client: httpx.AsyncClient, base_url: str, keywords: List[str], limit: int = 30
) -> List[str]:
    html = await _fetch(client, base_url)
    soup = BeautifulSoup(html, "lxml")
    hrefs: List[str] = []
    kws = [k.lower() for k in keywords if k.strip()]
    for a in soup.find_all("a", href=True):
        cand = urljoin(base_url, a["href"])
        if not _same_domain(base_url, cand):
            continue
        slug = cand.lower()
        if any(kw in slug for kw in kws):
            hrefs.append(cand)
    hrefs = [base_url] + list(dict.fromkeys(hrefs))
    return hrefs[:limit]


async def ingest_stage_a(
    pack_key: str,
    sources: List[str],
    subtopics: List[str],
    max_pages_per_source: int = 20,
) -> tuple[int, List[str]]:
    """
    Fetch a small, relevant slice quickly; return (num_pages_ingested, urls).
    """
    if not sources:
        return 0, []
    total = 0
    all_urls: List[str] = []
    async with httpx.AsyncClient(
        follow_redirects=True, headers={"User-Agent": "DocPack/1"}
    ) as c:
        for base in sources:
            try:
                links = await _candidate_links(
                    c, base, subtopics, limit=max_pages_per_source
                )
            except Exception:
                continue
            all_urls.extend(links)
            docs = []
            for url in links:
                try:
                    html = await _fetch(c, url)
                    text = _extract_text(html)
                    title = BeautifulSoup(html, "lxml").title
                    docs.append(
                        {
                            "title": (title.get_text(strip=True) if title else url),
                            "url": url,
                            "text": text,
                        }
                    )
                except Exception:
                    continue
            if docs:
                upsert_documents(pack_key, docs)
                total += len(docs)
    # dedupe, keep the last 100
    seen = set()
    dedup: List[str] = []
    for u in all_urls:
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    return total, dedup[-100:]
