from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List

from orchestrator.graph import run_turn
from rag.docpack_manager import pack_summaries
from rag.retriever import retrieve_from_pack

load_dotenv()

app = FastAPI(title="Enterprise Support Bot (Docs-Only)")


class ChatRequest(BaseModel):
    messages: List[Dict]


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/chat")
async def chat(payload: ChatRequest):
    async def gen():
        async for delta in run_turn(payload.model_dump()):
            yield "data: " + json.dumps(delta) + "\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- Debug endpoints (optional but very useful) ----------

class DebugSearchReq(BaseModel):
    pack_key: str
    query: str
    k: int = 8


@app.get("/debug/packs")
def debug_packs() -> Dict[str, List[Dict]]:
    return {"packs": pack_summaries()}


@app.post("/debug/search")
async def debug_search(body: DebugSearchReq) -> Dict[str, List[Dict]]:
    ctx, cits = await retrieve_from_pack(body.pack_key, body.query, k=body.k)
    return {"context": ctx, "citations": cits}
