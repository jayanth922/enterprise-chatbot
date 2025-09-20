from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from orchestrator.topic_agent import TopicDecision, classify
from rag.docpack_manager import ensure_pack_for_decision
from rag.retriever import retrieve_from_pack


class TurnState(BaseModel):
    user_msg: str
    context: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    pack_key: str | None = None
    decision: TopicDecision | None = None
    mode: str = "clarify"  # "clarify" | "grounded"


def _require_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)


async def plan(state: TurnState) -> TurnState:
    d = classify(state.user_msg)
    state.decision = d

    # If we lack sources or confidence is low → clarify
    if not d.candidate_sources or d.confidence < 0.35:
        state.mode = "clarify"
        return state

    key, _ = await ensure_pack_for_decision(d, lang="en")
    state.pack_key = key
    state.mode = "grounded"
    return state


async def retrieve(state: TurnState) -> TurnState:
    if state.mode != "grounded" or not state.pack_key:
        return state
    ctx, cits = await retrieve_from_pack(state.pack_key, state.user_msg, k=20)
    state.context = ctx
    state.citations = cits
    if not cits:
        state.mode = "clarify"
    return state


async def stream_answer(state: TurnState) -> AsyncGenerator[str, None]:
    model = _require_model()

    if state.mode == "clarify":
        question = (
            state.decision.clarify
            if (state.decision and state.decision.clarify)
            else "Could you clarify the exact aspect you want help with?"
        )
        msgs = [
            SystemMessage(content="Ask the following exactly as a question and stop."),
            HumanMessage(content=question),
        ]
        async for chunk in model.astream(msgs):
            if getattr(chunk, "content", None):
                yield chunk.content
        return

    system = (
        "You are a grounded support assistant.\n"
        "- Use ONLY the provided context snippets and cite their URLs.\n"
        "- If context is insufficient, say what’s missing and stop."
    )
    payload = {
        "decision": state.decision.model_dump() if state.decision else None,
        "context": state.context,
        "citations": state.citations,
    }
    msgs = [
        SystemMessage(content=system),
        HumanMessage(content=f"User: {state.user_msg}\n\nGrounding:\n{json.dumps(payload)}"),
    ]
    async for chunk in model.astream(msgs):
        if getattr(chunk, "content", None):
            yield chunk.content


async def run_turn(payload: dict) -> AsyncGenerator[dict, None]:
    user_msg = payload["messages"][-1]["content"]
    state = TurnState(user_msg=user_msg)

    try:
        state = await plan(state)
        state = await retrieve(state)
    except Exception as e:
        yield {"role": "assistant", "content": f"(internal error handled) {e}"}
        return

    yield {"role": "assistant", "content": "(stream start)"}
    async for piece in stream_answer(state):
        yield {"role": "assistant", "content": piece}
    yield {"role": "assistant", "content": "(done)"}
