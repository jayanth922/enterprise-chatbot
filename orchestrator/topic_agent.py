from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError


class TopicDecision(BaseModel):
    # "tech" is a generic domain/subject (Python, Kubernetes, Redis, History, etc.)
    tech: Optional[str] = Field(None, description="Domain/subject inferred from query")
    subtopics: List[str] = Field(default_factory=list, description="Aspects within domain")
    problem_focus: str = Field("unknown", description="specific_issue|general_overview|unknown")
    version: Optional[str] = Field(None, description="Version if implied, else null")
    candidate_sources: List[str] = Field(default_factory=list, description="Official base URLs")
    confidence: float = Field(0.0, description="0..1")
    query_type: str = Field(
        "unknown",
        description=(
            "general_explanation|definition|how_to|troubleshoot|compare|code_help|qa|unknown"
        ),
    )
    needs_grounding: bool = Field(False, description="Should we RAG on this turn?")
    clarify: str = Field("", description="Single clarifying question if ambiguous; else ''")
    rationale: str = Field("", description="Short reason (debug)")


def _model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0)) if m else {}


def _coerce(d: dict) -> dict:
    out = dict(d or {})
    out["subtopics"] = out.get("subtopics") or []
    out["candidate_sources"] = out.get("candidate_sources") or []
    pf = str(out.get("problem_focus", "unknown")).lower()
    out["problem_focus"] = pf if pf in {"specific_issue", "general_overview", "unknown"} else "unknown"
    qt = str(out.get("query_type", "unknown")).lower()
    allowed = {"general_explanation", "definition", "how_to", "troubleshoot", "compare", "code_help", "qa", "unknown"}
    out["query_type"] = qt if qt in allowed else "unknown"
    try:
        c = float(out.get("confidence", 0.0))
    except Exception:
        c = 0.0
    out["confidence"] = max(0.0, min(1.0, c))
    t = out.get("tech")
    out["tech"] = (str(t).strip() or None) if t is not None else None
    v = out.get("version")
    out["version"] = (str(v).strip() or None) if v is not None else None
    out["needs_grounding"] = bool(out.get("needs_grounding", False))
    out["clarify"] = str(out.get("clarify", "")).strip()
    outs = []
    for s in out["candidate_sources"]:
        s = str(s).strip()
        if s.startswith("http://") or s.startswith("https://"):
            outs.append(s)
    out["candidate_sources"] = outs
    return out


def classify(user_msg: str) -> TopicDecision:
    """
    LLM decides domain/subtopics, whether to ground via docs, and (if needed)
    one clarifying question. Returns a safe TopicDecision even on malformed JSON.
    """
    system = (
        "Extract intent and scope for a docs-grounded assistant. "
        "Prefer official documentation sources (kubernetes.io, docs.python.org, "
        "redis.io, postgresql.org, react.dev, ubuntu.com, elastic.co, etc.). "
        "Return STRICT JSON with: tech, subtopics, problem_focus "
        "(specific_issue|general_overview|unknown), version, candidate_sources, "
        "confidence (0..1), query_type (general_explanation|definition|how_to|"
        "troubleshoot|compare|code_help|qa|unknown), needs_grounding (bool), "
        "clarify (one short question or ''), rationale."
    )
    user = f"User query:\n{user_msg}"

    try:
        resp = _model().invoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw = getattr(resp, "content", "{}")
        data = _coerce(_extract_json(raw))
        return TopicDecision.model_validate(data)
    except ValidationError:
        return TopicDecision()
    except Exception:
        return TopicDecision()
