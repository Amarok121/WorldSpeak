"""Narrator + ConsistencyCritic graph.

Uses OpenAI when OPENAI_API_KEY is set; otherwise a deterministic mock LLM
so reviewers can run the demo without cloud credentials.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .rag import LoreChunk, format_context, retrieve


HARD_RULES = [
    ("magic", re.compile(r"\b(spell|wizard|mana|cast magic|마법)\b", re.I)),
    ("teleport", re.compile(r"\b(teleport|순간이동|워프)\b", re.I)),
    ("flying_car", re.compile(r"\b(flying car|비행차)\b", re.I)),
    # Positive rule breaks only: Mira actually enters harbor alone.
    (
        "mira_alone_harbor",
        re.compile(
            r"mira[^\.]{0,80}\b(enters|entered|goes|went|walks)\b[^\.]{0,40}\b(deep )?harbor\b[^\.]{0,40}\balone\b",
            re.I,
        ),
    ),
    # Positive rule breaks only: Orren accepts a personal bribe.
    (
        "orren_bribe",
        re.compile(
            r"orren[^\.]{0,80}\b(accepts|accepted|takes|took|pockets)\b[^\.]{0,40}\b(bribe|personal (scrap )?tokens?)\b",
            re.I,
        ),
    ),
]


class GraphState(TypedDict, total=False):
    user_input: str
    location: str
    flags: dict[str, Any]
    lore_context: str
    retrieved_ids: list[str]
    draft: str
    critique: str
    ok: bool
    retries: int
    final: str
    choices: list[str]


def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _chat(system: str, user: str) -> str:
    if _has_openai():
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0.4)
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(msg.content)

    # Deterministic mock: stays inside Ironwick rules.
    lower = user.lower()
    if "reject" in lower or "거절" in lower:
        scene = (
            "Mira taps her copper ear-cuff and lets the sealed chip stay on the stall. "
            "Fog thickens over Hook Street. Orren watches from a lantern post but does not approach. "
            "You keep your scrap tokens. The night market noise feels suddenly too loud."
        )
        choices = [
            "Ask Mira what the chip would have cost",
            "Walk toward Coil Yards",
            "Signal Orren without speaking",
        ]
    elif "bargain" in lower or "흥정" in lower:
        scene = (
            "You push a partial token stack forward. Mira weighs it, then slides the chip halfway back. "
            "No magic, no shortcuts—only contract language. A drone rattles overhead with cargo, not passengers."
        )
        choices = [
            "Accept the partial deal",
            "Refuse and leave Hook Street",
            "Offer information instead of tokens",
        ]
    else:
        scene = (
            "On Hook Street, Mira Vale sets a sealed memory-chip between you. "
            "Her ear-cuff buzzes once - contract tension, not sorcery. "
            "Deep Harbor lanterns glow far away; she will not go there alone. "
            "Orren Kite keeps distance, unpaid by personal scrap tokens."
        )
        choices = [
            "Accept the sealed chip",
            "Refuse the chip",
            "Bargain with information",
        ]
    return json.dumps({"scene": scene, "choices": choices}, ensure_ascii=False)


def _parse_story(raw: str) -> tuple[str, list[str]]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        scene = str(data.get("scene", "")).strip()
        choices = [str(c).strip() for c in data.get("choices", []) if str(c).strip()]
        if scene and len(choices) >= 3:
            return scene, choices[:3]
    except json.JSONDecodeError:
        pass
    # Fallback: split trailing choices if model ignored JSON.
    lines = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    choices = lines[-3:] if len(lines) >= 3 else [
        "Look around",
        "Ask a question",
        "Leave the area",
    ]
    scene = "\n".join(lines[:-3]) if len(lines) > 3 else raw
    return scene, choices


def critic_violations(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in HARD_RULES:
        if pattern.search(text):
            hits.append(name)
    return hits


def build_graph(chunks: list[LoreChunk]):
    def retrieve_node(state: GraphState) -> GraphState:
        query = f"{state.get('user_input', '')} {state.get('location', '')}"
        picked = retrieve(query, chunks, top_k=3)
        return {
            "lore_context": format_context(picked),
            "retrieved_ids": [c.doc_id for c in picked],
        }

    def narrator_node(state: GraphState) -> GraphState:
        system = (
            "You are Narrator for a text virtual world. "
            "Obey lore_context hard rules. No magic. No teleport. No flying cars. "
            "Keep 80-120 words. Reply ONLY as JSON: "
            '{"scene":"...","choices":["...","...","..."]}'
        )
        user = (
            f"location={state.get('location')}\n"
            f"flags={json.dumps(state.get('flags', {}), ensure_ascii=False)}\n"
            f"player={state.get('user_input')}\n"
            f"critique={state.get('critique', '')}\n"
            f"lore_context:\n{state.get('lore_context', '')}"
        )
        raw = _chat(system, user)
        scene, choices = _parse_story(raw)
        return {"draft": scene, "choices": choices}

    def critic_node(state: GraphState) -> GraphState:
        draft = state.get("draft", "")
        violations = critic_violations(draft)
        # Also reject empty or choice-less drafts.
        choices = state.get("choices") or []
        if len(choices) < 3:
            violations.append("missing_choices")
        if not draft.strip():
            violations.append("empty_scene")
        if violations:
            return {
                "ok": False,
                "critique": "Fix violations: " + ", ".join(violations),
                "retries": int(state.get("retries", 0)) + 1,
            }
        return {
            "ok": True,
            "critique": "",
            "final": draft,
            "choices": choices,
        }

    def route_after_critic(state: GraphState) -> str:
        if state.get("ok"):
            return "done"
        if int(state.get("retries", 0)) >= 2:
            return "force"
        return "retry"

    def force_safe_node(state: GraphState) -> GraphState:
        safe = (
            "Mira stays on Hook Street with the sealed chip between you. "
            "No magic, no teleport, no flying cars. Deep Harbor remains distant lantern light. "
            "Orren watches unpaid. The contract tension is the whole scene."
        )
        choices = [
            "Accept the sealed chip",
            "Refuse the chip",
            "Walk toward Coil Yards",
        ]
        return {"final": safe, "choices": choices, "ok": True, "critique": "forced_safe"}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("narrator", narrator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("force_safe", force_safe_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "narrator")
    graph.add_edge("narrator", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"done": END, "retry": "narrator", "force": "force_safe"},
    )
    graph.add_edge("force_safe", END)
    return graph.compile()
