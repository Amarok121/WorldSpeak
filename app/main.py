from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .graph import build_graph
from .rag import load_lore
from .session_store import STORE

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

lore_chunks = load_lore()
story_graph = build_graph(lore_chunks)

app = FastAPI(
    title="WorldSpeak",
    description="Worldview RAG + session state + Narrator/ConsistencyCritic demo",
    version="0.1.0",
)


class TurnRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=500)


class TurnResponse(BaseModel):
    session_id: str
    location: str
    turn: int
    scene: str
    choices: list[str]
    flags: dict
    retrieved_docs: list[str]
    critique: str
    llm_mode: str


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "lore_docs": len(lore_chunks),
        "llm_mode": "openai" if os.getenv("OPENAI_API_KEY") else "mock",
    }


@app.post("/api/session")
def create_session() -> dict:
    session = STORE.create()
    session.flags = {"met_mira": False, "has_chip": False}
    STORE.save(session)
    return {"session_id": session.session_id, "location": session.location}


@app.post("/api/turn", response_model=TurnResponse)
def take_turn(body: TurnRequest) -> TurnResponse:
    if body.session_id:
        session = STORE.get(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
    else:
        session = STORE.create()
        session.flags = {"met_mira": False, "has_chip": False}

    message = body.message.strip()
    lower = message.lower()
    if "accept" in lower or "수락" in lower or "받아" in lower:
        session.flags["has_chip"] = True
        session.flags["met_mira"] = True
    if "refuse" in lower or "거절" in lower:
        session.flags["refused_chip"] = True
        session.flags["met_mira"] = True
    if "coil" in lower or "야드" in lower:
        session.location = "Coil Yards"
    if "harbor" in lower or "항구" in lower:
        # Player may move toward harbor; Mira still will not enter alone in narration.
        session.location = "Deep Harbor approach"

    result = story_graph.invoke(
        {
            "user_input": message,
            "location": session.location,
            "flags": session.flags,
            "retries": 0,
        }
    )

    scene = str(result.get("final") or result.get("draft") or "")
    choices = list(result.get("choices") or [])
    session.turn += 1
    session.history.append({"user": message, "assistant": scene})
    STORE.save(session)

    return TurnResponse(
        session_id=session.session_id,
        location=session.location,
        turn=session.turn,
        scene=scene,
        choices=choices[:3],
        flags=session.flags,
        retrieved_docs=list(result.get("retrieved_ids") or []),
        critique=str(result.get("critique") or ""),
        llm_mode="openai" if os.getenv("OPENAI_API_KEY") else "mock",
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
