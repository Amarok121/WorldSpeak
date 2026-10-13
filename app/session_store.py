"""In-memory session flags. Tiny stand-in for personalized persistent worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class SessionState:
    session_id: str
    flags: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    location: str = "Hook Street"
    turn: int = 0


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}

    def create(self) -> SessionState:
        session = SessionState(session_id=str(uuid4()))
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.session_id] = session


STORE = SessionStore()
