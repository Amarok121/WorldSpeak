"""Keyword RAG over local lore markdown files. No embedding service required."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


LORE_DIR = Path(__file__).resolve().parent / "lore"


@dataclass(frozen=True)
class LoreChunk:
    doc_id: str
    title: str
    text: str
    tokens: frozenset[str]


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9가-힣]+", text.lower()))


def load_lore(lore_dir: Path | None = None) -> list[LoreChunk]:
    root = lore_dir or LORE_DIR
    chunks: list[LoreChunk] = []
    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = path.stem
        for line in raw.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        chunks.append(
            LoreChunk(
                doc_id=path.name,
                title=title,
                text=raw.strip(),
                tokens=_tokenize(raw),
            )
        )
    return chunks


def retrieve(query: str, chunks: list[LoreChunk], top_k: int = 3) -> list[LoreChunk]:
    q = _tokenize(query)
    if not q:
        return chunks[:top_k]
    scored: list[tuple[int, LoreChunk]] = []
    for chunk in chunks:
        overlap = len(q & chunk.tokens)
        # Prefer hard rules when conflicts are likely.
        if "rule" in chunk.doc_id or "character" in chunk.doc_id:
            overlap += 1
        scored.append((overlap, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [chunk for score, chunk in scored if score > 0][:top_k]
    return picked or chunks[:top_k]


def format_context(chunks: list[LoreChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        parts.append(f"### {chunk.title} ({chunk.doc_id})\n{chunk.text}")
    return "\n\n".join(parts)
