# WorldSpeak

CookApps LLM & Backend portfolio bridge demo.

**Worldview docs → RAG → session flags → Narrator + ConsistencyCritic**

Keeps a tiny text world coherent while the player makes short, consequential choices.

## What it shows

| Piece | Role |
| --- | --- |
| Lore markdown (5 docs) | Fixed worldview contract |
| Keyword RAG | Retrieve city / character / hard rules |
| Session store | Personalized non-volatile flags (in-memory) |
| Narrator | Draft the next scene + 3 choices |
| ConsistencyCritic | Reject magic / teleport / rule breaks, retry or force-safe |

## Limits (honest)

- Not Hermes Agent
- No commercial-model weight fine-tuning
- No multi-thousand concurrent personalized worlds
- Session state is in-memory (process lifetime only)
- Default LLM is a deterministic **mock**; set `OPENAI_API_KEY` for real generation

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8765
```

Open http://127.0.0.1:8765

Optional `.env`:

```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

## API

- `GET /api/health`
- `POST /api/session`
- `POST /api/turn` `{ "session_id"?: "...", "message": "Accept the sealed chip" }`

## Design note

Free LLM generation breaks fun when lore collapses. WorldSpeak treats lore as a contract, retrieves it, then criticizes the draft before the player sees it—the same habit as schema gates and multi-agent critics in larger projects.
