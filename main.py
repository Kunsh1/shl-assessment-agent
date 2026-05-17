"""
SHL Assessment Recommender — FastAPI v4
GET  /health
POST /chat   — schema exactly matches assignment PDF spec (unchanged from v3)

Change vs v3:
  - CORS restricted to known origins (P2 security hardening)
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from agent import SHLAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational SHL assessment recommendation API",
    version="4.0.0",
)

# ── P2: Restricted CORS (was allow_origins=["*"] in v3) ──────────────────────
# Add your production UI origin here; keep localhost entries for local dev and
# any preview/staging URLs as needed.
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:8000",
    "http://localhost:3000",
    "https://shl-recommender-ui.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

CATALOG_PATH = Path(__file__).parent / "catalog.json"
agent = SHLAgent(str(CATALOG_PATH))
logger.info("Agent ready")


# ── Schemas (PDF-spec compliant — MUST NOT change) ────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]   # full conversation including the latest user turn


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str   # comma-separated codes e.g. "K" or "P,C"


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]   # empty list (not null) when no recs
    end_of_conversation: bool

@app.get("/")
def root():
    """Redirects the base URL directly to the Swagger UI docs."""
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silences the browser's favicon 404 error."""
    return {}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "assessments_loaded": len(agent.catalog),
        "model": agent._model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=422, detail="`messages` list must not be empty.")
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="Last message must have role='user'.")

    try:
        result = agent.chat([m.model_dump() for m in req.messages])
        return ChatResponse(
            reply=result["reply"],
            recommendations=[RecommendationItem(**r) for r in result["recommendations"]],
            end_of_conversation=result["end_of_conversation"],
        )
    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
