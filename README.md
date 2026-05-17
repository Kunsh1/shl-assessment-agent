# SHL Assessment Recommender

Conversational API recommending SHL Individual Test Solutions.

## What Challenges I faced and how they were handled 

| Issue | Limitation | Improvement |
|-------|----|----|
| **RAG/FAISS** | Full 377-item catalog injected into every system prompt (~32k tokens/turn) | ✅ Agentic loop — LLM calls `search_catalog` tool → Python filters in-memory → results returned in `<search_results>` XML tags |
| **Token usage** | ~32,000 tokens per turn, triggered HTTP 429 rate limits on free tier | ✅ ~1400 tokens per turn (LLM only sees filtered results) |
| **Hard constraints** | Missed duration/type filters even with full catalog | ✅ Python enforces exact logical filtering before LLM sees any data |
| **Hallucinated metadata** | LLM could invent URLs or test type codes | ✅ Catalog ground-truth lookup on every submitted recommendation |
| **Duplicate recommendations** | Not handled | ✅ Deduplication by name using a set before final output |
| **JSON failures** | LLM occasionally emitted invalid JSON in tool calls | ✅ `temperature=0.0` + 3-attempt retry loop |
| **Rate limit bursts** | No delay between agentic turns | ✅ 2.5 s sleep between turns to respect Groq TPM burst limits |

---

## Architecture

```
POST /chat  { messages: [{role, content}, ...] }
     │
     ├─ System prompt = behavior rules only (no catalog)
     │
     ├─ Agentic loop (up to 8 turns)
     │    ├─ LLM calls search_catalog(test_type, keyword, job_level, max_duration)
     │    │    └─ Python filters 377-item in-memory catalog → returns XML-wrapped JSON
     │    ├─ LLM may call search_catalog multiple times (e.g., K-type, then A-type, then P-type)
     │    └─ LLM calls submit_recommendations({recommendations, end_of_conversation})
     │
     └─ Response:
          msg.content          → reply (conversational text)
          tool_call.arguments  → recommendations (structured, exact names)
          tool_call.arguments  → end_of_conversation (bool)
```

**Why agentic tool-based retrieval beats a static catalog-in-prompt:**
- Token cost drops from ~32k → ~300 per turn — eliminates rate-limit crashes on free-tier providers
- Python enforces hard filters (duration cap, test_type, job_level) with 100% accuracy
- LLM can issue multiple targeted searches in one turn (e.g., skills + cognitive + personality)
- Catalog ground-truth lookup prevents hallucinated URLs and test-type codes

---

## Tool Flow (typical happy path)

```
Turn 1: LLM emits search_catalog(test_type="K", keyword="java spring")
Turn 2: LLM receives filtered results; emits search_catalog(test_type="P") + search_catalog(test_type="A")
Turn 3: LLM receives both result sets; emits submit_recommendations(...)
  → caller receives {reply, recommendations, end_of_conversation}
```

---

## Endpoints

### `GET /health`
```json
{ "status": "ok", "assessments_loaded": 377, "model": "qwen/qwen3-32b" }
```

### `POST /chat`

**Request** (PDF spec):
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a senior Java backend engineer."}
  ]
}
```

**Response** (PDF spec):
```json
{
  "reply": "For a senior Java backend engineer I recommend...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://...", "test_type": "K"},
    {"name": "Spring (New)",                      "url": "https://...", "test_type": "K"},
    {"name": "SHL Verify Interactive G+",          "url": "https://...", "test_type": "A"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is an **empty list** (not null) on clarification or comparison turns.

---

## Local Setup

```bash
git clone <repo> && cd shl-v4
pip install -r requirements.txt

# Get a free Groq key at https://console.groq.com
export GROQ_API_KEY=gsk_...

python main.py
# → http://localhost:8000
# → Swagger: http://localhost:8000/docs

# Or run with uvicorn directly:
python -m uvicorn main:app --reload
```

---

## Deploy to Render

1. Push to GitHub
2. New Web Service → repo → `render.yaml` auto-detected
3. Add env var: `GROQ_API_KEY` (free at console.groq.com)
4. Deploy

No build-time model downloads. Cold start is near-instant.

---

## Free API keys

| Provider | URL | Free limit | Model used |
|----------|-----|-----------|------------|
| **Groq** (recommended) | console.groq.com | 1,000 req/day | qwen/qwen3-32b |
| OpenRouter | openrouter.ai | Daily credits | meta-llama/llama-3.3-70b-instruct:free |

Set exactly one: `GROQ_API_KEY` or `OPENROUTER_API_KEY`.
