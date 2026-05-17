"""
SHL Assessment Recommender — Agent v4
────────────────────────────────────────────────────────────────────────────────
Architecture change vs v3
  v3: 377-item catalog injected into every system-prompt call (~32k tokens each)
  v4: Agentic loop  →  LLM calls search_catalog  →  Python filters in-memory
                    →  filtered results returned in <search_results> XML tags
                    →  LLM calls submit_recommendations with final shortlist

Tool flow (typical happy path):
  Turn 1: LLM emits search_catalog(type="K", keyword="java spring")
  Turn 2: LLM receives filtered catalog; emits search_catalog(type="P") + search_catalog(type="A")
  Turn 3: LLM receives both result sets; emits submit_recommendations(...)
  → caller receives {reply, recommendations, end_of_conversation}

Providers (try in order): GROQ_API_KEY → OPENROUTER_API_KEY → ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Map full category names to PDF-required letter codes ──
TYPE_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B", 
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S"
}
# ── LLM client factory ────────────────────────────────────────────────────────

def _make_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) using whichever key is available."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if groq_key:
        logger.info("LLM provider: Groq (qwen/qwen3-32b)")
        return (
            OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key),
            "qwen/qwen3-32b",
        )
    if openrouter_key:
        logger.info("LLM provider: OpenRouter (llama-3.3-70b-instruct:free)")
        return (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                default_headers={"HTTP-Referer": "https://shl-recommender.app"},
            ),
            "meta-llama/llama-3.3-70b-instruct:free",
        )
    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY, OPENROUTER_API_KEY."
    )


# ── Duration parser ───────────────────────────────────────────────────────────

def _parse_duration_minutes(duration_str: str) -> int:
    """
    Parse a human-readable duration string into minutes.
    Returns 0 if the value is unknown, untimed, or cannot be parsed
    (0 is treated as "no limit" by the filter — keep these items in).

    Handles: "30 minutes", "25-35 minutes" (uses upper bound),
             "1 hour", "1 hour 30 minutes", "45 mins", "Untimed", ""
    """
    if not duration_str:
        return 0
    s = duration_str.lower().strip()
    if any(word in s for word in ("untimed", "variable", "varies", "flexible")):
        return 0

    # For ranges like "25-35 minutes" take the upper bound (conservative)
    s = re.sub(r"\d+\s*[-–]\s*(\d+)", r"\1", s)

    hours_match = re.search(r"(\d+)\s*h(?:our|r)?s?", s)
    mins_match = re.search(r"(\d+)\s*m(?:in(?:ute)?s?)?", s)

    total = 0
    if hours_match:
        total += int(hours_match.group(1)) * 60
    if mins_match:
        total += int(mins_match.group(1))
    return total


# ── Tool definitions ──────────────────────────────────────────────────────────

SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_catalog",
        "description": (
            "Search the SHL assessment catalog by filter criteria. "
            "Call this BEFORE making any recommendation. "
            "You may call it multiple times with different parameters to build a comprehensive shortlist "
            "(e.g., once for K-type knowledge tests, once for A-type cognitive, once for P-type personality). "
            "Results are returned in <search_results> XML tags."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_level": {
                    "type": "string",
                    "description": (
                        "Filter by target job level (case-insensitive, partial match). "
                        "Examples: 'Graduate', 'Manager', 'Director', 'Entry-Level', 'Professional'. "
                        "Leave empty to include all levels."
                    ),
                },
                "test_type": {
                    "type": "string",
                    "description": (
                        "Filter by test-type code(s), comma-separated. "
                        "A=Ability & Aptitude | B=Biodata & Situational Judgment | C=Competencies | "
                        "D=Development & 360 | E=Assessment Exercises | K=Knowledge & Skills | "
                        "P=Personality & Behavior | S=Simulations. "
                        "Leave empty for all types."
                    ),
                },
                "max_duration": {
                    "type": "string",
                    "description": (
                        "Maximum assessment duration in minutes. "
                        "Pass '0' (default) for no limit."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Space- or comma-separated keywords to match against the assessment name or "
                        "description (case-insensitive OR logic). "
                        "Use for domain/technology searches: 'java spring sql', 'HIPAA', 'contact center'. "
                        "Leave empty to skip keyword filtering."
                    ),
                },
            },
            "required": [],
        },
    },
}

SUBMIT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_recommendations",
        "description": (
            "Call this tool WHENEVER you recommend, suggest, or present a list of assessments to the user. "
            "Do NOT wait for a 'final' list. If you name an assessment as a viable option in your reply, "
            "you MUST simultaneously call this tool to output the structured data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "description": "Ordered list of recommended assessments (1–10 items).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": (
                                    "EXACT assessment name from search_catalog results — "
                                    "copy it character-for-character."
                                ),
                            },
                            "url": {
                                "type": "string",
                                "description": "Assessment URL from search_catalog results.",
                            },
                            "test_type": {
                                "type": "string",
                                "description": "Test type code(s) from search_catalog results.",
                            },
                        },
                        "required": ["name", "url", "test_type"],
                    },
                },
                "end_of_conversation": {
                    "type": "boolean",
                    "description": (
                        "Set true ONLY when the user has explicitly confirmed or finalised "
                        "the shortlist and needs nothing more "
                        "(e.g. 'confirmed', 'that's all', 'lock it in', 'perfect')."
                    ),
                },
            },
            "required": ["recommendations", "end_of_conversation"],
        },
    },
}


# ── System prompt (no catalog; LLM must call search_catalog) ─────────────────

SYSTEM_PROMPT = """\
You are an expert SHL assessment consultant. You have access to the full SHL Individual Test Solutions
catalog via the `search_catalog` tool. You do NOT have the catalog memorised — always search before
recommending.

## Turn-by-turn workflow

1. **CLARIFY** — If the request is too vague to search meaningfully, ask exactly ONE targeted question
   before calling any tool. Good topics: job level, exact role, language requirement, duration cap,
   selection vs. development, senior-IC vs. tech-lead.

2. **SEARCH** — Call `search_catalog` with appropriate filters. You may call it **multiple times**
   in the same turn to cover different dimensions:
   - Separate calls by test_type (e.g., K for skills, A for cognitive, P for personality)
   - Use keyword for technology/domain lookups ("java spring sql", "HIPAA", "contact center")
   - Use job_level only when the user specified a target audience
   - Combine keyword + test_type for precision (e.g., type="K", keyword="docker aws")

3. **RECOMMEND** — After reviewing <search_results>, select 1–10 assessments and write a short conversational reply introducing the options.
   
   CRITICAL RULES FOR RECOMMENDING:
   - You MUST call the `submit_recommendations` tool ANY time you suggest assessments.
   - Do NOT just list them in the text. You must use the tool so the backend receives the structured JSON array.
   - Set `end_of_conversation` to false if you are just presenting the first draft of the list.

4. **REFINE** — When the user adds/removes items, call `search_catalog` again if needed, then
   call `submit_recommendations` with the updated full list.

5. **COMPARE** — When the user asks to compare assessments, reply conversationally only —
   do NOT call any tool.

6. **DECLINE & OUT OF SCOPE** - Legal/compliance interpretation questions → redirect to the user's legal team.
   - NEVER recommend competitor products or tools outside of the SHL catalog (e.g., HackerRank, Codility, LeetCode). 
   - If a specific technology has no catalog test (e.g., Rust, Java), state explicitly that SHL does not currently offer a specific test for that requirement. Instead, recommend the closest general alternatives FROM YOUR SEARCH RESULTS (e.g., general cognitive aptitude or problem-solving tests).

## end_of_conversation
Set to `true` in `submit_recommendations` ONLY when the user explicitly confirms/finalises the
shortlist ("confirmed", "that's all", "lock it in", "perfect", "that works", "good", etc.).

## Test type codes
A=Ability & Aptitude | B=Biodata & Situational Judgment | C=Competencies
D=Development & 360 | E=Assessment Exercises | K=Knowledge & Skills
P=Personality & Behavior | S=Simulations
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class SHLAgent:
    # Maximum LLM turns per user request (each search call uses one turn).
    # Allows up to ~6 search calls + 1 submit call + 1 slack turn.
    _MAX_TURNS: int = 8

    def __init__(self, catalog_path: str) -> None:
        self.catalog: list[dict] = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        
        # --- DATA NORMALIZATION FIX ---
        for item in self.catalog:
            # 1. Fix missing URL
            if "url" not in item and "link" in item:
                item["url"] = item["link"]
                
            # 2. Fix missing test_type (Translate "keys" to letter codes)
            if "test_type" not in item and "keys" in item:
                # Look up the letter for each key. e.g. "Knowledge & Skills" -> "K"
                codes = [TYPE_MAP.get(k.strip(), "") for k in item["keys"]]
                # Join them with commas e.g. "K,A"
                item["test_type"] = ",".join(c for c in codes if c) or "Unknown"
        # ------------------------------

        self._catalog_by_name: dict[str, dict] = {
            item["name"]: item for item in self.catalog
        }
        self._client, self._model = _make_client()
        logger.info(
            "SHLAgent v4 ready: %d items in catalog, model=%s",
            len(self.catalog),
            self._model,
        )

    # ── Catalog search (executed locally in Python) ───────────────────────────

    def _search_catalog(
        self,
        job_level: str = "",
        test_type: str = "",
        max_duration: int = 0,
        keyword: str = "",
    ) -> list[dict]:
        """
        Filter the in-memory catalog and return concise records for the LLM.
        All parameters are optional; omitting a parameter skips that filter.
        Results are capped at 30 items to keep tool responses manageable.
        """
        results: list[dict] = list(self.catalog)

        # ── job_level filter (partial, case-insensitive) ──────────────────────
        if job_level:
            jl = job_level.lower().strip()
            results = [
                item for item in results
                # Items with an empty/absent job_levels list are treated as
                # "unrestricted" and always pass the filter.
                if not item.get("job_levels")
                or any(jl in level.lower() for level in item["job_levels"])
            ]

        # ── test_type filter (exact code match) ───────────────────────────────
        if test_type:
            wanted = {t.strip().upper() for t in re.split(r"[,\s]+", test_type) if t.strip()}
            results = [
                item for item in results
                if wanted & {c.strip().upper() for c in item.get("test_type", "").split(",")}
            ]

        # ── max_duration filter ───────────────────────────────────────────────
        if max_duration > 0:
            filtered: list[dict] = []
            for item in results:
                mins = _parse_duration_minutes(item.get("duration", "") or "")
                # Keep items where duration is unknown (0) or within the cap
                if mins == 0 or mins <= max_duration:
                    filtered.append(item)
            results = filtered

        # ── keyword filter (OR logic across space/comma-separated terms) ──────
        if keyword:
            tokens = [
                t.strip().lower()
                for t in re.split(r"[,\s]+", keyword)
                if t.strip()
            ]
            results = [
                item for item in results
                if any(
                    tok in item["name"].lower()
                    or tok in (item.get("description") or "").lower()
                    for tok in tokens
                )
            ]

        # ── Return concise records (cap at 30) ────────────────────────────────
        return [
            {
                "name": item.get("name", "Unknown"),
                "url": item.get("url", item.get("link", "URL missing")),
                "test_type": item.get("test_type", "Unknown"),
                "duration": item.get("duration", "") or "—",
                "job_levels": item.get("job_levels", []),
                "languages": item.get("languages", [])[:5],
                "description": (item.get("description") or "")[:160].replace("\n", " "),
            }
            for item in results[:10]
        ]

    # ── Agentic chat loop ─────────────────────────────────────────────────────

    def chat(self, messages: list[dict]) -> dict:
        """
        Run the multi-turn agentic loop for one user request.

        Args:
            messages: Full conversation in OpenAI format including the latest user turn.
                      [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            {
              "reply": str,                  # conversational text for the user
              "recommendations": list[dict], # [{name, url, test_type}, ...]
              "end_of_conversation": bool,
            }
        """
        # Build the working message list: system prompt + full conversation history
        api_messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + messages

        reply: str = ""
        recommendations: list[dict] = []
        end_of_conversation: bool = False
        total_prompt_tokens: int = 0
        total_completion_tokens: int = 0

        for turn in range(self._MAX_TURNS):
            # --- NEW: Anti-Burst Rate Limiter ---
            if turn > 0:
                logger.debug("Sleeping for 2.5s to respect Groq TPM burst limits...")
                time.sleep(2.5) 
            # ------------------------------------
            max_retries = 3
            completion = None
            for attempt in range(max_retries):
                try:
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=api_messages,
                        tools=[SEARCH_TOOL, SUBMIT_TOOL],
                        tool_choice="auto",
                        max_tokens=2048,
                        temperature=0.0,  # Lowered to 0.0 to prevent JSON syntax hallucinations
                    )
                    break  # Success! Break out of the retry loop
                except Exception as e:
                    logger.warning(f"Groq API Error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise  # If it fails 3 times, crash gracefully
            # ------------------------------------------------------

            # ── P1: Log token usage for every LLM call ────────────────────────
            if completion.usage:
                pt = completion.usage.prompt_tokens
                ct = completion.usage.completion_tokens
                total_prompt_tokens += pt
                total_completion_tokens += ct
                logger.info(
                    "Turn %d/%d — Tokens used: Prompt=%d, Completion=%d",
                    turn + 1, self._MAX_TURNS, pt, ct,
                )

            msg = completion.choices[0].message
            # Capture any conversational text the LLM included alongside tool calls
            if msg.content:
                reply = msg.content.strip()
                reply = re.sub(r"<function>.*?</function>:\s*\{.*?\}", "", reply, flags=re.DOTALL).strip()

            # ── No tool calls → LLM replied directly (clarification / compare) ─
            if not msg.tool_calls:
                logger.debug("Turn %d: no tool calls, exiting loop.", turn + 1)
                break

            # ── Append the assistant turn (with tool_calls) to history ─────────
            api_messages.append({
                "role": "assistant",
                "content": msg.content,  # may be None; that is valid
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # ── Process each tool call in this turn ───────────────────────────
            tool_results: list[dict] = []
            found_submit: bool = False

            for tc in msg.tool_calls:
                fn_name = tc.function.name

                # ── search_catalog: execute locally, return XML-wrapped results ─
                if fn_name == "search_catalog":
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as exc:
                        logger.warning("search_catalog arg parse error: %s", exc)
                        args = {}

                    search_results = self._search_catalog(
                        job_level=args.get("job_level", "") or "",
                        test_type=args.get("test_type", "") or "",
                        max_duration=int(args.get("max_duration", 0) or 0),
                        keyword=args.get("keyword", "") or "",
                    )

                    result_json = json.dumps(search_results, indent=2)
                    content = (
                        f"<search_results count=\"{len(search_results)}\">\n"
                        f"{result_json}\n"
                        f"</search_results>"
                    )
                    logger.info(
                        "search_catalog(job_level=%r, test_type=%r, max_duration=%r, keyword=%r) "
                        "→ %d results",
                        args.get("job_level"), args.get("test_type"),
                        args.get("max_duration"), args.get("keyword"),
                        len(search_results),
                    )

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    })

                # ── submit_recommendations: extract, deduplicate, finish loop ──
                elif fn_name == "submit_recommendations":
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as exc:
                        logger.warning("submit_recommendations arg parse error: %s", exc)
                        args = {}

                    raw_recs: list[dict] = args.get("recommendations", [])
                    end_of_conversation = bool(args.get("end_of_conversation", False))

                    # ── P0 Bug fix: deduplicate by name using a set ────────────
                    seen_names: set[str] = set()
                    for r in raw_recs:
                        name = r.get("name", "").strip()
                        if not name or name in seen_names:
                            continue
                        seen_names.add(name)

                        # Defensive lookup: use catalog ground truth to prevent
                        # hallucinated URLs / test_types (pattern from v3, retained)
                        catalog_item = self._catalog_by_name.get(name)
                        if catalog_item:
                            recommendations.append({
                                "name": catalog_item.get("name", "Unknown"),
                                "url": catalog_item.get("url", "URL missing"),
                                "test_type": catalog_item.get("test_type", "Unknown"),
                            })
                        else:
                            # LLM name slightly off — accept as-is rather than drop
                            logger.warning(
                                "Assessment %r not found in catalog by exact name; "
                                "using LLM-provided values.",
                                name,
                            )
                            recommendations.append({
                                "name": name,
                                "url": r.get("url", ""),
                                "test_type": r.get("test_type", ""),
                            })

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Recommendations received.",
                    })

                    found_submit = True
                    # ── P0 Bug fix: break immediately — no duplicate processing ─
                    break

                else:
                    logger.warning("Unknown tool call: %s", fn_name)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool '{fn_name}'.",
                    })

            # Append all tool results before the next LLM call
            api_messages.extend(tool_results)

            if found_submit:
                logger.debug("submit_recommendations processed; exiting loop.")
                break

        else:
            # MAX_TURNS exhausted without a submit — log and return what we have
            logger.warning(
                "Agentic loop reached MAX_TURNS=%d without submit_recommendations.",
                self._MAX_TURNS,
            )

        # ── P1: Summary token log ──────────────────────────────────────────────
        logger.info(
            "Request complete. Total tokens — Prompt=%d, Completion=%d",
            total_prompt_tokens,
            total_completion_tokens,
        )

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation,
        }
