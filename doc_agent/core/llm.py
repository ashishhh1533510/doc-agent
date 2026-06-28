"""
Model layer: the single place that connects to Gemini.

Provides build_agent() for chat agents and embed_texts() for embeddings --
both go through Gemini, so the model config lives in exactly one file.
"""

import os
import re
import json
import asyncio
import contextvars
import httpx
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

# gemini-2.5-flash-lite input context window (tokens). Used only for the UI's
# "context usage per call" gauge -- it is the per-call limit, not a daily quota.
CONTEXT_LIMIT = 1_048_576

# Per-request token-usage collector. Scoped with a ContextVar (not a module global)
# because FastAPI serves requests concurrently and a global would blend the token
# counts of two simultaneous generations together. Each API handler calls
# start_run_metrics() at the top, every run_agent() records into the active
# collector, and summarize_run_metrics() shapes the totals for the response.
_run_metrics: contextvars.ContextVar = contextvars.ContextVar("run_metrics", default=None)


def start_run_metrics() -> None:
    """Begin a fresh per-request token-usage collector for this async context."""
    _run_metrics.set({
        "per_agent": {},
        "calls": 0,
        "input": 0,
        "output": 0,
        "total": 0,
        "max_call_input": 0,
    })


def get_run_metrics():
    """Return the active collector dict, or None if start_run_metrics() wasn't called."""
    return _run_metrics.get()


def _record_usage(agent_name: str, usage) -> None:
    """Fold one Gemini response's usage_details into the active collector.

    `usage` is the agent_framework UsageDetails dict (mapped straight from Gemini's
    own response usage -- the authoritative count, not an estimate). No-ops when no
    collector is active or the response carried no usage, so nothing ever breaks.
    """
    m = _run_metrics.get()
    if not m or not usage:
        return
    i = usage.get("input_token_count") or 0
    o = usage.get("output_token_count") or 0
    t = usage.get("total_token_count") or (i + o)
    m["calls"] += 1
    m["input"] += i
    m["output"] += o
    m["total"] += t
    m["max_call_input"] = max(m["max_call_input"], i)
    a = m["per_agent"].setdefault(agent_name, {"calls": 0, "tokens": 0})
    a["calls"] += 1
    a["tokens"] += t


def summarize_run_metrics() -> dict:
    """Shape the active collector into the token_usage payload the UI renders.

    Returns zeroed fields if no calls were recorded so the caller can attach it
    unconditionally without guarding for None.
    """
    m = get_run_metrics() or {}
    calls = m.get("calls", 0)
    total = m.get("total", 0)
    per_agent = sorted(
        (
            {
                "name": name,
                "tokens": v["tokens"],
                "calls": v["calls"],
                "pct": round(100 * v["tokens"] / total) if total else 0,
            }
            for name, v in m.get("per_agent", {}).items()
        ),
        key=lambda x: -x["tokens"],
    )
    max_call_input = m.get("max_call_input", 0)
    return {
        "model": CHAT_MODEL,
        "total_tokens": total,
        "input_tokens": m.get("input", 0),
        "output_tokens": m.get("output", 0),
        "llm_calls": calls,
        "avg_tokens_per_call": round(total / calls, 1) if calls else 0,
        "context_limit": CONTEXT_LIMIT,
        "max_call_input": max_call_input,
        "context_pct_per_call": round(100 * max_call_input / CONTEXT_LIMIT, 2),
        "per_agent": per_agent,
        "dashboard_url": "https://aistudio.google.com/usage",
    }

# Free-tier quota is 250k input tokens/minute; target a safety margin under it so
# we pace ourselves before Gemini ever has to reject a call.
_INPUT_TPM_BUDGET = 180_000
_PROMPT_OVERHEAD_TOKENS = 4_000  # system instructions + framework overhead per call
_MAX_CONCURRENT_CALLS = 3
# The binding free-tier constraint for gemini-2.5-flash-lite is the REQUEST count,
# not tokens — our prompts are small enough to never trip the token budget, yet a
# multi-view run can fire 20+ tiny requests in a burst and hit the per-minute /
# per-day request quota. So we also pace raw request count per rolling 60s. Free
# tier is ~15 RPM; default to a margin under it. Override with GEMINI_MAX_RPM.
_MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "12"))


class _RateLimiter:
    """Module-level rolling-60s token bucket + request-count bucket + concurrency cap.

    Every run_agent() call estimates its input tokens and waits here until both
    (a) that many tokens and (b) a request slot are available in the trailing 60s
    window. This makes a run self-pace under the per-minute quota instead of firing
    ahead and reacting to 429s after the fact. The request bucket is what keeps a
    multi-view LLD run from bursting past the free-tier request cap.
    """

    def __init__(self, budget: int, rpm: int = _MAX_RPM):
        self._budget = budget
        self._rpm = max(rpm, 1)
        self._usage: list[tuple[float, int]] = []  # (monotonic_time, tokens)
        self._requests: list[float] = []           # request timestamps (60s window)
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CALLS)

    def _prune(self, now: float) -> int:
        cutoff = now - 60.0
        self._usage = [(t, n) for t, n in self._usage if t > cutoff]
        return sum(n for _, n in self._usage)

    def _prune_requests(self, now: float) -> int:
        cutoff = now - 60.0
        self._requests = [t for t in self._requests if t > cutoff]
        return len(self._requests)

    async def acquire(self, tokens: int):
        await self._semaphore.acquire()
        async with self._lock:
            # First gate: request-count per minute. Wait until a slot frees in the
            # trailing 60s window before we even consider tokens.
            while True:
                now = asyncio.get_event_loop().time()
                if self._prune_requests(now) < self._rpm:
                    break
                wait = max(self._requests[0] + 60.0 - now, 0.5)
                await asyncio.sleep(min(wait, 5.0))

            # Second gate: input tokens per minute.
            # A single call larger than the whole budget can never "fit"; pacing
            # cannot help it. Wait only for the window to drain to empty, then let
            # it through and rely on the 429 retry/backoff rather than hang forever.
            if tokens >= self._budget:
                now = asyncio.get_event_loop().time()
                while self._prune(now) > 0:
                    await asyncio.sleep(min(self._usage[0][0] + 60.0 - now, 5.0))
                    now = asyncio.get_event_loop().time()
                self._usage.append((now, tokens))
                self._requests.append(now)
                return
            while True:
                now = asyncio.get_event_loop().time()
                used = self._prune(now)
                if used + tokens <= self._budget:
                    self._usage.append((now, tokens))
                    self._requests.append(now)
                    return
                oldest_time = self._usage[0][0] if self._usage else now
                wait = max(oldest_time + 60.0 - now, 0.5)
                await asyncio.sleep(min(wait, 5.0))

    def release(self):
        self._semaphore.release()


_rate_limiter = _RateLimiter(_INPUT_TPM_BUDGET)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3 chars/token).

    Code and compact JSON tokenize denser than prose; the optimistic 4-chars/token
    heuristic undercounts and lets a call slip over the free-tier per-minute quota.
    Using 3 keeps the rate limiter on the safe side.
    """
    return len(text) // 3 + _PROMPT_OVERHEAD_TOKENS


def build_agent(instructions: str, name: str):
    """Build a Gemini-backed chat agent with the given instructions and name.

    The client's HTTP timeouts are set explicitly: the OpenAI SDK defaults to a
    5s connect timeout, which an intermittent network/DNS blip on a busy
    multi-agent run can blow, surfacing as a misleading APITimeoutError. A 15s
    connect window absorbs those blips; run_agent() retries any that remain.
    OpenAIChatCompletionClient has no timeout kwarg, so we inject a
    pre-configured AsyncOpenAI client via async_client.
    """
    async_client = AsyncOpenAI(
        base_url=GEMINI_BASE_URL,
        api_key=os.environ["GEMINI_API_KEY"],
        timeout=httpx.Timeout(120.0, connect=15.0),
    )
    return OpenAIChatCompletionClient(
        model=CHAT_MODEL,
        async_client=async_client,
    ).as_agent(name=name, instructions=instructions)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each input text (used for RAG retrieval)."""
    client = OpenAI(base_url=GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
    vectors = []
    for text in texts:  # one at a time for simplicity; batch later for large codebases
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        vectors.append(response.data[0].embedding)
    return vectors


def compact_json(value) -> str:
    """Serialize `value` as minimal-whitespace JSON for prompts.

    Pretty-printing (indent=2) on a large nested facts blob adds a lot of
    whitespace tokens; compact separators drop ~25-40% of input tokens with
    zero information loss. Used by every prompt builder that embeds facts.
    """
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _parse_retry_delay(msg: str) -> float | None:
    """Pull a retry-after delay (seconds) out of a provider error message.

    Gemini 429s carry both 'Please retry in 21.3s.' and a 'retryDelay: 21s'
    field; either tells us how long the per-minute window needs to reset.
    """
    m = re.search(r"retry in ([\d.]+)\s*s", msg)
    if m:
        return float(m.group(1))
    m = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+)", msg)
    if m:
        return float(m.group(1))
    return None


async def run_agent(
    agent,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_rate_limit_retries: int = 3,
):
    """Run the provided agent with `prompt` and return the reply text.

    Retries transient failures (e.g. 503/unavailable) with exponential backoff.
    A per-minute rate limit (429) is recoverable — the provider ships a
    retryDelay and the window resets on its own — so we honor that delay and
    retry. A hard/daily quota carries no usable retry hint, so we fail fast
    rather than burn more budget re-sending a huge prompt.
    Normalizes different agent return types by extracting `text` when
    available, otherwise falling back to str(result).
    """
    last_exc = None
    transient_attempt = 0
    rate_limit_attempt = 0
    tokens = estimate_tokens(prompt)
    while True:
        await _rate_limiter.acquire(tokens)
        try:
            try:
                result = await agent.run(prompt)
                # Capture Gemini's authoritative token count (not the rate-limiter
                # estimate) attributed to this agent, for the per-run usage panel.
                _record_usage(getattr(agent, "name", "unknown"),
                              getattr(result, "usage_details", None))
                return getattr(result, "text", str(result))
            finally:
                _rate_limiter.release()
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            is_rate_limit = any(
                t in msg for t in ("quota", "resource_exhausted", "429", "exceeded", "rate limit", "rate_limit")
            )
            if is_rate_limit:
                delay = _parse_retry_delay(msg)
                # Per-minute caps either ship a retryDelay or name a *-per-minute
                # quota; either way waiting clears them. Daily caps don't.
                recoverable = delay is not None or "perminute" in msg or "per minute" in msg
                hard = any(t in msg for t in ("perday", "per day", "per_day", "daily"))
                if recoverable and not hard and rate_limit_attempt < max_rate_limit_retries:
                    rate_limit_attempt += 1
                    wait = min((delay if delay is not None else 30.0) + 1.0, 65.0)
                    await asyncio.sleep(wait)
                    continue
                # hard quota, or rate-limit retries exhausted
                raise
            # treat genuine transient server hiccups as retryable. Timeouts
            # (connect/read) are network blips, not logic errors -- a retry with
            # backoff almost always succeeds, so don't 500 the whole run on one.
            if any(token in msg for token in (
                "500", "503", "unavailable", "high demand", "temporar",
                "timed out", "timeout", "connecttimeout", "readtimeout",
            )):
                transient_attempt += 1
                if transient_attempt >= max_retries:
                    raise last_exc
                delay = base_delay * (2 ** (transient_attempt - 1))
                # jitter
                delay = delay * (0.8 + 0.4 * (os.urandom(1)[0] / 255.0))
                await asyncio.sleep(delay)
                continue
            # non-transient error: re-raise immediately
            raise


def extract_json(text: str) -> str:
    """Best-effort isolation of a JSON object/array from a model reply.

    Strips ```json fences and, if the reply still has surrounding prose, slices
    from the first `{`/`[` to the last `}`/`]`. Lets us salvage a reply that has
    JSON buried in commentary before falling back to a retry.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) >= 2 else s[3:]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    if not (s.startswith("{") or s.startswith("[")):
        candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
        if candidates:
            start = min(candidates)
            end = max(s.rfind("}"), s.rfind("]"))
            if end > start:
                s = s[start:end + 1]
    return s


async def run_agent_json(agent, prompt: str, max_retries: int = 2, base_delay: float = 0.5,
                         fallback=None):
    """Run an agent and parse its reply as JSON, retrying when it isn't valid.

    Small models (e.g. gemini-flash-lite) occasionally answer the prompt in prose
    instead of JSON. We salvage a buried JSON block first; if that fails we retry
    with a stronger JSON-only instruction. Raises ValueError after `max_retries`
    unless `fallback` is provided, in which case `fallback` is returned instead.
    """
    last_text = ""
    attempt_prompt = prompt
    for attempt in range(1, max_retries + 1):
        last_text = await run_agent(agent, attempt_prompt)
        try:
            return json.loads(extract_json(last_text))
        except (json.JSONDecodeError, TypeError):
            if attempt < max_retries:
                attempt_prompt = (
                    prompt
                    + "\n\nIMPORTANT: your previous reply was not valid JSON. Respond with "
                    "ONLY the JSON value, starting with { and ending with }. No prose, no "
                    "markdown fences, no phase-by-phase analysis."
                )
                await asyncio.sleep(base_delay)
    if fallback is not None:
        return fallback
    raise ValueError(
        f"Agent did not return valid JSON after {max_retries} attempts: {last_text[:200]}"
    )