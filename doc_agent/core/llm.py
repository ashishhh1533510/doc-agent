"""
Model layer: the single place that connects to Gemini.

Provides build_agent() for chat agents and embed_texts() for embeddings --
both go through Gemini, so the model config lives in exactly one file.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = "gemini-2.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def build_agent(instructions: str, name: str):
    """Build a Gemini-backed chat agent with the given instructions and name."""
    return OpenAIChatCompletionClient(
        base_url=GEMINI_BASE_URL,
        api_key=os.environ["GEMINI_API_KEY"],
        model=CHAT_MODEL,
    ).as_agent(name=name, instructions=instructions)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each input text (used for RAG retrieval)."""
    client = OpenAI(base_url=GEMINI_BASE_URL, api_key=os.environ["GEMINI_API_KEY"])
    vectors = []
    for text in texts:  # one at a time for simplicity; batch later for large codebases
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        vectors.append(response.data[0].embedding)
    return vectors


async def run_agent(agent, prompt: str, max_retries: int = 5, base_delay: float = 1.0):
    """Run the provided agent with `prompt` and return the reply text.

    Retries transient failures (e.g. 503/unavailable) with exponential backoff.
    Normalizes different agent return types by extracting `text` when
    available, otherwise falling back to str(result).
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            result = await agent.run(prompt)
            return getattr(result, "text", str(result))
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # treat common transient indicators as retryable
            if any(token in msg for token in ("500","503", "unavailable", "high demand", "rate limit", "temporar")):
                delay = base_delay * (2 ** (attempt - 1))
                # jitter
                delay = delay * (0.8 + 0.4 * (os.urandom(1)[0] / 255.0))
                await asyncio.sleep(delay)
                continue
            # non-transient error: re-raise immediately
            raise
    # exhausted retries
    raise last_exc