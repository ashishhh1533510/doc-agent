"""
Model layer: the single place that connects to Gemini.

Provides build_agent() for chat agents and embed_texts() for embeddings --
both go through Gemini, so the model config lives in exactly one file.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = "gemini-2.5-flash"
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