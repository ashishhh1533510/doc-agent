"""
QA agent (LLM-driven): answers questions about a codebase from retrieved facts.

It receives a question plus the facts RAG retrieved, and answers using only
those facts -- the same no-hallucination discipline as the writer.
"""

from doc_agent.integrations.llm_provider import build_agent

INSTRUCTIONS = """You answer questions about a Python codebase. You are given a question and a set of
retrieved facts (functions and classes) from the codebase. Answer using ONLY those facts. If the
retrieved facts don't contain the answer, say so plainly. Be concise and mention the relevant
function or class names."""


class QAAgent:
    """An LLM agent that answers codebase questions from retrieved context."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="CodebaseQA")

    async def answer(self, question: str, chunks: list[dict]) -> str:
        """Answer the question using the retrieved chunks as context."""
        context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
        prompt = f"Retrieved facts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        result = await self._agent.run(prompt)
        return getattr(result, "text", str(result))