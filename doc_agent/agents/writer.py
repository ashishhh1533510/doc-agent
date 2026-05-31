"""
Writer agent (LLM-driven): turns extracted facts into a README.

It knows nothing about how facts are extracted or how reviewing works -- it
only writes and revises a README from facts. Documents ONLY what's there.
"""

import json

from doc_agent.core.llm import build_agent

INSTRUCTIONS = """You are a technical writer. You are given a JSON description of a Python
codebase's structure: its modules, functions, classes, signatures, and docstrings.

Write a clear, well-organized README.md in Markdown. Include:
- A project title and a short description of what the project does.
- An overview of the main modules and what each is responsible for.
- A usage / API reference section listing the key functions and classes with their signatures.

Important: document ONLY what appears in the provided facts. Do not invent functions, parameters,
return values, or behavior that isn't there."""


def _facts_block(facts) -> str:
    """Render the extracted facts as a prompt block."""
    return "Codebase facts (JSON):\n\n" + json.dumps(facts, indent=2, ensure_ascii=False)


class WriterAgent:
    """An LLM agent that writes and revises READMEs from extracted facts."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="ReadmeWriter")

    async def write(self, facts) -> str:
        """Write a README from scratch using the facts."""
        result = await self._agent.run(_facts_block(facts) + "\n\nWrite the complete README.md now.")
        return getattr(result, "text", str(result))

    async def revise(self, facts, draft: str, issues: list[str]) -> str:
        """Rewrite the README, fixing the issues the reviewer raised."""
        prompt = (
            _facts_block(facts)
            + "\n\nCurrent README draft:\n" + draft
            + "\n\nA reviewer found these issues:\n- " + "\n- ".join(issues)
            + "\n\nRewrite the complete README.md fixing every issue. Document only what's in the facts."
        )
        result = await self._agent.run(prompt)
        return getattr(result, "text", str(result))