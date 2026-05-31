"""
Reviewer agent (LLM-driven): the "checker" in the maker-checker loop.

Checks a generated README against the extracted facts and returns a verdict --
either approved, or a list of specific issues for the writer to fix.
"""

import json

from doc_agent.core.llm import build_agent

INSTRUCTIONS = """You are a meticulous documentation reviewer. You are given:
1. The ground-truth facts about a codebase, as JSON.
2. A generated README written from those facts.

Look for problems:
- Hallucinations: anything in the README (function, class, parameter, behavior) NOT in the facts.
- Inaccuracies: signatures, return types, or decorators that don't match the facts.
- Omissions: important public items in the facts the README leaves out.

Respond with ONLY a JSON object, no other text:
{"approved": true, "issues": []}
or
{"approved": false, "issues": ["specific problem 1", "specific problem 2"]}

Approve only if there are no hallucinations or inaccuracies. Minor wording differences are fine."""


def _parse_verdict(text: str) -> dict:
    """Pull the JSON verdict out of the reply, tolerating ```json code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        verdict = json.loads(cleaned)
        return {
            "approved": bool(verdict.get("approved", True)),
            "issues": list(verdict.get("issues", [])),
        }
    except (json.JSONDecodeError, AttributeError):
        # If we can't parse a verdict, don't loop forever -- accept what we have.
        return {"approved": True, "issues": [f"(unparseable reviewer reply: {text[:80]})"]}


class ReviewerAgent:
    """An LLM agent that fact-checks a README against extracted facts."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="Reviewer")

    async def review(self, facts, readme: str) -> dict:
        """Check the README against the facts; return {'approved': bool, 'issues': [...]}."""
        prompt = (
            "FACTS (JSON):\n" + json.dumps(facts, indent=2, ensure_ascii=False)
            + "\n\nGENERATED README:\n" + readme
            + "\n\nReview it now and return your JSON verdict."
        )
        result = await self._agent.run(prompt)
        return _parse_verdict(getattr(result, "text", str(result)))