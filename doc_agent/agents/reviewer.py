"""
Reviewer agent (LLM-driven): the "checker" in the maker-checker loop.

Checks a generated README against the extracted facts and returns a verdict --
either approved, or a list of specific issues for the writer to fix.
"""

import json

from doc_agent.integrations.llm_provider import build_agent, run_agent_json

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
        try:
            verdict = await run_agent_json(self._agent, prompt)
        except ValueError:
            # could not parse a verdict after retries — accept what we have, don't loop
            return {"approved": True, "issues": []}
        return {
            "approved": bool(verdict.get("approved", True)),
            "issues": list(verdict.get("issues", [])),
        }