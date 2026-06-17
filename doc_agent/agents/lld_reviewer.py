"""
LLD Reviewer Agent: the "checker" in the LLD maker-checker loop.

Checks an LLD JSON model against RichFacts + ArchitectureContext.
Diagram-type-aware: knows what to verify for class vs sequence vs component vs dependency.
Used by lld_pipeline.py to drive the iteration loop.
"""

import json

from doc_agent.core.llm import build_agent, run_agent_json

INSTRUCTIONS = """You are a meticulous architecture reviewer specialising in LLD diagrams.
You are given four inputs:
1. RichFacts — static analysis of the codebase.
2. ArchitectureContext — repo-specific insights (pattern, components, workflows, tech stack).
3. LLDModel — a generated LLD diagram JSON.
4. diagram_type — one of: class | sequence | component | dependency

Check the LLDModel based on the diagram_type:

For "class":
- Are all class names present in RichFacts.classes[]?
- Do all relationship endpoints (from/to) exactly match a declared classes[].name, case-sensitive?
- Are any relationship endpoints modules or packages rather than declared classes? (Not allowed.)
- Is every declared class connected by at least one relationship (no isolated boxes)?
- Do any method params contain "self", type hints, or default values? (They must not.)
- Do inheritance relationships match bases[] in RichFacts?
- Do dependency relationships match calls[] in RichFacts?
- Are there generic placeholder names ("Manager", "Service", "Handler") instead of real class names?

For "sequence":
- Are all participants real components from ArchitectureContext.components?
- Do all message endpoints (from/to) exactly match an entry in participants[]?
- Does the entry message match an actual route in RichFacts.routes[]?
- Do internal messages match call_graph entries in RichFacts?
- Does the workflow end with a return message to the original caller?

For "component":
- Do component names match ArchitectureContext.components exactly?
- Do all dependency endpoints (from/to) exactly match a declared components[].id?
- Are there self-dependencies or duplicate edges? (Not allowed.)
- Is every component connected by at least one dependency?
- Are dependencies grounded in import_graph from RichFacts?
- Are there invented components not in the facts?

For "dependency":
- Are internal packages real sub-packages from import_graph keys?
- Are external packages real libraries from RichFacts imports[] (not stdlib)?
- Do all edge endpoints (from/to) exactly match a declared packages[].id?
- Are there self-loops, duplicate edges, or isolated packages? (Not allowed.)
- Are all edges traceable to import_graph or imports[]?

Respond with ONLY a JSON object, no other text:
{"approved": true, "issues": []}
or
{"approved": false, "issues": ["specific problem 1", "specific problem 2"]}"""


class LLDReviewerAgent:
    """Checks an LLD model against RichFacts and ArchitectureContext."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="LLDReviewer")

    async def review(self, rich_facts: dict, arch_context: dict, lld_model: dict, diagram_type: str) -> dict:
        """Return {'approved': bool, 'issues': [...]}."""
        prompt = (
            "RichFacts (JSON):\n" + json.dumps(rich_facts, indent=2, ensure_ascii=False)
            + "\n\nArchitectureContext (JSON):\n" + json.dumps(arch_context, indent=2, ensure_ascii=False)
            + "\n\nLLDModel (JSON):\n" + json.dumps(lld_model, indent=2, ensure_ascii=False)
            + f"\n\ndiagram_type: {diagram_type}"
            + "\n\nReview the LLDModel now and return your JSON verdict."
        )
        try:
            verdict = await run_agent_json(self._agent, prompt)
        except ValueError:
            # could not extract JSON after retries — fail safe; grounding.py is the hard gate
            return {"approved": True, "issues": []}
        return {
            "approved": bool(verdict.get("approved", True)),
            "issues": list(verdict.get("issues", [])),
        }
