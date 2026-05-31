"""
The documentation pipeline: orchestration.

Wires the pieces into the sequential, maker-checker flow:

    extract facts (tool) -> write draft (agent) -> review (agent)
                                  ^                      |
                                  +---- revise <---------+  (if issues)

The pipeline owns the control flow; tools and agents stay unaware of each other.
"""

import json

from doc_agent.tools.extractor import extract_from_directory
from doc_agent.agents.writer import WriterAgent
from doc_agent.agents.reviewer import ReviewerAgent


class DocumentationPipeline:
    """Runs the full extract -> write -> review -> revise loop."""

    def __init__(self, max_rounds: int = 2):
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.max_rounds = max_rounds

    async def run(self, project_path) -> dict:
        # 1. TOOL: deterministic fact extraction (no LLM).
        facts = extract_from_directory(project_path)

        # 2. AGENT: write the first draft.
        draft = await self.writer.write(facts)

        # 3. AGENT: maker-checker loop -- review, and revise if needed.
        trace = []
        for round_num in range(1, self.max_rounds + 1):
            verdict = await self.reviewer.review(facts, draft)
            trace.append({
                "round": round_num,
                "approved": verdict["approved"],
                "issues": verdict["issues"],
            })
            if verdict["approved"]:
                break
            draft = await self.writer.revise(facts, draft, verdict["issues"])

        return {"readme": draft, "review_trace": trace}


# Run end to end from the terminal: python -m doc_agent.workflow.pipeline <path>
if __name__ == "__main__":
    import asyncio
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    out = asyncio.run(DocumentationPipeline().run(target))
    print(out["readme"])
    print("\n\n--- REVIEW TRACE ---")
    print(json.dumps(out["review_trace"], indent=2))