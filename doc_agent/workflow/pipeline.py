"""
The documentation pipeline: orchestration.

Wires the pieces into the sequential, maker-checker flow:

    extract facts (tool) -> write draft (agent) -> review (agent)
                                  ^                      |
                                  +---- revise <---------+  (if issues)

Optionally saves the final README to a .md file via the output tool.
"""

import json

from doc_agent.tools.extractor import extract_from_directory
from doc_agent.tools.output import save_markdown, strip_code_fence
from doc_agent.agents.writer import WriterAgent
from doc_agent.agents.reviewer import ReviewerAgent


class DocumentationPipeline:
    """Runs the full extract -> write -> review -> revise loop."""

    def __init__(self, max_rounds: int = 2):
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.max_rounds = max_rounds

    async def run(self, project_path, output_path=None) -> dict:
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

        readme = strip_code_fence(draft)
        result = {"readme": readme, "review_trace": trace}

        # 4. TOOL: optionally save the README to a .md file.
        if output_path:
            result["saved_to"] = save_markdown(output_path, readme)

        return result


# Generate + save: python -m doc_agent.workflow.pipeline doc_agent README.md
if __name__ == "__main__":
    import asyncio
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    output = sys.argv[2] if len(sys.argv) > 2 else None
    out = asyncio.run(DocumentationPipeline().run(project, output))
    print(out["readme"])
    if out.get("saved_to"):
        print(f"\n\nSaved README to: {out['saved_to']}")
    print("\n--- REVIEW TRACE ---")
    print(json.dumps(out["review_trace"], indent=2))