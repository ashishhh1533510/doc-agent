"""
HLD pipeline: deep extraction → architecture context → C4 model → review loop → render.

output_type controls what gets rendered:
  "combined"   → C4 Context + C4 Container (the only supported output)
"""

import json
from doc_agent.tools.component_clusters import slim_components
from doc_agent.tools.extractor import extract_rich_from_directory
from doc_agent.tools.input_resolver import resolve_input
from doc_agent.tools.output import render_c4_combined, save_text
from doc_agent.agents.arch_context import ArchitectureContextAgent
from doc_agent.agents.hld_architect import HLDArchitectureAgent
from doc_agent.agents.hld_reviewer import HLDReviewerAgent
from doc_agent.tools.diagram_validator import validate_mermaid
from doc_agent.workflow.grounding import check_grounding



_RENDERERS = {
    "combined":  render_c4_combined,
}
def _slim_for_hld(rich_facts: dict) -> dict:
    """Strip per-method detail the HLD agent doesn't need, keeping the prompt small.
    Carries the deterministic components/edges/pattern so the agent must ground
    its diagram in them instead of inventing generic shape."""
    slim_files = []
    for f in rich_facts.get("files", []):
        slim_files.append({
            "file": f.get("file"),
            "imports": f.get("imports", []),
            "routes": f.get("routes", []),
            "classes": [
                {"name": c["name"], "bases": c.get("bases", []), "is_db_model": c.get("is_db_model", False)}
                for c in f.get("classes", [])
            ],
        })
    return {
        "primary_language": rich_facts.get("primary_language"),
        "framework": rich_facts.get("framework"),
        "frameworks": rich_facts.get("frameworks", []),
        "languages": rich_facts.get("languages"),
        "architecture_signals": rich_facts.get("architecture_signals", {}),
        "components": slim_components(rich_facts.get("components", [])),
        "component_edges": rich_facts.get("edges", []),
        "import_graph": rich_facts.get("import_graph", {}),
        "files": slim_files,
    }



async def run_hld(
    project_path: str,
    output_type: str = "combined",
    output_path: str | None = None,
    token: str | None = None,
    max_rounds: int = 2,
) -> dict:
    """
    Run the HLD pipeline for a project and return the Mermaid diagram.

    Returns:
        {
            "output_type": str,
            "content": str,          # Mermaid text
            "arch_context": dict,    # repo-specific insights (for debugging)
            "review_trace": [...],   # per-round reviewer verdicts
            "saved_to": str | None,
        }
    """
    if output_type not in _RENDERERS:
        raise ValueError(f"output_type must be one of {list(_RENDERERS)}; got {output_type!r}")

    with resolve_input(project_path, token) as code_dir:
        # Stage 1 — deep extraction (deterministic)
        rich_facts = extract_rich_from_directory(code_dir)

        # Stage 2a — architecture context (one LLM call, shared)
        slim_facts = _slim_for_hld(rich_facts)
        arch_ctx = await ArchitectureContextAgent().analyze(slim_facts)


        # Stage 2b + review loop
        hld_agent = HLDArchitectureAgent()
        reviewer  = HLDReviewerAgent()

        model = await hld_agent.analyze(slim_facts, arch_ctx)
        trace = []

        for round_num in range(1, max_rounds + 1):
            verdict = await reviewer.review(slim_facts, arch_ctx, model)
            ground_issues = check_grounding(model, slim_facts, arch_ctx)
            if ground_issues:
                verdict["approved"] = False
                verdict["issues"] = verdict.get("issues", []) + ground_issues
            trace.append({
                "round": round_num,
                "approved": verdict["approved"],
                "issues": verdict["issues"],
            })
            if verdict["approved"]:
                break
            model = await hld_agent.revise(slim_facts, arch_ctx, model, verdict["issues"])



        # Stage 3 — deterministic rendering
        content = _RENDERERS[output_type](model)

        # Stage 4 — syntax validation
        validation = validate_mermaid(content)

        result = {
            "output_type": output_type,
            "content": content,
            "arch_context": arch_ctx,
            "review_trace": trace,
            "validation": validation,
            "saved_to": None,
        }

        if output_path and content:
            result["saved_to"] = save_text(output_path, content)

        return result





# Quick test: python -m doc_agent.workflow.hld_pipeline <project_path> [combined|context|container]
if __name__ == "__main__":
    import asyncio
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    otype   = sys.argv[2] if len(sys.argv) > 2 else "combined"
    out = asyncio.run(run_hld(project, otype))
    print(out["content"])
    print("\n--- ArchitectureContext ---")
    print(json.dumps(out["arch_context"], indent=2))
    print("\n--- Review Trace ---")
    print(json.dumps(out["review_trace"], indent=2))
