"""
LLD pipeline: deep extraction → architecture context → LLD model → review loop → render.

diagram_type controls which diagram is generated:
  "class"       → UML class diagram
  "sequence"    → sequence diagram (most important workflow)
  "component"   → component/module diagram
  "dependency"  → package dependency diagram
"""

import json

from doc_agent.tools.extractor import extract_rich_from_directory
from doc_agent.tools.input_resolver import resolve_input
from doc_agent.tools.output import (
    render_class_diagram, render_sequence_diagram,
    render_component_diagram, render_dependency_diagram, save_text,
)
from doc_agent.agents.arch_context import ArchitectureContextAgent
from doc_agent.agents.lld_agents import (
    ClassDiagramAgent, SequenceDiagramAgent,
    ComponentDiagramAgent, DependencyDiagramAgent,
)
from doc_agent.agents.lld_reviewer import LLDReviewerAgent
from doc_agent.tools.diagram_validator import validate_mermaid


_AGENT_MAP = {
    "class":      ClassDiagramAgent,
    "sequence":   SequenceDiagramAgent,
    "component":  ComponentDiagramAgent,
    "dependency": DependencyDiagramAgent,
}

_RENDERER_MAP = {
    "class":      render_class_diagram,
    "sequence":   render_sequence_diagram,
    "component":  render_component_diagram,
    "dependency": render_dependency_diagram,
}

def _slim_for_lld(rich_facts: dict) -> dict:
    """Keep class/method/call detail for LLD but drop raw function bodies and trim noise."""
    slim_files = []
    for f in rich_facts.get("files", []):
        slim_files.append({
            "file": f.get("file"),
            "imports": f.get("imports", []),
            "routes": f.get("routes", []),
            "classes": [
                {
                    "name": c["name"],
                    "bases": c.get("bases", []),
                    "is_db_model": c.get("is_db_model", False),
                    "fields": c.get("fields", []),
                    "methods": [
                        {
                            "name": m["name"],
                            "signature": m.get("signature"),
                            "is_async": m.get("is_async", False),
                            "calls": m.get("calls", []),
                        }
                        for m in c.get("methods", [])
                    ],
                }
                for c in f.get("classes", [])
            ],
        })
    return {
        "primary_language": rich_facts.get("primary_language"),
        "framework": rich_facts.get("framework"),
        "import_graph": rich_facts.get("import_graph", {}),
        "files": slim_files,
    }



async def run_lld(
    project_path: str,
    diagram_type: str = "class",
    output_path: str | None = None,
    token: str | None = None,
    max_rounds: int = 2,
) -> dict:
    """
    Run the LLD pipeline for a project and return the Mermaid diagram.

    Returns:
        {
            "diagram_type": str,
            "content": str,          # Mermaid text
            "arch_context": dict,    # repo-specific insights (for debugging)
            "review_trace": [...],   # per-round reviewer verdicts
            "saved_to": str | None,
        }
    """
    if diagram_type not in _AGENT_MAP:
        raise ValueError(f"diagram_type must be one of {list(_AGENT_MAP)}; got {diagram_type!r}")

    with resolve_input(project_path, token) as code_dir:
        # Stage 1 — deep extraction (deterministic)
        rich_facts = extract_rich_from_directory(code_dir)

        # Stage 2a — architecture context (one LLM call, shared)
        slim_facts = _slim_for_lld(rich_facts)
        arch_ctx = await ArchitectureContextAgent().analyze(slim_facts)

        # Stage 2b + review loop
        lld_agent = _AGENT_MAP[diagram_type]()
        reviewer  = LLDReviewerAgent()

        model = await lld_agent.analyze(slim_facts, arch_ctx)
        trace = []

        for round_num in range(1, max_rounds + 1):
            verdict = await reviewer.review(slim_facts, arch_ctx, model, diagram_type)
            trace.append({
                "round": round_num,
                "approved": verdict["approved"],
                "issues": verdict["issues"],
            })
            if verdict["approved"]:
                break
            model = await lld_agent.revise(slim_facts, arch_ctx, model, verdict["issues"])

        # Stage 3 — deterministic rendering
        content = _RENDERER_MAP[diagram_type](model)

        # Stage 4 — syntax validation
        validation = validate_mermaid(content)

        result = {
            "diagram_type": diagram_type,
            "content": content,
            "arch_context": arch_ctx,
            "review_trace": trace,
            "validation": validation,
            "saved_to": None,
        }

        if output_path and content:
            result["saved_to"] = save_text(output_path, content)

        return result


# Quick test: python -m doc_agent.workflow.lld_pipeline <project_path> [class|sequence|component|dependency]
if __name__ == "__main__":
    import asyncio
    import sys

    project     = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    dtype       = sys.argv[2] if len(sys.argv) > 2 else "class"
    out = asyncio.run(run_lld(project, dtype))
    print(out["content"])
    print("\n--- ArchitectureContext ---")
    print(json.dumps(out["arch_context"], indent=2))
    print("\n--- Review Trace ---")
    print(json.dumps(out["review_trace"], indent=2))
