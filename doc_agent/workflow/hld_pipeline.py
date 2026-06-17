"""
HLD pipeline: deep extraction → deterministic container model → LLM enrichment → render.

output_type controls what gets rendered:
  "combined"   → C4 Context + C4 Container (the only supported output)

Pipeline shape (post Change D):
  facts → build_container_model (deterministic node-set)
        → HLDEnrichmentAgent   (text-only: system_purpose, descriptions, edge labels)
        → apply_enrichment     (merge by id — structural invention is discarded)
        → strip_technology_nodes (guard)
        → render_c4_combined   (shape-by-kind)
        → validate_mermaid
"""

import json
from doc_agent.tools.component_clusters import select_files_stratified, budget_facts_blob
from doc_agent.tools.architecture_model import build_system_digest
from doc_agent.tools.extractor import extract_rich_from_directory
from doc_agent.tools.input_resolver import resolve_input
from doc_agent.tools.output import render_c4_combined, save_text, strip_technology_nodes
from doc_agent.tools.container_model import build_container_model, apply_enrichment
from doc_agent.agents.hld_enrich import HLDEnrichmentAgent
from doc_agent.tools.diagram_validator import validate_mermaid



_RENDERERS = {
    "combined":  render_c4_combined,
}
def _slim_for_hld(rich_facts: dict, repo_root: str) -> dict:
    """Strip per-method detail the HLD agent doesn't need, keeping the prompt small.

    HLD discovers runtime *containers* (web/API hosts, services, persistence,
    datastores, external systems) from entry/route/persistence evidence. It must
    NOT be fed the import-graph component-clustering model (components/edges/
    architecture_signals) — that is the LLD/component-diagram model, and feeding it
    here collapses runtime containers into source-code business capabilities (the
    regression this avoids). Routes + is_db_model on the per-file list are the
    signals the container/datastore discovery needs.

    system_digest is a whole-repo, one-row-per-area structural summary (see
    build_system_digest) — it is the breadth signal that keeps a large repo's
    full set of areas visible to the agent regardless of the file cap below.

    File cap keeps the per-minute token budget low; select_files_stratified takes
    the most architecturally-relevant files from EVERY area first (then fills any
    remaining budget by global relevance), so no single dense area can monopolize
    the sample and hide the rest of the system (the regression this avoids)."""
    selected, omitted = select_files_stratified(rich_facts.get("files", []), repo_root, per_area=4, cap=40)
    slim_files = []
    for f in selected:
        slim_files.append({
            "file": f.get("file"),
            "imports": f.get("imports", []),
            "routes": f.get("routes", []),
            "classes": [
                {"name": c["name"], "bases": c.get("bases", []), "is_db_model": c.get("is_db_model", False)}
                for c in f.get("classes", [])
            ],
        })
    blob = {
        "primary_language": rich_facts.get("primary_language"),
        "framework": rich_facts.get("framework"),
        "frameworks": rich_facts.get("frameworks", []),
        "languages": rich_facts.get("languages"),
        "system_digest": build_system_digest(rich_facts, repo_root),
        "import_graph": rich_facts.get("import_graph", {}),
        "files": slim_files,
        "files_omitted": omitted,
    }
    return budget_facts_blob(blob)



async def run_hld(
    project_path: str,
    output_type: str = "combined",
    output_path: str | None = None,
    token: str | None = None,
    max_rounds: int = 2,   # kept for API compatibility; not used in new pipeline
) -> dict:
    """
    Run the HLD pipeline for a project and return the Mermaid diagram.

    Returns:
        {
            "output_type": str,
            "content": str,          # Mermaid text
            "arch_context": dict,    # enrichment output (for debugging)
            "review_trace": [...],   # always [] in new pipeline
            "saved_to": str | None,
        }
    """
    if output_type not in _RENDERERS:
        raise ValueError(f"output_type must be one of {list(_RENDERERS)}; got {output_type!r}")

    with resolve_input(project_path, token) as code_dir:
        # Stage 1 — deep extraction (deterministic)
        rich_facts = extract_rich_from_directory(code_dir)

        # Stage 2 — deterministic container-level topology (no LLM, no drift)
        model = build_container_model(rich_facts, code_dir)

        # Stage 2a — slim facts for enrichment context (small prompt budget)
        slim_facts = _slim_for_hld(rich_facts, code_dir)

        # Stage 2b — LLM enrichment (text only: system_purpose, descriptions, labels)
        enrichment = await HLDEnrichmentAgent().enrich(model, slim_facts)
        model = apply_enrichment(model, enrichment)

        # Stage 2c — remove framework/library nodes that are technology metadata,
        # not C4 architecture nodes (guard against library boxes leaking in).
        model = strip_technology_nodes(model, slim_facts.get("frameworks"))
        # Note: collapse_layers is NOT called — the deterministic node-set has no
        # architectural-layer boxes to collapse. resolve_floating_externals is also
        # not needed — all relationships are wired deterministically in Stage 2.

        # Stage 3 — deterministic rendering (shape-by-kind)
        content = _RENDERERS[output_type](model)

        # Stage 4 — syntax validation
        validation = validate_mermaid(content)

        result = {
            "output_type": output_type,
            "content": content,
            "arch_context": enrichment,   # enrichment dict for debugging
            "review_trace": [],
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
