"""
HLD pipeline v4: C4 Combined (Context + Container) architecture synthesis.

Produces exactly ONE diagram that merges the C4 Context and Container viewpoints
into a single hierarchical picture — readable by a CXO or newcomer to the code.

Pipeline shape:
  Stage 1  - deterministic candidate model
               build_candidate_model → merge_orchestration
               → infer_communication_graph → infer_entrypoint
               → synthesize_architecture_backbone (guarantee connectivity)
  Stage 2  - slim facts for LLM (capped, stratified)
  Stage 3  - HLDGroundedArchitect: classifies/labels candidates, assigns domains
  Stage 4  - guardrails + readability passes
               apply_grounding → apply_enrichment → strip_technology_nodes
               → enforce_c4_levels → assign_architecture_layers
               → drop_operational_noise → assign_domains
               → reduce_edges_for_readability → curate_significant_edges
               → assign_container_roles
  Stage 5  - one faithful combined diagram (render_c4_combined)

output_type: "combined" (the only supported value)
"""

import json
import logging

from doc_agent.tools.component_clusters import select_files_stratified, budget_facts_blob
from doc_agent.tools.architecture_model import build_system_digest
from doc_agent.tools.extractor import extract_rich_from_directory
from doc_agent.tools.input_resolver import resolve_input, _parse_git_url, is_git_url
from doc_agent.tools.output import (
    render_c4_combined,
    save_text,
    strip_technology_nodes,
)
from doc_agent.tools.container_model import (
    build_candidate_model,
    apply_grounding,
    apply_enrichment,
    enforce_c4_levels,
    discover_orchestration,
    merge_orchestration,
    infer_communication_graph,
    infer_entrypoint,
    synthesize_architecture_backbone,
    enforce_narrative_spine,
    enforce_connectivity,
    assign_architecture_layers,
    drop_operational_noise,
    assign_domains,
    consolidate_containers_for_abstraction,
    reduce_edges_for_readability,
    curate_significant_edges,
    assign_container_roles,
    validate_model,
)
from doc_agent.tools.manifest_parser import parse_all_manifests
from doc_agent.agents.hld_grounded_architect import HLDGroundedArchitect
from doc_agent.tools.diagram_validator import validate_mermaid
from doc_agent.evaluation.fidelity_scorer import compute_accuracy

log = logging.getLogger(__name__)

_RENDERERS = {
    "combined": render_c4_combined,
}


def _derive_repo_name(project_path: str) -> str | None:
    if not project_path:
        return None
    p = project_path.strip()
    if is_git_url(p):
        try:
            clone_url, _branch, _subpath, _is_file = _parse_git_url(p)
            base = clone_url.rstrip("/").rsplit("/", 1)[-1]
            return base[:-4] if base.endswith(".git") else base or None
        except Exception:
            return None
    from pathlib import Path as _Path
    try:
        return _Path(p).resolve().name or None
    except Exception:
        return None


def _slim_for_hld(rich_facts: dict, repo_root: str) -> dict:
    """Strip per-method detail the HLD agent doesn't need."""
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
    # Tight ceiling: dense code/JSON tokenizes nearer len/3 than len/4, so keep the
    # facts blob small enough that candidate_model + facts + instructions stays well
    # under the Gemini free-tier 250k input-tokens/minute quota in a single call.
    return budget_facts_blob(blob, max_tokens=70_000)


def _view(type_: str, label: str, content: str) -> dict:
    return {
        "type": type_,
        "label": label,
        "content": content,
        "validation": validate_mermaid(content),
    }


async def run_hld(
    project_path: str,
    output_type: str = "combined",
    output_path: str | None = None,
    token: str | None = None,
    max_rounds: int = 2,
) -> dict:
    """
    Run the HLD pipeline (v4 native C4, spec §3 escalation) for a project.

    Returns:
        {
            "output_type":  str,
            "content":      str,          # first diagram content (backward compat)
            "diagrams":     [
                {"type": str, "label": str, "content": str, "validation": dict},
                ...
            ],
            "arch_context": dict,
            "review_trace": [],
            "saved_to":     str | None,
        }
    """
    if output_type not in _RENDERERS:
        raise ValueError(f"output_type must be one of {list(_RENDERERS)}; got {output_type!r}")

    repo_name = _derive_repo_name(project_path)

    with resolve_input(project_path, token) as code_dir:
        # ── Stage 1: Deterministic candidate model ────────────────────────────
        rich_facts    = extract_rich_from_directory(code_dir)
        manifests     = parse_all_manifests(code_dir)
        orchestration = discover_orchestration(code_dir)
        model         = build_candidate_model(
            rich_facts, code_dir, repo_name=repo_name, manifests=manifests,
            orchestration=orchestration,  # evidence fusion — nodes only
        )
        model         = merge_orchestration(model, orchestration)     # adds nodes only
        model         = infer_communication_graph(model, rich_facts, orchestration)  # all edges
        model         = infer_entrypoint(model)                       # actor→entrypoint
        model         = synthesize_architecture_backbone(model)       # guarantee connectivity
        model         = enforce_narrative_spine(model)                # primary path, immune to thinning

        # ── Stage 2: Slim facts for LLM ───────────────────────────────────────
        slim_facts = _slim_for_hld(rich_facts, code_dir)

        # ── Stage 3: Grounded LLM architect ──────────────────────────────────
        enrichment = await HLDGroundedArchitect().classify(model, slim_facts)

        # ── Stage 4: Guardrails + architectural validation ────────────────────
        model = apply_grounding(model, model)
        model = apply_enrichment(model, enrichment)
        model = strip_technology_nodes(model, slim_facts.get("frameworks"))
        model = enforce_c4_levels(model)
        model = assign_architecture_layers(model)      # layer field for tiered rendering
        model = drop_operational_noise(model)          # remove load-gen / telemetry nodes
        model = assign_domains(model)                  # ensure every container has a group
        model = consolidate_containers_for_abstraction(model)  # fold module hairball into domains
        model = reduce_edges_for_readability(model)    # transitive reduction + collapse
        model = curate_significant_edges(model)        # per-source fan-out cap (redundant-only)
        model = enforce_connectivity(model)            # R2-R5 repair: no floating subsystems
        model = assign_container_roles(model)          # persist node["role"] (ingress face)

        validation_report = validate_model(model)
        if not validation_report["passed"]:
            for finding in validation_report["findings"]:
                log.warning("HLD validation [%s]: %s", finding["level"], finding["message"])
        else:
            for finding in validation_report["findings"]:
                log.info("HLD validation [%s]: %s", finding["level"], finding["message"])

        # ── Stage 5: one faithful combined diagram ────────────────────────────
        sys_label = (
            model.get("containers", {}).get("system_label")
            or model.get("context", {}).get("system_name")
            or "System"
        )
        diagram_content = render_c4_combined(model)
        diagrams = [_view("combined", f"{sys_label} - Containers", diagram_content)]

        # Completeness guard — log any model entity id absent from the rendered output
        all_ids = (
            [c["id"] for c in model.get("containers", {}).get("containers", [])]
            + [d["id"] for d in model.get("containers", {}).get("databases", [])]
            + [e["id"] for e in model.get("containers", {}).get("external_services", [])]
            + [a["id"] for a in model.get("context", {}).get("actors", [])]
        )
        missing = [eid for eid in all_ids if eid not in diagram_content]
        if missing:
            log.warning("HLD coverage: %d entity ids absent from diagram: %s", len(missing), missing)

        content = diagrams[0]["content"] if diagrams else ""

        accuracy = compute_accuracy(
            "combined",
            facts=slim_facts,
            model=model,
            validation_report=validation_report,
            missing_ids=missing,
        )

        result = {
            "output_type":        output_type,
            "content":            content,
            "diagrams":           diagrams,
            "arch_context":       enrichment,
            "review_trace":       [],
            "validation":         diagrams[0]["validation"] if diagrams else {},
            "validation_report":  validation_report,
            "accuracy":           accuracy,
            "coverage":           {"missing_ids": missing},
            "saved_to":           None,
        }

        if output_path and content:
            result["saved_to"] = save_text(output_path, content)

        return result


# Quick test: python -m doc_agent.orchestrators.hld_pipeline <project_path> [combined]
if __name__ == "__main__":
    import asyncio
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    otype   = sys.argv[2] if len(sys.argv) > 2 else "combined"
    out = asyncio.run(run_hld(project, otype))
    print(f"=== {len(out.get('diagrams', []))} diagram(s) ===")
    for d in out.get("diagrams", []):
        print(f"\n--- [{d['type']}] {d['label']} ---")
        print(d["content"])
        print("validation:", d["validation"])
    print("\n--- ArchitectureContext ---")
    print(json.dumps(out["arch_context"], indent=2))
