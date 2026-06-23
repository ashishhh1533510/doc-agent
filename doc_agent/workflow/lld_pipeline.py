"""
LLD pipeline: deep extraction → architecture context → LLD model → review loop → render.

diagram_type controls which diagram is generated:
  "class"       → UML class diagram
  "sequence"    → sequence diagram (most important workflow)
  "component"   → component/module diagram
  "dependency"  → package dependency diagram
"""
import asyncio
from doc_agent.tools.diagram_graph import (
    build_class_graph, plan_class_views, plan_sequence_views,
)

import json
from doc_agent.tools.component_clusters import slim_components, select_files, partition_for_class_diagram, budget_facts_blob

from doc_agent.tools.extractor import extract_rich_from_directory
from doc_agent.tools.input_resolver import resolve_input
from doc_agent.tools.manifest_parser import parse_all_manifests
from doc_agent.workflow.hld_pipeline import _derive_repo_name
from doc_agent.tools.output import (
    render_class_diagram, render_sequence_diagram,
    render_component_diagram, render_dependency_diagram, save_text,
    render_component_plantuml, plantuml_server_url,
    render_component_view_set,
)
from doc_agent.tools.view_planner import plan_views
from doc_agent.agents.arch_context import ArchitectureContextAgent
from doc_agent.agents.lld_agents import (
    ClassDiagramAgent, SequenceDiagramAgent,
    ComponentDiagramAgent, DependencyDiagramAgent,
)
from doc_agent.tools.component_arch import discover_components, validate_architecture_model
from doc_agent.tools.dependency_graph import build_dependency_model

from doc_agent.agents.lld_reviewer import LLDReviewerAgent
from doc_agent.tools.diagram_validator import validate_mermaid
from doc_agent.workflow.grounding import check_grounding



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

def _slim_for_lld(rich_facts: dict, files_override: list | None = None) -> dict:

    """Keep class/method/call detail for LLD but drop raw bodies and trim noise.
    Also carries the deterministic components/edges/pattern for grounding."""
    if files_override is not None:
        selected, omitted = select_files(files_override, cap=35)
    else:
        selected, omitted = select_files(rich_facts.get("files", []), cap=35)
    slim_files = []
    for f in selected:
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
                        {"name": m["name"], "signature": m.get("signature"),
                         "is_async": m.get("is_async", False), "calls": m.get("calls", [])}
                        for m in c.get("methods", [])
                    ],
                }
                for c in f.get("classes", [])
            ],
        })
    blob = {
        "primary_language": rich_facts.get("primary_language"),
        "framework": rich_facts.get("framework"),
        "frameworks": rich_facts.get("frameworks", []),
        "architecture_signals": rich_facts.get("architecture_signals", {}),
        "components": slim_components(rich_facts.get("components", [])),
        "component_edges": rich_facts.get("edges", []),
        "import_graph": rich_facts.get("import_graph", {}),
        "files": slim_files,
        "files_omitted": omitted,
    }
    return budget_facts_blob(blob)

async def _review_once(lld_agent, reviewer, facts, arch_ctx, model, diagram_type):
    """One review; a single revise only if the reviewer or grounding rejects it."""
    verdict = await reviewer.review(facts, arch_ctx, model, diagram_type)
    ground_issues = check_grounding(model, facts, arch_ctx)
    issues = list(verdict.get("issues", [])) + ground_issues
    approved = bool(verdict.get("approved", True)) and not ground_issues
    trace = [{"round": 1, "approved": approved, "issues": issues}]
    if not approved:
        model = await lld_agent.revise(facts, arch_ctx, model, issues)
        trace.append({"round": 2, "approved": None, "issues": [], "revised": True})
    return model, trace



async def run_lld(
    project_path: str,
    diagram_type: str = "class",
    output_path: str | None = None,
    token: str | None = None,
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

    repo_name = _derive_repo_name(project_path)

    with resolve_input(project_path, token) as code_dir:
        # Stage 1 — deep extraction (deterministic)
        rich_facts = extract_rich_from_directory(code_dir)

        # Stage 1b — parse build manifests (dep-grounded naming + datastore detection)
        manifests = parse_all_manifests(code_dir)
        # Attach manifest deps as extra import signals so component discovery
        # benefits from the same manifest-grounded detection as HLD.
        all_manifest_deps: list[str] = []
        for parsed in manifests.values():
            all_manifest_deps.extend(parsed.get("dependencies") or [])
        if all_manifest_deps:
            rich_facts = dict(rich_facts)
            rich_facts["manifest_deps"] = all_manifest_deps
            rich_facts["repo_name"] = repo_name or ""

        # Stage 2a — architecture context (one LLM call, shared)
        slim_facts = _slim_for_lld(rich_facts)
        arch_ctx = await ArchitectureContextAgent().analyze(slim_facts)

        lld_agent = _AGENT_MAP[diagram_type]()
        reviewer  = LLDReviewerAgent()

        # ── Class diagram: partition by folder, one focused diagram per partition ──
        # ── Class diagram: graph-analysis → bounded, cohesive views ──
        if diagram_type == "class":
            graph = build_class_graph(rich_facts)
            views = plan_class_views(graph)

            async def _do_view(view):
                agent = ClassDiagramAgent()          # fresh per task (parallel-safe)
                model = await agent.refine(view, arch_ctx)
                content = render_class_diagram(model)
                return {
                    "label": model.get("label") or view.get("label", ""),
                    "content": content,
                    "validation": validate_mermaid(content),
                }

            all_diagrams = (
                list(await asyncio.gather(*[_do_view(v) for v in views]))
                if views else []
            )
            primary = all_diagrams[0]["content"] if all_diagrams else ""
            result = {
                "diagram_type": "class",
                "content": primary,
                "diagrams": all_diagrams,
                "arch_context": arch_ctx,
                "review_trace": [],
                "validation": validate_mermaid(primary),
                "saved_to": None,
            }
            if output_path and primary:
                result["saved_to"] = save_text(output_path, primary)
            return result

            # single partition — fall through to standard flow below
                    # ── Sequence diagram: call-graph trace → bounded workflows ──
        if diagram_type == "sequence":
            candidates = plan_sequence_views(rich_facts)

            async def _do_seq(cand):
                agent = SequenceDiagramAgent()      # fresh per task (parallel-safe)
                model = await agent.refine(cand, arch_ctx)
                content = render_sequence_diagram(model)
                return {
                    "label": model.get("name") or cand.get("name", ""),
                    "content": content,
                    "validation": validate_mermaid(content),
                }

            all_diagrams = (
                list(await asyncio.gather(*[_do_seq(c) for c in candidates]))
                if candidates else []
            )
            primary = all_diagrams[0]["content"] if all_diagrams else ""
            result = {
                "diagram_type": "sequence",
                "content": primary,
                "diagrams": all_diagrams,
                "arch_context": arch_ctx,
                "review_trace": [],
                "validation": validate_mermaid(primary),
                "saved_to": None,
            }
            if output_path and primary:
                result["saved_to"] = save_text(output_path, primary)
            return result

                # ── Component diagram: discover → name → view-plan → render hierarchy ──
        if diagram_type == "component":
            candidate = discover_components(rich_facts, code_dir)
            if candidate.get("components"):
                checks  = validate_architecture_model(candidate, rich_facts)
                agent   = ComponentDiagramAgent()
                model   = await agent.refine(candidate, arch_ctx)
                viewset = plan_views(model)
                views   = render_component_view_set(viewset)

                # Back-compat top-level fields = L1 overview
                l1      = views[0] if views else {}
                content   = l1.get("content", "")
                image_url = l1.get("image_url")
                fmt       = "plantuml"

                diagrams = []
                for v in views:
                    d = {
                        "label":   v["title"],
                        "level":   v["level"],
                        "content": v["content"],
                        "omitted": v.get("omitted", {"nodes": 0, "edges": 0}),
                    }
                    if v.get("image_url"):
                        d["image_url"] = v["image_url"]
                    diagrams.append(d)

                result = {
                    "diagram_type": "component",
                    "format":       fmt,
                    "content":      content,
                    "image_url":    image_url,
                    "diagrams":     diagrams,
                    "arch_context": arch_ctx,
                    "review_trace": [],
                    "validation":   checks,
                    "saved_to":     None,
                }
                if output_path and content:
                    result["saved_to"] = save_text(output_path, content)
                return result
            # no resolvable components — fall through to the old whole-repo flow below

        # ── Dependency diagram: deterministic architecture model → flat package graph ──
        if diagram_type == "dependency":
            candidate = discover_components(rich_facts, code_dir)
            if candidate.get("components"):
                agent   = ComponentDiagramAgent()
                named   = await agent.refine(candidate, arch_ctx)
                model   = build_dependency_model(named, rich_facts, code_dir)
                content = render_dependency_diagram(model)
                result  = {
                    "diagram_type": "dependency",
                    "content":      content,
                    "arch_context": arch_ctx,
                    "review_trace": [],
                    "validation":   validate_mermaid(content),
                    "saved_to":     None,
                }
                if output_path and content:
                    result["saved_to"] = save_text(output_path, content)
                return result
            # no resolvable components — fall through to legacy flow below

        # ── All other diagram types + single-partition class diagrams ──
        model = await lld_agent.analyze(slim_facts, arch_ctx)
        model, trace = await _review_once(
            lld_agent, reviewer, slim_facts, arch_ctx, model, diagram_type
        )

        content = _RENDERER_MAP[diagram_type](model)
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
