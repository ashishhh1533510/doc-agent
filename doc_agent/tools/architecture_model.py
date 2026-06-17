"""
Architecture model — the missing transform that turns repository STRUCTURE into
architectural STRUCTURE before any diagram is drawn.

Deterministic and repo/language/framework-agnostic. Consumes the same RichFacts
the rest of the pipeline uses (per-file routes/classes/imports + the resolved
import_graph) and produces an ArchitectureModel: runtime components grouped by
RESPONSIBILITY (not folder), architecturally-significant edges only, ordered
layers, and reported cycles. The LLM downstream only NAMES — it never decides
scope, grouping, layering, or which edges survive.
"""
from __future__ import annotations

import os
import re
import statistics

import networkx as nx

from doc_agent.tools.import_graph import module_id
from doc_agent.tools.component_clusters import _seed_areas, _file_relevance

# Rule 2 — non-runtime artifacts, classified by CONVENTION, never by repo name.
_NONRUNTIME_DIR = re.compile(
    r"(^|/)("
    r"tests?|__tests__|testing|spec|specs|e2e|"            # tests
    r"bench|benchmarks?|"                                  # benchmarks
    r"examples?|samples?|demos?|"                          # examples/samples
    r"scripts?|tools?|tooling|build|dist|ci|\.github|"     # build/ci/tooling
    r"migrations?|seeds?|seeders?|fixtures?|mocks?|stubs?" # data/migration tooling
    r")(/|$)", re.I,
)
_NONRUNTIME_STEM = re.compile(
    r"(test|tests|spec|specs|benchmark|benchmarks|mock|mocks|fixture|fixtures)$", re.I
)

# Canonical layer order — generic architectural roles, never repo-specific names.
_LAYER_ORDER = ["presentation", "application", "domain", "infrastructure", "persistence"]
_ROLE_LABEL = {
    "presentation":   "Presentation / API",
    "application":    "Application Services",
    "domain":         "Domain",
    "infrastructure": "Infrastructure",
    "persistence":    "Persistence",
}
_ENTRY_NAMES = {"main", "program", "index", "__main__", "app", "manage", "cli", "server"}


def _is_nonruntime(path: str) -> bool:
    p = (path or "").replace("\\", "/")
    if _NONRUNTIME_DIR.search(p):
        return True
    stem = os.path.splitext(p.rsplit("/", 1)[-1])[0]
    return bool(_NONRUNTIME_STEM.search(stem))


def _is_entry(mid: str, fact: dict) -> bool:
    base = mid.split("/")[-1].lower()
    if base in _ENTRY_NAMES:
        return True
    return any((fn.get("name") or "").lower() == "main" for fn in fact.get("functions", []))


def _behavioral(fact: dict) -> bool:
    """Carries behaviour (methods/functions) vs. a pure data carrier (DTO/constants).
    Method-less carriers are utility noise and get suppressed (Rule 5)."""
    if fact.get("functions"):
        return True
    for c in fact.get("classes", []):
        if [m for m in c.get("methods", []) if (m.get("name") or "") != "__init__"]:
            return True
    return False


def runtime_facts(rich_facts: dict, repo_root: str) -> dict:
    """module_id -> fact, runtime artifacts only (Rule 2)."""
    return {
        module_id(f["file"], repo_root): f
        for f in rich_facts.get("files", [])
        if not f.get("error") and not _is_nonruntime(f.get("file", ""))
    }


def build_internal_graph(facts: dict, raw: dict):
    """Internal import DiGraph (runtime-restricted) + per-module external-import pressure."""
    G = nx.DiGraph()
    G.add_nodes_from(facts)
    for s, deps in raw.items():
        if s in facts:
            for d in deps:
                if d in facts and d != s:
                    G.add_edge(s, d)
    ext_count = {}
    for mid, f in facts.items():
        internal = set(raw.get(mid, []))
        ext_count[mid] = max(0, len({*f.get("imports", [])}) - len(internal))
    return G, ext_count


def classify_roles(facts: dict, G: "nx.DiGraph", ext_count: dict) -> dict:
    """Each module -> shared|presentation|persistence|domain|infrastructure|application.
    Structural evidence first (routes/db/entrypoint); topology decides the middle band,
    with NO name matching:
      internal sink + framework-light  → domain  (depended upon; depends on ~nothing)
      external-heavy (top quartile)    → infrastructure (adapters to the outside)
      everything else                  → application (orchestration in between)
    """
    role: dict[str, str] = {}
    undecided = []
    for mid, f in facts.items():
        has_route = bool(f.get("routes"))
        has_db = any(c.get("is_db_model") for c in f.get("classes", []))
        if not _behavioral(f) and not has_route and not has_db:
            role[mid] = "shared"                     # DTO/constants → suppressed
        elif has_route or _is_entry(mid, f):
            role[mid] = "presentation"
        elif has_db:
            role[mid] = "persistence"
        else:
            undecided.append(mid)

    if undecided:
        ext_vals = sorted(ext_count[m] for m in undecided)
        p75 = ext_vals[int(0.75 * (len(ext_vals) - 1))]
        ext_med = statistics.median(ext_vals)
        for mid in undecided:
            if G.out_degree(mid) == 0 and ext_count[mid] <= ext_med:
                role[mid] = "domain"
            elif ext_count[mid] >= p75 and p75 > 0:
                role[mid] = "infrastructure"
            else:
                role[mid] = "application"
    return role


def build_architecture_model(rich_facts: dict, repo_root: str) -> dict:
    """Repository STRUCTURE -> architectural STRUCTURE, deterministically."""
    facts = runtime_facts(rich_facts, repo_root)
    if not facts:
        return {"diagram_type": "component", "components": [], "edges": [],
                "layers": [], "cycles": []}

    raw = rich_facts.get("import_graph", {})
    G, ext_count = build_internal_graph(facts, raw)
    role = classify_roles(facts, G, ext_count)

    # ── group modules into ONE component per responsibility (Rules 1/4/6) ────
    members: dict[str, list] = {}
    for mid, r in role.items():
        members.setdefault(r, []).append(mid)

    # ── architectural edges only (Rules 5/7) ─────────────────────────────────
    # aggregate module edges to role edges; drop self-edges and anything touching
    # the suppressed 'shared' utility band; collapse multiedges; keep direction.
    weights: dict[tuple, int] = {}
    for u, v in G.edges():
        ru, rv = role[u], role[v]
        if ru != rv and "shared" not in (ru, rv):
            weights[(ru, rv)] = weights.get((ru, rv), 0) + 1

    present = [r for r in _LAYER_ORDER if r in members]
    components = [
        {"id": r, "label": _ROLE_LABEL[r], "layer": r,
         "member_count": len(members[r]), "members": sorted(members[r])[:12]}
        for r in present
    ]
    edges = [{"from": a, "to": b, "weight": w}
             for (a, b), w in sorted(weights.items(), key=lambda kv: -kv[1])]

    # ── cycles: detect + report, never normalise (Rule 7) ────────────────────
    CG = nx.DiGraph()
    CG.add_nodes_from(present)
    CG.add_edges_from((e["from"], e["to"]) for e in edges)
    cycles = [sorted(scc) for scc in nx.strongly_connected_components(CG) if len(scc) > 1]

    return {
        "diagram_type": "component",
        "components": components,
        "edges": edges,
        "layers": [{"id": r, "label": _ROLE_LABEL[r], "rank": i} for i, r in enumerate(present)],
        "cycles": cycles,
    }


def _area_score(area: dict) -> tuple:
    """Rank areas by architectural significance — same evidence weighting as
    _file_relevance, aggregated per area rather than per file."""
    return (
        area["route_count"],
        area["db_model_count"],
        1 if area["has_entry_point"] else 0,
        area["file_count"],
    )


def build_system_digest(rich_facts: dict, repo_root: str) -> dict:
    """Whole-repo structural digest: one row per top-level architectural area,
    not per file, so it stays small even on thousand-file repos and never drops
    an area regardless of repo size. This is the breadth signal the HLD context
    agent uses to avoid collapsing the system into whichever single subtree has
    the densest route/DB evidence (the keyhole that caused the regression).

    Areas are seeded with the same deterministic per-subtree prefix logic used
    for LLD component clustering (_seed_areas, the directory-only sibling of
    _seed_clusters), so e.g. Catalog.API, Basket.API, Ordering.API surface as
    distinct areas instead of collapsing under a shared root."""
    facts = runtime_facts(rich_facts, repo_root)
    if not facts:
        return {"areas": []}

    mids = sorted(facts)
    area_of = _seed_areas(mids)

    grouped: dict[str, list[str]] = {}
    for mid in mids:
        grouped.setdefault(area_of[mid], []).append(mid)

    areas = []
    for area, members in grouped.items():
        members_facts = [facts[m] for m in members]
        routes = [r for f in members_facts for r in (f.get("routes") or [])]
        classes = [c for f in members_facts for c in f.get("classes", [])]
        db_classes = [c for c in classes if c.get("is_db_model")]
        languages = sorted({f.get("language") for f in members_facts if f.get("language")})
        internal = set(members)
        external_imports: dict[str, int] = {}
        for f in members_facts:
            for imp in f.get("imports", []):
                if imp not in internal:
                    external_imports[imp] = external_imports.get(imp, 0) + 1
        top_external = sorted(external_imports, key=lambda i: -external_imports[i])[:5]

        areas.append({
            "area": area,
            "file_count": len(members),
            "route_count": len(routes),
            "db_model_count": len(db_classes),
            "has_entry_point": any(_is_entry(m, facts[m]) for m in members),
            "languages": languages,
            "sample_routes": sorted({r.get("path") for r in routes if r.get("path")})[:5],
            "sample_classes": sorted({c.get("name") for c in classes if c.get("name")})[:5],
            "external_imports": top_external,
        })

    areas.sort(key=_area_score, reverse=True)
    return {"areas": areas}
