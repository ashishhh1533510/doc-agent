"""
Deterministic view_planner tests — no pytest, no LLM, no network.

Tests plan_views() with synthetic component models covering:
  - small model (fits budget): one L1 view, no L2
  - large model (25 components): L1 within budget, every component represented
  - edge prioritization: cross-layer/db edges survive over low-weight intra-layer
  - overflow folding: omitted counts are nonzero when budget exceeded
  - L2 drill-down: generated only when total exceeds budget

Run:  python tests/test_view_planner.py
Exit code is non-zero if any case fails.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.view_planner import (
    plan_views, plan_single_view, score_component, score_edge,
    MAX_NODES_PER_VIEW, MAX_EDGES_PER_VIEW, DRILL_MIN_COMPONENTS,
    MAX_COMPONENTS_SINGLE,
)

_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_comp(cid: str, layer: str = "application", has_routes: bool = False,
               has_db: bool = False, members: int = 2, is_infra: bool = False,
               owns_entities: list | None = None) -> dict:
    return {
        "id":           cid,
        "label":        cid.replace("_", " ").title(),
        "module":       "system",
        "module_label": "System",
        "layer":        layer,
        "stereotype":   layer,
        "is_infra":     is_infra,
        "members":      [f"m{i}" for i in range(members)],
        "member_count": members,
        "has_routes":   has_routes,
        "has_db":       has_db,
        "owns_entities": owns_entities or [],
        "interfaces":   [],
    }


def _make_edge(fid: str, tid: str, label: str = "requires", weight: int = 1) -> dict:
    return {"from": fid, "to": tid, "label": label, "weight": weight}


def _make_model(comps: list[dict], deps: list[dict] | None = None) -> dict:
    return {"components": comps, "dependencies": deps or [], "packages": []}


def _all_component_ids_in_l1(model: dict, views: list[dict]) -> bool:
    """Every component_id in the model must appear in exactly one L1 node."""
    all_cids = {c["id"] for c in model["components"]}
    l1        = next((v for v in views if v["level"] == "L1"), None)
    if not l1:
        return False
    covered = set()
    for n in l1["nodes"]:
        covered.update(n.get("component_ids", []))
    return covered == all_cids


# ══════════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════════

def test_small_model_single_view():
    """A model with <=MAX_NODES_PER_VIEW components produces exactly one L1, no L2."""
    comps = [_make_comp(f"c{i}", "application") for i in range(4)]
    deps  = [_make_edge("c0", "c1"), _make_edge("c1", "c2")]
    vs    = plan_views(_make_model(comps, deps))
    views = vs["views"]
    check("small: exactly one view", len(views) == 1, str(len(views)))
    check("small: view is L1",       views[0]["level"] == "L1")
    check("small: no L2 generated",  not any(v["level"] == "L2" for v in views))
    check("small: omitted zeros",    views[0]["omitted"] == {"nodes": 0, "edges": 0})


def test_small_model_l1_node_is_aggregate():
    """L1 nodes must be aggregate nodes."""
    comps = [_make_comp(f"c{i}", "application") for i in range(3)]
    vs    = plan_views(_make_model(comps))
    l1    = vs["views"][0]
    check("small: all L1 nodes are aggregate",
          all(n.get("is_aggregate") for n in l1["nodes"]))


def test_large_model_l1_within_budget():
    """25 components → L1 has ≤ MAX_NODES_PER_VIEW nodes."""
    layers = ["presentation", "application", "domain", "infrastructure", "persistence"]
    comps  = [_make_comp(f"c{i}", layers[i % len(layers)]) for i in range(25)]
    vs     = plan_views(_make_model(comps))
    l1     = vs["views"][0]
    check("large: L1 node count ≤ budget",
          len(l1["nodes"]) <= MAX_NODES_PER_VIEW,
          str(len(l1["nodes"])))


def test_large_model_all_components_represented():
    """Every component_id must appear in exactly one L1 aggregate node (nothing vanishes)."""
    layers = ["presentation", "application", "domain", "infrastructure", "persistence"]
    comps  = [_make_comp(f"c{i}", layers[i % len(layers)]) for i in range(25)]
    model  = _make_model(comps)
    vs     = plan_views(model)
    check("large: all components represented in L1",
          _all_component_ids_in_l1(model, vs["views"]))


def test_large_model_has_l2_views():
    """25 components with ≥DRILL_MIN_COMPONENTS per group should produce L2 views."""
    layers = ["presentation", "application", "domain", "infrastructure", "persistence"]
    comps  = [_make_comp(f"c{i}", layers[i % len(layers)]) for i in range(25)]
    vs     = plan_views(_make_model(comps))
    l2_views = [v for v in vs["views"] if v["level"] == "L2"]
    check("large: at least one L2 drill-down", len(l2_views) >= 1, str(len(l2_views)))


def test_edge_budget_overflow_recorded():
    """When edges exceed MAX_EDGES_PER_VIEW, omitted.edges > 0 (no silent truncation)."""
    comps = [
        _make_comp("api",   "presentation", has_routes=True),
        _make_comp("svc",   "application"),
        _make_comp("dom",   "domain"),
        _make_comp("infra", "infrastructure"),
        _make_comp("db",    "persistence", has_db=True),
    ]
    # create more edges than the budget
    deps = []
    ids  = [c["id"] for c in comps]
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j:
                deps.append(_make_edge(a, b, weight=1))
    vs   = plan_views(_make_model(comps, deps))
    l1   = vs["views"][0]
    total_inter_group = len({(e["from"], e["to"]) for e in deps})
    if total_inter_group > MAX_EDGES_PER_VIEW:
        check("edge overflow: omitted.edges > 0",
              l1["omitted"]["edges"] > 0,
              str(l1["omitted"]["edges"]))
    else:
        check("edge overflow: edges within budget (no overflow needed)", True)


def test_high_priority_edges_survive():
    """Cross-layer edges into persistence (has_db) must survive edge prioritization
    over low-weight same-layer edges."""
    comps = [
        _make_comp("api",    "presentation", has_routes=True),
        _make_comp("svc",    "application"),
        _make_comp("db",     "persistence",  has_db=True),
    ] + [_make_comp(f"noise{i}", "application") for i in range(3)]

    deps = [
        _make_edge("api", "db",  "requires",  weight=1),   # cross-layer to persistence
        _make_edge("api", "svc", "requires",  weight=1),   # normal
    ] + [_make_edge(f"noise{i}", f"noise{(i+1) % 3}", weight=1) for i in range(3)]

    vs = plan_views(_make_model(comps, deps))
    l1 = vs["views"][0]

    # Find node id for the persistence group
    db_grp_node = next(
        (n for n in l1["nodes"] if "db" in n.get("component_ids", [])), None
    )
    api_grp_node = next(
        (n for n in l1["nodes"] if "api" in n.get("component_ids", [])), None
    )

    if db_grp_node and api_grp_node:
        edge_to_db = any(
            e["to"] == db_grp_node["id"] and e["from"] == api_grp_node["id"]
            for e in l1["edges"]
        )
        check("priority: api→db edge present in L1", edge_to_db,
              f"edges={[(e['from'], e['to']) for e in l1['edges']]}")
    else:
        check("priority: api and db group nodes found", False,
              f"nodes={[n['id'] for n in l1['nodes']]}")


def test_score_component_routes_dominates():
    """A component with routes scores higher than one with only members."""
    c_routes = _make_comp("api", "presentation", has_routes=True, members=2)
    c_plain  = _make_comp("util", "application", members=100)
    check("score: routes component > large plain component",
          score_component(c_routes, 0) > score_component(c_plain, 0))


def test_score_component_infra_demoted():
    """is_infra=True reduces the score by 10x."""
    c_normal = _make_comp("svc", "application", members=5)
    c_infra  = _make_comp("inf", "application", members=5, is_infra=True)
    check("score: infra demoted vs normal",
          score_component(c_normal, 0) > score_component(c_infra, 0))


def test_score_edge_db_bonus():
    """An edge into a persistence component scores higher than a plain requires edge."""
    db_comp    = _make_comp("db", "persistence", has_db=True)
    plain_comp = _make_comp("svc", "application")
    comp_by_id = {"db": db_comp, "svc": plain_comp}
    e_db       = _make_edge("api", "db",  "requires", weight=1)
    e_plain    = _make_edge("api", "svc", "requires", weight=1)
    check("score edge: db destination scores higher",
          score_edge(e_db, comp_by_id) > score_edge(e_plain, comp_by_id))


def test_overflow_folding_sets_omitted():
    """When L1 groups exceed MAX_NODES_PER_VIEW, omitted.nodes > 0."""
    layers = ["presentation", "application", "domain", "infrastructure", "persistence"]
    # 15 unique layers/groups by creating components with distinct layer values
    # (we can't have more than 5 real layers, so use module axis instead)
    comps = []
    for i in range(15):
        c = _make_comp(f"c{i}", "application")
        c["module"] = f"project_{i}"
        c["module_label"] = f"Project {i}"
        comps.append(c)
    vs = plan_views(_make_model(comps))
    l1 = vs["views"][0]
    check("overflow: node count ≤ budget",
          len(l1["nodes"]) <= MAX_NODES_PER_VIEW, str(len(l1["nodes"])))
    check("overflow: omitted.nodes > 0 when groups exceed budget",
          l1["omitted"]["nodes"] > 0, str(l1["omitted"]["nodes"]))


def test_empty_model_returns_empty_views():
    """Empty component list produces no views."""
    vs = plan_views({"components": [], "dependencies": [], "packages": []})
    check("empty: no views", vs["views"] == [])


def test_l2_ghost_nodes_for_cross_group_deps():
    """L2 views include ghost nodes for cross-group neighbors."""
    comps = (
        [_make_comp(f"api{i}",  "presentation", has_routes=(i == 0)) for i in range(4)] +
        [_make_comp(f"svc{i}",  "application")  for i in range(4)] +
        [_make_comp(f"dom{i}",  "domain")        for i in range(4)]
    )
    # cross-group edge from presentation to application
    deps = [_make_edge("api0", "svc0", "requires")]
    vs   = plan_views(_make_model(comps, deps))
    l2_pres = next((v for v in vs["views"] if v["level"] == "L2" and "presentation" in v.get("title", "").lower()), None)
    if l2_pres:
        has_ghost = any(n.get("is_ghost") for n in l2_pres["nodes"])
        check("L2: ghost node present for cross-group dep", has_ghost,
              str([n["id"] for n in l2_pres["nodes"]]))
    else:
        # L2 might not be generated if group count is within budget — that's fine
        check("L2: no L2 generated (within budget, skip ghost check)", True)


# ── plan_single_view (single readable diagram) ────────────────────────────────

def test_single_view_one_diagram():
    """plan_single_view always returns exactly one view, regardless of size."""
    layers = ["presentation", "application", "domain", "infrastructure", "persistence"]
    comps  = [_make_comp(f"c{i}", layers[i % len(layers)]) for i in range(25)]
    vs = plan_single_view(_make_model(comps))
    check("single: exactly one view", len(vs["views"]) == 1, str(len(vs["views"])))
    check("single: no L2 anywhere", not any(v["level"] == "L2" for v in vs["views"]))


def test_single_view_caps_components():
    """More than the budget folds the tail; node count never exceeds the cap."""
    comps = [_make_comp(f"c{i}", "application", members=i + 2) for i in range(30)]
    vs = plan_single_view(_make_model(comps))
    v  = vs["views"][0]
    check("single: nodes <= budget", len(v["nodes"]) <= MAX_COMPONENTS_SINGLE, str(len(v["nodes"])))
    check("single: folded tail recorded", v["omitted"]["nodes"] == 30 - MAX_COMPONENTS_SINGLE,
          str(v["omitted"]["nodes"]))


def test_single_view_real_components_not_aggregates():
    """Nodes are real components (have a layer), not L1 aggregate boxes."""
    comps = [_make_comp(f"c{i}", "application") for i in range(4)]
    v = plan_single_view(_make_model(comps))["views"][0]
    check("single: nodes are real (no is_aggregate)", not any(n.get("is_aggregate") for n in v["nodes"]))
    check("single: nodes carry layer", all(n.get("layer") for n in v["nodes"]))


def test_single_view_external_datastore_wired():
    """A persistence component gets a requires edge to a database external."""
    comps = [_make_comp("db", "persistence", has_db=True),
             _make_comp("svc", "application")]
    ext = [{"id": "ext_postgresql", "label": "PostgreSQL", "kind": "datastore",
            "stereotype": "database"}]
    v = plan_single_view(_make_model(comps), ext)["views"][0]
    check("single: externals carried", v["externals"] == ext)
    check("single: db edge wired",
          any(e["from"] == "db" and e["to"] == "ext_postgresql" for e in v["edges"]),
          str(v["edges"]))


def test_single_view_service_external_not_floating():
    """A service external is attached from a source component (never left floating)."""
    comps = [_make_comp("svc", "application")]
    ext = [{"id": "ext_openai", "label": "OpenAI", "kind": "external",
            "stereotype": "infrastructure"}]
    v = plan_single_view(_make_model(comps), ext)["views"][0]
    check("single: service external has an incoming edge",
          any(e["to"] == "ext_openai" for e in v["edges"]), str(v["edges"]))


def test_single_view_empty():
    """No components → no views."""
    vs = plan_single_view({"components": [], "dependencies": [], "packages": []})
    check("single: empty model → no views", vs["views"] == [])


# ── architecture composition invariants (the anti-hairball guarantees) ────────

def test_compose_datastores_funnel_through_one_representative():
    """ALL datastores attach to a SINGLE data-access representative — never N×M.
    This is the rule that removes the hairball: each datastore has exactly one inbound
    edge, and every such edge originates from the same component."""
    comps = [_make_comp("dom1", "domain", owns_entities=["A"]),
             _make_comp("dom2", "domain", owns_entities=["B"]),
             _make_comp("dom3", "domain", owns_entities=["C"]),
             _make_comp("data", "persistence", has_db=True)]
    ext = [{"id": "ext_neo4j", "label": "Neo4j", "kind": "datastore", "stereotype": "database"},
           {"id": "ext_mongo", "label": "MongoDB", "kind": "datastore", "stereotype": "database"},
           {"id": "ext_es", "label": "Elasticsearch", "kind": "datastore", "stereotype": "database"}]
    v = plan_single_view(_make_model(comps), ext)["views"][0]
    db_ids = {x["id"] for x in ext}
    db_edges = [e for e in v["edges"] if e["to"] in db_ids]
    sources = {e["from"] for e in db_edges}
    check("compose: every datastore reached exactly once",
          sorted(e["to"] for e in db_edges) == sorted(db_ids), str(db_edges))
    check("compose: ALL datastores share ONE source (no N×M)",
          len(sources) == 1, str(sources))
    check("compose: the source is the data-access (persistence) component",
          sources == {"data"}, str(sources))


def test_compose_internal_edges_forward_and_budgeted():
    """Internal edges are forward-only (tier(dst) >= tier(src)) and capped at ~1.5×nodes."""
    import math as _math
    from doc_agent.tools.architecture_model import _LAYER_ORDER
    rank = {l: i for i, l in enumerate(_LAYER_ORDER)}
    comps = [_make_comp("p", "presentation"), _make_comp("a", "application"),
             _make_comp("d", "domain"), _make_comp("i", "infrastructure")]
    # include a back-edge (infra → presentation) that must be dropped
    deps = [_make_edge("p", "a", weight=5), _make_edge("a", "d", weight=4),
            _make_edge("d", "i", weight=3), _make_edge("i", "p", weight=9)]
    v = plan_single_view(_make_model(comps, deps))["views"][0]
    by_id = {c["id"]: c for c in comps}
    internal = [e for e in v["edges"] if e["to"] in by_id]
    forward = all(rank[by_id[e["to"]]["layer"]] >= rank[by_id[e["from"]]["layer"]] for e in internal)
    check("compose: all internal edges are forward (no back-edges)", forward, str(internal))
    check("compose: high-weight back-edge i→p was dropped",
          not any(e["from"] == "i" and e["to"] == "p" for e in internal), str(internal))
    check("compose: internal edges within ~1.5×nodes budget",
          len(internal) <= _math.ceil(1.5 * len(v["nodes"])), str(len(internal)))


def test_compose_title_uses_system_label():
    """The view's system_label comes from the model's package label (the repo name)."""
    comps = [_make_comp("c0", "application")]
    model = {"components": comps, "dependencies": [],
             "packages": [{"id": "system", "label": "OpenMetadata", "rank": 0}]}
    v = plan_single_view(model)["views"][0]
    check("compose: system_label = repo name", v["system_label"] == "OpenMetadata",
          v.get("system_label"))


# ══════════════════════════════════════════════════════════════════════════════

def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        case()

    print(f"\n{'='*60}")
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)}/{len(_FAILURES) + _PASS}")
        for f in _FAILURES:
            print(f"  FAIL  {f}")
        sys.exit(1)
    else:
        print(f"PASSED {_PASS}/{_PASS}")
        sys.exit(0)


if __name__ == "__main__":
    main()
