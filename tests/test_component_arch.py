"""
Deterministic component_arch tests — no pytest, no LLM, no network.

Tests _derive_modules, _cap_groups, and discover_components with synthetic data.

Run:  ./venv/Scripts/python.exe tests/test_component_arch.py
Exit code is non-zero if any case fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from doc_agent.tools.component_arch import (
    _derive_modules,
    _cap_groups,
    _fallback_label,
    _qualify_clusters,
    _consolidate_groups,
    discover_external_systems,
    _MAX_MODULES,
    _MAX_COMPONENTS_PER_MODULE,
    _SURFACE_COVERAGE,
)


# ── tiny harness (matches test_container_model.py style) ─────────────────────
_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_monorepo_runtime() -> list:
    """Synthetic monorepo with:
    - packages/nocodb/src/lib/<dir>/f<i>.py  — one big package, 600 files across 10 dirs
    - packages/gui/src/a.py .. f50.py        — medium package, 50 files
    - packages/cli/src/a.py .. f10.py        — small package, 10 files
    All paths use forward slashes (as _module_key / _common_dir_prefix expect).
    """
    files = []
    # big package: 600 files across 10 directories
    for d in range(10):
        for i in range(60):
            files.append(f"packages/nocodb/src/lib/dir{d}/f{i}.py")
    # medium package: 50 files
    for i in range(50):
        files.append(f"packages/gui/src/f{i}.py")
    # small package: 10 files
    for i in range(10):
        files.append(f"packages/cli/src/f{i}.py")
    return files


def _make_small_runtime() -> list:
    """Normal small repo — should descend zero times and return 1-2 modules."""
    return [
        "src/a.py",
        "src/b.py",
        "src/api/c.py",
        "src/api/d.py",
        "src/models/e.py",
    ]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_derive_modules_monorepo_bounded():
    """Large monorepo: _derive_modules must stay <= _MAX_MODULES."""
    runtime = _make_monorepo_runtime()
    mods = _derive_modules(runtime)
    check(
        "_derive_modules monorepo: module count <= _MAX_MODULES",
        len(mods) <= _MAX_MODULES,
        f"got {len(mods)} modules, cap is {_MAX_MODULES}",
    )
    # Ensure we didn't explode to leaf dirs (600 dirs/files would be >> 12)
    check(
        "_derive_modules monorepo: did not explode to leaf dirs",
        len(mods) < 50,
        f"got {len(mods)} modules",
    )
    # All files must be covered (no files lost)
    all_files = [f for files in mods.values() for f in files]
    check(
        "_derive_modules monorepo: all files covered",
        len(all_files) == len(runtime),
        f"got {len(all_files)} covered, expected {len(runtime)}",
    )


def test_derive_modules_monorepo_real_packages():
    """Large monorepo: the three real top-level packages should each appear."""
    runtime = _make_monorepo_runtime()
    mods = _derive_modules(runtime)
    mod_keys = list(mods.keys())
    # At minimum, the three distinct packages should be distinguishable
    # (either as direct module keys or ancestors of module keys)
    has_nocodb = any("nocodb" in k for k in mod_keys)
    has_gui = any("gui" in k for k in mod_keys)
    has_cli = any("cli" in k for k in mod_keys)
    check(
        "_derive_modules monorepo: packages/nocodb present",
        has_nocodb,
        f"keys: {mod_keys[:8]}",
    )
    check(
        "_derive_modules monorepo: packages/gui present",
        has_gui,
        f"keys: {mod_keys[:8]}",
    )
    check(
        "_derive_modules monorepo: packages/cli present",
        has_cli,
        f"keys: {mod_keys[:8]}",
    )


def test_derive_modules_small_repo():
    """Small normal repo: should stay as a single shallow module (descend zero times)."""
    runtime = _make_small_runtime()
    mods = _derive_modules(runtime)
    # small repo has <<60 files per module, so no split should happen.
    # The initial common-root+1 partition of these 5 paths under "src/" yields <= 4 modules.
    check(
        "_derive_modules small repo: no oversized module -> no recursion -> <= 5 modules",
        len(mods) <= 5,
        f"got {len(mods)} modules: {list(mods.keys())}",
    )
    all_files = [f for files in mods.values() for f in files]
    check(
        "_derive_modules small repo: all files covered",
        sorted(all_files) == sorted(runtime),
        f"coverage mismatch",
    )


def test_derive_modules_no_files():
    """Empty runtime: should return empty dict without error."""
    mods = _derive_modules([])
    check("_derive_modules empty: returns dict", isinstance(mods, dict), str(type(mods)))
    check("_derive_modules empty: no modules", len(mods) == 0, str(mods))


def test_cap_groups_under_cap():
    """_cap_groups with fewer groups than k: no change."""
    groups = [{"a", "b"}, {"c"}, {"d", "e", "f"}]
    result = _cap_groups(groups, 5)
    check("_cap_groups under cap: returns same groups", len(result) == len(groups), str(len(result)))


def test_cap_groups_at_cap():
    """_cap_groups with exactly k groups: no change."""
    groups = [{f"m{i}"} for i in range(6)]
    result = _cap_groups(groups, 6)
    check("_cap_groups at cap: keeps all 6", len(result) == 6, str(len(result)))


def test_cap_groups_over_cap():
    """_cap_groups with more than k groups: folds tail into largest, no members lost."""
    groups = [{"a", "b", "c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}, {"j"}]
    total_members = sum(len(g) for g in groups)
    k = 4
    result = _cap_groups(groups, k)
    check("_cap_groups over cap: exactly k groups returned", len(result) == k, str(len(result)))
    # No members lost
    result_members = sum(len(g) for g in result)
    check(
        "_cap_groups over cap: no members lost",
        result_members == total_members,
        f"before={total_members} after={result_members}",
    )


def test_cap_groups_max_components_per_module():
    """_cap_groups respects _MAX_COMPONENTS_PER_MODULE."""
    # More groups than the module constant
    groups = [{f"file{i}.py"} for i in range(20)]
    result = _cap_groups(groups, _MAX_COMPONENTS_PER_MODULE)
    check(
        "_cap_groups: respects _MAX_COMPONENTS_PER_MODULE",
        len(result) <= _MAX_COMPONENTS_PER_MODULE,
        f"got {len(result)}, cap={_MAX_COMPONENTS_PER_MODULE}",
    )
    before = sum(len(g) for g in groups)
    after = sum(len(g) for g in result)
    check("_cap_groups: no members lost", before == after, f"before={before} after={after}")


# ── dominant-surface qualification (_qualify_clusters) ───────────────────────

def _make_facts_with_routes(clusters: list[list[tuple]]) -> tuple:
    """
    clusters: list of clusters, each cluster is a list of (file_id, n_routes) tuples.
    Returns (comms, G, facts) ready for _qualify_clusters.
    """
    facts = {}
    comms = []
    for cluster in clusters:
        members = set()
        for fid, n_routes in cluster:
            facts[fid] = {"routes": [{"path": f"/r{i}"} for i in range(n_routes)],
                          "classes": [], "functions": [], "imports": []}
            members.add(fid)
        comms.append(members)
    G = nx.DiGraph()
    G.add_nodes_from(facts.keys())
    return comms, G, facts


def test_qualify_single_dominant_surface():
    """One cluster has almost all routes → only it qualifies as Tier 1."""
    # Cluster 0: 50 routes (dominant), clusters 1-4: 2 routes each → total 58, 80% = 46.4
    # Cumulative: 50 >= 46.4 → only cluster 0 qualifies via Tier 1
    clusters = [[(f"ctrl_{i}", 1) for i in range(50)]]   # cluster 0: 50 route files, 1 route each
    for j in range(4):
        clusters.append([(f"misc_{j}_{k}", 1) for k in range(2)])  # clusters 1-4: 2 routes each
    comms, G, facts = _make_facts_with_routes(clusters)
    q = _qualify_clusters(comms, G, facts)
    check("dominant single: cluster 0 qualifies", 0 in q, str(q))
    check("dominant single: only 1 route-qualified (Tier1)", len(q) == 1, str(q))


def test_qualify_balanced_surfaces():
    """Three clusters with similar route counts all qualify."""
    # 45, 43, 40, 2 → total=130, 80%=104 → 45+43+40=128≥104 → 3 qualify
    clusters = [
        [(f"a{i}", 1) for i in range(45)],
        [(f"b{i}", 1) for i in range(43)],
        [(f"c{i}", 1) for i in range(40)],
        [(f"d{i}", 1) for i in range(2)],
    ]
    comms, G, facts = _make_facts_with_routes(clusters)
    q = _qualify_clusters(comms, G, facts)
    # The three dominant clusters (0,1,2) should all qualify
    check("balanced: all 3 dominant clusters qualify", {0, 1, 2}.issubset(q), str(q))
    check("balanced: thin cluster 3 does not qualify via Tier 1",
          3 not in q or len(comms[3]) > 2, str(q))  # 3 can only qualify via Tier 2


def test_qualify_tiny_repo_fallback():
    """< 3 total routes → no Tier 1 surface → falls through to Tier 2 / library fallback."""
    clusters = [[(f"f{i}", 0) for i in range(3)], [(f"g{i}", 0) for i in range(5)]]
    comms, G, facts = _make_facts_with_routes(clusters)
    q = _qualify_clusters(comms, G, facts)
    # Should not crash; must qualify something (library fallback)
    check("tiny: qualification returns non-empty", len(q) > 0, str(q))


# ── global responsibility consolidation (_consolidate_groups) ────────────────

def test_consolidate_folds_controller_into_entity_context():
    """A non-entity (controller) group folds into the entity-owning context it couples to."""
    # group 0: entity context (owns Order), group 1: controller (no entity) importing group 0
    final_groups = [{"order_svc.py"}, {"order_ctrl.py"}]
    group_module = ["m", "m"]
    group_is_infra = [False, False]
    ent_refs = {"order_svc.py": {"Order"}, "order_ctrl.py": set()}
    G = nx.DiGraph()
    G.add_edge("order_ctrl.py", "order_svc.py")  # controller imports the context
    groups, mods, infra = _consolidate_groups(final_groups, group_module, group_is_infra, G, ent_refs)
    check("consolidate: folded to 1 group", len(groups) == 1, str(groups))
    check("consolidate: controller merged into context",
          {"order_svc.py", "order_ctrl.py"} == groups[0], str(groups))


def test_consolidate_keeps_infra_separate():
    """Infra groups are shared sinks and are never merged into anchors."""
    final_groups = [{"order_svc.py"}, {"db.py"}]
    ent_refs = {"order_svc.py": {"Order"}, "db.py": set()}
    G = nx.DiGraph()
    G.add_edge("order_svc.py", "db.py")
    groups, mods, infra = _consolidate_groups(final_groups, ["m", "m"], [False, True], G, ent_refs)
    check("consolidate: infra stays separate", len(groups) == 2, str(groups))
    check("consolidate: infra flag preserved", True in infra, str(infra))


def test_consolidate_island_not_merged():
    """A non-entity group with no coupling to any anchor stays its own component."""
    final_groups = [{"a.py"}, {"island.py"}]
    ent_refs = {"a.py": {"X"}, "island.py": set()}
    G = nx.DiGraph()  # no edges → island has no coupling
    G.add_nodes_from(["a.py", "island.py"])
    groups, mods, infra = _consolidate_groups(final_groups, ["m", "m"], [False, False], G, ent_refs)
    check("consolidate: uncoupled island preserved", len(groups) == 2, str(groups))


def test_consolidate_entityless_size_dominant():
    """Entity-less repo: small groups fold into the size-dominant anchor(s)."""
    final_groups = [set(f"big{i}.py" for i in range(10)), {"small.py"}]
    ent_refs = {m: set() for g in final_groups for m in g}  # no entities anywhere
    G = nx.DiGraph()
    G.add_edge("small.py", "big0.py")
    groups, mods, infra = _consolidate_groups(final_groups, ["m", "m"], [False, False], G, ent_refs)
    check("consolidate: entity-less folds into dominant", len(groups) == 1, str(len(groups)))


def test_consolidate_collapses_all_infra_to_one():
    """Per-module infra sinks (the 14-box explosion) collapse into ONE infra component."""
    final_groups = [{"order_svc.py"}, {"infra_a.py"}, {"infra_b.py"}, {"infra_c.py"}]
    ent_refs = {"order_svc.py": {"Order"}, "infra_a.py": set(),
                "infra_b.py": set(), "infra_c.py": set()}
    G = nx.DiGraph()
    G.add_nodes_from(["order_svc.py", "infra_a.py", "infra_b.py", "infra_c.py"])
    groups, mods, infra = _consolidate_groups(
        final_groups, ["m1", "m1", "m2", "m3"], [False, True, True, True], G, ent_refs)
    check("consolidate: exactly one infra component", infra.count(True) == 1, str(infra))
    infra_files = groups[infra.index(True)]
    check("consolidate: all infra files unioned into the one sink",
          {"infra_a.py", "infra_b.py", "infra_c.py"} <= infra_files, str(infra_files))


def test_consolidate_caps_contexts():
    """More than _MAX_CONTEXTS entity contexts → tail folds into surviving contexts."""
    from doc_agent.tools.component_arch import _MAX_CONTEXTS
    n = _MAX_CONTEXTS + 4
    final_groups = [{f"ctx{i}.py"} for i in range(n)]
    ent_refs = {f"ctx{i}.py": {f"Ent{i}"} for i in range(n)}
    G = nx.DiGraph()
    # chain-couple every context so the tail has a coupled survivor to fold into
    for i in range(n - 1):
        G.add_edge(f"ctx{i+1}.py", f"ctx{i}.py")
    groups, mods, infra = _consolidate_groups(
        final_groups, ["m"] * n, [False] * n, G, ent_refs)
    check("consolidate: non-infra capped at _MAX_CONTEXTS",
          sum(1 for x in infra if not x) <= _MAX_CONTEXTS, str(len(groups)))
    check("consolidate: no files lost to the cap",
          set().union(*groups) == {f"ctx{i}.py" for i in range(n)}, str(groups))


# ── deterministic fallback labels ─────────────────────────────────────────────

def test_fallback_label_uses_entity():
    """Owned entity drives the label, suffixed by the capability surface."""
    lbl = _fallback_label("persistence", has_routes=False, has_db=True,
                           owns=["User"], module_label="core")
    check("_fallback_label: entity + persistence suffix", "User" in lbl and "Persistence" in lbl, lbl)


def test_fallback_label_uses_module_when_no_entity():
    """No entity → falls back to module label + suffix; never empty / never a comp id."""
    lbl = _fallback_label("application", has_routes=False, has_db=False,
                          owns=[], module_label="billing")
    check("_fallback_label: module + services suffix", "billing" in lbl and "Services" in lbl, lbl)
    check("_fallback_label: non-empty", bool(lbl.strip()), lbl)


def test_fallback_label_routes_api():
    """A routed component is labelled an API."""
    lbl = _fallback_label("presentation", has_routes=True, has_db=False,
                          owns=["Order"], module_label="web")
    check("_fallback_label: API suffix for routes", lbl.endswith("API"), lbl)


# ── external system discovery ─────────────────────────────────────────────────

def test_external_systems_from_manifest_deps():
    """Datastore + service SDKs in manifest deps surface as external nodes."""
    rf = {"files": [{"imports": ["psycopg2"]}], "manifest_deps": ["redis", "openai"]}
    ext = discover_external_systems(rf)
    labels = {e["label"] for e in ext}
    check("external: postgres detected", any("PostgreSQL" in l for l in labels), str(labels))
    stereos = {e["stereotype"] for e in ext}
    check("external: stereotypes are db/infra", stereos <= {"database", "infrastructure"}, str(stereos))
    check("external: every node has an id", all(e.get("id") for e in ext), str(ext))


def test_external_systems_none():
    """No deps / no imports → no external systems, no error."""
    ext = discover_external_systems({"files": [], "manifest_deps": []})
    check("external: empty → empty list", ext == [], str(ext))


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        try:
            case()
        except Exception as e:
            _FAILURES.append(f"{case.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail[:200]}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
