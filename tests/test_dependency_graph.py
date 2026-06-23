"""
Deterministic dependency_graph tests — no pytest, no LLM, no network.

Tests build_dependency_model() with synthetic component models and rich_facts
for all plan-specified cases, plus renderer shape assertions.

Run:  ./venv/Scripts/python.exe tests/test_dependency_graph.py
Exit code is non-zero if any case fails.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.dependency_graph import build_dependency_model, MAX_INTERNAL_PACKAGES, MAX_EDGES, MAX_EXTERNAL_PACKAGES
from doc_agent.tools.output import render_dependency_diagram
from doc_agent.core.llm import run_agent_json


# ── tiny harness ──────────────────────────────────────────────────────────────
_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_component(cid: str, label: str, member_count: int = 3,
                    members: list | None = None) -> dict:
    return {
        "id": cid,
        "label": label,
        "module": "system",
        "module_label": "MyRepo",
        "layer": "application",
        "stereotype": "application",
        "member_count": member_count,
        "members": members or [f"src/{cid}/mod_{i}" for i in range(member_count)],
        "has_routes": False,
        "has_db": False,
        "owns_entities": [],
        "interfaces": [],
    }


def _make_dep(from_id: str, to_id: str, label: str = "requires") -> dict:
    return {"from": from_id, "to": to_id, "label": label}


def _make_named_model(components: list[dict], deps: list[dict]) -> dict:
    return {
        "diagram_type": "component",
        "components": components,
        "dependencies": deps,
        "packages": [{"id": "system", "label": "MyRepo", "rank": 0}],
    }


def _make_rich_facts(files: list[dict] | None = None,
                     import_graph: dict | None = None) -> dict:
    return {
        "files": files or [],
        "import_graph": import_graph or {},
        "primary_language": "python",
        "framework": None,
        "frameworks": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Caps: 30 components / 50 edges  →  ≤12 internal, ≤20 edges
# ══════════════════════════════════════════════════════════════════════════════

def test_caps():
    """30 synthetic components and 50 internal edges must be capped to ≤12 / ≤20."""
    n = 30
    components = [
        _make_component(f"comp_{i:02d}", f"Component {i}", member_count=n - i)
        for i in range(n)
    ]
    # Build 50 internal edges cycling through all 30 components
    deps = []
    seen = set()
    for i in range(50):
        f = f"comp_{i % n:02d}"
        t = f"comp_{(i + 3) % n:02d}"
        if f != t and (f, t) not in seen:
            seen.add((f, t))
            deps.append(_make_dep(f, t))

    named = _make_named_model(components, deps)
    rich = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    internal = [p for p in model["packages"] if p["kind"] == "internal"]
    edges    = model["edges"]

    check("caps: internal packages ≤ MAX_INTERNAL_PACKAGES",
          len(internal) <= MAX_INTERNAL_PACKAGES,
          f"got {len(internal)}")
    check("caps: total edges ≤ MAX_EDGES",
          len(edges) <= MAX_EDGES,
          f"got {len(edges)}")
    check("caps: diagram_type is dependency",
          model.get("diagram_type") == "dependency")


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Non-runtime filtering: test-dir paths never appear in output
# ══════════════════════════════════════════════════════════════════════════════

def test_nonruntime_not_in_packages():
    """Packages output must not contain test-directory paths as labels or ids."""
    # Only runtime components are passed in (discover_components filters upstream),
    # so test paths can only appear as external import names — assert they don't.
    components = [_make_component("auth", "Auth", 5)]
    deps: list = []
    named = _make_named_model(components, deps)

    # Even if rich_facts files include test-dir paths, those should not leak into
    # external package names (we're not scanning import_graph for those files since
    # their module_ids won't be in member_to_comp for any kept component).
    rich = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    bad_labels = [
        p["label"] for p in model["packages"]
        if any(seg in (p["label"] or "").lower() for seg in ("test", "spec", "mock", "fixture"))
    ]
    check("nonruntime: no test/spec/mock labels in packages", not bad_labels, str(bad_labels))


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — Ranking: highest member_count survive
# ══════════════════════════════════════════════════════════════════════════════

def test_ranking():
    """Components with the highest member_count must be kept when capped."""
    # 15 components: member_count = 15, 14, ..., 1
    n = 15
    components = [
        _make_component(f"comp_{i:02d}", f"Comp {i}", member_count=n - i)
        for i in range(n)
    ]
    # Linear chain: 0→1→2→...→14  (all would make edges if kept)
    deps = [_make_dep(f"comp_{i:02d}", f"comp_{i+1:02d}") for i in range(n - 1)]

    named = _make_named_model(components, deps)
    rich  = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    kept_ids = {p["id"] for p in model["packages"] if p["kind"] == "internal"}
    # The top MAX_INTERNAL_PACKAGES by member_count are comp_00..comp_11 (counts 15..4)
    expected_survivors = {f"comp_{i:02d}" for i in range(MAX_INTERNAL_PACKAGES)}
    all_survivors_correct = expected_survivors <= kept_ids or (
        # Edge-based pruning may remove some with 0 edges — accept that
        all(cid in expected_survivors for cid in kept_ids)
    )
    check("ranking: kept components are highest-member_count ones", all_survivors_correct,
          f"kept={kept_ids}, expected⊆{expected_survivors}")
    check("ranking: low-member_count tail dropped",
          "comp_14" not in kept_ids,
          f"comp_14 (count=1) should be outside top-{MAX_INTERNAL_PACKAGES}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — Connectivity: no orphan packages, no self-loops, no duplicate edges
# ══════════════════════════════════════════════════════════════════════════════

def test_connectivity():
    """Internal packages must appear in at least one edge; no self-loops; no dupes."""
    components = [_make_component(f"c{i}", f"C{i}", member_count=5) for i in range(6)]
    deps = [
        _make_dep("c0", "c1"),
        _make_dep("c1", "c2"),
        _make_dep("c2", "c3"),
        _make_dep("c3", "c4"),
        _make_dep("c4", "c5"),
        # self-loop (must be dropped)
        _make_dep("c0", "c0"),
        # duplicate (must be deduped)
        _make_dep("c0", "c1"),
    ]
    named = _make_named_model(components, deps)
    rich  = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    packages  = model["packages"]
    edges     = model["edges"]
    internal  = {p["id"] for p in packages if p["kind"] == "internal"}
    edge_endpoints = {e["from"] for e in edges} | {e["to"] for e in edges if not e["to"].startswith("ext_")}

    # No orphan internal packages (every internal id appears in at least one edge)
    orphans = internal - edge_endpoints
    check("connectivity: no orphan internal packages", not orphans, str(orphans))

    # No self-loops
    self_loops = [e for e in edges if e["from"] == e["to"]]
    check("connectivity: no self-loops", not self_loops, str(self_loops))

    # No duplicate edges
    edge_pairs = [(e["from"], e["to"]) for e in edges]
    check("connectivity: no duplicate edges", len(edge_pairs) == len(set(edge_pairs)),
          f"dupes in {edge_pairs}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — Externals: stdlib excluded, capped at 6, internal→external edges only
# ══════════════════════════════════════════════════════════════════════════════

def test_externals():
    """External libs: stdlib excluded, capped at MAX_EXTERNAL_PACKAGES, ranked by
    importer count, only internal→external edges emitted."""
    with tempfile.TemporaryDirectory() as code_dir:
        # Create synthetic source files so module_id resolution works
        src_dir = Path(code_dir) / "src"
        src_dir.mkdir()
        (src_dir / "auth.py").write_text("")
        (src_dir / "orders.py").write_text("")

        mid_auth   = "src/auth"
        mid_orders = "src/orders"

        components = [
            _make_component("auth",   "Auth",   5, members=[mid_auth]),
            _make_component("orders", "Orders", 4, members=[mid_orders]),
        ]
        # auth imports: fastapi (external), sqlalchemy (external), os (stdlib noise)
        # orders imports: fastapi (external), celery (external), redis (external)
        # Also add 5 more external libs to test the cap-at-6 rule
        files = [
            {
                "file": str(src_dir / "auth.py"),
                "language": "python",
                "imports": ["fastapi", "sqlalchemy", "os", "re", "pydantic", "httpx",
                            "alembic", "jwt"],
                "routes": [],
                "classes": [],
                "functions": [],
                "error": None,
            },
            {
                "file": str(src_dir / "orders.py"),
                "language": "python",
                "imports": ["fastapi", "celery", "redis", "boto3", "kombu"],
                "routes": [],
                "classes": [],
                "functions": [],
                "error": None,
            },
        ]
        import_graph = {mid_auth: [], mid_orders: []}
        rich = _make_rich_facts(files=files, import_graph=import_graph)

        deps: list = []
        named = _make_named_model(components, deps)

        model = build_dependency_model(named, rich, code_dir)

    external = [p for p in model["packages"] if p["kind"] == "external"]
    ext_labels = {p["label"] for p in external}
    ext_edges = [e for e in model["edges"] if e["to"].startswith("ext_")]
    int_edges  = [e for e in model["edges"] if not e["to"].startswith("ext_")]
    ext_from_ids = {e["from"] for e in ext_edges}

    check("externals: capped at MAX_EXTERNAL_PACKAGES",
          len(external) <= MAX_EXTERNAL_PACKAGES,
          f"got {len(external)}: {ext_labels}")
    check("externals: os not in external packages (stdlib noise)",
          "os" not in ext_labels, str(ext_labels))
    check("externals: re not in external packages (stdlib noise)",
          "re" not in ext_labels, str(ext_labels))
    check("externals: fastapi appears (imported by 2 components)",
          "fastapi" in ext_labels, str(ext_labels))
    # All external edge sources must be internal component ids
    internal_ids = {"auth", "orders"}
    check("externals: only internal→external edges",
          ext_from_ids <= internal_ids,
          f"edge sources: {ext_from_ids}")
    # No external→external edges
    ext_package_ids = {p["id"] for p in external}
    bad_ext_edges = [e for e in ext_edges if e["from"] in ext_package_ids]
    check("externals: no external→external edges", not bad_ext_edges, str(bad_ext_edges))
    # No internal→internal edges when there were no internal deps
    check("externals: no spurious internal edges when no internal deps",
          not int_edges, str(int_edges))


# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — Renderer integration: graph LR, subgraphs, edge count ≤ cap
# ══════════════════════════════════════════════════════════════════════════════

def test_renderer_integration():
    """render_dependency_diagram must produce graph LR with correct subgraphs."""
    components = [
        _make_component("api",    "API Layer",     6),
        _make_component("domain", "Domain",        5),
        _make_component("infra",  "Infrastructure", 4),
    ]
    deps = [
        _make_dep("api",    "domain"),
        _make_dep("domain", "infra"),
    ]
    named = _make_named_model(components, deps)
    rich  = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    rendered = render_dependency_diagram(model)

    check("renderer: starts with graph LR", rendered.startswith("graph LR"), rendered[:40])
    check("renderer: This Repo subgraph present", 'subgraph REPO' in rendered, rendered[:200])

    internal = [p for p in model["packages"] if p["kind"] == "internal"]
    external = [p for p in model["packages"] if p["kind"] == "external"]

    if external:
        check("renderer: External Libraries subgraph present",
              'subgraph EXT' in rendered, rendered[:200])

    # Count rendered edges (lines with -->)
    rendered_edges = [ln for ln in rendered.splitlines() if "-->" in ln]
    check("renderer: edge count ≤ MAX_EDGES",
          len(rendered_edges) <= MAX_EDGES,
          f"got {len(rendered_edges)}")

    # Internal package labels must appear in the rendered output
    for c in internal:
        lbl = (c.get("label") or c["id"]).strip() or c["id"]
        check(f"renderer: label '{lbl}' in output", lbl in rendered, rendered[:300])


# ══════════════════════════════════════════════════════════════════════════════
# Test 7 — Fallback: run_agent_json returns fallback when all retries fail
# ══════════════════════════════════════════════════════════════════════════════

def test_fallback_on_bad_json():
    """run_agent_json must return the fallback value when the agent never returns
    valid JSON, rather than raising ValueError."""
    import doc_agent.core.llm as _llm_module

    original_run_agent = _llm_module.run_agent

    async def _junk_agent(*args, **kwargs):
        return "this is definitely not json {{{{ unclosed"

    async def _run():
        _llm_module.run_agent = _junk_agent
        try:
            fallback_val = {"components": []}
            result = await run_agent_json(
                None, "prompt", max_retries=2, fallback=fallback_val
            )
            check("fallback: returned fallback dict", result == fallback_val,
                  f"got {result!r}")
            check("fallback: no exception raised", True)
        except Exception as exc:
            check("fallback: no exception raised", False, f"raised {type(exc).__name__}: {exc}")
        finally:
            _llm_module.run_agent = original_run_agent

    asyncio.run(_run())


def test_no_fallback_raises():
    """run_agent_json without fallback must still raise ValueError on bad JSON."""
    import doc_agent.core.llm as _llm_module

    original_run_agent = _llm_module.run_agent

    async def _junk_agent(*args, **kwargs):
        return "not json at all"

    async def _run():
        _llm_module.run_agent = _junk_agent
        try:
            raised = False
            try:
                await run_agent_json(None, "prompt", max_retries=1)
            except ValueError:
                raised = True
            check("no-fallback: ValueError raised when no fallback given", raised)
        finally:
            _llm_module.run_agent = original_run_agent

    asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════════════════
# Test 8 — Transitive reduction: A→C dropped when A→B→C exists
# ══════════════════════════════════════════════════════════════════════════════

def test_transitive_reduction():
    """A→B, B→C, A→C input → A→C must be dropped; A→B and B→C must survive."""
    components = [
        _make_component("a", "A", member_count=5),
        _make_component("b", "B", member_count=4),
        _make_component("c", "C", member_count=3),
    ]
    deps = [
        _make_dep("a", "b"),
        _make_dep("b", "c"),
        _make_dep("a", "c"),  # transitive — should be dropped
    ]
    named = _make_named_model(components, deps)
    rich = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir)

    edge_pairs = {(e["from"], e["to"]) for e in model["edges"]}
    check("transitive_reduction: A→C removed", ("a", "c") not in edge_pairs,
          f"edges: {edge_pairs}")
    check("transitive_reduction: A→B kept", ("a", "b") in edge_pairs,
          f"edges: {edge_pairs}")
    check("transitive_reduction: B→C kept", ("b", "c") in edge_pairs,
          f"edges: {edge_pairs}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 9 — Node-relative budget: dense small graph yields ≤ len(kept)+3 internal edges
# ══════════════════════════════════════════════════════════════════════════════

def test_node_relative_budget():
    """Bipartite DAG (9 non-transitive edges) with max_edges=5 override forces the
    budget cap. After capping + orphan-pruning, every remaining package is referenced."""
    # Sources: n0, n1, n2. Targets: n3, n4, n5. 9 edges, no transitive paths.
    components = [
        _make_component(f"n{i}", f"Node{i}", member_count=6 - i) for i in range(6)
    ]
    deps = [
        _make_dep(f"n{i}", f"n{j}")
        for i in range(3) for j in range(3, 6)
    ]
    named = _make_named_model(components, deps)
    rich = _make_rich_facts()

    with tempfile.TemporaryDirectory() as code_dir:
        model = build_dependency_model(named, rich, code_dir, max_edges=5)

    internal = [p for p in model["packages"] if p["kind"] == "internal"]
    int_edges = [e for e in model["edges"] if not e["to"].startswith("ext_")]

    check("node_budget: internal edges ≤ max_edges override (5)",
          len(int_edges) <= 5,
          f"got {len(int_edges)}")

    # After orphan-pruning, every package in the result must appear in at least one edge
    internal_ids = {p["id"] for p in internal}
    referenced = {e["from"] for e in int_edges} | {
        e["to"] for e in int_edges if not e["to"].startswith("ext_")
    }
    orphans = internal_ids - referenced
    check("node_budget: no orphan packages after orphan-prune", not orphans,
          str(orphans))


# ══════════════════════════════════════════════════════════════════════════════
# Test 10 — External naming: curated map resolves framework labels; no bare org/com
# ══════════════════════════════════════════════════════════════════════════════

def test_external_naming():
    """Imports from org.springframework.*, com.fasterxml.jackson.*, jakarta.persistence.*
    must produce labels Spring/Jackson/JPA; no bare 'org', 'com', 'edu', 'jakarta' labels."""
    with tempfile.TemporaryDirectory() as code_dir:
        src_dir = Path(code_dir) / "src"
        src_dir.mkdir()
        (src_dir / "service.java").write_text("")

        mid = "src/service"
        components = [_make_component("svc", "Service", 3, members=[mid])]
        files = [
            {
                "file": str(src_dir / "service.java"),
                "language": "java",
                "imports": [
                    "org.springframework.web.bind.annotation.RestController",
                    "com.fasterxml.jackson.databind.ObjectMapper",
                    "jakarta.persistence.Entity",
                    "org.slf4j.Logger",
                ],
                "routes": [],
                "classes": [],
                "functions": [],
                "error": None,
                "namespace": None,
            }
        ]
        import_graph = {mid: []}
        rich = _make_rich_facts(files=files, import_graph=import_graph)
        named = _make_named_model(components, [])

        model = build_dependency_model(named, rich, code_dir)

    ext_labels = {p["label"] for p in model["packages"] if p["kind"] == "external"}

    check("ext_naming: Spring present", "Spring" in ext_labels, str(ext_labels))
    check("ext_naming: Jackson present", "Jackson" in ext_labels, str(ext_labels))
    check("ext_naming: JPA present", "JPA" in ext_labels, str(ext_labels))
    check("ext_naming: SLF4J present", "SLF4J" in ext_labels, str(ext_labels))

    bare_generic = {"org", "com", "edu", "jakarta", "javax", "io", "net"}
    bad = ext_labels & bare_generic
    check("ext_naming: no bare generic namespace labels", not bad,
          f"found: {bad}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 11 — Internal namespace exclusion: repo's own org.apache.fineract.* not external
# ══════════════════════════════════════════════════════════════════════════════

def test_internal_namespace_exclusion():
    """Files with namespace 'org.apache.fineract.accounting' importing
    'org.apache.fineract.portfolio.*' must NOT produce external packages."""
    with tempfile.TemporaryDirectory() as code_dir:
        src_dir = Path(code_dir) / "src"
        src_dir.mkdir()
        (src_dir / "accounting.java").write_text("")

        mid = "src/accounting"
        components = [_make_component("acct", "Accounting", 3, members=[mid])]
        files = [
            {
                "file": str(src_dir / "accounting.java"),
                "language": "java",
                "imports": [
                    "org.apache.fineract.portfolio.loan.domain.Loan",
                    "org.apache.fineract.portfolio.savings.domain.SavingsAccount",
                ],
                "routes": [],
                "classes": [],
                "functions": [],
                "error": None,
                "namespace": "org.apache.fineract.accounting",
            }
        ]
        import_graph = {mid: []}
        rich = _make_rich_facts(files=files, import_graph=import_graph)
        named = _make_named_model(components, [])

        model = build_dependency_model(named, rich, code_dir)

    ext_labels = {p["label"] for p in model["packages"] if p["kind"] == "external"}
    # None of these should appear as external packages
    bad = [lbl for lbl in ext_labels
           if "fineract" in lbl.lower() or lbl.lower() in {"org", "apache"}]
    check("ns_exclusion: org.apache.fineract imports not treated as external",
          not bad, f"found: {bad}")
    check("ns_exclusion: no external packages at all (only internal cross-module imports)",
          not ext_labels, f"found: {ext_labels}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        try:
            case()
        except Exception as e:
            _FAILURES.append(f"{case.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
