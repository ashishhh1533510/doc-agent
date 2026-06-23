"""
dependency_graph.py — deterministic projector: architecture component model → dependency diagram.

Takes the named component model from ComponentDiagramAgent.refine() and projects it to
the flat package-graph shape that render_dependency_diagram() consumes:
  {diagram_type, packages: [{id, label, kind}], edges: [{from, to, label}]}

Internal packages come from the named component model (architecture-centric, bounded).
External libraries are derived deterministically from import evidence, stdlib-filtered,
bounded to max_external. The LLM is never trusted for structure; only labels.
"""
from __future__ import annotations

from collections import deque

from doc_agent.tools.import_graph import module_id as _module_id

MAX_INTERNAL_PACKAGES = 12
MAX_EXTERNAL_PACKAGES = 6
MAX_EDGES = 20

# Cross-language safety net for stdlib/runtime noise that per-language extractors
# may not strip (Python stdlib is already stripped at extraction time).
# Note: javax is intentionally NOT here so javax.persistence → JPA via the curated map.
_STDLIB_NOISE: frozenset[str] = frozenset({
    # generic / multi-language
    "os", "sys", "re", "json", "io", "math", "time", "datetime", "collections",
    "itertools", "functools", "typing", "pathlib", "logging", "copy", "abc",
    "threading", "subprocess", "hashlib", "base64", "uuid", "enum", "dataclasses",
    "contextlib", "traceback", "inspect", "ast", "warnings", "string", "struct",
    # Java / Kotlin / Scala builtins
    "java", "kotlin", "scala",
    # JS / TS built-ins
    "fs", "path", "http", "https", "url", "util", "events", "stream", "buffer",
    "crypto", "assert", "process", "console", "child_process", "os", "cluster",
})

# Generic namespace prefixes that alone carry no architectural meaning.
_GENERIC_NS: frozenset[str] = frozenset({
    "org", "com", "io", "net", "edu", "gov", "gnu", "co", "uk", "jakarta", "javax",
})

# Curated framework map: longest dotted-prefix match wins.
# Key: dotted import prefix (lowercase). Value: display label.
_FRAMEWORK_MAP: list[tuple[str, str]] = sorted([
    ("org.springframework.boot",      "Spring Boot"),
    ("org.springframework",           "Spring"),
    ("org.hibernate",                 "Hibernate"),
    ("org.apache.commons",            "Apache Commons"),
    ("org.apache.kafka",              "Kafka"),
    ("org.apache.logging",            "Log4j"),
    ("org.slf4j",                     "SLF4J"),
    ("org.mapstruct",                 "MapStruct"),
    ("org.mockito",                   "Mockito"),
    ("org.junit",                     "JUnit"),
    ("com.fasterxml.jackson",         "Jackson"),
    ("com.google.common",             "Guava"),
    ("com.google.gson",               "Gson"),
    ("com.google.inject",             "Guice"),
    ("com.amazonaws",                 "AWS SDK"),
    ("com.azure",                     "Azure SDK"),
    ("jakarta.persistence",           "JPA"),
    ("javax.persistence",             "JPA"),
    ("jakarta.validation",            "Jakarta Validation"),
    ("jakarta.ws.rs",                 "Jakarta REST"),
    ("jakarta.servlet",               "Jakarta Servlet"),
], key=lambda kv: -len(kv[0]))  # longest prefix first


def _external_lib_key(imp: str) -> tuple[str, str] | None:
    """Map a raw import string to (key, label) for external package display.

    Returns None when the import should be hidden (stdlib noise or bare generic namespace).
    """
    low = imp.lower().strip()
    if not low:
        return None

    # Stdlib noise — exact first segment match
    first_seg = low.split(".")[0]
    if first_seg in _STDLIB_NOISE:
        return None

    # Curated map: longest dotted-prefix match
    for prefix, label in _FRAMEWORK_MAP:
        if low == prefix or low.startswith(prefix + "."):
            key = label.lower().replace(" ", "_")
            return key, label

    # Heuristic fallback for unmapped namespaced imports
    segs = low.split(".")
    if len(segs) > 1:
        # Skip leading generic-namespace segments
        idx = 0
        while idx < len(segs) and segs[idx] in _GENERIC_NS:
            idx += 1
        if idx >= len(segs):
            # All segments were generic — hide it
            return None
        key = segs[idx]
        label = key.capitalize()
        return key, label

    # Simple non-namespaced import (Python/JS): keep as-is
    return low, imp.strip()


def _reachable_without(
    node: str,
    target: str,
    adjacency: dict[str, list[str]],
) -> bool:
    """Return True iff `target` is reachable from `node` via paths that don't use
    the direct edge node→target (i.e. through at least one intermediate hop)."""
    visited = {node}
    queue: deque[str] = deque()
    for nb in adjacency.get(node, []):
        if nb != target and nb not in visited:
            visited.add(nb)
            queue.append(nb)
    while queue:
        cur = queue.popleft()
        if cur == target:
            return True
        for nb in adjacency.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return False


def _transitive_reduce(edges: list[dict]) -> list[dict]:
    """Drop edge (u,v) iff v is reachable from u through another path.

    Processes edges in a stable sorted order for determinism.
    Cycle-safe for our tiny (≤12 node) graphs.
    """
    # Build adjacency list
    adjacency: dict[str, list[str]] = {}
    for e in edges:
        adjacency.setdefault(e["from"], []).append(e["to"])

    kept = []
    for e in sorted(edges, key=lambda x: (x["from"], x["to"])):
        if not _reachable_without(e["from"], e["to"], adjacency):
            kept.append(e)
    return kept


def build_dependency_model(
    named_component_model: dict,
    rich_facts: dict,
    code_dir: str,
    *,
    max_internal: int = MAX_INTERNAL_PACKAGES,
    max_external: int = MAX_EXTERNAL_PACKAGES,
    max_edges: int = MAX_EDGES,
) -> dict:
    """Project a named component model + rich_facts into a flat dependency diagram.

    Returns {"diagram_type": "dependency", "packages": [...], "edges": [...]} in
    the exact shape render_dependency_diagram() expects.
    """
    components = named_component_model.get("components", [])
    dependencies = named_component_model.get("dependencies", [])

    # ── Internal packages: rank by member_count desc, cap at max_internal ────────
    ranked = sorted(
        components,
        key=lambda c: (-(c.get("member_count") or len(c.get("members", [])) or 1), c["id"]),
    )
    kept = ranked[:max_internal]
    kept_ids: set[str] = {c["id"] for c in kept}

    # Build member_count lookup for edge ranking
    member_count: dict[str, int] = {
        c["id"]: (c.get("member_count") or len(c.get("members", [])) or 1)
        for c in kept
    }

    internal_packages = [
        {
            "id": c["id"],
            "label": (c.get("label") or "").strip() or c["id"],
            "kind": "internal",
        }
        for c in kept
    ]

    # ── Internal edges: filter to kept_ids, no self-loops, dedupe ────────────────
    seen_pairs: set[tuple] = set()
    internal_edges: list[dict] = []
    for e in dependencies:
        f, t = e.get("from"), e.get("to")
        if not f or not t or f not in kept_ids or t not in kept_ids or f == t:
            continue
        key = (f, t)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        internal_edges.append({
            "from": f,
            "to": t,
            "label": e.get("label", "depends on"),
            "weight": e.get("weight", 1),
        })

    # ── 1a: Tree-leaning thinning ─────────────────────────────────────────────────
    # Pass 1: transitive reduction — drop A→C when A→B→C exists.
    # Fallback: cycle-heavy graphs produce zero-edge results; skip reduction in that case.
    reduced = _transitive_reduce(internal_edges)
    internal_edges = reduced if reduced else internal_edges

    # Pass 2: node-relative budget — keep highest-weight edges within budget.
    internal_edge_budget = min(max_edges, len(kept) + 3)
    if len(internal_edges) > internal_edge_budget:
        internal_edges = sorted(
            internal_edges,
            key=lambda e: (
                -e.get("weight", 1),
                -(member_count.get(e["from"], 1) + member_count.get(e["to"], 1)),
                e["from"],
                e["to"],
            ),
        )[:internal_edge_budget]

    # ── External packages: from per-file import evidence in rich_facts ────────────
    import_graph: dict = rich_facts.get("import_graph", {})
    internal_module_ids: set[str] = set(import_graph.keys())

    # Top-level path segments of internal module_ids — used to tell internal from external.
    # E.g. for module_ids like "doc_agent/core/llm" the top segment is "doc_agent".
    internal_tops: set[str] = {mid.split("/")[0] for mid in internal_module_ids if mid}

    # ── 1b: Namespace-aware internal detection ────────────────────────────────────
    # Collect the repo's own namespaces (Java/C# namespace declarations) from rich_facts.
    # E.g. "org.apache.fineract.accounting" → also register "org.apache.fineract" (3-seg root).
    _internal_ns_roots: set[str] = set()
    for f in rich_facts.get("files", []):
        ns = f.get("namespace")
        if not ns:
            continue
        ns_low = ns.lower().strip()
        _internal_ns_roots.add(ns_low)
        # Also add up to 3-segment prefix for broad matching
        segs = ns_low.split(".")
        for depth in range(1, min(4, len(segs))):
            _internal_ns_roots.add(".".join(segs[:depth]))

    def _is_internal_import(imp: str) -> bool:
        """Return True if imp is an import of the repo's own code (not an external lib)."""
        low = imp.lower().strip()
        # Check path-segment tops (Python/JS style)
        if low.split(".")[0].split("/")[0] in internal_tops:
            return True
        # Check namespace roots (Java/C# style)
        for root in _internal_ns_roots:
            if low == root or low.startswith(root + "."):
                return True
        return False

    # Map module_id → file-level raw imports from rich_facts
    file_imports_map: dict[str, list[str]] = {}
    for f in rich_facts.get("files", []):
        if f.get("error"):
            continue
        fpath = f.get("file") or ""
        if not fpath:
            continue
        try:
            mid = _module_id(fpath, code_dir)
            file_imports_map[mid] = f.get("imports", [])
        except Exception:
            pass

    # Build component-member lookup from kept components only
    member_to_comp: dict[str, str] = {}
    for c in kept:
        for m in c.get("members", []):
            member_to_comp[m] = c["id"]

    # ── 1c: Curated framework map + heuristic external keys ───────────────────────
    # ext_label_map[key] = display label
    ext_label_map: dict[str, str] = {}
    # ext_comp_set[key] = set of component_ids that import it (for ranking)
    ext_comp_set: dict[str, set[str]] = {}
    # ext_comp_count[key][comp_id] = import count (for picking the edge source)
    ext_comp_count: dict[str, dict[str, int]] = {}

    for mid, comp_id in member_to_comp.items():
        raw_imports = file_imports_map.get(mid, [])
        for imp in raw_imports:
            if not imp:
                continue
            # Skip relative imports — those resolve to internal modules
            if imp.startswith("."):
                continue
            # Skip if it looks internal (path-segment or namespace match)
            if _is_internal_import(imp):
                continue
            result = _external_lib_key(imp)
            if result is None:
                continue
            lib_key, lib_label = result
            ext_label_map[lib_key] = lib_label
            ext_comp_set.setdefault(lib_key, set()).add(comp_id)
            ext_comp_count.setdefault(lib_key, {})
            ext_comp_count[lib_key][comp_id] = ext_comp_count[lib_key].get(comp_id, 0) + 1

    # Rank external libs by number of distinct kept components importing them
    ranked_ext = sorted(
        ext_comp_set.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    selected_ext = ranked_ext[:max_external]

    external_packages = [
        {"id": f"ext_{lib_key}", "label": ext_label_map.get(lib_key, lib_key), "kind": "external"}
        for lib_key, _ in selected_ext
    ]

    # One internal→external edge per library, from the most-frequent importing component
    seen_ext: set[tuple] = set()
    external_edges: list[dict] = []
    for lib_key, _ in selected_ext:
        ext_id = f"ext_{lib_key}"
        counts = ext_comp_count.get(lib_key, {})
        # most-frequent importer; stable tie-break by comp_id
        best_comp = max(counts, key=lambda cid: (counts[cid], cid), default=None)
        if not best_comp:
            continue
        pair = (best_comp, ext_id)
        if pair in seen_ext:
            continue
        seen_ext.add(pair)
        external_edges.append({"from": best_comp, "to": ext_id, "label": "uses"})

    # ── Final cap: internal edges first (architecture-primary), then external ─────
    # Strip internal weight field before output (not part of render contract)
    clean_internal = [{"from": e["from"], "to": e["to"], "label": e["label"]} for e in internal_edges]
    all_edges = clean_internal + external_edges
    all_edges = all_edges[:max_edges]

    # Drop internal packages that ended up with zero edges (readability), but never
    # empty the internal set entirely.
    referenced_in_edges: set[str] = set()
    for e in all_edges:
        referenced_in_edges.add(e["from"])
        if not e["to"].startswith("ext_"):
            referenced_in_edges.add(e["to"])

    if referenced_in_edges:
        pruned = [p for p in internal_packages if p["id"] in referenced_in_edges]
        if pruned:
            internal_packages = pruned

    return {
        "diagram_type": "dependency",
        "packages": internal_packages + external_packages,
        "edges": all_edges,
    }
