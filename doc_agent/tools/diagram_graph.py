"""
Graph-analysis toolkit for diagram generation.

Deterministic and repository-agnostic. Turns extracted RichFacts into a typed,
weighted class-relationship graph, finds natural communities (Louvain), and
projects bounded, cohesive *views* a human can actually read. The LLM later
refines each bounded view — it never does whole-repo decomposition.

The only baked-in constants are human-perception budgets (nodes/edges per
diagram). Everything else (cluster count, boundaries) is discovered from the graph.
"""
from __future__ import annotations

import os
import re

import networkx as nx
from networkx.algorithms.community import louvain_communities

# ── perception budget — the only universal constants ──────────────────────
MAX_NODES_PER_VIEW = 20
MAX_EDGES_PER_VIEW = 30
MIN_COMMUNITY_SIZE = 3
MAX_VIEWS = 8

# structural relationships matter far more than transient calls
_TYPE_WEIGHT = {
    "inheritance": 4, "realization": 4,
    "composition": 3, "aggregation": 2,
    "dependency": 1,
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# ── deterministic noise filters (language-agnostic, no repo specifics) ─────
# test code: a path segment that is exactly a test dir, or a file whose stem
# ends in Test/Tests/Spec/Specs (covers .cs, .java, .py, .ts, .js conventions)
_TEST_DIR = re.compile(r"(^|/)(tests?|__tests__|testing|spec|specs)(/|$)", re.I)
_TEST_FILE = re.compile(r"(test|tests|spec|specs)$", re.I)
# pure data carriers: a class whose name ends in one of these is a DTO/contract
_DTO_SUFFIX = re.compile(r"(dto|response|request|payload|args|eventargs)$", re.I)


def _identifiers(text: str) -> list[str]:
    return _IDENT.findall(text or "")


def _is_test_file(file_path: str) -> bool:
    p = (file_path or "").replace("\\", "/")
    if _TEST_DIR.search(p):
        return True
    stem = os.path.splitext(os.path.basename(p))[0]
    return bool(_TEST_FILE.search(stem))


def _is_data_carrier(cls: dict) -> bool:
    """DTO/response/request class with no behaviour — drop from the design graph.
    A method-less class whose name carries a data-contract suffix is a carrier;
    domain entities (which have methods) and ViewModels are kept."""
    name = cls.get("name") or ""
    methods = [m for m in cls.get("methods", []) if (m.get("name") or "") != "__init__"]
    return bool(_DTO_SUFFIX.search(name)) and not methods


def _package(file_path: str) -> str:
    """Best-effort top-level package key for the same-package prior."""
    p = (file_path or "").replace("\\", "/")
    parts = [x for x in os.path.dirname(p).split("/") if x and x != "."]
    return "/".join(parts[:2]) or "root"


def build_class_graph(rich_facts: dict) -> nx.DiGraph:
    """Typed, weighted class graph derived only from extracted facts."""
    classes: dict[str, dict] = {}
    pkg: dict[str, str] = {}
    for f in rich_facts.get("files", []):
        if f.get("error"):
            continue
        if _is_test_file(f.get("file", "")):
            continue  # exclude test code from the design graph
        for c in f.get("classes", []):
            name = c.get("name")
            if name and name not in classes and not _is_data_carrier(c):
                classes[name] = c
                pkg[name] = _package(f.get("file", ""))
    declared = set(classes)

    g = nx.DiGraph()
    for name, c in classes.items():
        g.add_node(name, cls=c, package=pkg[name])

    def add_edge(src: str, dst: str, etype: str) -> None:
        if src == dst or dst not in declared:
            return
        w = _TYPE_WEIGHT.get(etype, 1)
        if pkg.get(src) == pkg.get(dst):
            w += 1  # same-package prior (a hint, NOT the partition key)
        if g.has_edge(src, dst):
            if w > g[src][dst]["weight"]:
                g[src][dst].update(type=etype, weight=w)
        else:
            g.add_edge(src, dst, type=etype, weight=w)

    for name, c in classes.items():
        for base in c.get("bases", []):
            for tok in _identifiers(base):
                add_edge(name, tok, "inheritance")
        for fld in c.get("fields", []):
            for tok in _identifiers(fld.get("type", "")):
                add_edge(name, tok, "composition")
        for m in c.get("methods", []):
            for call in m.get("calls", []):
                for tok in _identifiers(call):
                    add_edge(name, tok, "dependency")
    return g


def detect_communities(graph: nx.DiGraph) -> list[set]:
    if graph.number_of_nodes() == 0:
        return []
    ug = nx.Graph()
    ug.add_nodes_from(graph.nodes())
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 1)
        if ug.has_edge(u, v):
            ug[u][v]["weight"] += w
        else:
            ug.add_edge(u, v, weight=w)
    if ug.number_of_edges() == 0:
        return [{n} for n in ug.nodes()]
    return louvain_communities(ug, weight="weight", seed=42)  # seed → reproducible


def centrality(graph: nx.DiGraph) -> dict:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, weight="weight")
    except Exception:
        return dict(graph.degree(weight="weight"))


def project_view(graph: nx.DiGraph, members: set, cent: dict, label: str = "") -> dict:
    """Bounded, orphan-free candidate class-model for one view."""
    ranked = sorted(members, key=lambda n: cent.get(n, 0), reverse=True)
    keep = set(ranked[:MAX_NODES_PER_VIEW])

    scored = [
        (d.get("weight", 1), u, v, d.get("type", "dependency"))
        for u, v, d in graph.edges(data=True)
        if u in keep and v in keep
    ]
    scored.sort(key=lambda e: e[0], reverse=True)
    scored = scored[:MAX_EDGES_PER_VIEW]

    connected = {u for _, u, v, _ in scored} | {v for _, u, v, _ in scored}
    if len(keep) > 1:
        keep = {n for n in keep if n in connected}  # drop orphans

    classes = [
        {"name": n,
         "fields": graph.nodes[n]["cls"].get("fields", []),
         "methods": graph.nodes[n]["cls"].get("methods", [])}
        for n in keep
    ]
    relationships = [
        {"from": u, "to": v, "type": t}
        for _, u, v, t in scored if u in keep and v in keep
    ]
    return {"diagram_type": "class", "label": label,
            "classes": classes, "relationships": relationships}


def plan_class_views(graph: nx.DiGraph) -> list[dict]:
    """Decide how many class diagrams to emit, from measured graph size."""
    cent = centrality(graph)
    n = graph.number_of_nodes()
    if n == 0:
        return []
    if n <= MAX_NODES_PER_VIEW:
        v = project_view(graph, set(graph.nodes()), cent, "Overview")
        return [v] if v["classes"] else []

    comms = [c for c in detect_communities(graph) if len(c) >= MIN_COMMUNITY_SIZE]
    comms.sort(key=len, reverse=True)
    comms = comms[:MAX_VIEWS]
    if not comms:
        top = set(sorted(graph.nodes(), key=lambda x: cent.get(x, 0),
                         reverse=True)[:MAX_NODES_PER_VIEW])
        v = project_view(graph, top, cent, "Core")
        return [v] if v["classes"] else []

    views = []
    for i, c in enumerate(comms):
        v = project_view(graph, set(c), cent, f"Cluster {i + 1}")
        if v["classes"]:
            views.append(v)
    return views


# ══════════════════════════════════════════════════════════════════════════
# SEQUENCE — entry-point enumeration + call-graph BFS/DFS → bounded workflows
# ══════════════════════════════════════════════════════════════════════════

# perception budgets for a readable sequence
MAX_PARTICIPANTS = 8
MAX_MESSAGES = 16
MAX_DEPTH = 6
MAX_SEQUENCE_VIEWS = 3

_ENTRY_FUNC_NAMES = {"main", "run", "handle", "execute", "start"}


def _short_module(file_path: str) -> str:
    """Module short name (file stem) — owner label for free functions."""
    base = os.path.basename((file_path or "").replace("\\", "/"))
    return os.path.splitext(base)[0] or "module"


def _last_segment(call: str) -> str:
    """Last identifier of a (possibly dotted) call expression: 'a.b.run' -> 'run'."""
    toks = _identifiers(call)
    return toks[-1] if toks else ""


def build_call_graph(rich_facts: dict) -> dict:
    """Owner-keyed call graph built only from extracted facts.

    Returns:
      {
        "callables": { callable_key: {name, owner, is_async, calls:[...], is_entry, route} },
        "by_name":   { simple_name: [callable_key, ...] },   # for call resolution
        "entries":   [callable_key, ...],                    # route handlers + entry funcs
      }
    A 'callable' is a method (owner = its class) or a free function (owner = its module).
    """
    callables: dict[str, dict] = {}
    by_name: dict[str, list] = {}
    entries: list[str] = []

    def _add(key, name, owner, is_async, calls, is_entry, route=None):
        callables[key] = {"name": name, "owner": owner, "is_async": bool(is_async),
                          "calls": list(calls or []), "is_entry": is_entry, "route": route}
        by_name.setdefault(name, []).append(key)
        if is_entry:
            entries.append(key)

    for f in rich_facts.get("files", []):
        if f.get("error") or _is_test_file(f.get("file", "")):
            continue
        module = _short_module(f.get("file", ""))
        handlers = {r.get("handler"): r for r in f.get("routes", []) if r.get("handler")}

        # free functions (owner = module)
        for fn in f.get("functions", []):
            name = fn.get("name")
            if not name:
                continue
            route = handlers.get(name)
            is_entry = route is not None or name.lower() in _ENTRY_FUNC_NAMES
            _add(f"{module}.{name}", name, module, fn.get("is_async"),
                 fn.get("calls", []), is_entry, route)

        # methods (owner = class)
        for c in f.get("classes", []):
            owner = c.get("name") or module
            for m in c.get("methods", []):
                name = m.get("name")
                if not name:
                    continue
                route = handlers.get(name)
                is_entry = route is not None or name.lower() in _ENTRY_FUNC_NAMES
                _add(f"{owner}.{name}", name, owner, m.get("is_async"),
                     m.get("calls", []), is_entry, route)

    return {"callables": callables, "by_name": by_name, "entries": entries}


def _resolve_call(cg: dict, call: str, from_owner: str) -> str | None:
    """Resolve a call expression to a declared callable key, preferring a
    cross-owner target so messages connect participants. Returns None for
    external/stdlib calls (not declared) — they are dropped from the diagram."""
    name = _last_segment(call)
    cands = cg["by_name"].get(name)
    if not cands:
        return None
    # prefer a callable owned by someone other than the caller
    for k in cands:
        if cg["callables"][k]["owner"] != from_owner:
            return k
    return cands[0]


def _reach(cg: dict, start: str) -> int:
    """Count distinct owners reachable from an entry (BFS, owner-level)."""
    seen_keys, owners, stack = {start}, set(), [start]
    while stack:
        k = stack.pop()
        c = cg["callables"][k]
        owners.add(c["owner"])
        for call in c["calls"]:
            t = _resolve_call(cg, call, c["owner"])
            if t and t not in seen_keys:
                seen_keys.add(t)
                stack.append(t)
    return len(owners)


def _trace_sequence(cg: dict, entry_key: str) -> dict:
    """DFS from an entry in source-call order, bounded by perception budget."""
    participants: list[str] = []
    messages: list[dict] = []
    visited_pairs: set = set()

    def _see(owner):
        if owner not in participants and len(participants) < MAX_PARTICIPANTS:
            participants.append(owner)

    def visit(key, depth):
        if len(messages) >= MAX_MESSAGES:
            return
        c = cg["callables"][key]
        owner = c["owner"]
        _see(owner)
        for call in c["calls"]:
            if len(messages) >= MAX_MESSAGES:
                return
            t = _resolve_call(cg, call, owner)
            if not t:
                continue
            callee = cg["callables"][t]
            to_owner = callee["owner"]
            if to_owner not in participants and len(participants) >= MAX_PARTICIPANTS:
                continue  # would exceed participant budget — skip this branch
            _see(to_owner)
            messages.append({"from": owner, "to": to_owner,
                             "label": callee["name"],
                             "type": "async" if callee["is_async"] else "sync"})
            pair = (owner, to_owner, callee["name"])
            if pair not in visited_pairs and depth < MAX_DEPTH and to_owner != owner:
                visited_pairs.add(pair)
                visit(t, depth + 1)

    visit(entry_key, 0)
    # closing return to the originator
    if messages and participants:
        messages.append({"from": messages[-1]["to"], "to": participants[0],
                         "label": "result", "type": "return"})

    entry = cg["callables"][entry_key]
    route = entry.get("route") or {}
    name = (f'{route.get("method","")} {route.get("path","")}'.strip()
            or f'{entry["owner"]}.{entry["name"]}')
    return {"diagram_type": "sequence", "name": name,
            "participants": participants, "messages": messages}


def plan_sequence_views(rich_facts: dict) -> list[dict]:
    """Top-K grounded workflows, each a bounded candidate sequence model."""
    cg = build_call_graph(rich_facts)
    if not cg["entries"]:
        return []
    # rank entries by reach, take the most connected ones
    ranked = sorted(cg["entries"], key=lambda k: _reach(cg, k), reverse=True)

    views, seen_signatures = [], set()
    for key in ranked:
        cand = _trace_sequence(cg, key)
        if len(cand["participants"]) < 2 or len(cand["messages"]) < 2:
            continue  # need at least two participants for a real interaction
        sig = frozenset(cand["participants"])
        if sig in seen_signatures:
            continue  # same set of participants as an earlier view — skip dup
        seen_signatures.add(sig)
        views.append(cand)
        if len(views) >= MAX_SEQUENCE_VIEWS:
            break
    return views

# ══════════════════════════════════════════════════════════════════════════
# COMPONENT — deterministic rollup of compute_components() into one bounded,
# layered view. The LLM only names/confirms; it never re-scopes or re-wires.
# ══════════════════════════════════════════════════════════════════════════

MAX_COMPONENTS_PER_VIEW = 12
MAX_COMPONENT_EDGES = 20


def _component_layer(comp: dict) -> str:
    """Deterministic layer from structural signals only (no naming, no repo specifics)."""
    if comp.get("has_routes") or comp.get("has_main_entry"):
        return "entry"
    if comp.get("has_db_models"):
        return "infrastructure"
    return "core"


def _rollup_key(comp_id: str) -> str:
    """Top-level package-prefix key used to merge components when over budget."""
    return comp_id.split("/")[0] if "/" in comp_id else comp_id


def _merge_components(components: list[dict], edges: list[dict], cap: int) -> tuple[list[dict], list[dict]]:
    """Roll components up to their package prefix until within `cap`. Pure
    grouping by the repo's own folder structure — no naming/semantics involved.
    If the rollup alone doesn't fit the budget (e.g. a flat layout), keep the
    highest fan-in+fan-out groups and drop the rest."""
    if len(components) <= cap:
        return components, edges

    groups: dict[str, list[dict]] = {}
    for c in components:
        groups.setdefault(_rollup_key(c["id"]), []).append(c)

    if len(groups) > cap:
        ranked = sorted(groups.items(),
                         key=lambda kv: sum(m["fan_in"] + m["fan_out"] for m in kv[1]),
                         reverse=True)
        keep = {k for k, _ in ranked[:cap]}
        groups = {k: v for k, v in groups.items() if k in keep}

    merged: list[dict] = []
    id_map: dict[str, str] = {}
    for key, members in groups.items():
        for m in members:
            id_map[m["id"]] = key
        merged.append({
            "id": key,
            "files": [f for m in members for f in m.get("files", [])],
            "languages": sorted({l for m in members for l in m.get("languages", [])}),
            "fan_in": sum(m.get("fan_in", 0) for m in members),
            "fan_out": sum(m.get("fan_out", 0) for m in members),
            "has_routes": any(m.get("has_routes") for m in members),
            "has_db_models": any(m.get("has_db_models") for m in members),
            "has_main_entry": any(m.get("has_main_entry") for m in members),
        })

    merged_edges: dict[tuple, int] = {}
    for e in edges:
        a, b = id_map.get(e["from"]), id_map.get(e["to"])
        if a and b and a != b:
            key = (a, b)
            merged_edges[key] = merged_edges.get(key, 0) + e.get("weight", 1)
    new_edges = [{"from": a, "to": b, "weight": w} for (a, b), w in merged_edges.items()]
    return merged, new_edges


def _component_routes(comp: dict, files_by_path: dict) -> list[str]:
    """Real exposed routes for this component — grounds the 'interfaces' field."""
    out = []
    for fp in comp.get("files", []):
        f = files_by_path.get(fp)
        if not f:
            continue
        for r in f.get("routes", []):
            label = f'{r.get("method", "")} {r.get("path", "")}'.strip()
            if label:
                out.append(label)
    return out


def plan_component_view(rich_facts: dict) -> dict:
    """Bounded, layered candidate component-model — deterministic scope/structure;
    the LLM only names components and confirms interfaces. Returns {} if the
    repo has no resolvable components."""
    components = rich_facts.get("components", [])
    edges = rich_facts.get("edges", [])
    if not components:
        return {}

    components, edges = _merge_components(components, edges, MAX_COMPONENTS_PER_VIEW)

    edges = sorted(edges, key=lambda e: e.get("weight", 1), reverse=True)[:MAX_COMPONENT_EDGES]
    ids = {c["id"] for c in components}
    edges = [e for e in edges if e["from"] in ids and e["to"] in ids]

    connected = {e["from"] for e in edges} | {e["to"] for e in edges}
    if len(components) > 1:
        components = [c for c in components if c["id"] in connected]
        ids = {c["id"] for c in components}
        edges = [e for e in edges if e["from"] in ids and e["to"] in ids]

    files_by_path = {f.get("file"): f for f in rich_facts.get("files", [])}
    candidate_components = [
        {
            "id": c["id"],
            "layer": _component_layer(c),
            "languages": c.get("languages", []),
            "files": c.get("files", [])[:8],
            "routes": _component_routes(c, files_by_path)[:8],
            "fan_in": c.get("fan_in", 0),
            "fan_out": c.get("fan_out", 0),
        }
        for c in components
    ]
    candidate_edges = [{"from": e["from"], "to": e["to"]} for e in edges]

    return {
        "diagram_type": "component",
        "components": candidate_components,
        "edges": candidate_edges,
    }
