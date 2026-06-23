"""
C4 view helpers: decision procedure functions for the HLD pipeline.

count_elements  — count containers, externals, actors in a model
detect_systems  — split model into independently-deployable systems
group_containers — split containers into groups <= max_per
"""
from __future__ import annotations

import re
from pathlib import Path


# ── count_elements ────────────────────────────────────────────────────────────

def count_elements(model: dict) -> tuple[int, int, int]:
    """Return (n_containers, n_externals, n_actors) for a model."""
    cont = model.get("containers", {})
    ctx  = model.get("context", {})
    n_containers = len(cont.get("containers", [])) + len(cont.get("databases", []))
    n_externals  = len(cont.get("external_services", [])) + len(ctx.get("external_systems", []))
    n_actors     = len(ctx.get("actors", []))
    return n_containers, n_externals, n_actors


# ── detect_systems ────────────────────────────────────────────────────────────

def detect_systems(model: dict, orchestration: dict | None = None) -> list[dict]:
    """Split a model into independently-deployable sub-models.

    Deterministic, model-only rule: two containers belong to the SAME system if
    they are connected (directly, or transitively via a shared datastore/queue)
    in the relationship graph. Externals and actors are excluded from the
    connectivity graph because they can be legitimately shared across systems.

    Returns >=2 sub-models only when containers form >=2 connected components;
    otherwise [model] (single system).

    `orchestration` is the dict from discover_orchestration() (or None). It is
    accepted for API compatibility / future hints and is NOT iterated as a list.
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    if len(containers) < 2:
        return [model]

    # Connectivity graph over containers + infra (datastores/queues) only.
    infra = cont.get("databases", [])
    node_ids = {c["id"] for c in containers} | {d["id"] for d in infra}
    adjacency: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for r in cont.get("relationships", []):
        f, t = r.get("from", ""), r.get("to", "")
        if f in node_ids and t in node_ids:
            adjacency[f].add(t)
            adjacency[t].add(f)

    # Connected components (iterative DFS)
    visited: set[str] = set()
    components: list[set[str]] = []
    for nid in node_ids:
        if nid in visited:
            continue
        comp: set[str] = set()
        stack = [nid]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            stack.extend(adjacency[n] - visited)
        components.append(comp)

    # Keep only components that contain at least one container
    container_ids = {c["id"] for c in containers}
    sys_components = [comp for comp in components if comp & container_ids]
    if len(sys_components) <= 1:
        return [model]

    # One sub-model per component (reuse _submodel for infra/ext/rel assembly)
    id_to_container = {c["id"]: c for c in containers}
    sub_models = [
        _submodel(model, [id_to_container[cid] for cid in comp if cid in id_to_container])
        for comp in sys_components
    ]
    return sub_models if sub_models else [model]


# ── group_containers ──────────────────────────────────────────────────────────

def group_containers(model: dict, max_per: int = 8) -> list[dict]:
    """Split a model's containers into groups of <=max_per.

    Fallback chain (always terminates):
    1. Folder/ownership prefix groups
    2. Connected components in the relationship graph
    3. Name-chunk (sort + chunk by max_per)
    """
    cont = model.get("containers", {})
    containers = list(cont.get("containers", []))

    if len(containers) <= max_per:
        return [model]

    groups = _group_by_prefix(containers)
    if len(groups) >= 2 and all(len(g) <= max_per for g in groups):
        return [_submodel(model, grp) for grp in groups]

    groups = _group_by_connected_components(containers, cont.get("relationships", []))
    if len(groups) >= 2:
        # Each component may still be oversized; chunk if needed
        chunked = []
        for g in groups:
            chunked.extend(_chunk(g, max_per))
        return [_submodel(model, grp) for grp in chunked]

    # Last resort: sort by name and chunk
    chunks = _chunk(sorted(containers, key=lambda c: c.get("label", c["id"])), max_per)
    return [_submodel(model, grp) for grp in chunks]


def _group_by_prefix(containers: list[dict]) -> list[list[dict]]:
    """Group containers by their top-level folder/ownership prefix."""
    groups: dict[str, list[dict]] = {}
    for c in containers:
        prefix = _prefix(c)
        groups.setdefault(prefix, []).append(c)
    return list(groups.values())


def _prefix(c: dict) -> str:
    """Extract a prefix from container id or manifest_dir evidence."""
    evidence = c.get("evidence", {})
    mdir = evidence.get("manifest_dir", "")
    if mdir:
        parts = Path(mdir).parts
        return parts[0] if parts else "__other__"
    cid = c.get("id", "")
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9]*)", cid)
    return m.group(1).lower() if m else "__other__"


def _group_by_connected_components(containers: list[dict], rels: list[dict]) -> list[list[dict]]:
    """Split containers into connected components via their relationships."""
    cids = {c["id"] for c in containers}
    adjacency: dict[str, set[str]] = {c["id"]: set() for c in containers}
    for r in rels:
        f, t = r.get("from", ""), r.get("to", "")
        if f in cids and t in cids:
            adjacency[f].add(t)
            adjacency[t].add(f)

    visited: set[str] = set()
    components: list[list[str]] = []
    for cid in cids:
        if cid not in visited:
            component: list[str] = []
            stack = [cid]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(adjacency[node] - visited)
            components.append(component)

    id_to_container = {c["id"]: c for c in containers}
    return [[id_to_container[cid] for cid in comp] for comp in components]


def _chunk(lst: list, size: int) -> list[list]:
    """Split a list into chunks of at most size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _submodel(model: dict, sub_containers: list[dict]) -> dict:
    """Build a sub-model that includes only the given containers + their connected infra/externals."""
    cont = model.get("containers", {})
    ctx  = model.get("context", {})

    sub_ids = {c["id"] for c in sub_containers}
    all_dbs  = cont.get("databases", [])
    all_exts = cont.get("external_services", [])
    rels     = cont.get("relationships", [])

    # Include dbs/queues with edges to this group
    db_ids = {d["id"] for d in all_dbs}
    sub_db_ids = {r.get("to") for r in rels if r.get("from") in sub_ids and r.get("to") in db_ids}
    sub_dbs  = [d for d in all_dbs if d["id"] in sub_db_ids]
    sub_exts_ids = {r.get("to") for r in rels if r.get("from") in sub_ids} - sub_ids - db_ids
    sub_exts = [e for e in all_exts if e["id"] in sub_exts_ids]
    all_sub_ids = sub_ids | sub_db_ids | sub_exts_ids | {a["id"] for a in ctx.get("actors", [])}
    sub_rels = [r for r in rels if r.get("from") in all_sub_ids and r.get("to") in all_sub_ids]

    return {
        "context": ctx,
        "containers": {
            "system_label": cont.get("system_label", ctx.get("system_name", "System")),
            "containers": sub_containers,
            "databases": sub_dbs,
            "external_services": sub_exts,
            "relationships": sub_rels,
        },
    }
