"""
view_planner.py — deterministic view-planning layer for component diagrams.

Sits between ComponentDiagramAgent.refine() and the renderer. Takes the named
model and produces a ViewSet: an L1 overview (one aggregate node per layer/module
group) plus L2 drill-down diagrams for groups that exceed the readability budget.

No LLM calls. No name/keyword matching. Pure structural signals.
"""
from __future__ import annotations

from doc_agent.tools.architecture_model import _LAYER_ORDER, _ROLE_LABEL

# ── readability budget constants (single source of truth) ───────────────────
MAX_NODES_PER_VIEW  = 10   # aggregate or real nodes rendered per diagram
MAX_EDGES_PER_VIEW  = 16   # dependency arrows per diagram
DRILL_MIN_COMPONENTS = 2   # a group needs >=N components to warrant its own L2


# ── scoring functions ────────────────────────────────────────────────────────

def score_component(c: dict, fan_in: int) -> int:
    """Importance score for one component. Higher = keep when space is tight."""
    s  = 1000 if c.get("has_routes") else 0
    s +=  500 if c.get("has_db")     else 0
    s +=   50 * len(c.get("owns_entities") or [])
    s +=    5 * fan_in
    s +=        c.get("member_count", len(c.get("members") or []))
    if c.get("is_infra"):          # consolidated orphan sink — demote
        s = s // 10
    return s


def score_edge(e: dict, comp_by_id: dict) -> int:
    """Importance score for one dependency edge. Higher = keep when space is tight."""
    label = e.get("label", "requires")
    base  = {"communicates_with": 30, "implements": 20, "requires": 10}.get(label, 10)
    dst   = comp_by_id.get(e.get("to", ""), {})
    if dst.get("has_db"):          # X -> persistence is always meaningful
        base += 40
    base += e.get("weight", 1)
    return base


# ── main entry point ─────────────────────────────────────────────────────────

def plan_views(model: dict) -> dict:
    """
    Convert a named component model into a ViewSet.

    Returns:
        {
          "views": [
            { "level": "L1"|"L2", "title": str, "group_key": str,
              "nodes": [...], "edges": [...],
              "omitted": {"nodes": int, "edges": int} }
          ]
        }
    """
    components   = [c for c in model.get("components", []) if (c.get("id") or "").strip()]
    dependencies = list(model.get("dependencies", model.get("edges", [])))

    if not components:
        return {"views": []}

    comp_by_id: dict[str, dict] = {c["id"]: c for c in components}

    # fan_in count per component
    fan_in: dict[str, int] = {c["id"]: 0 for c in components}
    for e in dependencies:
        tid = e.get("to", "")
        if tid in fan_in:
            fan_in[tid] += 1

    # ── determine grouping axis ──────────────────────────────────────────────
    modules        = {c.get("module") or "" for c in components}
    use_module_axis = len(modules) > 1

    def _gkey(c: dict) -> str:
        return (c.get("module") or "") if use_module_axis else (c.get("layer") or "application")

    # ── build ordered groups ─────────────────────────────────────────────────
    groups: dict[str, list[dict]] = {}
    if use_module_axis:
        for c in components:
            groups.setdefault(_gkey(c), []).append(c)
    else:
        for layer in _LAYER_ORDER:
            bucket = [c for c in components if (c.get("layer") or "application") == layer]
            if bucket:
                groups[layer] = bucket
        leftover = [c for c in components if _gkey(c) not in groups]
        if leftover:
            groups.setdefault("application", []).extend(leftover)

    group_keys = list(groups.keys())

    # ── pre-score edges (descending) ─────────────────────────────────────────
    scored_deps = sorted(
        ((score_edge(e, comp_by_id), e) for e in dependencies),
        reverse=True,
        key=lambda t: t[0],
    )

    # ── L1 aggregate nodes ───────────────────────────────────────────────────
    l1_nodes: list[dict] = []
    for k in group_keys:
        grp  = groups[k]
        total_fi = sum(fan_in.get(c["id"], 0) for c in grp)
        l1_nodes.append({
            "id":            f"grp_{k}",
            "label":         f"{_group_label(k, grp, use_module_axis)} ({len(grp)})",
            "is_aggregate":  True,
            "group_key":     k,
            "component_ids": [c["id"] for c in grp],
            "has_routes":    any(c.get("has_routes") for c in grp),
            "has_db":        any(c.get("has_db")     for c in grp),
            "score":         sum(score_component(c, fan_in.get(c["id"], 0)) for c in grp),
        })

    # fold if L1 itself exceeds budget
    l1_omitted_nodes = 0
    if len(l1_nodes) > MAX_NODES_PER_VIEW:
        ranked_l1 = sorted(l1_nodes, key=lambda n: n["score"], reverse=True)
        keep, fold = ranked_l1[:MAX_NODES_PER_VIEW - 1], ranked_l1[MAX_NODES_PER_VIEW - 1:]
        folded_keys  = {n["group_key"] for n in fold}
        folded_cids  = [cid for n in fold for cid in n["component_ids"]]
        keep.append({
            "id":            "grp___overflow",
            "label":         f"+{len(fold)} more groups",
            "is_aggregate":  True,
            "is_overflow":   True,
            "group_key":     "__overflow",
            "component_ids": folded_cids,
            "has_routes":    any(n.get("has_routes") for n in fold),
            "has_db":        any(n.get("has_db")     for n in fold),
            "score":         0,
        })
        l1_omitted_nodes = len(fold)
        l1_nodes  = keep
        group_keys = [k for k in group_keys if k not in folded_keys] + ["__overflow"]

    # comp_id -> L1 node id map
    comp_to_grp_node: dict[str, str] = {}
    for n in l1_nodes:
        for cid in n.get("component_ids", []):
            comp_to_grp_node[cid] = n["id"]

    # ── L1 inter-group edges ─────────────────────────────────────────────────
    l1_edge_acc: dict[tuple, dict] = {}
    for _, e in scored_deps:
        fg = comp_to_grp_node.get(e.get("from", ""))
        tg = comp_to_grp_node.get(e.get("to",   ""))
        if not fg or not tg or fg == tg:
            continue
        key = (fg, tg)
        if key not in l1_edge_acc:
            l1_edge_acc[key] = {"from": fg, "to": tg, "label": e.get("label", "requires"), "weight": 0}
        l1_edge_acc[key]["weight"] += e.get("weight", 1)

    l1_edges = sorted(l1_edge_acc.values(), key=lambda e: e["weight"], reverse=True)
    l1_omitted_edges = 0
    if len(l1_edges) > MAX_EDGES_PER_VIEW:
        l1_omitted_edges = len(l1_edges) - MAX_EDGES_PER_VIEW
        l1_edges = l1_edges[:MAX_EDGES_PER_VIEW]

    views: list[dict] = [{
        "level":   "L1",
        "title":   "System Overview",
        "nodes":   l1_nodes,
        "edges":   l1_edges,
        "omitted": {"nodes": l1_omitted_nodes, "edges": l1_omitted_edges},
    }]

    # ── L2 drill-down views ───────────────────────────────────────────────────
    if len(components) > MAX_NODES_PER_VIEW:
        for gk in [k for k in group_keys if k != "__overflow"]:
            grp_comps = groups.get(gk, [])
            if len(grp_comps) < DRILL_MIN_COMPONENTS:
                continue

            ranked = sorted(grp_comps, key=lambda c: score_component(c, fan_in.get(c["id"], 0)), reverse=True)

            l2_omitted_nodes = 0
            overflow_node    = None
            if len(ranked) > MAX_NODES_PER_VIEW:
                keep_c, fold_c    = ranked[:MAX_NODES_PER_VIEW - 1], ranked[MAX_NODES_PER_VIEW - 1:]
                l2_omitted_nodes  = len(fold_c)
                overflow_node = {
                    "id":            f"grp_{gk}__overflow",
                    "label":         f"+{len(fold_c)} more",
                    "is_aggregate":  True,
                    "is_overflow":   True,
                    "group_key":     f"{gk}__overflow",
                    "component_ids": [c["id"] for c in fold_c],
                    "has_routes":    any(c.get("has_routes") for c in fold_c),
                    "has_db":        any(c.get("has_db")     for c in fold_c),
                }
                ranked = keep_c

            grp_ids  = {c["id"] for c in ranked}
            l2_nodes = [dict(c) for c in ranked]
            if overflow_node:
                grp_ids.update(overflow_node["component_ids"])
                l2_nodes.append(overflow_node)

            # cross-group neighbor groups as ghost nodes
            ghost_grps: dict[str, dict] = {}
            for e in dependencies:
                fid, tid = e.get("from", ""), e.get("to", "")
                fc = comp_by_id.get(fid, {})
                tc = comp_by_id.get(tid, {})
                fc_gk = _gkey(fc) if fc else None
                tc_gk = _gkey(tc) if tc else None
                if fc_gk == gk and tc_gk and tc_gk != gk and tc_gk not in ghost_grps:
                    ghost_grps[tc_gk] = {
                        "id":        f"ghost_{tc_gk}",
                        "label":     _group_label(tc_gk, groups.get(tc_gk, []), use_module_axis),
                        "is_ghost":  True,
                        "group_key": tc_gk,
                    }
                elif tc_gk == gk and fc_gk and fc_gk != gk and fc_gk not in ghost_grps:
                    ghost_grps[fc_gk] = {
                        "id":        f"ghost_{fc_gk}",
                        "label":     _group_label(fc_gk, groups.get(fc_gk, []), use_module_axis),
                        "is_ghost":  True,
                        "group_key": fc_gk,
                    }

            l2_nodes.extend(ghost_grps.values())

            # build comp_id -> rendered node id map
            vis_id_map: dict[str, str] = {c["id"]: c["id"] for c in ranked}
            if overflow_node:
                for cid in overflow_node["component_ids"]:
                    vis_id_map[cid] = overflow_node["id"]
            for gk2, ghost in ghost_grps.items():
                for c2 in groups.get(gk2, []):
                    vis_id_map[c2["id"]] = ghost["id"]

            # L2 edges
            seen_l2: set[tuple] = set()
            l2_edges: list[dict] = []
            for _, e in scored_deps:
                fid, tid = e.get("from", ""), e.get("to", "")
                fvid = vis_id_map.get(fid)
                tvid = vis_id_map.get(tid)
                if not fvid or not tvid or fvid == tvid:
                    continue
                # at least one end must be a real (non-overflow) group component
                if fid not in grp_ids and tid not in grp_ids:
                    continue
                pair = (fvid, tvid)
                if pair in seen_l2:
                    continue
                seen_l2.add(pair)
                l2_edges.append({"from": fvid, "to": tvid, "label": e.get("label", "requires"), "weight": e.get("weight", 1)})

            l2_omitted_edges = 0
            if len(l2_edges) > MAX_EDGES_PER_VIEW:
                l2_omitted_edges = len(l2_edges) - MAX_EDGES_PER_VIEW
                l2_edges = l2_edges[:MAX_EDGES_PER_VIEW]

            views.append({
                "level":     "L2",
                "title":     f"{_group_label(gk, grp_comps, use_module_axis)} — Detail",
                "group_key": gk,
                "nodes":     l2_nodes,
                "edges":     l2_edges,
                "omitted":   {"nodes": l2_omitted_nodes, "edges": l2_omitted_edges},
            })

    return {"views": views}


# ── helpers ──────────────────────────────────────────────────────────────────

def _group_label(key: str, comps: list[dict], use_module_axis: bool) -> str:
    if use_module_axis:
        if comps:
            return comps[0].get("module_label") or key.split("/")[-1] or key
        return key.split("/")[-1] or key
    return _ROLE_LABEL.get(key, key.replace("_", " ").title())
