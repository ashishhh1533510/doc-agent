"""
Deterministic component clustering — the anti-generic foundation.

Turns the resolved import graph into real, file-grounded components (clusters of
files that actually depend on each other) plus an evidence-based architecture
pattern. The HLD/LLD agents are then required to NAME and MERGE these
components rather than invent generic shape, so two different repos can no
longer produce the same diagram.

compute_components(import_graph, files, repo_root) returns:
    {
      "components": [
        {id, files, module_ids, languages, fan_in, fan_out,
         has_routes, has_db_models, has_main_entry}, ...
      ],
      "edges": [{from, to, weight}, ...],          # weighted inter-component deps
      "architecture_signals": {"pattern": "...", "evidence": [...]},
    }

Clustering: seed by top-level source dir (after stripping the longest common
path prefix, so e.g. a Maven src/main/java/<pkg> root collapses to the feature
package); merge singletons into their best-connected neighbour; merge small,
highly-mutual pairs. All deterministic — no LLM.
"""

from doc_agent.tools.import_graph import module_id, detect_frameworks

_ENTRY_NAMES = {"main", "program", "index", "__main__", "app", "manage", "cli", "server"}


def _common_dir_prefix(mids):
    dir_lists = [m.split("/")[:-1] for m in mids]
    if not dir_lists:
        return []
    common = dir_lists[0]
    for d in dir_lists[1:]:
        i = 0
        while i < len(common) and i < len(d) and common[i] == d[i]:
            i += 1
        common = common[:i]
        if not common:
            break
    return common


def _is_entry(mid, fact) -> bool:
    base = mid.split("/")[-1].lower()
    if base in _ENTRY_NAMES:
        return True
    return any((fn.get("name") or "").lower() == "main" for fn in fact.get("functions", []))


def compute_components(import_graph, files, repo_root) -> dict:
    info = {}
    for f in files:
        if not f.get("error"):
            info[module_id(f["file"], repo_root)] = f
    mids = sorted(info)
    if not mids:
        return {"components": [], "edges": [], "architecture_signals": {"pattern": "unknown", "evidence": []}}

    cp = len(_common_dir_prefix(mids))
    graph = import_graph

    def seed_of(m):
        rem = m.split("/")[cp:]
        return rem[0] if len(rem) >= 2 else m   # file directly under root -> singleton seed

    cluster_of = {m: seed_of(m) for m in mids}

    def clusters():
        cl = {}
        for m, c in cluster_of.items():
            cl.setdefault(c, set()).add(m)
        return cl

    def weight(cl, a, b):
        A, B = cl[a], cl[b]
        w = 0
        for s in A:
            w += sum(1 for d in graph.get(s, []) if d in B)
        for s in B:
            w += sum(1 for d in graph.get(s, []) if d in A)
        return w

    # Step 2 — merge clusters with <2 files into their best-connected neighbor.
    changed = True
    while changed:
        changed = False
        cl = clusters()
        for c, ms in cl.items():
            if len(ms) >= 2:
                continue
            others = [o for o in cl if o != c]
            if not others:
                continue
            best = sorted(others, key=lambda o: (-weight(cl, c, o), o))[0]
            for m in list(cl[c]):
                cluster_of[m] = best
            changed = True
            break

    # Step 3 — merge small, highly-mutual pairs (>=80% mutual, combined <6).
    changed = True
    while changed:
        changed = False
        cl = clusters()
        names = sorted(cl)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if len(cl[a]) + len(cl[b]) >= 6:
                    continue
                w = weight(cl, a, b)
                out_a = sum(1 for s in cl[a] for d in graph.get(s, []) if d not in cl[a])
                out_b = sum(1 for s in cl[b] for d in graph.get(s, []) if d not in cl[b])
                denom = out_a + out_b
                if denom and w / denom >= 0.8:
                    for m in list(cl[b]):
                        cluster_of[m] = a
                    changed = True
                    break
            if changed:
                break

    cl = clusters()
    edge_w = {}
    for s, deps in graph.items():
        cs = cluster_of.get(s)
        for d in deps:
            cd = cluster_of.get(d)
            if cs and cd and cs != cd:
                edge_w[(cs, cd)] = edge_w.get((cs, cd), 0) + 1

    components = []
    for cid in sorted(cl):
        ms = sorted(cl[cid])
        components.append({
            "id": cid,
            "files": [info[m]["file"] for m in ms],
            "module_ids": ms,
            "languages": sorted({info[m].get("language") for m in ms}),
            "fan_in": sum(w for (a, b), w in edge_w.items() if b == cid),
            "fan_out": sum(w for (a, b), w in edge_w.items() if a == cid),
            "has_routes": any(info[m].get("routes") for m in ms),
            "has_db_models": any(c.get("is_db_model") for m in ms for c in info[m].get("classes", [])),
            "has_main_entry": any(_is_entry(m, info[m]) for m in ms),
        })
    edges = [{"from": a, "to": b, "weight": w} for (a, b), w in sorted(edge_w.items())]

    signals = _classify(components, edges, info, detect_frameworks(files))
    return {"components": components, "edges": edges, "architecture_signals": signals}


def _classify(components, edges, info, frameworks):
    langs = {l for c in components for l in c["languages"]}
    route_comps = [c["id"] for c in components if c["has_routes"]]
    db_comps = [c["id"] for c in components if c["has_db_models"]]
    entry_comps = [c["id"] for c in components if c["has_main_entry"]]
    web_langs = {"typescript", "javascript"}
    has_frontend_fw = any(fw in frameworks for fw in ("react", "nextjs"))
    # a real frontend component: predominantly web-lang, no persistence, and the
    # repo actually uses a frontend framework (not just incidental static .js).
    frontend = [c["id"] for c in components
                if set(c["languages"]) <= web_langs and not c["has_db_models"]
                and not c["has_routes"]] if has_frontend_fw else []
    backend = [c["id"] for c in components if c["has_routes"] or c["has_db_models"]]

    # fullstack monorepo: a frontend framework + a real frontend component
    # separate from a route/persistence backend
    if has_frontend_fw and frontend and backend and not (set(frontend) & set(backend)):
        return {"pattern": "fullstack_monorepo",
                "evidence": [f"frontend: {frontend}", f"backend: {backend}",
                             f"frontend frameworks: {[f for f in frameworks if f in ('react', 'nextjs')]}"]}
    # layered api: routes + persistence across components
    if route_comps and db_comps:
        return {"pattern": "layered_api",
                "evidence": [f"route components: {route_comps}", f"db-model components: {db_comps}"]}
    # cli: an entry point, no web surface
    if entry_comps and not route_comps:
        return {"pattern": "cli",
                "evidence": [f"entry components: {entry_comps}", "no route handlers found"]}
    # pipeline: a mostly-linear dependency chain, no routes
    if not route_comps and edges and len(edges) <= len(components):
        return {"pattern": "pipeline",
                "evidence": [f"{len(components)} components in a {len(edges)}-edge chain", "no route handlers"]}
    # library: imported code, no entry/routes
    if not route_comps and not entry_comps:
        return {"pattern": "library", "evidence": ["no routes", "no main entry point"]}
    if route_comps:
        return {"pattern": "layered_api", "evidence": [f"route components: {route_comps}"]}
    return {"pattern": "library", "evidence": ["no distinctive signals"]}

def slim_components(components, file_cap=12):
    """Cap files per component (keep all metrics) so large repos don't blow the
    prompt. file_count preserves each component's true size for the agent."""
    out = []
    for c in components:
        out.append({
            "id": c["id"],
            "languages": c.get("languages", []),
            "file_count": len(c.get("files", [])),
            "files": c.get("files", [])[:file_cap],
            "fan_in": c.get("fan_in", 0),
            "fan_out": c.get("fan_out", 0),
            "has_routes": c.get("has_routes", False),
            "has_db_models": c.get("has_db_models", False),
            "has_main_entry": c.get("has_main_entry", False),
        })
    return out
