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
# UI/frontend frameworks: a component running one of these is independently
# runnable (a SPA build) even without HTTP routes or a main entrypoint.
_UI_FRAMEWORK_NAMES = {"react", "nextjs", "vue", "angular", "svelte", "sveltekit", "remix"}


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

def _seed_clusters(mids):
    """Deterministic seed cluster per module id.

    Group modules by top-level directory, then within EACH subtree strip that
    subtree's OWN common directory prefix and seed by the next segment (the
    project/module). Computing the prefix per-subtree rather than once globally
    means a sibling tree (tests/ beside src/, frontend/ beside backend/) can't
    drag the shared prefix to empty and collapse an entire tree into a single
    cluster. Membership matches the old global-prefix seeding on single-root
    repos; it only diverges — correctly — when more than one top-level tree exists.
    """
    groups: dict[str, list] = {}
    for m in mids:
        groups.setdefault(m.split("/")[0], []).append(m)
    cluster_of = {}
    for members in groups.values():
        cp = len(_common_dir_prefix(members))          # this subtree's shared prefix
        for m in members:
            parts = m.split("/")
            # seed = prefix + next segment; a module with nothing below the
            # prefix is its own singleton (Step 2/3 merges consolidate it).
            cluster_of[m] = "/".join(parts[:cp + 1]) if len(parts) > cp + 1 else m
    return cluster_of


def _seed_areas(mids):
    """Directory-only area grouping for breadth analysis (system digest,
    stratified file sampling).

    _seed_clusters seeds at file granularity when a file sits directly in its
    subtree's common directory (no deeper subfolder) — every sibling file in a
    flat folder becomes its own singleton "cluster". That's fine for
    compute_components, which runs import-graph merge passes afterward to fix
    up singletons, but breadth analysis doesn't run those passes, so it would
    see one area per file in any flat folder (e.g. Controllers/*.cs) — the same
    collapse-to-one-subtree failure mode this is meant to prevent, just shaped
    differently. This keys areas purely by directory (always drops the
    filename), so a flat folder of siblings groups into one area."""
    groups: dict[str, list] = {}
    for m in mids:
        groups.setdefault(m.split("/")[0], []).append(m)
    area_of = {}
    for members in groups.values():
        cp = len(_common_dir_prefix(members))
        for m in members:
            dirs = m.split("/")[:-1]
            if len(dirs) > cp:
                area_of[m] = "/".join(dirs[:cp + 1])
            else:
                area_of[m] = "/".join(dirs) if dirs else m
    return area_of


def compute_components(import_graph, files, repo_root) -> dict:
    info = {}
    for f in files:
        if not f.get("error"):
            info[module_id(f["file"], repo_root)] = f
    mids = sorted(info)
    if not mids:
        return {"components": [], "edges": [], "architecture_signals": {"pattern": "unknown", "evidence": []}}

    graph = import_graph
    cluster_of = _seed_clusters(mids)   # per-subtree seeding (robust to sibling trees)


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
            # Only absorb a small cluster into a neighbor it is ACTUALLY connected
            # to. A singleton with no import links to anything is an independent
            # island (e.g. a separate frontend app) and must stay its own component.
            if weight(cl, c, best) == 0:
                continue
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
            "has_ui": bool(set(detect_frameworks([info[m] for m in ms])) & _UI_FRAMEWORK_NAMES),
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
            "has_ui": c.get("has_ui", False),
        })
    return out


def _file_relevance(f) -> int:
    """Rank a file by architectural significance for the prompt budget."""
    score = 0
    if f.get("routes"):
        score += 1000 + 10 * len(f["routes"])
    for c in f.get("classes", []):
        score += 8 + len(c.get("methods", []))
        if c.get("is_db_model"):
            score += 200
    score += len(f.get("functions", []))
    return score


def select_files(files, cap):
    """Pick the `cap` most architecturally-relevant files (routes/DB models first,
    then by structure) to bound the prompt size on large repos. Files with errors
    are dropped. Returns (selected_files, omitted_count)."""
    real = [f for f in files if not f.get("error")]
    if len(real) <= cap:
        return real, 0
    ranked = sorted(real, key=_file_relevance, reverse=True)
    return ranked[:cap], len(real) - cap


def select_files_stratified(files, repo_root, per_area=4, cap=40):
    """Like select_files, but takes the top `per_area` files from EVERY top-level
    architectural area first, then fills any remaining budget by global relevance.

    select_files alone ranks globally by routes/DB-model density, so on a repo
    with many areas the single densest area (e.g. an admin/config module) can
    win every slot and the rest of the system becomes invisible to the LLM. This
    guarantees every area contributes naming evidence before depth is added to
    the most significant ones. Files with errors are dropped. Returns
    (selected_files, omitted_count)."""
    real = [f for f in files if not f.get("error")]
    if len(real) <= cap:
        return real, 0

    items = []
    for f in real:
        try:
            mid = module_id(f["file"], repo_root)
        except ValueError:
            mid = f["file"]
        items.append((f, mid))

    area_of = _seed_areas([mid for _, mid in items])
    groups: dict[str, list] = {}
    for f, mid in items:
        groups.setdefault(area_of[mid], []).append(f)

    selected = []
    selected_ids = set()
    for area in sorted(groups):
        top = sorted(groups[area], key=_file_relevance, reverse=True)[:per_area]
        for f in top:
            selected.append(f)
            selected_ids.add(id(f))

    if len(selected) < cap:
        leftover = [f for f in real if id(f) not in selected_ids]
        leftover_ranked = sorted(leftover, key=_file_relevance, reverse=True)
        selected.extend(leftover_ranked[:cap - len(selected)])
    elif len(selected) > cap:
        selected = sorted(selected, key=_file_relevance, reverse=True)[:cap]

    selected.sort(key=_file_relevance, reverse=True)
    return selected, len(real) - len(selected)


def budget_facts_blob(blob: dict, max_tokens: int = 120_000) -> dict:
    """Hard ceiling on a slimmed facts blob's prompt size, on top of the file cap.

    select_files()/slim_components() already cap file *count*, but import_graph,
    component_edges, component count, and per-class fields/methods/calls are left
    unbounded — on a large repo these alone can push a single prompt over the
    Gemini per-minute token quota, which no retry can ever recover from. This
    trims those dimensions, in priority order, re-measuring after each step with
    the same `len(json)//4` heuristic used elsewhere. No-op when already small
    (small/medium repos are returned untouched)."""
    import copy
    from doc_agent.core.llm import compact_json

    def size(b) -> int:
        return len(compact_json(b)) // 4

    if size(blob) <= max_tokens:
        return blob

    b = copy.deepcopy(blob)

    # Stage 1 — import_graph: keep the highest-degree (most architecturally
    # connected) nodes first.
    def trim_import_graph(n):
        ig = b.get("import_graph")
        if not ig:
            return
        ranked = sorted(
            ig.items(),
            key=lambda kv: len(kv[1]) if isinstance(kv[1], list) else 0,
            reverse=True,
        )
        b["import_graph"] = {k: v for k, v in ranked[:n]}

    for n in (60, 30, 10, 0):
        trim_import_graph(n)
        if size(b) <= max_tokens:
            return b

    # Stage 2 — component_edges: keep the heaviest edges first.
    def trim_edges(n):
        edges = b.get("component_edges")
        if not edges:
            return
        ranked = sorted(edges, key=lambda e: e.get("weight", 0), reverse=True)
        b["component_edges"] = ranked[:n]

    for n in (40, 20, 5, 0):
        trim_edges(n)
        if size(b) <= max_tokens:
            return b

    # Stage 3 — components: keep the most connected components first.
    def trim_components(n):
        comps = b.get("components")
        if not comps:
            return
        ranked = sorted(
            comps, key=lambda c: c.get("fan_in", 0) + c.get("fan_out", 0), reverse=True
        )
        b["components"] = ranked[:n]

    for n in (25, 12, 5):
        trim_components(n)
        if size(b) <= max_tokens:
            return b

    # Stage 4 — per-class detail: drop call traces, then cap methods/fields.
    def trim_classes(max_methods, max_fields, keep_calls):
        for f in b.get("files", []):
            for c in f.get("classes", []):
                methods = c.get("methods")
                if isinstance(methods, list):
                    if not keep_calls:
                        for m in methods:
                            if isinstance(m, dict):
                                m.pop("calls", None)
                    c["methods"] = methods[:max_methods]
                fields = c.get("fields")
                if isinstance(fields, list):
                    c["fields"] = fields[:max_fields]

    for max_methods, max_fields, keep_calls in (
        (5, 5, True), (5, 5, False), (3, 3, False), (1, 1, False), (0, 0, False),
    ):
        trim_classes(max_methods, max_fields, keep_calls)
        if size(b) <= max_tokens:
            return b

    return b  # best effort — already far smaller than the input

def partition_for_class_diagram(
    files: list,
    components: list,
    repo_root: str,
    min_classes: int = 3,
    max_classes_per_diagram: int = 15,
) -> list:
    """
    Group files into cohesive partitions for per-partition class diagram generation.
    Driven entirely by the repo's own folder structure and component membership.
    Returns list of {"label": str, "files": [...], "class_count": int},
    ordered by descending class count (most significant first).
    """
    import os
    from collections import defaultdict

    def _prefix(file_path, depth=2):
        try:
            rel = os.path.relpath(file_path, repo_root).replace("\\", "/")
        except ValueError:
            rel = file_path.replace("\\", "/")
        parts = [p for p in rel.split("/") if p and p != "."]
        return "/".join(parts[:-1][:depth]) or "root"

    def _class_count(fs):
        return sum(len(f.get("classes", [])) for f in fs)

    real_files = [f for f in files if not f.get("error") and f.get("classes")]

    # Step 1 — group by top-2 folder prefix
    groups: dict[str, list] = defaultdict(list)
    for f in real_files:
        groups[_prefix(f["file"])].append(f)

    # component membership: file_path -> component_id
    file_to_comp = {fp: c["id"] for c in components for fp in c.get("files", [])}

    # Step 2 — merge tiny groups into nearest neighbour by shared component
    changed = True
    while changed:
        changed = False
        for k in list(groups):
            if k not in groups or _class_count(groups[k]) >= min_classes:
                continue
            others = [o for o in groups if o != k]
            if not others:
                break
            best = max(
                others,
                key=lambda o: (
                    sum(1 for f in groups[k]
                        if file_to_comp.get(f["file"]) in
                        {file_to_comp.get(g["file"]) for g in groups[o]}),
                    _class_count(groups[o])
                )
            )
            groups[best].extend(groups.pop(k))
            changed = True
            break

    # Step 3 — split oversized groups one level deeper
    final: dict[str, list] = {}
    for key, gfiles in groups.items():
        if _class_count(gfiles) <= max_classes_per_diagram:
            final[key] = gfiles
            continue
        sub: dict[str, list] = defaultdict(list)
        for f in gfiles:
            sub[_prefix(f["file"], depth=3)].append(f)
        for sk, sf in sub.items():
            if _class_count(sf) >= min_classes:
                final[sk] = sf
            else:
                largest = max(sub, key=lambda k: _class_count(sub[k]))
                final.setdefault(largest, []).extend(sf)

    # Step 4 — build result, drop empties, sort descending
    result = []
    for key, pfiles in final.items():
        count = _class_count(pfiles)
        if count >= min_classes:
            result.append({
                "label": key.replace("/", " / "),
                "files": pfiles,
                "class_count": count,
            })
    result.sort(key=lambda p: p["class_count"], reverse=True)
    return result
