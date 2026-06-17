"""
component_arch.py — responsibility-based component discovery for UML component diagrams.

Repository STRUCTURE -> architectural STRUCTURE, deterministically and repo/language/
framework-agnostically:
    runtime modules -> cohesive responsibility clusters (graph communities)
                    -> components carrying layer + stereotype METADATA
                    -> provided interfaces + classified inter-component edges
                    -> validation gate
The LLM downstream only NAMES; it never decides scope, grouping, layering, or edges.
"""
from __future__ import annotations

import re
from collections import Counter

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from doc_agent.tools.architecture_model import (
    runtime_facts, build_internal_graph, classify_roles,
    _is_nonruntime, _LAYER_ORDER, _ROLE_LABEL,
)
from doc_agent.tools.component_clusters import _seed_clusters, _common_dir_prefix


# ── responsibility clustering (deterministic) ────────────────────────────────
def _folder_seed(mids: list) -> list:
    """Convention-based degrade path, shared with HLD seeding."""
    groups: dict[str, set] = {}
    for m, k in _seed_clusters(mids).items():
        groups.setdefault(k, set()).add(m)
    return list(groups.values())


# ── module (project) partitioning (deterministic, directory-tree only) ──────
# A "module" is the repo's physical project/deployable unit (ApplicationCore,
# Web, PublicApi, Infrastructure, ...). This is the TOP-LEVEL package axis for
# the component diagram; responsibility clusters become components INSIDE
# each module. Derived purely from the directory tree (global common-root
# prefix + one more segment) -- no manifest files, no name matching.
def _module_key(mid: str, cp: int) -> str:
    parts = mid.split("/")
    return "/".join(parts[:cp + 1]) if len(parts) > cp + 1 else mid


def _module_label(mkey: str) -> str:
    return mkey.split("/")[-1] or mkey


def _derive_modules(runtime: list) -> dict:
    """module_key -> sorted member ids. Directory-tree only, deterministic."""
    cp = len(_common_dir_prefix(sorted(runtime)))
    mods: dict[str, list] = {}
    for m in sorted(runtime):
        mods.setdefault(_module_key(m, cp), []).append(m)
    return mods


# ── entity ownership (deterministic) ─────────────────────────────────────────
# A component is a set of files that share ownership of one capability. For
# domain/data files, ownership is measured by which repo-defined entity TYPES a
# file defines, persists, or operates on -- type-level references only (calling
# a method is not ownership), so a controller that merely USES an entity isn't
# pulled into that entity's bounded context.
_ID = re.compile(r"[A-Za-z_]\w*")


def _defined_entities(facts: dict) -> set:
    """Every class name declared anywhere in the repo = a candidate entity type."""
    return {c["name"] for f in facts.values() for c in f.get("classes", []) if c.get("name")}


def _entity_refs(fact: dict, entities: set) -> set:
    """Repo-defined entity types this file DEFINES, PERSISTS, or OPERATES ON."""
    refs = set()
    for c in fact.get("classes", []):
        if c.get("name") in entities:
            refs.add(c["name"])
        for fld in c.get("fields", []):
            refs |= set(_ID.findall(fld.get("type") or "")) & entities
        for m in c.get("methods", []):
            refs |= set(_ID.findall(m.get("returns") or "")) & entities
            refs |= set(_ID.findall(m.get("signature") or "")) & entities
    return refs


def _plurality_module(members: list, mod_of: dict) -> str:
    """A component's home package = the project owning the plurality of its files."""
    counts = Counter(mod_of[m] for m in members)
    best = max(counts.values())
    return sorted(k for k, v in counts.items() if v == best)[0]


def _cap_groups(groups: list, k: int) -> list:
    """Readability guard: keep at most k components per module. If more than k
    clusters survived qualification inside one module, fold the smallest tail
    into the largest (convention-based threshold, not repo-specific)."""
    if len(groups) <= k:
        return groups
    ordered = sorted(groups, key=lambda g: -len(g))
    kept = [set(g) for g in ordered[:k]]
    for g in ordered[k:]:
        kept[0] |= set(g)
    return kept


def _absorb_singletons(comms: list, UG: "nx.Graph") -> list:
    """If clustering pulverises into mostly singletons, fold each singleton into the
    community it shares the most edges with (ties -> smallest member id)."""
    singles = [next(iter(c)) for c in comms if len(c) == 1]
    if len(singles) <= 0.6 * max(1, len(comms)):
        return comms
    big = [set(c) for c in comms if len(c) > 1]
    leftovers = []
    for m in singles:
        if big:
            i = min(range(len(big)),
                    key=lambda k: (-sum(1 for n in UG.neighbors(m) if n in big[k]), min(big[k])))
            if any(n in big[i] for n in UG.neighbors(m)):
                big[i].add(m)
                continue
        leftovers.append({m})
    return big + leftovers


def _cluster(G: "nx.DiGraph", runtime: list) -> list:
    """Cohesive dependency groups over the runtime coupling graph.
    greedy_modularity_communities is parameter-free + deterministic (CNM, no RNG).
    Thresholds below are convention-based readability guards, not repo-specific."""
    UG = G.subgraph(runtime).to_undirected()
    if UG.number_of_nodes() <= 3 or UG.number_of_edges() == 0:
        return _folder_seed(runtime)                       # tiny / edgeless
    comms = [set(c) for c in greedy_modularity_communities(UG)]
    if len(comms) == 1 and len(runtime) > 8:
        return _folder_seed(runtime)                       # single giant blob
    comms = _absorb_singletons(comms, UG)                  # pulverised -> merge
    placed = {m for c in comms for m in c}
    leftover = [m for m in runtime if m not in placed]     # safety: bucket any stragglers
    if leftover:
        comms.extend(_folder_seed(leftover))
    return [c for c in comms if c]


# ── architectural component qualification (deterministic, structural) ───────
# A cluster from _cluster() is EVIDENCE, not automatically a component. Only a cluster
# that owns a responsibility (a capability surface, or a substantial+central domain unit)
# qualifies as a standalone component. Unqualified clusters (peripheral mechanisms —
# helpers/utilities/validators/DTOs/etc., regardless of language or naming convention) are
# consolidated rather than rendered, so component count tracks architecture, not file count.
# Deliberately NOT name/keyword matched (no "*Helper"/"*Util" blocklist): that would be
# exactly the repo/language/framework-specific hardcoding this project forbids. Qualification
# is purely structural so it generalizes across stacks.
def _cluster_has_evidence(members: list, facts: dict) -> tuple:
    """Structural capability-surface evidence for a cluster (Tier-1 qualification)."""
    has_routes = any(facts[m].get("routes") for m in members)
    has_db = any(cc.get("is_db_model")
                 for m in members for cc in facts[m].get("classes", []))
    return has_routes, has_db


def _qualify_clusters(comms: list, G: "nx.DiGraph", facts: dict) -> set:
    """Indices of clusters that own a responsibility. A cluster is EVIDENCE; only a
    qualified cluster becomes a component. Structural only — no name/keyword matching."""
    cl_of = {m: i for i, c in enumerate(comms) for m in c}
    importers = [set() for _ in comms]                 # distinct other clusters importing in
    for u, v in G.edges():
        cu, cv = cl_of.get(u), cl_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            importers[cv].add(cu)
    masses = sorted(len(c) for c in comms)
    med = masses[len(masses) // 2] if masses else 0
    qualified = set()
    for i, c in enumerate(comms):
        has_routes, has_db = _cluster_has_evidence(c, facts)
        if has_routes or has_db:                       # Tier 1: capability surface
            qualified.add(i)
        elif len(c) >= max(2, med) and len(importers[i]) >= 2:   # Tier 2: substantial + central
            qualified.add(i)
    if not qualified:                                  # pure-library fallback
        ranked = sorted(range(len(comms)),
                        key=lambda i: (-len(comms[i]), -len(importers[i])))
        qualified = set(ranked[:min(6, len(comms))])
    return qualified


def _owns_entity(members: list, ent_refs: dict) -> bool:
    return any(ent_refs.get(m) for m in members)


def _qualify_contexts(clusters: list, facts: dict, ent_refs: dict) -> set:
    """Tier 1: owns >=1 entity type AND has >=2 files (a real bounded context).
    Tier 2: has a capability surface (routes/db). Else unqualified -> absorbed/pooled.
    Structural only — no name/keyword matching."""
    out = set()
    for i, c in enumerate(clusters):
        has_routes, has_db = _cluster_has_evidence(c, facts)
        if (_owns_entity(c, ent_refs) and len(c) >= 2) or has_routes or has_db:
            out.add(i)
    return out


def _consolidate_clusters(comms: list, qualified: set, G):
    """Absorb each unqualified cluster into the context that owns it (>=60% of its edge
    weight), else pool it into ONE infrastructure sink. Returns (groups, infra_members)
    where groups is a list of qualified member-sets (consolidation applied). Honors an
    edge 'weight' attribute when present (e.g. entity-ownership cohesion); edges without
    one default to weight 1."""
    cl_of = {m: i for i, c in enumerate(comms) for m in c}
    w: dict[tuple, int] = {}
    for u, v, data in G.edges(data=True):
        cu, cv = cl_of.get(u), cl_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            wt = data.get("weight", 1)
            w[(cu, cv)] = w.get((cu, cv), 0) + wt
    groups = {q: set(comms[q]) for q in qualified}
    infra: set = set()
    for i, c in enumerate(comms):
        if i in qualified:
            continue
        ties = {}
        for q in qualified:
            tw = w.get((i, q), 0) + w.get((q, i), 0)
            if tw:
                ties[q] = tw
        total = sum(ties.values())
        if ties and max(ties.values()) >= 0.6 * total:   # clearly one context's detail
            best = max(ties, key=lambda q: (ties[q], -q))
            groups[best] |= comms[i]
        else:
            infra |= c                                    # cross-cutting / orphan -> pool
    return list(groups.values()), infra

def _consolidate_architecture(groups: list, ent_refs: dict, G) -> tuple:
    """Architectural Consolidation (the stage the two-track rewrite bypassed).

    Entity-owning clusters ARE the architectural capabilities. Every cluster that owns
    no domain entity (presentation surface, interface bundle, utility/integration group)
    is EVIDENCE: it folds into the capability it collaborates with most. Structural and
    deterministic -- no names, no frameworks. Only ever REDUCES the component count.

    Returns (capability_member_sets, orphans) where orphans are non-owning clusters with
    no coupling to any capability (the caller pools them into the single infra sink)."""
    owners = [i for i, g in enumerate(groups) if any(ent_refs.get(m) for m in g)]
    if not owners:
        return [set(g) for g in groups], []          # entity-less repo: leave clusters as-is
    comp = {i: set(groups[i]) for i in owners}
    owner_of_file = {m: i for i in owners for m in groups[i]}
    orphans = []
    for j, g in enumerate(groups):
        if j in comp:
            continue                                  # already a capability
        coup: dict = {}                               # import coupling to each capability
        for m in g:
            for nb in set(G.successors(m)) | set(G.predecessors(m)):
                oi = owner_of_file.get(nb)
                if oi is not None:
                    coup[oi] = coup.get(oi, 0) + 1
        if coup:
            best = max(coup, key=lambda i: (coup[i], -i))   # most-coupled capability; deterministic tie-break
            comp[best] |= g
        else:
            orphans.append(g)
    return list(comp.values()), orphans



# ── per-component metadata ───────────────────────────────────────────────────
def _dominant_layer(layers: list) -> str:
    counts: dict[str, int] = {}
    for l in layers:
        counts[l] = counts.get(l, 0) + 1
    return sorted(counts, key=lambda l: (-counts[l],
                  _LAYER_ORDER.index(l) if l in _LAYER_ORDER else 99))[0]


def _stereotype(layer: str, has_db: bool, has_routes: bool) -> str:
    """Metadata only — never a grouping axis. Business components get no stereotype."""
    if layer == "persistence" or has_db:
        return "persistence"
    if layer == "infrastructure":
        return "infrastructure"
    if layer == "presentation" or has_routes:
        return "presentation"
    return ""


# A capability is a SERVICE CONTRACT over groups of operations — never a raw operation.
# The deterministic step decides HOW MANY capabilities a component exposes (one per
# architectural SURFACE it presents); the LLM later NAMES each from operation evidence.
# Count grows with architectural surfaces, never with method/handler/endpoint count.
_CAPABILITY_SUFFIX = {
    "presentation":   "API",
    "persistence":    "Persistence",
    "domain":         "Domain Services",
    "application":    "Services",
    "infrastructure": "Integration",
}


def _capabilities(layer: str, has_routes: bool, has_db: bool) -> list:
    """The capability SURFACES a component exposes — at most 2, structural not volumetric.
    Returns role-based suffixes (e.g. ['API'], ['API','Persistence'], ['Services'])."""
    surfaces = []
    if has_routes:
        surfaces.append("presentation")
    if has_db:
        surfaces.append("persistence")
    if not surfaces:
        surfaces.append(layer if layer in _CAPABILITY_SUFFIX else "application")
    out, seen = [], set()
    for s in surfaces:
        suf = _CAPABILITY_SUFFIX[s]
        if suf not in seen:
            seen.add(suf)
            out.append(suf)
        if len(out) >= 2:
            break
    return out


def _operation_evidence(members: list, facts: dict, cap: int = 12) -> list:
    """Raw operations — EVIDENCE for capability naming only. NEVER rendered as interfaces."""
    ops: list[str] = []
    for m in members:
        f = facts[m]
        for r in f.get("routes", []):
            p = (r.get("path") or r.get("handler") or "").strip()
            if p:
                ops.append(f'{r.get("method", "")} {p}'.strip())
        for c in f.get("classes", []):
            for meth in c.get("methods", []):
                nm = meth.get("name", "")
                if nm and not nm.startswith("_"):
                    ops.append(nm)
        for fn in f.get("functions", []):
            nm = fn.get("name", "")
            if nm and not nm.startswith("_"):
                ops.append(nm)
    seen, out = set(), []
    for o in ops:
        if o not in seen:
            seen.add(o)
            out.append(o)
        if len(out) >= cap:
            break
    return out


# ── dependency classification -> CLOSED vocabulary ───────────────────────────
def _classify_edge(src: dict, dst: dict) -> str:
    """Map import evidence into {requires, implements, communicates_with}. Heuristic;
    'requires' is the safe default. publishes/subscribes need pub-sub evidence the
    extractors don't produce -> never auto-emitted. Never emit a label outside the vocab."""
    sl, dl = src["layer"], dst["layer"]
    if dst.get("has_db") or dl == "persistence":
        return "requires"
    if sl == "infrastructure" and dl == "domain":
        return "implements"
    if dl == "presentation":
        return "communicates_with"
    return "requires"


def discover_components(rich_facts: dict, repo_root: str) -> dict:
    from pathlib import Path
    facts = runtime_facts(rich_facts, repo_root)
    empty = {"diagram_type": "component", "components": [], "edges": [],
             "packages": [], "layers": [], "cycles": []}
    if not facts:
        return empty
    raw = rich_facts.get("import_graph", {})
    G, ext_count = build_internal_graph(facts, raw)
    role = classify_roles(facts, G, ext_count)

    runtime = [m for m in facts if role[m] != "shared"]
    if not runtime:
        return empty

    # ── capability clustering: GLOBAL import-coupling modularity (restored) ──
    comms = _cluster(G, runtime)
    # ── qualification: PRESERVED — drops DTO/controller/utility noise ──
    qualified = _qualify_clusters(comms, G, facts)
    groups, infra = _consolidate_clusters(comms, qualified, G)
    # ── architectural consolidation: fold non-capability clusters into capabilities ──
    entities = _defined_entities(facts)
    ent_refs = {m: _entity_refs(facts[m], entities) for m in runtime}
    cap_groups, orphans = _consolidate_architecture(groups, ent_refs, G)
    infra_members = set(infra)
    for o in orphans:
        infra_members |= o
    final_groups = list(cap_groups)
    infra_idx = -1
    if infra_members:
        final_groups.append(infra_members)
        infra_idx = len(final_groups) - 1

    # ── system boundary: ONE package; the boxes inside are capabilities, not projects ──
    sys_label = Path(repo_root).name or "System"
    components: list = []
    comp_of: dict = {}
    for i, g in enumerate(final_groups):
        members = sorted(g)
        layer = "infrastructure" if i == infra_idx else _dominant_layer([role[m] for m in members])
        has_routes = any(facts[m].get("routes") for m in members)
        has_db = any(cc.get("is_db_model") for m in members for cc in facts[m].get("classes", []))
        owns = sorted({e for m in members for e in ent_refs.get(m, set())})[:8]
        cid = f"comp_{i:02d}"
        for m in g:
            comp_of[m] = cid
        components.append({
            "id": cid, "label": "", "module": "system", "module_label": sys_label,
            "layer": layer, "stereotype": layer, "is_infra": i == infra_idx,
            "member_count": len(members), "members": members[:12],
            "has_routes": has_routes, "has_db": has_db, "owns_entities": owns,
            "capabilities": _capabilities(layer, has_routes, has_db),
            "operation_evidence": _operation_evidence(members, facts),
        })

    weights: dict[tuple, int] = {}
    for u, v in G.edges():
        cu, cv = comp_of.get(u), comp_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            weights[(cu, cv)] = weights.get((cu, cv), 0) + 1
    by_id = {c["id"]: c for c in components}
    edge_items = sorted(weights.items(), key=lambda kv: -kv[1])
    edge_cap = 3 * len(components) if components else 0
    if edge_cap and len(edge_items) > edge_cap:
        edge_items = edge_items[:edge_cap]
    edges = [{"from": cu, "to": cv, "weight": w, "label": _classify_edge(by_id[cu], by_id[cv])}
             for (cu, cv), w in edge_items]

    components.sort(key=lambda c: (c["is_infra"], -c["member_count"], c["id"]))
    present = [r for r in _LAYER_ORDER if any(c["layer"] == r for c in components)]
    packages = [{"id": "system", "label": sys_label, "rank": 0}]

    CG = nx.DiGraph()
    CG.add_nodes_from(c["id"] for c in components)
    CG.add_edges_from((e["from"], e["to"]) for e in edges)
    cycles = [sorted(s) for s in nx.strongly_connected_components(CG) if len(s) > 1]

    return {"diagram_type": "component", "components": components, "edges": edges,
            "packages": packages,
            "layers": [{"id": r, "label": _ROLE_LABEL[r], "rank": i} for i, r in enumerate(present)],
            "cycles": cycles}



# ── validation gate (advisory — never hard-blocks rendering) ─────────────────
def validate_architecture_model(rich_facts_model: dict, rich_facts: dict | None = None) -> dict:
    model = rich_facts_model
    comps, edges = model.get("components", []), model.get("edges", model.get("dependencies", []))
    packages = model.get("packages", [])
    hard, soft = [], []
    # module-first: packages are the project axis, components are the responsibility axis
    # inside each. More packages than components would mean an empty package -- a structural
    # error -- and every component must carry its module.
    if packages and len(packages) > len(comps):
        hard.append(f"more packages ({len(packages)}) than components ({len(comps)})")
    if comps and any(not c.get("module") for c in comps):
        hard.append("a component has no module/package assignment")
    leaked = [m for c in comps for m in c.get("members", []) if _is_nonruntime(m)]
    if leaked:
        hard.append(f"non-runtime modules leaked: {leaked[:5]}")
    # count invariant: responsibilities (= total member files) must vastly outnumber
    # components, or the cohesion/consolidation stage was skipped (component explosion).
    responsibilities = sum(c.get("member_count", len(c.get("members", []))) for c in comps)
    if comps and len(comps) > max(12, 0.5 * responsibilities):
        hard.append(f"fragmentation: {len(comps)} components for {responsibilities} "
                    f"responsibilities (expected responsibilities >> components)")
    # capability-ownership leak: every component must own a domain entity or expose a
    # capability surface, else consolidation failed to absorb a peripheral mechanism
    # (DTO/utility/serializer/config/etc.).
    for c in comps:
        if c.get("is_infra"):
            continue   # the ONE consolidated infra sink may own no entity/surface — by design
        if not (c.get("owns_entities") or c.get("has_routes") or c.get("has_db")):
            hard.append(f'{c["id"]}: owns no entity and exposes no capability surface (leaked utility)')
    if any(c.get("layer") not in _LAYER_ORDER for c in comps):
        hard.append("a component has no inferrable architectural layer")
    for c in comps:
        caps = c.get("capabilities") or c.get("interfaces") or []
        if c.get("has_routes") and not caps:
            soft.append(f'{c["id"]} exposes routes but has no capabilities')
        # interface count must scale with architectural surfaces, never with operation count
        ops = len(c.get("operation_evidence") or [])
        if ops and len(caps) > 2:
            hard.append(f'{c["id"]}: {len(caps)} interfaces is operation-shaped, not capability-shaped')
    if comps and len(edges) > 3 * len(comps):
        soft.append(f"edge-dominated: {len(edges)} edges / {len(comps)} components")
    for c in comps:
        if c.get("is_infra"):
            continue   # the ONE consolidated infra sink is allowed to be a thin singleton
        if c.get("member_count", len(c.get("members", []))) <= 1 and not (c.get("has_routes") or c.get("has_db")):
            # a singleton domain/application unit (e.g. a single core interface or
            # aggregate) is architecturally legitimate; only infra/persistence leaks
            # indicate a qualification-gate miss -- those are the true "leaked utility" case.
            if c.get("layer") in ("infrastructure", "persistence"):
                hard.append(f'{c["id"]}: singleton with no capability surface (qualification leak)')
            else:
                soft.append(f'{c["id"]}: singleton domain/application component with no capability surface')
    if len(comps) > 20:
        soft.append(f"{len(comps)} components — likely under-consolidated (should scale with architecture, not files)")
    return {"ok": not hard, "warnings": hard + soft}
