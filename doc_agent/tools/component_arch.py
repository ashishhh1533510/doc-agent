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


# Adaptive-depth budget: a module holding more files than this is a giant
# monorepo package (e.g. packages/nocodb) that should fragment into its
# sub-projects rather than collapse into one over-folded box. Such a module is
# re-partitioned one directory segment deeper, repeated until every module fits
# the budget or no further structural split is possible. Convention-based
# readability guard, not repo-specific; a normal repo descends zero times.
_MODULE_FILE_BUDGET = 60
_MAX_MODULES = 12
_MAX_COMPONENTS_PER_MODULE = 6


def _split_deeper(mfiles: list) -> dict:
    """Re-partition a single module one directory segment past ITS own common
    prefix. Same directory-tree idiom as the top-level partition, just scoped to
    this module's members. Returns subkey -> sorted members."""
    cp = len(_common_dir_prefix(sorted(mfiles)))
    sub: dict[str, list] = {}
    for m in sorted(mfiles):
        sub.setdefault(_module_key(m, cp), []).append(m)
    return sub


def _derive_modules(runtime: list) -> dict:
    """module_key -> sorted member ids. Directory-tree only, deterministic.

    Module-count-bounded: after the shallow common-root+1 partition, iteratively
    pick the SINGLE LARGEST oversized module (> _MODULE_FILE_BUDGET files) and
    split it one level deeper — but ONLY if (a) the split produces >1 distinct
    sub-module AND (b) the resulting total module count stays < _MAX_MODULES.
    Stops when no qualifying split exists or the cap is reached.

    Net: a packages/* workspace splits into real sub-projects, but a single huge
    package stays ONE module rather than fragmenting into hundreds. A normal repo
    descends zero times (unchanged behavior)."""
    cp = len(_common_dir_prefix(sorted(runtime)))
    mods: dict[str, list] = {}
    for m in sorted(runtime):
        mods.setdefault(_module_key(m, cp), []).append(m)
    for _ in range(_MAX_MODULES):          # generous cap to prevent any infinite loop
        oversized = sorted(
            [k for k, v in mods.items() if len(v) > _MODULE_FILE_BUDGET],
            key=lambda k: -len(mods[k]),   # largest first
        )
        applied = False
        for k in oversized:
            sub = _split_deeper(mods[k])
            if len(sub) <= 1:
                continue                    # not splittable — try next
            new_total = len(mods) - 1 + len(sub)
            if new_total >= _MAX_MODULES:
                continue                    # would exceed cap — try next
            del mods[k]
            mods.update(sub)
            applied = True
            break                           # one split per pass; re-evaluate
        if not applied:
            break
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


_SURFACE_COVERAGE = 0.80   # cumulative route/db weight threshold for dominant-surface selection


def _qualify_clusters(comms: list, G: "nx.DiGraph", facts: dict) -> set:
    """Indices of clusters that own a responsibility. A cluster is EVIDENCE; only a
    qualified cluster becomes a component. Structural only — no name/keyword matching.

    Tier 1 — dominant capability surface:
      Route clusters are selected by cumulative route-count coverage, NOT by a simple
      has_routes flag. Only clusters whose combined route count covers ≥80% of the
      module's total route count qualify. This ensures feature controllers (Notifications,
      Sorts, Calendars — each with a handful of routes) fold into the dominant API
      surfaces, while genuinely distinct surfaces (Public API + Admin API) both survive.
      Same cumulative logic applies to DB-model clusters.
    Tier 2 — substantial + genuinely central application cluster (unchanged).
    """
    cl_of = {m: i for i, c in enumerate(comms) for m in c}
    importers = [set() for _ in comms]
    exporters = [set() for _ in comms]
    for u, v in G.edges():
        cu, cv = cl_of.get(u), cl_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            importers[cv].add(cu)
            exporters[cu].add(cv)
    masses = sorted(len(c) for c in comms)
    med = masses[len(masses) // 2] if masses else 0

    # ── Tier 1: dominant API / DB surfaces ──────────────────────────────────
    # Weight = pure count of routes / db-model classes in the cluster (not × member_count,
    # to avoid biasing toward large directories with few real routes).
    route_weight = {i: sum(len(facts[m].get("routes", [])) for m in c) for i, c in enumerate(comms)}
    db_weight    = {i: sum(1 for m in c for cc in facts[m].get("classes", [])
                           if cc.get("is_db_model")) for i, c in enumerate(comms)}

    def _dominant(weights: dict, min_total: int) -> set:
        """Cumulative coverage selection: add clusters (heaviest first) until
        their combined weight covers ≥ _SURFACE_COVERAGE of the total.
        Returns empty set when total < min_total (no meaningful surface)."""
        total = sum(weights.values())
        if total < min_total:
            return set()
        out, cum = set(), 0
        for idx, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            out.add(idx)
            cum += w
            if cum / total >= _SURFACE_COVERAGE:
                break
        return out

    dominant_routes = _dominant(route_weight, min_total=3)
    dominant_db     = _dominant(db_weight,    min_total=1)
    import sys
    if dominant_routes or dominant_db:
        rw_sorted = sorted(route_weight.items(), key=lambda kv: -kv[1])
        dw_sorted = sorted(db_weight.items(), key=lambda kv: -kv[1])
        print(f"[component_arch] route weights: {rw_sorted} → {len(dominant_routes)} dominant",
              file=sys.stderr)
        print(f"[component_arch] db weights:    {dw_sorted} → {len(dominant_db)} dominant",
              file=sys.stderr)

    qualified = set()
    for i, c in enumerate(comms):
        if i in dominant_routes or i in dominant_db:   # Tier 1: dominant surface only
            qualified.add(i)
        elif (len(c) >= max(3, med)                    # Tier 2: substantial AND genuinely central
              and len(importers[i]) >= 3
              and len(exporters[i]) >= 1):
            qualified.add(i)
    if not qualified:                                   # pure-library fallback
        ranked = sorted(range(len(comms)),
                        key=lambda i: (-len(comms[i]), -len(importers[i])))
        qualified = set(ranked[:min(4, len(comms))])
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
        # Entity-less module (SDK / utility package): keep only the top-3 by size
        # and fold the rest into orphans so they collapse into the infra sink.
        # Without this, every utility cluster survives as a separate component.
        ranked_all = sorted(range(len(groups)), key=lambda j: -len(groups[j]))
        keep_idx = set(ranked_all[:min(3, len(ranked_all))])
        orphans_el = [set(groups[j]) for j in ranked_all if j not in keep_idx]
        return [set(groups[j]) for j in ranked_all if j in keep_idx], orphans_el
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


def _fallback_label(layer: str, has_routes: bool, has_db: bool,
                    owns: list, module_label: str) -> str:
    """Deterministic, human-readable component name from structural evidence — used
    when the LLM names nothing (free-tier reject / offline). Generic across stacks:
    primary owned entity (or module) + the component's capability surface suffix.
    Guarantees a meaningful label instead of a raw `comp_NN` id."""
    suffix = _capabilities(layer, has_routes, has_db)[0]   # API / Persistence / Services / ...
    if owns:
        head = owns[0] if len(owns) == 1 else f"{owns[0]} & {owns[1]}"
    elif module_label:
        head = module_label
    else:
        head = ""
    return f"{head} {suffix}".strip() if head else suffix


# ── external system discovery (deterministic, reuses container_model scanners) ──
def discover_external_systems(rich_facts: dict) -> list:
    """Datastores / caches / queues / cloud-and-LLM SDKs the repo talks to, detected
    from import tokens + build-manifest deps. Reuses the same scanners HLD/container
    discovery uses, so detection is consistent and repo/stack-agnostic."""
    from doc_agent.tools.container_model import (
        _scan_db_engines, _scan_services,
        _consolidate_datastores, _consolidate_queues,
    )
    imports_flat: list = []
    for f in rich_facts.get("files", []):
        imports_flat.extend(f.get("imports", []) or [])
    mdeps = rich_facts.get("manifest_deps") or []

    db = _consolidate_queues(_consolidate_datastores(_scan_db_engines(imports_flat, mdeps)))
    svc = _scan_services(imports_flat, mdeps)

    out, seen = [], set()
    for lbl, (label, kind) in db.items():
        if label in seen:
            continue
        seen.add(label)
        stereo = "database" if kind in ("datastore", "cache") else "infrastructure"
        out.append({"id": f"ext_{_safe_ext_id(label)}", "label": label,
                    "kind": kind, "stereotype": stereo})
    for lbl, (label, kind, _verb) in svc.items():
        if label in seen:
            continue
        seen.add(label)
        out.append({"id": f"ext_{_safe_ext_id(label)}", "label": label,
                    "kind": kind, "stereotype": "infrastructure"})
    return out[:8]


def _safe_ext_id(label: str) -> str:
    return re.sub(r"\W+", "_", label).strip("_").lower() or "ext"


_GLOBAL_COVERAGE = 0.80   # cumulative size threshold for anchor selection in entity-less repos
_MAX_CONTEXTS    = 8      # global cap on internal (non-infra) components — readability budget


def _consolidate_groups(final_groups: list, group_module: list, group_is_infra: list,
                        G: "nx.DiGraph", ent_refs: dict) -> tuple:
    """Global responsibility consolidation across modules (the reference-diagram shape).

    Entity-owning, non-infra groups are ANCHOR contexts (bounded contexts). Every other
    non-infra group (route controllers, application helpers, utility bundles) folds into the
    anchor it is most import-coupled to — so a feature controller joins the context whose data
    it serves instead of standing alone. Infra groups are shared services/sinks and are left
    intact. Entity-less repos (SDKs) fall back to size-dominant anchors (cumulative 80%).

    Returns (groups, modules, is_infra) with the same parallel-list shape, only fewer entries.
    Only ever REDUCES the group count; deterministic, structural — no names/keywords.
    """
    n = len(final_groups)
    if n <= 1:
        return final_groups, group_module, group_is_infra

    file_grp = {m: i for i, g in enumerate(final_groups) for m in g}
    coup: dict[tuple, int] = {}
    for u, v in G.edges():
        gu, gv = file_grp.get(u), file_grp.get(v)
        if gu is not None and gv is not None and gu != gv:
            coup[(gu, gv)] = coup.get((gu, gv), 0) + 1

    def pair_w(a: int, b: int) -> int:
        return coup.get((a, b), 0) + coup.get((b, a), 0)

    owns = [any(ent_refs.get(m) for m in g) for g in final_groups]
    anchors = [i for i in range(n) if owns[i] and not group_is_infra[i]]
    if not anchors:
        # entity-less: pick size-dominant non-infra groups until 80% of files covered
        non_infra = [i for i in range(n) if not group_is_infra[i]]
        ordered = sorted(non_infra, key=lambda i: (-len(final_groups[i]), i))
        total = sum(len(final_groups[i]) for i in non_infra) or 1
        cum = 0
        for i in ordered:
            anchors.append(i)
            cum += len(final_groups[i])
            if cum / total >= _GLOBAL_COVERAGE:
                break
    anchor_set = set(anchors)

    # fold each non-anchor, non-infra group into its most-coupled anchor (only if coupled)
    parent = list(range(n))
    for i in range(n):
        if i in anchor_set or group_is_infra[i] or not anchors:
            continue
        best = max(anchors, key=lambda a: (pair_w(i, a), len(final_groups[a]), -a))
        if pair_w(i, best) > 0:
            parent[i] = best
        # else: an island with no coupling — keep as its own component

    merged: dict[int, dict] = {}
    order: list[int] = []
    for i in range(n):
        root = parent[i]
        if root not in merged:
            merged[root] = {"files": set(), "module": group_module[root],
                            "infra": group_is_infra[root]}
            order.append(root)
        merged[root]["files"] |= final_groups[i]

    # ── Pillar 1: collapse ALL per-module infra sinks into ONE global infra component ──
    # A module-first pass produces one infra sink per module; on a large monorepo that is
    # the 12-box `<<infrastructure>>` explosion. There is architecturally one shared
    # platform/infrastructure surface, so union every infra root into a single sink.
    ctx_roots   = [r for r in order if not merged[r]["infra"]]
    infra_roots = [r for r in order if merged[r]["infra"]]
    infra_files: set = set()
    for r in infra_roots:
        infra_files |= merged[r]["files"]
    infra_module = group_module[infra_roots[0]] if infra_roots else None

    # ── Pillar 2: cap internal contexts to _MAX_CONTEXTS, folding the tail in ──────────
    # Coupling recomputed at the merged-root level so a folded child counts toward its
    # parent. Rank entity owners first (real bounded contexts), then size, then fan-in;
    # fold the tail into the most-coupled survivor (fallback: largest).
    root_of: dict = {}
    for r in ctx_roots:
        for m in merged[r]["files"]:
            root_of[m] = r
    rcoup: dict[tuple, int] = {}
    for u, v in G.edges():
        ru, rv = root_of.get(u), root_of.get(v)
        if ru is not None and rv is not None and ru != rv:
            rcoup[(ru, rv)] = rcoup.get((ru, rv), 0) + 1

    def _rpair(a: int, b: int) -> int:
        return rcoup.get((a, b), 0) + rcoup.get((b, a), 0)

    def _fanin(r: int) -> int:
        return sum(w for (a, b), w in rcoup.items() if b == r)

    def _owns(r: int) -> bool:
        return any(ent_refs.get(m) for m in merged[r]["files"])

    ranked = sorted(ctx_roots,
                    key=lambda r: (_owns(r), len(merged[r]["files"]), _fanin(r)),
                    reverse=True)
    keep = ranked[:_MAX_CONTEXTS]
    fold = ranked[_MAX_CONTEXTS:]
    acc = {k: set(merged[k]["files"]) for k in keep}
    for r in fold:
        if keep:
            best = max(keep, key=lambda k: (_rpair(r, k), len(merged[k]["files"]), -k))
            acc[best] |= merged[r]["files"]
        else:                       # no surviving context → fold into the infra sink
            infra_files |= merged[r]["files"]

    out_files  = [acc[k] for k in keep]
    out_mod    = [merged[k]["module"] for k in keep]
    out_infra  = [False for _ in keep]
    if infra_files:
        out_files.append(infra_files)
        out_mod.append(infra_module or (group_module[0] if group_module else ""))
        out_infra.append(True)
    return out_files, out_mod, out_infra


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

    entities = _defined_entities(facts)
    ent_refs = {m: _entity_refs(facts[m], entities) for m in runtime}

    # ── module-first discovery ───────────────────────────────────────────────
    # Partition runtime files into physical modules (directory tree only), then
    # cluster + qualify + consolidate WITHIN each module. Consolidation never
    # crosses a module boundary, so a capability cannot absorb files from
    # unrelated packages into one mega-component — the failure mode that collapsed
    # large monorepos into a single box. Clustering is unchanged (just scoped per
    # module); a single-module repo loops once and behaves as global discovery did.
    final_groups: list[set] = []
    group_module: list[str] = []     # parallel: module key per final group
    group_is_infra: list[bool] = []  # parallel: is this the module's infra sink
    for mkey, mfiles in _derive_modules(runtime).items():
        subG = G.subgraph(mfiles)
        comms = _cluster(G, mfiles)
        qualified = _qualify_clusters(comms, subG, facts)
        groups, infra = _consolidate_clusters(comms, qualified, subG)
        cap_groups, orphans = _consolidate_architecture(groups, ent_refs, subG)
        cap_groups = _cap_groups(cap_groups, _MAX_COMPONENTS_PER_MODULE)
        infra_members = set(infra)
        for o in orphans:
            infra_members |= o
        for g in cap_groups:
            final_groups.append(set(g))
            group_module.append(mkey)
            group_is_infra.append(False)
        if infra_members:
            final_groups.append(infra_members)
            group_module.append(mkey)
            group_is_infra.append(True)

    # ── global responsibility consolidation (reference-diagram granularity) ──────
    # Fold cross-module feature controllers / helpers into the bounded context they
    # serve, so the diagram shows ~5-12 subsystems, not per-folder feature clusters.
    final_groups, group_module, group_is_infra = _consolidate_groups(
        final_groups, group_module, group_is_infra, G, ent_refs)

    # ── system boundary: ONE package; the boxes inside are capabilities, not projects ──
    # Prefer the human repo name (e.g. "OpenMetadata"); Path(repo_root).name is the temp
    # clone dir ("doc_agent_clone_xxxx") for URL inputs and must never surface in the diagram.
    sys_label = (rich_facts.get("repo_name") or "").strip() or Path(repo_root).name or "System"

    components: list = []
    comp_of: dict = {}
    for i, g in enumerate(final_groups):
        members = sorted(g)
        mkey = group_module[i]
        is_infra = group_is_infra[i]
        layer = "infrastructure" if is_infra else _dominant_layer([role[m] for m in members])
        has_routes = any(facts[m].get("routes") for m in members)
        has_db = any(cc.get("is_db_model") for m in members for cc in facts[m].get("classes", []))
        owns = sorted({e for m in members for e in ent_refs.get(m, set())})[:8]
        cid = f"comp_{i:02d}"
        for m in g:
            comp_of[m] = cid
        components.append({
            "id": cid,
            "label": _fallback_label(layer, has_routes, has_db, owns, _module_label(mkey)),
            "module": mkey, "module_label": _module_label(mkey),
            "layer": layer, "stereotype": layer, "is_infra": is_infra,
            "member_count": len(members), "members": members[:12],
            "member_files": members,   # FULL list for fidelity scoring (members[] is display-capped)
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
            elif not c.get("owns_entities"):
                # an entity-owning single-class unit is a legitimate bounded context
                # (cf. the reference's per-entity components); only entity-less singletons
                # signal an under-consolidated fragment.
                soft.append(f'{c["id"]}: singleton domain/application component with no capability surface')
    if len(comps) > 20:
        soft.append(f"{len(comps)} components — likely under-consolidated (should scale with architecture, not files)")
    return {"ok": not hard, "warnings": hard + soft}
