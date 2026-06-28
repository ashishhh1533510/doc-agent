"""
Output tool: render generated documentation into different formats and save it.

md / html   -> prose (html is converted from markdown)
json / yaml -> the extracted facts as a structured spec
mermaid     -> C4 combined HLD + LLD diagram text (saved as-is)

All text formats are written as plain text. Deterministic -- no LLM here.
"""
import re
import json
import zlib
from pathlib import Path

import markdown as _markdown
import yaml

from doc_agent.tools.architecture_model import _LAYER_ORDER

# Generic, repo-agnostic catalog of frameworks / libraries / SDKs. These are
# implementation technologies, never C4 architecture nodes — when one is discovered
# it belongs as `tech` metadata on its owning container, not as a box. The catalog
# is applied uniformly to every repository; it contains NO repo-specific names. It
# is matched alongside the repo's OWN detected frameworks (see strip_technology_nodes).
_FRAMEWORK_LABELS = {
    # web / app frameworks + UI
    "aspnet", "aspnetcore", "react", "angular", "vue", "vuejs", "next", "nextjs",
    "express", "nestjs", "fastify", "spring", "springboot", "fastapi", "flask",
    "django", "starlette", "blazor", "blazored", "razor", "wpf", "winforms",
    # ORMs / data-access libraries
    "entityframework", "entityframeworkcore", "efcore", "hibernate", "dapper",
    "sequelize", "prisma", "mongoose", "sqlalchemy", "tortoise", "peewee", "jpa",
    # cross-cutting libraries (mediation, mapping, validation, DI, logging, serialization, resilience, docs)
    "mediatr", "automapper", "fluentvalidation", "autofac", "ninject", "masstransit",
    "serilog", "nlog", "log4net", "log4j", "slf4j", "newtonsoft", "newtonsoftjson",
    "jackson", "lombok", "polly", "swashbuckle", "swagger", "redux", "axios",
}

def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def resolve_floating_externals(model: dict, frameworks: list | None = None) -> dict:
    """Remove framework orphans; wire genuine external orphans to a plausible container."""
    ctx  = model.get("context", {}) or {}
    cont = model.get("containers", {}) or {}

    all_rels = (ctx.get("relationships") or []) + (cont.get("relationships") or [])
    rel_ids = {_slug(r["from"]) for r in all_rels} | {_slug(r["to"]) for r in all_rels}

    deny = set(_FRAMEWORK_LABELS)
    for fw in (frameworks or []):
        deny.add(_norm_label(fw))

    containers = cont.get("containers") or []

    def _pick_container(ext_label: str) -> dict | None:
        if not containers:
            return None
        norm = _norm_label(ext_label)
        persist_kw = {"persistence", "infrastructure", "data", "repository", "ef",
                      "entity", "orm", "sql", "db", "storage", "database"}
        api_kw     = {"api", "web", "endpoint", "controller", "gateway", "http", "rest"}
        data_kw    = {"db", "database", "sql", "postgres", "mysql", "mongo", "redis",
                      "cosmos", "storage", "blob", "cache", "queue", "kafka", "rabbit"}
        is_datastore = any(k in norm for k in data_kw)
        search_kw = persist_kw if is_datastore else api_kw
        for c in containers:
            combined = _norm_label(c.get("label", "")) + _norm_label(c.get("tech", ""))
            if any(k in combined for k in search_kw):
                return c
        return containers[-1] if is_datastore else containers[0]

    def _process_ext_list(ext_list: list) -> list:
        kept = []
        for ext in ext_list:
            eid = _slug(ext.get("id", ""))
            if eid in rel_ids:
                kept.append(ext)
                continue
            norm = _norm_label(ext.get("label") or ext.get("id", ""))
            if norm in deny:
                continue  # pure framework — drop it
            # genuine orphan — wire it
            target = _pick_container(ext.get("label") or ext.get("id", ""))
            if target:
                data_kw = {"db", "database", "sql", "postgres", "mysql", "mongo", "redis",
                           "cosmos", "storage", "blob", "cache", "queue", "kafka", "rabbit"}
                edge_label = "reads/writes" if any(k in norm for k in data_kw) else "calls"
                cont.setdefault("relationships", []).append(
                    {"from": target["id"], "to": ext["id"], "label": edge_label}
                )
                rel_ids.add(eid)
            kept.append(ext)
        return kept

    ctx["external_systems"]   = _process_ext_list(ctx.get("external_systems") or [])
    cont["external_services"] = _process_ext_list(cont.get("external_services") or [])
    model["context"]    = ctx
    model["containers"] = cont
    return model


def _is_technology(label: str, deny: set) -> bool:
    """Deterministic classifier: is this entity a framework/library, not architecture?

    True when the whole normalized label, or any whitespace-delimited token of it,
    exactly matches a known technology in `deny` (the generic catalog ∪ the repo's
    detected frameworks). Exact (not substring) matching keeps genuine external
    systems whose names merely contain a tech token (e.g. a 'Razorpay' gateway is
    NOT matched by 'razor'). Repo-agnostic — no repository names are consulted.
    """
    whole = _norm_label(label)
    if not whole:
        return False
    if whole in deny:
        return True
    return any(_norm_label(tok) in deny for tok in re.split(r"[^A-Za-z0-9]+", label) if tok)


def strip_technology_nodes(model: dict, frameworks: list | None = None) -> dict:
    """Remove framework/library entities from the HLD model (deterministic, run pre-render).

    A C4 Context/Container diagram models people, software systems, containers,
    databases, and external systems — never frameworks, libraries, SDKs, or
    packages. Those are technology *metadata*. This pass classifies every external
    entity (context.external_systems + containers.external_services) against the
    generic technology catalog plus the repo's own detected frameworks, and for
    each match:
      - folds its name into the `tech` field of every container that referenced it,
      - deletes the entity from the external lists,
      - purges every relationship that touches it.
    Containers, databases, actors, and genuine external systems are left untouched.
    """
    ctx  = model.get("context", {}) or {}
    cont = model.get("containers", {}) or {}

    deny = set(_FRAMEWORK_LABELS)
    for fw in (frameworks or []):
        n = _norm_label(fw)
        if n:
            deny.add(n)

    tech_ids = set()              # slugged ids of entities to remove
    tech_label_by_id = {}         # slug id -> display label (for folding into tech)
    for lst in (ctx.get("external_systems") or [], cont.get("external_services") or []):
        for ext in lst:
            label = ext.get("label") or ext.get("id", "")
            if _is_technology(label, deny):
                sid = _slug(ext.get("id", ""))
                tech_ids.add(sid)
                tech_label_by_id[sid] = label

    if not tech_ids:
        return model

    cont_by_slug = {_slug(c.get("id", "")): c for c in (cont.get("containers") or [])}

    def _add_tech(container: dict, tech_label: str) -> None:
        parts = [p.strip() for p in (container.get("tech") or "").split(",") if p.strip()]
        if all(_norm_label(p) != _norm_label(tech_label) for p in parts):
            parts.append(tech_label)
        container["tech"] = ", ".join(parts)

    # fold each technology node into the container(s) it was wired to
    for block in (ctx, cont):
        for r in block.get("relationships") or []:
            a, b = _slug(r.get("from", "")), _slug(r.get("to", ""))
            for tech_slug, other in ((a, b), (b, a)):
                if tech_slug in tech_ids and other in cont_by_slug:
                    _add_tech(cont_by_slug[other], tech_label_by_id[tech_slug])

    # drop technology entities from the external lists
    ctx["external_systems"]   = [e for e in (ctx.get("external_systems") or [])
                                 if _slug(e.get("id", "")) not in tech_ids]
    cont["external_services"] = [e for e in (cont.get("external_services") or [])
                                 if _slug(e.get("id", "")) not in tech_ids]

    # purge every relationship touching a removed technology node
    for block in (ctx, cont):
        if block.get("relationships"):
            block["relationships"] = [
                r for r in block["relationships"]
                if _slug(r.get("from", "")) not in tech_ids
                and _slug(r.get("to", "")) not in tech_ids
            ]

    model["context"], model["containers"] = ctx, cont
    return model


# ── Deterministic LAYER vs CONTAINER classification ──────────────────────────
# A C4 container is a runtime/communication boundary: it exposes HTTP routes, has
# an entry point, or is a datastore. An architectural *layer* (domain, application,
# infrastructure, persistence) is library code consumed by a container — it has no
# such boundary and must not be a container node. We classify deterministically
# from static-analysis evidence (routes / entry filename / is_db_model), never from
# the label text, so a real container that happens to be named "...Layer" is kept.

_ENTRY_STEMS = {
    "main", "program", "index", "app", "cli", "server", "manage",
    "__main__", "startup", "global", "bootstrap", "run",
}


def _file_signals(slim_facts: dict) -> dict:
    """Map file path (and basename) -> {has_routes, has_db, has_entry} from facts."""
    sig: dict[str, dict] = {}
    for f in slim_facts.get("files", []) or []:
        path = (f.get("file") or "").replace("\\", "/")
        if not path:
            continue
        base = path.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0].lower()
        s = {
            "has_routes": bool(f.get("routes")),
            "has_db": any(c.get("is_db_model") for c in (f.get("classes") or [])),
            "has_entry": stem in _ENTRY_STEMS,
        }
        sig[path.lower()] = s
        sig.setdefault(base.lower(), s)   # basename fallback
    return sig


def _resolve_signal(ev_path: str, file_sig: dict) -> dict | None:
    """Best-effort match of one evidence path to a file's signals."""
    e = (ev_path or "").replace("\\", "/").lower()
    if not e:
        return None
    s = file_sig.get(e) or file_sig.get(e.rsplit("/", 1)[-1])
    if s is not None:
        return s
    for path, ps in file_sig.items():        # suffix match for partial paths
        if "/" in path and (path.endswith(e) or e.endswith(path)):
            return ps
    return None


def _evidence_is_runtime(evidence: list, file_sig: dict):
    """True/False if evidence resolves to files; None if it cannot be resolved.

    True  = at least one evidence file exposes a runtime boundary (routes/entry).
    False = evidence files resolved, but none expose a runtime boundary (= a layer).
    None  = no evidence file could be resolved -> caller should keep the node (safe).
    """
    if not evidence:
        return None
    resolved = False
    for ev in evidence:
        s = _resolve_signal(ev, file_sig)
        if s is None:
            continue
        resolved = True
        if s["has_routes"] or s["has_entry"]:
            return True
    return False if resolved else None


def collapse_layers(model: dict, arch_ctx: dict, slim_facts: dict) -> dict:
    """Remove architectural layers from the HLD container view (deterministic, pre-render).

    Each container is mapped to its HLD-context capability -> evidence files ->
    static-analysis signals. A container with NO runtime boundary (no routes, no
    entry point) is a LAYER. Each layer is contracted out of the graph by bypass-
    rewiring (every predecessor edge is reconnected to every successor edge, so a
    persistence layer's path to its database survives), its responsibility is folded
    into the consuming container's description, and its node is removed.

    Safety: containers whose evidence cannot be resolved are KEPT (classified None).
    If no runtime container exists at all (e.g. a pure library) nothing is collapsed.
    Uses only static evidence — no label text, no repository-specific rules.
    """
    cont = model.get("containers", {}) or {}
    ctx  = model.get("context", {}) or {}
    containers = cont.get("containers") or []
    if len(containers) <= 2:
        return model

    file_sig = _file_signals(slim_facts or {})
    cap_evidence = {
        _norm_label(c.get("name")): (c.get("evidence") or [])
        for c in (arch_ctx or {}).get("capabilities", []) or []
    }

    def _evidence_for(label: str):
        n = _norm_label(label)
        if n in cap_evidence:
            return cap_evidence[n]
        for cn, ev in cap_evidence.items():        # fuzzy fallback if names drifted
            if cn and (cn in n or n in cn):
                return ev
        return None

    cont_by_id = {_slug(c.get("id", "")): c for c in containers}
    layer_ids = set()
    for c in containers:
        if _evidence_is_runtime(_evidence_for(c.get("label")), file_sig) is False:
            layer_ids.add(_slug(c.get("id", "")))
    runtime_ids = set(cont_by_id) - layer_ids
    if not layer_ids or not runtime_ids:
        return model       # nothing to collapse, or no runtime anchor -> safe bail

    all_rels = list(cont.get("relationships") or []) + list(ctx.get("relationships") or [])

    # enrich: fold each layer into the description of its strongest runtime consumer
    for lid in layer_ids:
        counts: dict[str, int] = {}
        for r in all_rels:
            a, b = _slug(r.get("from", "")), _slug(r.get("to", ""))
            other = a if b == lid else (b if a == lid else None)
            if other in runtime_ids:
                counts[other] = counts.get(other, 0) + 1
        if counts:
            target = sorted(counts, key=lambda k: (-counts[k], k))[0]
            host = cont_by_id[target]
            lbl = cont_by_id[lid].get("label") or lid
            desc = (host.get("description") or "").strip()
            if _norm_label(lbl) not in _norm_label(desc):
                host["description"] = f"{desc}; includes {lbl}".lstrip("; ")

    # contract each layer out of the container relationship graph (bypass-rewire)
    rels = list(cont.get("relationships") or [])
    for lid in sorted(layer_ids):
        preds, succs, remaining = [], [], []
        for r in rels:
            a, b = _slug(r.get("from", "")), _slug(r.get("to", ""))
            if a == lid and b == lid:
                continue
            if b == lid:
                preds.append((a, r.get("label", "")))
            elif a == lid:
                succs.append((b, r.get("label", "")))
            else:
                remaining.append(r)
        for p, pl in preds:
            for s, sl in succs:
                if p != s:
                    remaining.append({"from": p, "to": s, "label": sl or pl or "uses"})
        rels = remaining

    seen, deduped = set(), []
    for r in rels:
        k = (_slug(r.get("from", "")), _slug(r.get("to", "")))
        if k[0] == k[1] or k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    cont["relationships"] = deduped
    cont["containers"] = [c for c in containers if _slug(c.get("id", "")) not in layer_ids]
    ctx["relationships"] = [
        r for r in (ctx.get("relationships") or [])
        if _slug(r.get("from", "")) not in layer_ids
        and _slug(r.get("to", "")) not in layer_ids
    ]
    model["containers"], model["context"] = cont, ctx
    return model


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ```...``` fence the model sometimes adds."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def to_json(data) -> str:
    """Serialize extracted facts to a JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_yaml(data) -> str:
    """Serialize extracted facts to a YAML string (spec/config-style docs)."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def markdown_to_html(text: str) -> str:
    """Convert a markdown document into a standalone HTML page."""
    body = _markdown.markdown(text, extensions=["fenced_code", "tables", "toc"])
    return (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        "<title>Documentation</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def save_text(path, content: str) -> str:
    """Write any text content to a file; return the absolute path written."""
    path = Path(path)
    try:
        # ensure parent directory exists
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())
    except PermissionError:
        # fallback: attempt to write to an outputs/ subfolder within cwd
        fallback_dir = Path("outputs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / path.name
        fallback_path.write_text(content, encoding="utf-8")
        return str(fallback_path.resolve())


def save_json(path, data) -> str:
    """Save a dict/list as pretty-printed JSON; return the absolute path written."""
    return save_text(path, to_json(data))


def slim_facts_for_llm(facts: list, max_tokens: int = 100_000) -> list:
    """
    Produce a token-trimmed copy of the extracted facts for LLM calls only.

    The full facts feed json/yaml (which need completeness and cost 0 LLM calls).
    This slimmed version is what goes to the writer/reviewer/diagrammer so large
    repos stay under Gemini's per-minute input-token limit.

    Trimming, applied progressively until under budget:
      1. public API only (drop _private / __dunder methods and functions)
      2. first line of each docstring, truncated
      3. drop modules with nothing public
      4. if still over budget, drop docstrings, then drop method lists
    """
    import json

    def build(keep_doc: bool, keep_methods: bool) -> list:
        out = []
        for f in facts:
            sclasses = []
            for c in f.get("classes", []):
                entry = {"name": c.get("name", "")}
                if keep_doc:
                    d = (c.get("docstring") or "").split("\n")[0].strip()
                    if d:
                        entry["doc"] = d[:80]
                if keep_methods:
                    methods = c.get("methods", []) or []
                    names = []
                    for m in methods:
                        mn = m.get("name") if isinstance(m, dict) else m
                        if mn and not mn.startswith("_"):
                            names.append(mn)
                    if names:
                        entry["methods"] = names
                sclasses.append(entry)
            sfuncs = []
            for fn in f.get("functions", []) or []:
                fnn = fn.get("name") if isinstance(fn, dict) else fn
                if fnn and not fnn.startswith("_"):
                    sfuncs.append(fnn)
            if not sclasses and not sfuncs and not f.get("module_docstring"):
                continue
            entry = {"file": f.get("file", ""), "functions": sfuncs, "classes": sclasses}
            if f.get("imports"):
                entry["imports"] = f["imports"]
            if keep_doc:
                md = (f.get("module_docstring") or "").split("\n")[0].strip()
                if md:
                    entry["doc"] = md[:100]
            out.append(entry)
        return out

    for kwargs in (
        {"keep_doc": True,  "keep_methods": True},
        {"keep_doc": True,  "keep_methods": False},
        {"keep_doc": False, "keep_methods": False},
    ):
        payload = build(**kwargs)
        if len(json.dumps(payload)) // 4 <= max_tokens:
            return payload
    return payload



# ── Shape-by-kind map for C4 nodes ────────────────────────────────────────────
# Selection is driven by the node's `kind` field set by build_container_model.
# Generic and extensible: adding a new kind is one map entry.
# Mermaid flowchart shape syntax (all valid in the current renderer):
#   person    → ([" "]) stadium
#   service/web_app/worker → [" "] rectangle (default)
#   datastore/cache → [(" ")] cylinder
#   queue     → [/" "/] parallelogram
#   external  → {{"" "}} hexagon  (double braces = literal { } in Mermaid)
_SHAPE_BY_KIND = {
    "person":    "stadium",
    "web_app":   "rect",
    "service":   "rect",
    "worker":    "rect",
    "datastore": "cylinder",
    "cache":     "cylinder",
    "queue":     "para",
    "external":  "hex",
}

# ── Per-kind color palette (deterministic, repo-agnostic) ─────────────────────
# Each kind maps to a CSS class name; classDefs are emitted once per diagram in
# _CLASSDEFS. external uses the pre-existing `ext` class. Adding a kind is one
# entry here + one matching classDef line below.
_CLASS_BY_KIND = {
    "person":    "actor",
    "web_app":   "webapp",
    "service":   "service",
    "worker":    "worker",
    "datastore": "data",
    "cache":     "data",
    "queue":     "queue",
    "external":  "ext",
}

# classDef lines emitted once per diagram. Soft fills + saturated strokes so the
# diagram reads like a designed reference graphic regardless of repo shape.
_CLASSDEFS = [
    "classDef actor   fill:#eceff1,stroke:#607d8b,color:#263238",
    "classDef webapp  fill:#e3f2fd,stroke:#1976d2,color:#0d47a1",
    "classDef service fill:#e0f2f1,stroke:#00897b,color:#004d40",
    "classDef worker  fill:#f3e5f5,stroke:#8e24aa,color:#4a148c",
    "classDef data    fill:#fff3e0,stroke:#f57c00,color:#e65100",
    "classDef queue   fill:#fce4ec,stroke:#d81b60,color:#880e4f",
    "classDef ext     fill:#f5f5f5,stroke:#999,color:#333",
]


def _emit_node(nid: str, label: str, kind: str, tech: str = "", indent: str = "    ") -> str:
    """Return a Mermaid flowchart node line with the correct shape for its kind.

    `tech` is appended as a small subtitle inside the node label when present.
    External nodes automatically receive the :::ext CSS class.
    """
    safe_label = label.replace('"', "'")
    display = f"{safe_label}<br/><small>{tech}</small>" if tech else safe_label
    shape = _SHAPE_BY_KIND.get(kind, "rect")

    if shape == "stadium":
        node = f'{nid}(["{display}"])'
    elif shape == "cylinder":
        node = f'{nid}[("{display}")]'
    elif shape == "para":
        node = f'{nid}[/"{display}"/]'
    elif shape == "hex":
        # {{"..."}} is Mermaid hexagon; {{{{ / }}}} = escaped { / } in f-strings
        node = f'{nid}{{{{"{display}"}}}}'
        return f'{indent}{node}:::ext'
    else:  # rect — default for web_app, service, worker, and unknown kinds
        node = f'{nid}["{display}"]'

    # Tag with a per-kind CSS class for deterministic color-coding (see _CLASSDEFS).
    cls = _CLASS_BY_KIND.get(kind)
    if cls:
        return f'{indent}{node}:::{cls}'
    return f'{indent}{node}'


# ── Tier taxonomy for hierarchical rendering ──────────────────────────────────
# Maps node kind → tier name. Extensible: append to TIER_ORDER for new tiers.
# These are hierarchy metadata; Mermaid controls actual visual layout.
_KIND_TO_TIER = {
    "person":    "Actors",
    "web_app":   "Applications",
    "service":   "Services",
    "worker":    "Workers",
    "datastore": "Data Stores",
    "cache":     "Data Stores",
    "queue":     "Messaging",
    "external":  "External Systems",
}

# Maps the deterministic layer field (set by assign_architecture_layers) → tier name.
# Takes priority over _KIND_TO_TIER when a node carries a layer field.
_LAYER_TO_TIER = {
    "presentation": "Clients",
    "gateway":      "API Gateway",
    "application":  "Services",
    "worker":       "Workers",
    "data":         "Data Stores",
    "messaging":    "Messaging",
}

# Order in which tiers are emitted inside (and outside) the system boundary.
# "Clients" and "API Gateway" precede "Services" to produce a top-down flow:
# Actor → Clients → API Gateway → Services → Workers → Data Stores
# "Applications" is kept for backward-compat (web_app nodes without a layer).
TIER_ORDER = [
    "Actors",
    "External Systems",
    "Clients",
    "API Gateway",
    "Applications",
    "Services",
    "Workers",
    "Data Stores",
    "Messaging",
]

# Tiers rendered inside the system boundary subgraph
_INTERNAL_TIERS = {"Clients", "API Gateway", "Applications", "Services", "Workers", "Data Stores", "Messaging"}


# ── HLD flowchart renderer (system-boundary subgraph) ─────────────────────────

_LAYER_RANK = {
    "presentation": 0,
    "gateway":      1,
    "application":  2,
    "worker":       3,
    "data":         4,
    "messaging":    5,
}

_KIND_RANK = {
    "web_app": 0,
    "service": 2,
    "worker":  3,
    "datastore": 4,
    "cache":   4,
    "queue":   5,
}


def _render_flowchart_combined(model: dict) -> str:
    """Render HLD as a flowchart TD with a system boundary subgraph.

    When containers carry a 'group' field (set by assign_domains), renders domain
    subgraphs ordered entry-first with datastores nested in their owner's domain.
    Falls back to layer-tier subgraphs when no 'group' fields are present.

    Architecture (domain mode):
      Actors (outside, stadium shape)
      subgraph SYS — system boundary
          subgraph <Domain A>  (entry domain first)
          subgraph <Domain B>
          ...
      External Systems (outside, hexagons) — only those with ≥1 relationship

    Architecture (tier mode — backward-compatible):
      Actors (outside, stadium shape)
      subgraph SYS — system boundary
          subgraph Applications  (web_app nodes)
          subgraph Services      (service / worker nodes)
          subgraph Data Stores   (datastore / cache nodes)
          subgraph Messaging     (queue nodes)
      External Systems (outside, hexagons) — only those with ≥1 relationship
    """
    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    system_label = (
        cont.get("system_label")
        or ctx.get("system_name")
        or "System"
    )

    all_rels_flat = (ctx.get("relationships") or []) + (cont.get("relationships") or [])

    # Determine which external ids are connected (participate in ≥1 relationship)
    rel_endpoints: set[str] = set()
    for r in all_rels_flat:
        rel_endpoints.add(_slug(r.get("from", "")))
        rel_endpoints.add(_slug(r.get("to", "")))

    all_externals = (
        list(ctx.get("external_systems", []))
        + list(cont.get("external_services", []))
    )
    # Deduplicate externals by slug id
    seen_ext_ids: set[str] = set()
    deduped_externals: list[dict] = []
    for ext in all_externals:
        eid = _slug(ext["id"])
        if eid not in seen_ext_ids:
            seen_ext_ids.add(eid)
            deduped_externals.append(ext)

    # Only render externals that participate in at least one relationship
    connected_externals = [e for e in deduped_externals if _slug(e["id"]) in rel_endpoints]

    lines = [
        "%%{init: {'theme':'base','flowchart':{'curve':'basis','nodeSpacing':40,'rankSpacing':55,'padding':12}}}%%",
        "flowchart TD",
    ]
    lines.append("")

    # ── Actors (outside SYS) ─────────────────────────────────────────────────
    actor_ids: list[str] = []
    for a in ctx.get("actors", []):
        aid = _slug(a["id"])
        actor_ids.append(aid)
        lines.append(_emit_node(aid, a["label"], a.get("kind", "person"), indent="    "))

    if actor_ids:
        lines.append("")

    # ── System boundary (the only subgraph; strict flat Container view inside) ──
    # SYS is uppercase; _slug lowercases all node ids, so it cannot collide.
    safe_sys = system_label.replace('"', "'")
    lines.append(f'    subgraph SYS["{safe_sys}"]')
    lines.append("    direction TB")

    cap_ids: list[str] = []

    # ── Layered Container view: group deployable units into horizontal lanes ───
    # Each node is bucketed into a tier by its architecture `layer` (falling back
    # to its `kind`), then lanes are emitted top-down in TIER_ORDER inside the SYS
    # boundary. Lane membership is derived metadata the model already assigns —
    # repo-agnostic, no fabricated nodes (a monolith yields a single lane).
    all_containers = cont.get("containers", [])

    def _node_rank(node: dict) -> int:
        layer = node.get("layer", "")
        if layer in _LAYER_RANK:
            return _LAYER_RANK[layer]
        return _KIND_RANK.get(node.get("kind", "service"), 2)

    def _tier_of(node: dict) -> str:
        layer = node.get("layer", "")
        if layer in _LAYER_TO_TIER:
            return _LAYER_TO_TIER[layer]
        return _KIND_TO_TIER.get(node.get("kind", "service"), "Services")

    # Collect all nodes that render inside the boundary (containers + connected
    # datastores/caches/queues), bucketed by tier.
    tier_nodes: dict[str, list[dict]] = {}
    for node in all_containers:
        tier_nodes.setdefault(_tier_of(node), []).append(node)
    for db in cont.get("databases", []):
        if _slug(db["id"]) not in rel_endpoints:
            continue  # skip unconnected datastores
        tier_nodes.setdefault(_tier_of(db), []).append(db)

    # Emit one lane subgraph per non-empty internal tier, in TIER_ORDER.
    for tier in TIER_ORDER:
        members = tier_nodes.get(tier)
        if not members or tier not in _INTERNAL_TIERS:
            continue
        lane_id = "lane_" + _slug(tier)
        safe_tier = tier.replace('"', "'")
        lines.append(f'        subgraph {lane_id}["{safe_tier}"]')
        lines.append("        direction LR")
        for node in sorted(members, key=_node_rank):
            nid = _slug(node["id"])
            cap_ids.append(nid)
            lines.append(_emit_node(
                nid, node["label"], node.get("kind", "service"),
                tech=node.get("tech", ""), indent="            ",
            ))
        lines.append("        end")

    lines.append("    end")
    lines.append("")

    # ── Connected external systems (outside boundary) ─────────────────────────
    ext_ids: set[str] = set()
    for ext in connected_externals:
        eid = _slug(ext["id"])
        ext_ids.add(eid)
        lines.append(_emit_node(eid, ext["label"], ext.get("kind", "external"), indent="    "))

    if ext_ids:
        lines.append("")

    for _cd in _CLASSDEFS:
        lines.append(f"    {_cd}")
    # Emphasize the synthesized ingress entrypoint, if one was marked
    _entry = next((c for c in all_containers if c.get("role") == "entrypoint"), None)
    if _entry:
        lines.append("    classDef entrypoint fill:#fff3cd,stroke:#d39e00,stroke-width:2px,color:#222")
        lines.append(f"    class {_slug(_entry['id'])} entrypoint")
    lines.append("")

    # ── Build declared-ID set ─────────────────────────────────────────────────
    declared = set(actor_ids) | set(cap_ids) | ext_ids

    # Auto-declare any relationship endpoint not yet in declared (as external hex)
    for _rel in all_rels_flat:
        for _key in ("from", "to"):
            _eid = _slug(_rel.get(_key, ""))
            if _eid and _eid not in declared:
                declared.add(_eid)
                lines.append(_emit_node(_eid, _rel[_key], "external", indent="    "))

    # ── Relationships (deduped, validated) ───────────────────────────────────
    seen_rels: set = set()

    def _add_rel(f: str, t: str, label: str) -> None:
        f, t = _slug(f), _slug(t)
        if f in declared and t in declared and (f, t) not in seen_rels:
            seen_rels.add((f, t))
            escaped = label.replace('"', "'")
            lines.append(f'    {f} -->|"{escaped}"| {t}')

    for rel in all_rels_flat:
        _add_rel(rel["from"], rel["to"], rel.get("label", ""))

    # ── Fallback: guarantee at least one actor → capability edge ─────────────
    if actor_ids and cap_ids:
        if not any((a, c) in seen_rels for a in actor_ids for c in cap_ids):
            _add_rel(actor_ids[0], cap_ids[0], "uses")

    return "\n".join(lines)


def render_c4_combined(model: dict) -> str:
    """Render HLD as a flowchart TD with a named system-boundary subgraph."""
    return _render_flowchart_combined(model)


def render_c4_container(model: dict) -> str:
    """Render a (sub-)model's containers as a flowchart TD subgraph."""
    return _render_flowchart_combined(model)


# ── HLD flowchart context view (system as one box) ───────────────────────────

def render_c4_context(model: dict) -> str:
    """Render HLD context as a flowchart TD: system as ONE box + actors + externals."""
    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    system_label = (
        cont.get("system_label")
        or ctx.get("system_name")
        or "System"
    )

    lines = ["flowchart TD"]

    # ── Actors ──────────────────────────────────────────────────────────────
    actor_ids = []
    for a in ctx.get("actors", []):
        aid = _slug(a["id"])
        actor_ids.append(aid)
        lines.append(_emit_node(aid, a["label"], a.get("kind", "person"), indent="    "))

    lines.append("")

    # ── System as a single box ────────────────────────────────────────────
    safe_label = system_label.replace('"', "'")
    sys_id = _slug(system_label) or "system"
    lines.append(f'    {sys_id}["{safe_label}"]')

    lines.append("")

    # ── External systems (outside the system, hexagons) ───────────────────
    # Context view: external systems and queues/services the system depends on
    # but NOT internal datastores (those are inside the container boundary)
    all_externals = list(ctx.get("external_systems", []))
    all_rels_flat = list(ctx.get("relationships", []))
    rel_node_ids = {_slug(r["from"]) for r in all_rels_flat} | {_slug(r["to"]) for r in all_rels_flat}

    # Add container-level external services to context view as well
    for ext in cont.get("external_services", []):
        if not any(e["id"] == ext["id"] for e in all_externals):
            all_externals.append(ext)
    for r in cont.get("relationships", []):
        rel_node_ids.add(_slug(r.get("from", "")))
        rel_node_ids.add(_slug(r.get("to", "")))

    ext_ids: set[str] = set()
    seen_ext: set[str] = set()
    for ext in all_externals:
        eid = _slug(ext["id"])
        if eid not in seen_ext:
            seen_ext.add(eid)
            ext_ids.add(eid)
            lines.append(_emit_node(eid, ext["label"], ext.get("kind", "external"), indent="    "))

    lines.append("")
    lines.append("    classDef ext fill:#f5f5f5,stroke:#999,color:#333")
    lines.append("")

    # ── Declared node set ─────────────────────────────────────────────────
    declared = set(actor_ids) | {sys_id} | ext_ids

    # ── Relationships ─────────────────────────────────────────────────────
    seen_rels: set = set()

    def _add_rel(f: str, t: str, label: str) -> None:
        # Reroute any internal container ids to the system box
        f_s, t_s = _slug(f), _slug(t)
        if f_s not in declared:
            f_s = sys_id
        if t_s not in declared:
            t_s = sys_id
        if f_s == t_s:
            return
        if (f_s, t_s) not in seen_rels:
            seen_rels.add((f_s, t_s))
            escaped = label.replace('"', "'")
            lines.append(f'    {f_s} -->|"{escaped}"| {t_s}')

    # Actor → system
    for a_id in actor_ids:
        _add_rel(a_id, sys_id, "uses")

    # System → externals (from any relationship involving the system or containers)
    container_ids = {_slug(c["id"]) for c in cont.get("containers", [])}
    for rel in list(ctx.get("relationships", [])) + list(cont.get("relationships", [])):
        f_s, t_s = _slug(rel.get("from", "")), _slug(rel.get("to", ""))
        f_is_internal = (f_s in container_ids or f_s == sys_id)
        t_is_external = t_s in ext_ids
        f_is_actor = f_s in set(actor_ids)
        t_is_internal = (t_s in container_ids or t_s == sys_id)

        if f_is_internal and t_is_external:
            _add_rel(sys_id, t_s, rel.get("label", "uses"))
        elif f_is_actor and t_is_internal:
            _add_rel(f_s, sys_id, rel.get("label", "uses"))

    return "\n".join(lines)

def _clean_params(params: str) -> str:
    """Mermaid class members cannot contain [ ] | or = — keep parameter names only."""
    if not params:
        return ""
    names, current, depth = [], "", 0
    for ch in params:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            names.append(current)
            current = ""
            continue
        current += ch
    names.append(current)
    out = []
    for n in names:
        n = n.split(":")[0].split("=")[0].strip()
        n = re.sub(r"<[^>]*>", "", n).strip()   # strip any surviving generics
        toks = n.split()
        n = toks[-1] if toks else ""             # "Type name" → keep the identifier
        if n and n not in ("self", "cls") and re.match(r"^[A-Za-z_]\w*$", n):
            out.append(n)
    return ", ".join(out)


def _clean_type(typ: str) -> str:
    """Reduce an annotation to a Mermaid-safe name.

    Mermaid classDiagram members cannot contain < > [ ] | = (it uses ~Generic~,
    not <Generic>), so reduce to the base type name and drop nullability markers:
      'dict[str, bool] | None' -> 'dict'
      'List<BasketItem>'       -> 'List'
      'Task<int?>'             -> 'Task'
      'string?'                -> 'string'
    """
    if not typ:
        return ""
    base = typ.split("|")[0].split("=")[0].split("[")[0].split("<")[0].strip()
    return base.rstrip("?")


def _safe_class_id(name: str) -> str:
    """Sanitize a class name to a Mermaid-safe identifier.

    Strips generic parameters and replaces any non-identifier characters so that
    class names like IRepository<T> or Microsoft.EntityFrameworkCore.DbContext
    don't break the classDiagram parser.
    """
    base = re.sub(r"<[^>]*>", "", name).strip()   # IRepository<T> → IRepository
    base = re.sub(r"[^\w]", "_", base).strip("_")  # dots, dashes, etc. → _
    return base or "Class_"


def render_class_diagram(model: dict) -> str:
    """Render a class diagram JSON model as Mermaid classDiagram."""
    _REL = {
        "inheritance": "<|--", "composition": "*--", "aggregation": "o--",
        "dependency": "-->", "realization": "<|..",
    }

    seen_classes: set[str] = set()
    deduped_classes = []
    for cls in model.get("classes", []):
        if cls["name"] not in seen_classes:
            seen_classes.add(cls["name"])
            deduped_classes.append(cls)

    # safe Mermaid identifier for each original name (strips <>, dots, etc.)
    name_map: dict[str, str] = {c["name"]: _safe_class_id(c["name"]) for c in deduped_classes}

    # which classes survive in a valid relationship (both endpoints declared)
    connected: set[str] = set()
    rels = []
    for rel in model.get("relationships", []):
        src = re.sub(r"<[^>]*>", "", rel["from"]).strip()
        dst = re.sub(r"<[^>]*>", "", rel["to"]).strip()
        if src in seen_classes and dst in seen_classes:
            rels.append((rel, src, dst))
            connected.add(src); connected.add(dst)

    # drop orphans unless the whole view is a single class
    keep = (lambda n: True) if len(deduped_classes) <= 1 else (lambda n: n in connected)

    lines = ["classDiagram"]
    for cls in deduped_classes:
        if not keep(cls["name"]):
            continue
        safe_id = name_map[cls["name"]]
        original = cls["name"]
        fields = cls.get("fields", [])
        methods = cls.get("methods", [])
        # use "ClassName["Label"]" form if the name needed sanitizing
        label_annot = f'["{original}"]' if safe_id != original else ""
        if not fields and not methods:
            lines.append(f'  class {safe_id}{label_annot}')
            continue
        lines.append(f'  class {safe_id}{label_annot} {{')
        for f in fields:
            vis = f.get("visibility", "+")
            typ = _clean_type(f.get("type") or "")
            type_suffix = f" : {typ}" if typ else ""
            lines.append(f'    {vis}{f["name"]}{type_suffix}')
        for m in methods:
            vis = m.get("visibility", "+")
            ret = _clean_type(m.get("return_type") or "")
            ret_suffix = f" {ret}" if ret else ""
            params = _clean_params(m.get("params", ""))
            lines.append(f'    {vis}{m["name"]}({params}){ret_suffix}')
        lines.append("  }")

    for rel, src, dst in rels:
        if not (keep(src) and keep(dst)):
            continue
        src_id = name_map.get(src, _safe_class_id(src))
        dst_id = name_map.get(dst, _safe_class_id(dst))
        rtype = rel.get("type", "dependency")
        arrow = _REL.get(rtype, "-->")
        lbl = _safe_label(rel.get("label") or "")
        label_str = f" : {lbl}" if lbl else ""
        if rtype in ("inheritance", "realization"):
            src_id, dst_id = dst_id, src_id
        lines.append(f'  {src_id} {arrow} {dst_id}{label_str}')

    return "\n".join(lines)




def render_sequence_diagram(model: dict) -> str:
    """Render a sequence diagram JSON model as Mermaid sequenceDiagram."""
    _ARROW = {"sync": "->>", "async": "->>", "return": "-->>"}

    def _short(name: str) -> str:
        """Use only the last dotted segment as the participant alias."""
        return _slug(name.split(".")[-1])

    lines = ["sequenceDiagram"]

    for p in model.get("participants", []):
        alias = _short(p)
        label = p.split(".")[-1]
        lines.append(f'  participant {alias} as {label}')

    for msg in model.get("messages", []):
        arrow = _ARROW.get(msg.get("type", "sync"), "->>")
        label = msg.get("label", "")
        lines.append(f'  {_short(msg["from"])}{arrow}{_short(msg["to"])}: {label}')

    return "\n".join(lines)



def render_component_diagram(model: dict) -> str:
    """Render a component diagram JSON model as Mermaid graph LR.

    One subgraph per module/project (module-first, matching the PlantUML renderer).
    Components with layer == 'external' are rendered outside with ::ext style.
    """
    seen_ids: dict[str, str] = {}   # original_id -> slug
    by_module: dict[str, list] = {}
    module_order: list[str] = []
    external_comps: list[dict] = []

    for comp in model.get("components", []):
        raw_id = (comp.get("id") or "").strip()
        if not raw_id:
            continue
        slug = _safe_id(raw_id)
        if slug in seen_ids.values():
            continue  # duplicate after slugging
        seen_ids[raw_id] = slug
        tech = (comp.get("tech") or "").strip()
        label = _safe_label(comp.get("label") or raw_id)
        entry = {"slug": slug, "label": label, "tech": tech}
        if (comp.get("layer") or "").lower() == "external":
            external_comps.append(entry)
            continue
        mkey = comp.get("module") or "Application"
        if mkey not in by_module:
            by_module[mkey] = []
            module_order.append(mkey)
        by_module[mkey].append({**entry, "module_label": comp.get("module_label") or mkey})

    lines = ["graph LR"]

    pkgs = model.get("packages") or []
    order = [p["id"] for p in pkgs if p["id"] in by_module] or module_order
    for mkey in order:
        members = by_module[mkey]
        sub_label = members[0]["module_label"]
        sub_id = _safe_id(mkey)
        lines.append(f'  subgraph {sub_id}["{_safe_label(sub_label)}"]')
        for c in members:
            tech_str = f" ({c['tech']})" if c["tech"] else ""
            lines.append(f'    {c["slug"]}["{c["label"]}{tech_str}"]')
        lines.append("  end")
        lines.append("")

    for c in external_comps:
        tech_str = f" ({c['tech']})" if c["tech"] else ""
        lines.append(f'  {c["slug"]}["{c["label"]}{tech_str}"]:::ext')

    if external_comps:
        lines.append("  classDef ext fill:#f5f5f5,stroke:#999,color:#333")
        lines.append("")

    declared = set(seen_ids.values())
    seen_edges: set[tuple] = set()

    for dep in model.get("dependencies", []):
        fraw = (dep.get("from") or "").strip()
        traw = (dep.get("to") or "").strip()
        fid = seen_ids.get(fraw) or _safe_id(fraw)
        tid = seen_ids.get(traw) or _safe_id(traw)
        if fid not in declared or tid not in declared or fid == tid:
            continue
        if (fid, tid) in seen_edges:
            continue
        seen_edges.add((fid, tid))
        lbl = _safe_label(dep.get("label") or "")
        label_str = f'|"{lbl}"|' if lbl else ""
        lines.append(f'  {fid} -->{label_str} {tid}')

    return "\n".join(lines)



def render_component_plantuml(model: dict) -> str:
    """Render the architecture model as a TRUE UML component diagram (PlantUML).

    Top-level packages are the repo's physical projects/modules (module-first);
    layer (presentation/domain/persistence/...) is rendered as a <<stereotype>>
    on each component, not as the grouping axis.
    """
    comps = [c for c in model.get("components", []) if (c.get("id") or "").strip()]
    by_module: dict[str, list] = {}
    for c in comps:
        by_module.setdefault(c.get("module") or "", []).append(c)

    lines = ["@startuml", "skinparam componentStyle uml2", "left to right direction", ""]
    pkgs = model.get("packages") or []
    order = [p["id"] for p in pkgs] or list(by_module)
    for mkey in order:
        if mkey not in by_module:
            continue
        members = by_module[mkey]
        label = members[0].get("module_label") or mkey or "Components"
        lines.append(f'package "{_safe_label(label)}" {{')
        for c in members:
            cid = "c_" + _safe_id(c["id"])
            stereo = c.get("stereotype")
            stereo_str = f" <<{stereo}>>" if stereo else ""
            lines.append(f'  component "{_safe_label(c.get("label") or c["id"])}" as {cid}{stereo_str}')
        lines.append("}")
    lines.append("")

    for c in comps:                                        # capability interface as a lollipop
        cid = "c_" + _safe_id(c["id"])
        has_surface = c.get("has_routes") or c.get("has_db") or c.get("owns_entities")
        ifaces = c.get("interfaces") or c.get("capabilities") or []
        if has_surface and ifaces:                          # max 1/component — readability budget
            iid = f"{cid}_i0"
            lines.append(f'interface "{_safe_label(str(ifaces[0]))}" as {iid}')
            lines.append(f"{cid} -- {iid}")
    lines.append("")

    declared = {"c_" + _safe_id(c["id"]) for c in comps}
    seen = set()
    for e in model.get("dependencies", model.get("edges", [])):
        a = "c_" + _safe_id((e.get("from") or "").strip())
        b = "c_" + _safe_id((e.get("to") or "").strip())
        if a not in declared or b not in declared or a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        lines.append(f'{a} ..> {b} : {_safe_label(e.get("label") or "requires")}')

    lines.append("@enduml")
    return "\n".join(lines)


_IFACE_BY_LAYER = {
    "presentation":   "API",
    "application":    "Services",
    "domain":         "Domain",
    "infrastructure": "Integration",
    "persistence":    "Persistence",
}


def _component_iface_name(n: dict) -> str:
    """The provided-interface (contract) name a component exposes. Prefers the
    LLM-named interfaces/capabilities; falls back to a layer-based contract so a
    depended-upon component always presents a meaningful lollipop, never a bare id."""
    ifaces = n.get("interfaces") or n.get("capabilities") or []
    if ifaces and str(ifaces[0]).strip():
        return str(ifaces[0]).strip()
    return _IFACE_BY_LAYER.get(n.get("layer") or "application", "API")


def _render_view_plantuml(view: dict) -> str:
    """Render one ViewSet view (L1 or L2) as a PlantUML component diagram string."""
    nodes = view.get("nodes", [])
    edges = view.get("edges", [])
    title = view.get("title", "")
    omitted = view.get("omitted", {})

    lines = [
        "@startuml",
        "skinparam componentStyle uml2",
        "left to right direction",
        "skinparam component {",
        "  BackgroundColor<<boundary>> #F0F0F0",
        "  BorderColor<<boundary>> #AAAAAA",
        "  FontColor<<boundary>> #888888",
        "  BackgroundColor<<aggregate>> #FFFBE6",
        "  BorderColor<<aggregate>> #C8A000",
        "  BackgroundColor<<overflow>> #F5F5F5",
        "  BorderColor<<overflow>> #BBBBBB",
        "  FontColor<<overflow>> #888888",
        "  BackgroundColor<<database>> #E8F0FE",
        "  BorderColor<<database>> #4A6FA5",
        "  BackgroundColor<<infrastructure>> #F0F0F0",
        "  BorderColor<<infrastructure>> #888888",
        "}",
        "",
    ]

    if title:
        lines.append(f'title {_safe_label(title)}')
        lines.append("")

    declared: set[str] = set()
    # provider node id -> its provided-interface id. Dependency arrows are routed
    # INTO these lollipops (ball-and-socket assembly), so a consumer's required
    # arrow lands on the provider's provided interface instead of bypassing it.
    provided_iface: dict[str, str] = {}

    # group real components by layer for sub-packaging; aggregates/ghosts rendered flat
    real_nodes   = [n for n in nodes if not n.get("is_aggregate") and not n.get("is_ghost")]
    agg_nodes    = [n for n in nodes if n.get("is_aggregate")]
    ghost_nodes  = [n for n in nodes if n.get("is_ghost")]

    # any node that RECEIVES a dependency in this view is a provider -> it must
    # expose a provided interface for the incoming arrow(s) to assemble onto.
    edge_targets = {"c_" + _safe_id((e.get("to") or "").strip()) for e in edges}

    def _emit_component(n: dict, indent: str = "") -> None:
        """Declare one component + its provided-interface lollipop (assembly target)."""
        nid    = "c_" + _safe_id(n["id"])
        stereo = n.get("stereotype") or n.get("layer") or ""
        stereo_str = f" <<{stereo}>>" if stereo else ""
        lines.append(f'{indent}component "{_safe_label(n.get("label") or n["id"])}" as {nid}{stereo_str}')
        declared.add(nid)
        has_surface = n.get("has_routes") or n.get("has_db") or n.get("owns_entities")
        if has_surface or nid in edge_targets:
            iid = f"{nid}_i0"
            lines.append(f'{indent}interface "{_safe_label(_component_iface_name(n))}" as {iid}')
            lines.append(f"{indent}{nid} -- {iid}")
            provided_iface[nid] = iid

    if real_nodes:
        # ONE unified system: internal (non-infra) components live INSIDE a single
        # system-boundary rectangle; the shared platform/infrastructure component sits
        # OUTSIDE on the sink side, alongside external datastores/cloud — exactly the
        # reference shape. This is a SINGLE outer box, not the per-layer packages that
        # forced a vertical stack. Layer remains visible via each <<stereotype>>.
        _sink_rank = {"presentation": 0, "application": 1, "domain": 2,
                      "infrastructure": 3, "persistence": 4}
        ordered_nodes = sorted(
            real_nodes,
            key=lambda n: (_sink_rank.get(n.get("layer") or "application", 1),
                           -(n.get("member_count") or 0), n["id"]),
        )
        # Data-access / platform tier renders OUTSIDE the system box on the sink side
        # (next to the datastores it funnels to), exactly like the reference's
        # Persistence/Security services. Everything else is a business component inside.
        internal = [n for n in ordered_nodes
                    if not (n.get("is_infra") or (n.get("layer") in ("infrastructure", "persistence")))]
        platform = [n for n in ordered_nodes if n not in internal]

        sys_label = _safe_label(view.get("system_label") or "System")
        if internal:
            lines.append(f'rectangle "{sys_label}" as sys_boundary {{')
            for n in internal:
                _emit_component(n, indent="  ")
            lines.append("}")
        for n in platform:                 # shared platform/infra service — outside the box
            _emit_component(n)
        lines.append("")

    for n in agg_nodes:
        nid = "c_" + _safe_id(n["id"])
        stereo = "overflow" if n.get("is_overflow") else "aggregate"
        lines.append(f'component "{_safe_label(n.get("label") or n["id"])}" as {nid} <<{stereo}>>')
        declared.add(nid)
    if agg_nodes:
        lines.append("")

    for n in ghost_nodes:
        nid = "c_" + _safe_id(n["id"])
        lines.append(f'component "{_safe_label(n.get("label") or n["id"])}" as {nid} <<boundary>>')
        declared.add(nid)
        # a neighbouring module that is depended upon also exposes a provided
        # interface, so cross-module wiring reads as an assembly into its contract.
        if nid in edge_targets:
            iid = f"{nid}_i0"
            lines.append(f'interface "{_safe_label(n.get("label") or n["id"])}" as {iid}')
            lines.append(f"{nid} -- {iid}")
            provided_iface[nid] = iid
    if ghost_nodes:
        lines.append("")

    # external systems (datastores / services) — rendered as stereotyped components
    # that dependency arrows assemble into. Ids prefixed identically to component ids.
    for x in view.get("externals", []):
        nid = "c_" + _safe_id(x["id"])
        stereo = x.get("stereotype") or "infrastructure"
        lines.append(f'component "{_safe_label(x.get("label") or x["id"])}" as {nid} <<{stereo}>>')
        declared.add(nid)
    if view.get("externals"):
        lines.append("")

    seen_edges: set[tuple] = set()
    for e in edges:
        a = "c_" + _safe_id((e.get("from") or "").strip())
        b = "c_" + _safe_id((e.get("to")   or "").strip())
        if a not in declared or b not in declared or a == b or (a, b) in seen_edges:
            continue
        seen_edges.add((a, b))
        # route the required arrow into the provider's provided interface when one
        # exists (assembly); else fall back to a plain box-to-box dependency.
        target = provided_iface.get(b, b)
        lines.append(f'{a} ..> {target} : {_safe_label(e.get("label") or "requires")}')

    if omitted.get("nodes") or omitted.get("edges"):
        parts = []
        if omitted.get("nodes"):
            parts.append(f'{omitted["nodes"]} group(s) folded')
        if omitted.get("edges"):
            parts.append(f'{omitted["edges"]} edge(s) omitted')
        lines.append("")
        lines.append(f'note as omit_note')
        lines.append(f'  [{", ".join(parts)}]')
        lines.append(f'end note')

    lines.append("@enduml")
    return "\n".join(lines)


def render_component_view_set(viewset: dict) -> list:
    """Render each view in a ViewSet as a PlantUML diagram.

    Returns a list of dicts: [{title, level, content, image_url, omitted}]
    image_url may be None if the PlantUML server URL fails.
    """
    result = []
    for view in viewset.get("views", []):
        content = _render_view_plantuml(view)
        try:
            image_url = plantuml_server_url(content)
        except Exception:
            image_url = None
        result.append({
            "title":     view.get("title", ""),
            "level":     view.get("level", "L1"),
            "content":   content,
            "image_url": image_url,
            "omitted":   view.get("omitted", {"nodes": 0, "edges": 0}),
        })
    return result


def _pkg_label(layer_id: str) -> str:
    return {"presentation": "Presentation / API", "application": "Application",
            "domain": "Domain", "infrastructure": "Infrastructure",
            "persistence": "Persistence"}.get(layer_id, layer_id.title())


# PlantUML server text encoding: UTF-8 -> raw DEFLATE -> PlantUML base64 (NOT standard b64).
_PLANTUML_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"

def _plantuml_encode64(data: bytes) -> str:
    out = []
    for i in range(0, len(data), 3):                       # always 4 chars / 3-byte group
        b0 = data[i]
        b1 = data[i + 1] if i + 1 < len(data) else 0
        b2 = data[i + 2] if i + 2 < len(data) else 0
        out.append(_PLANTUML_ALPHABET[(b0 >> 2) & 0x3F])
        out.append(_PLANTUML_ALPHABET[(((b0 & 0x3) << 4) | (b1 >> 4)) & 0x3F])
        out.append(_PLANTUML_ALPHABET[(((b1 & 0xF) << 2) | (b2 >> 6)) & 0x3F])
        out.append(_PLANTUML_ALPHABET[b2 & 0x3F])
    return "".join(out)

def plantuml_server_url(puml_text: str, fmt: str = "svg",
                        server: str = "https://www.plantuml.com/plantuml") -> str:
    raw = zlib.compress(puml_text.encode("utf-8"), 9)[2:-4]   # strip zlib hdr + adler32 -> raw deflate
    return f"{server}/{fmt}/{_plantuml_encode64(raw)}"


def render_dependency_diagram(model: dict) -> str:
    """Render a dependency diagram JSON model as Mermaid graph LR.

    Internal packages go in 'This Repo' subgraph.
    External packages go in 'External Libraries' subgraph.
    """
    seen_ids: dict[str, str] = {}   # original_id -> slug
    internal_pkgs: list[dict] = []
    external_pkgs: list[dict] = []

    for pkg in model.get("packages", []):
        raw_id = (pkg.get("id") or "").strip()
        if not raw_id:
            continue
        slug = _safe_id(raw_id)
        if slug in seen_ids.values():
            continue  # duplicate
        seen_ids[raw_id] = slug
        entry = {"slug": slug, "label": _safe_label(pkg.get("label") or raw_id)}
        if (pkg.get("kind") or "").lower() == "internal":
            internal_pkgs.append(entry)
        else:
            external_pkgs.append(entry)

    lines = ["graph LR"]

    if internal_pkgs:
        lines.append('  subgraph REPO["This Repo"]')
        for p in internal_pkgs:
            lines.append(f'    {p["slug"]}["{p["label"]}"]')
        lines.append("  end")
        lines.append("")

    if external_pkgs:
        lines.append('  subgraph EXT["External Libraries"]')
        for p in external_pkgs:
            lines.append(f'    {p["slug"]}["{p["label"]}"]')
        lines.append("  end")
        lines.append("")

    declared = set(seen_ids.values())
    seen_edges: set[tuple] = set()

    for edge in model.get("edges", []):
        fraw = (edge.get("from") or "").strip()
        traw = (edge.get("to") or "").strip()
        fid = seen_ids.get(fraw) or _safe_id(fraw)
        tid = seen_ids.get(traw) or _safe_id(traw)
        if fid not in declared or tid not in declared or fid == tid:
            continue
        if (fid, tid) in seen_edges:
            continue
        seen_edges.add((fid, tid))
        lbl = _safe_label(edge.get("label") or "")
        label_str = f'|"{lbl}"|' if lbl else ""
        lines.append(f'  {fid} -->{label_str} {tid}')

    return "\n".join(lines)


def _slug(name: str) -> str:
    """Convert a display name to a safe Mermaid node ID.

    Strips ALL non-word characters (not just space/-/.) so ids derived from
    arbitrary names (e.g. '@scope/pkg', 'my.app:core', '(x)') can never leak a
    Mermaid-breaking character into node-id position.
    """
    return re.sub(r"[^\w]", "_", (name or "").lower()).strip("_") or "node"

_MERMAID_RESERVED = {
    "end", "subgraph", "loop", "alt", "else", "opt",
    "par", "break", "critical", "note", "rect", "ref",
}

def _safe_id(name: str) -> str:
    """Slug a name and append '_' if it's a Mermaid reserved word."""
    slug = re.sub(r"[^\w]", "_", (name or "").lower()).strip("_") or "node"
    return slug + "_" if slug in _MERMAID_RESERVED else slug

def _safe_label(text: str) -> str:
    """Collapse multi-line text, replace double-quotes, strip semicolons."""
    return (text or "").replace("\n", " ").replace('"', "'").replace(";", ",").strip()
