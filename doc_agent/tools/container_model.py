"""
Deterministic container-level topology builder.

build_container_model(rich_facts, repo_root) -> dict

Returns the same model shape that render_c4_combined expects:
  {"context": {...}, "containers": {...}}

The node-set (containers / datastores / externals / actors) is computed
purely from deployment evidence — build manifests, framework signals, DB
driver imports, third-party SDK imports. The LLM (HLDEnrichmentAgent)
fills in text-only fields downstream (system_purpose, descriptions,
edge labels) and cannot add, split, or rename any node, so drift back to
capability-style boxes is structurally impossible.

Node kind taxonomy (complete, fixed set):
  person · web_app · service · worker · datastore · cache · queue · external
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from doc_agent.tools.architecture_model import runtime_facts, _is_entry
from doc_agent.tools.import_graph import detect_frameworks
from doc_agent.tools.language_detector import SKIP_DIRS

# ── UI vs backend framework sets (Change E content embedded here) ─────────────
_UI_FRAMEWORKS = {
    "react", "nextjs", "vue", "angular", "svelte", "sveltekit", "remix",
}
_FULLSTACK_FRAMEWORKS = {"nextjs", "remix", "sveltekit"}   # single deploy — UI+API
_BACKEND_FRAMEWORKS = {
    "express", "fastify", "nestjs", "spring", "aspnetcore",
    "fastapi", "flask", "django", "starlette", "tornado",
}

# ── Build manifest file names / globs ────────────────────────────────────────
_MANIFEST_NAMES = frozenset({
    "package.json", "pom.xml", "build.gradle", "build.gradle.kts",
    "go.mod", "pyproject.toml", "setup.py", "Gemfile",
})
_MANIFEST_GLOBS = ("*.csproj", "*.fsproj")

# ── DB engine catalog: import-token -> (display label, kind) ─────────────────
# kind is one of: "datastore" | "cache" | "queue"
_DB_ENGINES: dict[str, tuple[str, str]] = {
    # Document stores
    "mongoose":          ("MongoDB",           "datastore"),
    "mongodb":           ("MongoDB",           "datastore"),
    "pymongo":           ("MongoDB",           "datastore"),
    "motor":             ("MongoDB",           "datastore"),
    # Relational — concrete drivers
    "pg":                ("PostgreSQL",        "datastore"),
    "psycopg2":          ("PostgreSQL",        "datastore"),
    "asyncpg":           ("PostgreSQL",        "datastore"),
    "npgsql":            ("PostgreSQL",        "datastore"),
    "mysql":             ("MySQL",             "datastore"),
    "mysql2":            ("MySQL",             "datastore"),
    "pymysql":           ("MySQL",             "datastore"),
    "sqlite3":           ("SQLite",            "datastore"),
    "better-sqlite3":    ("SQLite",            "datastore"),
    "aiosqlite":         ("SQLite",            "datastore"),
    "mssql":             ("SQL Server",        "datastore"),
    "tedious":           ("SQL Server",        "datastore"),
    "system.data.sqlclient": ("SQL Server",    "datastore"),
    "oracledb":          ("Oracle",            "datastore"),
    "cx_oracle":         ("Oracle",            "datastore"),
    # Cache / key-value
    "redis":             ("Redis",             "cache"),
    "ioredis":           ("Redis",             "cache"),
    "memcached":         ("Memcached",         "cache"),
    # Column / search / time-series
    "cassandra-driver":  ("Cassandra",         "datastore"),
    "cassandra":         ("Cassandra",         "datastore"),
    "elasticsearch":     ("Elasticsearch",     "datastore"),
    "influxdb":          ("InfluxDB",          "datastore"),
    # ORMs without a known concrete engine
    "sequelize":         ("Relational Database", "datastore"),
    "typeorm":           ("Relational Database", "datastore"),
    "knex":              ("Relational Database", "datastore"),
    "sqlalchemy":        ("Relational Database", "datastore"),
    "hibernate":         ("Relational Database", "datastore"),
    "jakarta.persistence": ("Relational Database", "datastore"),
    "javax.persistence": ("Relational Database", "datastore"),
    "prisma":            ("Database",          "datastore"),
    "entityframework":   ("SQL Server",        "datastore"),
    "efcore":            ("SQL Server",        "datastore"),
    # Message queues / brokers
    "kafkajs":           ("Kafka",             "queue"),
    "amqplib":           ("RabbitMQ",          "queue"),
    "bullmq":            ("Redis Queue",       "queue"),
    "pika":              ("RabbitMQ",          "queue"),
    "celery":            ("Celery Queue",      "queue"),
}

# ── Third-party service SDK catalog: token -> (label, kind, verb) ────────────
# Only catalog matches become external nodes — conservative to avoid hallucinations.
_SERVICE_SDKS: dict[str, tuple[str, str, str]] = {
    "stripe":            ("Stripe",       "payment", "processes payments via"),
    "cloudinary":        ("Cloudinary",   "media",   "stores media via"),
    "@sendgrid/mail":    ("SendGrid",     "email",   "sends email via"),
    "sendgrid":          ("SendGrid",     "email",   "sends email via"),
    "nodemailer":        ("Email/SMTP",   "email",   "sends email via"),
    "twilio":            ("Twilio",       "sms",     "sends SMS via"),
    "boto3":             ("AWS",          "cloud",   "uses"),
    "@aws-sdk":          ("AWS",          "cloud",   "uses"),
    "aws-sdk":           ("AWS",          "cloud",   "uses"),
    "googleapis":        ("Google APIs",  "service", "calls"),
    "@googleapis":       ("Google APIs",  "service", "calls"),
    "@google-cloud":     ("Google Cloud", "cloud",   "uses"),
    "firebase":          ("Firebase",     "service", "uses"),
    "firebase-admin":    ("Firebase",     "service", "uses"),
    "openai":            ("OpenAI",       "llm",     "calls"),
    "@anthropic-ai":     ("Anthropic",    "llm",     "calls"),
    "paypal-rest-sdk":   ("PayPal",       "payment", "processes payments via"),
    "razorpay":          ("Razorpay",     "payment", "processes payments via"),
    "braintree":         ("Braintree",    "payment", "processes payments via"),
    "pusher":            ("Pusher",       "service", "uses"),
    "mailgun":           ("Mailgun",      "email",   "sends email via"),
    "algolia":           ("Algolia",      "search",  "searches via"),
    "algoliasearch":     ("Algolia",      "search",  "searches via"),
    "aws-sdk/client-sqs": ("Amazon SQS", "queue",   "queues via"),
    "@aws-sdk/client-sqs": ("Amazon SQS", "queue",  "queues via"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Deterministic node id: lowercase, non-alphanumeric runs → underscore."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "node"


def _skip_path(parts: tuple) -> bool:
    return any(p in SKIP_DIRS for p in parts)


def _find_manifests(repo_root: str) -> list[str]:
    """Return absolute paths to build manifests, skipping SKIP_DIRS."""
    root = Path(repo_root)
    manifests = []
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            continue
        rel_parts = candidate.relative_to(root).parts
        if _skip_path(rel_parts[:-1]):   # skip dirs only, not the file itself
            continue
        if candidate.name in _MANIFEST_NAMES:
            manifests.append(str(candidate.resolve()))
        elif any(candidate.match(g) for g in _MANIFEST_GLOBS):
            manifests.append(str(candidate.resolve()))
    return manifests


def _nearest_manifest_dir(file_path: str, manifest_dirs: list[str], repo_root_abs: str) -> str:
    """The deepest ancestor manifest directory for a given file."""
    try:
        fpath = Path(file_path).resolve()
    except Exception:
        return repo_root_abs
    best, best_depth = repo_root_abs, len(Path(repo_root_abs).parts)
    for mdir in manifest_dirs:
        mdirp = Path(mdir)
        try:
            fpath.relative_to(mdirp)   # raises ValueError if not an ancestor
            depth = len(mdirp.parts)
            if depth > best_depth:
                best, best_depth = str(mdirp), depth
        except ValueError:
            continue
    return best


def _classify_unit(frameworks: list, has_routes: bool, has_entry: bool) -> str:
    """Assign a container kind from framework evidence + runtime signals."""
    fw_set = set(frameworks)
    if fw_set & _FULLSTACK_FRAMEWORKS:
        return "web_app"
    if fw_set & _UI_FRAMEWORKS and not (fw_set & _BACKEND_FRAMEWORKS):
        return "web_app"
    if (fw_set & _BACKEND_FRAMEWORKS) or has_routes:
        return "service"
    if has_entry:
        return "worker"
    return "service"   # safe fallback


def _scan_db_engines(imports_flat: list[str]) -> dict[str, tuple[str, str]]:
    """Match imports against _DB_ENGINES; return {label -> (label, kind)}."""
    found: dict[str, tuple[str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind) in _DB_ENGINES.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind)
    return found


def _scan_services(imports_flat: list[str]) -> dict[str, tuple[str, str, str]]:
    """Match imports against _SERVICE_SDKS; return {label -> (label, kind, verb)}."""
    found: dict[str, tuple[str, str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind, verb) in _SERVICE_SDKS.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind, verb)
    return found


def _container_label(dir_name: str, kind: str) -> str:
    """Generate a human-readable container label from the manifest directory name."""
    clean = dir_name.replace("-", " ").replace("_", " ").strip()
    if not clean or clean == ".":
        return "Application"
    title = " ".join(w.capitalize() for w in clean.split())
    if kind == "web_app":
        lower = title.lower()
        if lower in ("web", "frontend", "client", "ui", "spa"):
            return title
        return f"{title} Web App"
    if kind == "service":
        lower = title.lower()
        if lower in ("api", "backend", "server", "service"):
            return title
        return f"{title} API"
    if kind == "worker":
        return f"{title} Worker"
    return title


# ── Public API ────────────────────────────────────────────────────────────────

def build_container_model(rich_facts: dict, repo_root: str) -> dict:
    """Deterministically build the C4 container model from repository facts.

    Returns the same dict shape render_c4_combined expects, with a 'kind'
    field added to every node for shape-by-kind rendering (Change C).
    Text fields (system_purpose, descriptions, edge labels) are left blank
    for HLDEnrichmentAgent to fill in.
    """
    facts = runtime_facts(rich_facts, repo_root)
    repo_root_abs = str(Path(repo_root).resolve())

    # ── A1: discover deployable roots ────────────────────────────────────────
    manifest_files = _find_manifests(repo_root)
    manifest_dirs = sorted(
        {str(Path(m).parent.resolve()) for m in manifest_files},
        key=lambda d: len(d),    # shortest first → fallback wins ties
    )

    # Group every runtime file by its nearest manifest directory
    units: dict[str, dict] = {}
    for mid, f in facts.items():
        fp = f.get("file", "")
        mdir = _nearest_manifest_dir(fp, manifest_dirs, repo_root_abs)
        if mdir not in units:
            units[mdir] = {
                "files": [], "imports": [], "routes": [], "has_entry": False
            }
        units[mdir]["files"].append(f)
        units[mdir]["imports"].extend(f.get("imports", []))
        units[mdir]["routes"].extend(f.get("routes", []))
        if _is_entry(mid, f):
            units[mdir]["has_entry"] = True

    if not units:
        # Repo has no extractable runtime facts → emit a minimal placeholder
        units[repo_root_abs] = {
            "files": list(facts.values()),
            "imports": [],
            "routes": [],
            "has_entry": False,
        }

    # Flatten all imports for datastore / external-service scanning
    all_imports: list[str] = [imp for u in units.values() for imp in u["imports"]]

    # ── A2: datastores ────────────────────────────────────────────────────────
    db_from_classes = any(
        c.get("is_db_model")
        for f in facts.values()
        for c in f.get("classes", [])
    )
    db_found = _scan_db_engines(all_imports)
    if not db_found and db_from_classes:
        # ORM usage detected but driver not in catalog → generic node
        db_found["Relational Database"] = ("Relational Database", "datastore")

    datastore_nodes = [
        {"id": _slug(lbl), "label": lbl, "kind": kind, "tech": lbl, "description": ""}
        for lbl, (_, kind) in db_found.items()
    ]

    # ── A3: external systems ─────────────────────────────────────────────────
    svc_found = _scan_services(all_imports)
    external_nodes = []
    ext_verb: dict[str, str] = {}
    for label, (lbl, _kind, verb) in svc_found.items():
        nid = _slug(lbl)
        external_nodes.append(
            {"id": nid, "label": lbl, "kind": "external", "tech": _kind, "description": ""}
        )
        ext_verb[nid] = verb

    # ── A1 (continued): container nodes from deployable roots ────────────────
    container_nodes = []
    for mdir, udata in units.items():
        fws = detect_frameworks(udata["files"])
        has_routes = bool(udata["routes"])
        has_entry = udata["has_entry"]
        kind = _classify_unit(fws, has_routes, has_entry)

        rel = os.path.relpath(mdir, repo_root_abs).replace("\\", "/")
        dir_name = Path(rel).name if rel not in (".", "") else Path(repo_root_abs).name

        label = _container_label(dir_name, kind)
        nid = _slug(dir_name) or _slug(label)
        tech = ", ".join(fws) if fws else ""

        container_nodes.append({
            "id": nid,
            "label": label,
            "kind": kind,
            "tech": tech,
            "description": "",
            # Internal-only fields stripped before output
            "_has_routes": has_routes,
            "_mdir": mdir,
        })

    # ── A4: actors ────────────────────────────────────────────────────────────
    any_routes = any(c["_has_routes"] for c in container_nodes) or any(
        f.get("routes") for f in facts.values()
    )
    any_cli = any(
        u["has_entry"] and not u["routes"] for u in units.values()
    )

    actors: list[dict] = []
    if any_routes:
        actors.append(
            {"id": "user", "label": "User", "kind": "person", "description": "Client caller"}
        )
    elif any_cli:
        actors.append(
            {"id": "operator", "label": "Operator", "kind": "person", "description": "CLI operator"}
        )

    # ── A5: relationships ─────────────────────────────────────────────────────
    ctx_rels: list[dict] = []
    cont_rels: list[dict] = []

    # Actor → first web/service container with routes (or first container)
    if actors and container_nodes:
        actor_id = actors[0]["id"]
        target = next(
            (c for c in container_nodes if c["kind"] in ("web_app", "service") and c["_has_routes"]),
            container_nodes[0],
        )
        ctx_rels.append({"from": actor_id, "to": target["id"], "label": "uses"})

    # web_app → service when both exist
    web_apps = [c for c in container_nodes if c["kind"] == "web_app"]
    services = [c for c in container_nodes if c["kind"] == "service"]
    for wa in web_apps:
        for svc in services:
            cont_rels.append({"from": wa["id"], "to": svc["id"], "label": "calls API"})

    # all containers → every datastore
    for c in container_nodes:
        for ds in datastore_nodes:
            cont_rels.append({"from": c["id"], "to": ds["id"], "label": "reads/writes"})

    # all containers → every external service
    for c in container_nodes:
        for ext in external_nodes:
            verb = ext_verb.get(ext["id"], "uses")
            cont_rels.append({"from": c["id"], "to": ext["id"], "label": verb})

    # ── Strip internal-only fields before handing off to renderer ─────────────
    clean_containers = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in container_nodes
    ]

    return {
        "context": {
            "system_name": "",          # HLDEnrichmentAgent fills this
            "system_description": "",
            "architecture_style": "",
            "actors": actors,
            "external_systems": [
                {"id": e["id"], "label": e["label"], "kind": e["kind"], "description": e["description"]}
                for e in external_nodes
            ],
            "relationships": ctx_rels,
        },
        "containers": {
            "system_label": "",         # HLDEnrichmentAgent fills this
            "containers": clean_containers,
            "databases": datastore_nodes,
            "external_services": [
                {"id": e["id"], "label": e["label"], "kind": e["kind"], "description": e["description"]}
                for e in external_nodes
            ],
            "relationships": cont_rels,
        },
    }


def apply_enrichment(model: dict, enrichment: dict) -> dict:
    """Merge LLM text enrichment into the deterministic model (text-only, no structure).

    Applies: system_purpose -> system_name/system_label,
             descriptions{id: text} -> each node's description,
             edge_labels{from__to: text} -> relationship labels.
    Any id in enrichment not present in the model is silently discarded.
    """
    system_purpose = (enrichment.get("system_purpose") or "").strip()
    descriptions   = enrichment.get("descriptions") or {}
    edge_labels    = enrichment.get("edge_labels") or {}

    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    if system_purpose:
        ctx["system_name"]    = system_purpose
        cont["system_label"]  = system_purpose

    # Collect all known node ids for safety guard
    all_node_ids: set[str] = set()
    for lst in (
        ctx.get("actors", []),
        ctx.get("external_systems", []),
        cont.get("containers", []),
        cont.get("databases", []),
        cont.get("external_services", []),
    ):
        for node in (lst or []):
            if node.get("id"):
                all_node_ids.add(node["id"])

    def _apply_desc(nodes: list) -> None:
        for node in (nodes or []):
            nid = node.get("id", "")
            if nid in descriptions:
                node["description"] = descriptions[nid]

    _apply_desc(ctx.get("actors", []))
    _apply_desc(ctx.get("external_systems", []))
    _apply_desc(cont.get("containers", []))
    _apply_desc(cont.get("databases", []))
    _apply_desc(cont.get("external_services", []))

    def _apply_edge_labels(rels: list) -> None:
        for rel in (rels or []):
            key = f"{rel.get('from', '')}__{rel.get('to', '')}"
            if key in edge_labels:
                rel["label"] = edge_labels[key]

    _apply_edge_labels(ctx.get("relationships", []))
    _apply_edge_labels(cont.get("relationships", []))

    model["context"]    = ctx
    model["containers"] = cont
    return model
