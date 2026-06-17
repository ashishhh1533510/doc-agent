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
    # JVM / Spring import prefixes (matched as substrings against full FQCNs)
    "org.springframework.data": ("Relational Database", "datastore"),
    "org.mybatis":       ("Relational Database", "datastore"),
    "mybatis":           ("Relational Database", "datastore"),
    "org.hibernate":     ("Relational Database", "datastore"),
    "org.jooq":          ("Relational Database", "datastore"),
    "org.sqlite":        ("SQLite",            "datastore"),
    "org.xerial":        ("SQLite",            "datastore"),
    "org.postgresql":    ("PostgreSQL",        "datastore"),
    "com.mysql":         ("MySQL",             "datastore"),
    "com.zaxxer.hikari": ("Relational Database", "datastore"),
    "org.h2":            ("H2 Database",       "datastore"),
    "com.h2database":    ("H2 Database",       "datastore"),
    # Message queues / brokers
    "kafkajs":           ("Kafka",             "queue"),
    "amqplib":           ("RabbitMQ",          "queue"),
    "bullmq":            ("Redis Queue",       "queue"),
    "pika":              ("RabbitMQ",          "queue"),
    "celery":            ("Celery Queue",      "queue"),
    # JVM message queues
    "org.springframework.kafka": ("Kafka",     "queue"),
    "org.springframework.amqp":  ("RabbitMQ",  "queue"),
    "com.rabbitmq":      ("RabbitMQ",          "queue"),
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
    "anthropic":         ("Anthropic",    "llm",     "calls"),
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


# ── Manifest-coordinate fragment catalog ────────────────────────────────────
# Matched as case-insensitive substrings against Maven "groupId:artifactId"
# coordinates, npm package names, pip package names, NuGet package IDs, etc.
# This catches dependencies that never show up as source-level imports (e.g.
# JDBC drivers configured in pom.xml and loaded reflectively at runtime).
_MANIFEST_DB_FRAGMENTS: list[tuple[str, str, str]] = [
    # SQLite
    ("sqlite",         "SQLite",            "datastore"),
    ("xerial",         "SQLite",            "datastore"),
    # PostgreSQL
    ("postgresql",     "PostgreSQL",        "datastore"),
    ("npgsql",         "PostgreSQL",        "datastore"),
    ("pg",             "PostgreSQL",        "datastore"),   # Node.js pg driver (exact/prefix matched)
    # MySQL
    ("mysql",          "MySQL",             "datastore"),
    # SQL Server
    ("sqlserver",      "SQL Server",        "datastore"),
    ("mssql",          "SQL Server",        "datastore"),
    ("tedious",        "SQL Server",        "datastore"),
    # MongoDB
    ("mongodb",        "MongoDB",           "datastore"),
    ("mongoose",       "MongoDB",           "datastore"),
    # Redis
    ("redis",          "Redis",             "cache"),
    ("ioredis",        "Redis",             "cache"),
    # Kafka
    ("kafka",          "Kafka",             "queue"),
    # RabbitMQ
    ("rabbitmq",       "RabbitMQ",          "queue"),
    ("amqp",           "RabbitMQ",          "queue"),
    # Generic ORM / JPA / MyBatis / H2
    ("mybatis",        "Relational Database", "datastore"),
    ("hibernate",      "Relational Database", "datastore"),
    ("spring-data",    "Relational Database", "datastore"),
    ("spring.data",    "Relational Database", "datastore"),
    ("h2database",     "H2 Database",       "datastore"),
    (":h2",            "H2 Database",       "datastore"),
    ("jooq",           "Relational Database", "datastore"),
    ("liquibase",      "Relational Database", "datastore"),
    ("flyway",         "Relational Database", "datastore"),
    ("prisma",         "Database",          "datastore"),
    ("typeorm",        "Relational Database", "datastore"),
    ("sequelize",      "Relational Database", "datastore"),
    ("sqlalchemy",     "Relational Database", "datastore"),
    ("efcore",         "SQL Server",        "datastore"),
    ("entityframework","SQL Server",        "datastore"),
    # Elasticsearch
    ("elasticsearch",  "Elasticsearch",     "datastore"),
]

_MANIFEST_SVC_FRAGMENTS: list[tuple[str, str, str, str]] = [
    ("stripe",         "Stripe",      "payment", "processes payments via"),
    ("sendgrid",       "SendGrid",    "email",   "sends email via"),
    ("nodemailer",     "Email/SMTP",  "email",   "sends email via"),
    ("twilio",         "Twilio",      "sms",     "sends SMS via"),
    ("cloudinary",     "Cloudinary",  "media",   "stores media via"),
    ("boto3",          "AWS",         "cloud",   "uses"),
    ("aws-sdk",        "AWS",         "cloud",   "uses"),
    ("amazonaws",      "AWS",         "cloud",   "uses"),
    ("spring-cloud-aws", "AWS",       "cloud",   "uses"),
    ("googleapis",     "Google APIs", "service", "calls"),
    ("google-cloud",   "Google Cloud","cloud",   "uses"),
    ("firebase",       "Firebase",    "service", "uses"),
    ("openai",         "OpenAI",      "llm",     "calls"),
    ("anthropic",      "Anthropic",   "llm",     "calls"),
    ("paypal",         "PayPal",      "payment", "processes payments via"),
    ("razorpay",       "Razorpay",    "payment", "processes payments via"),
    ("algolia",        "Algolia",     "search",  "searches via"),
    ("pusher",         "Pusher",      "service", "uses"),
    ("mailgun",        "Mailgun",     "email",   "sends email via"),
]


def _manifest_matches(fragment: str, lowered_deps: list[str]) -> bool:
    """True if `fragment` plausibly identifies a dependency in `lowered_deps`.

    Short tokens (≤ 4 chars, e.g. "pg", "h2") use exact-or-prefix matching to
    avoid false positives ("pg" must not match "plugin" or "page").
    Longer tokens use substring matching (reliable because they're specific
    enough not to collide accidentally).
    """
    if len(fragment) <= 4:
        return any(
            dep == fragment
            or dep.startswith(fragment + "/")
            or dep.startswith(fragment + "-")
            or dep.startswith(fragment + ".")
            for dep in lowered_deps
        )
    return any(fragment in dep for dep in lowered_deps)


def _scan_db_engines(
    imports_flat: list[str],
    manifest_deps: list[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Match imports (and optionally manifest dependencies) against _DB_ENGINES.

    Returns {label -> (label, kind)}.
    manifest_deps are matched as case-insensitive substrings against the
    _MANIFEST_DB_FRAGMENTS catalog — catching drivers loaded reflectively at
    runtime and never appearing as source-level imports (e.g. sqlite-jdbc in pom.xml).
    """
    found: dict[str, tuple[str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind) in _DB_ENGINES.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind)

    # Also scan manifest dependency coordinates
    if manifest_deps:
        mdeps = [d.lower() for d in manifest_deps]
        for fragment, label, kind in _MANIFEST_DB_FRAGMENTS:
            if label in found:
                continue
            fl = fragment.lower()
            if _manifest_matches(fl, mdeps):
                found[label] = (label, kind)

    return found


def _scan_services(
    imports_flat: list[str],
    manifest_deps: list[str] | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Match imports (and optionally manifest dependencies) against _SERVICE_SDKS.

    Returns {label -> (label, kind, verb)}.
    """
    found: dict[str, tuple[str, str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind, verb) in _SERVICE_SDKS.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind, verb)

    # Also scan manifest dependency coordinates
    if manifest_deps:
        mdeps = [d.lower() for d in manifest_deps]
        for fragment, label, kind, verb in _MANIFEST_SVC_FRAGMENTS:
            if label in found:
                continue
            fl = fragment.lower()
            if _manifest_matches(fl, mdeps):
                found[label] = (label, kind, verb)

    return found


# ── Non-deployable module segments ───────────────────────────────────────────
# Manifest dirs whose repo-relative path contains any of these segments are
# build-tooling / test scaffolding, not deployable containers.
_NON_DEPLOYABLE_SEGMENTS = frozenset({
    "test", "tests", "test-suite", "testsuite", "integration-test",
    "bom", "tools", "buildsrc",
})


def _is_non_deployable(mdir: str, repo_root_abs: str) -> bool:
    """True if mdir is a test/build-tooling module that should not become a container."""
    try:
        rel = os.path.relpath(mdir, repo_root_abs).replace("\\", "/")
        parts = {p.lower() for p in rel.replace("\\", "/").split("/")}
        return bool(parts & _NON_DEPLOYABLE_SEGMENTS)
    except Exception:
        return False


def _consolidate_datastores(db_found: dict) -> dict:
    """Collapse multiple relational engines to a single node.

    Many repos list PostgreSQL, MySQL, H2, etc. as *optional* SQL dialects — they
    are not all running at once. Emitting N relational nodes creates an N×M edge
    mesh that mermaid renders as a flat strip. Rules:
      - If exactly one concrete relational engine is detected (ignoring the generic
        "Relational Database" placeholder and the embedded/test "H2 Database"),
        keep that engine's name.
      - If zero or multiple concrete engines are detected, emit a single
        "Relational Database" node.
      - Non-relational stores (MongoDB, Elasticsearch, InfluxDB, Cassandra) and
        caches/queues (Redis, Kafka, RabbitMQ) are left untouched.
    """
    _RELATIONAL = frozenset({
        "PostgreSQL", "MySQL", "SQL Server", "Oracle",
        "H2 Database", "Relational Database",
    })
    relational = {lbl: v for lbl, v in db_found.items() if lbl in _RELATIONAL}
    non_relational = {lbl: v for lbl, v in db_found.items() if lbl not in _RELATIONAL}

    if not relational:
        return db_found

    # concrete = known engines that are not the generic fallback or embedded H2
    concrete = [lbl for lbl in relational if lbl not in ("Relational Database", "H2 Database")]
    if len(concrete) == 1:
        chosen = concrete[0]
    else:
        chosen = "Relational Database"

    return {**non_relational, chosen: (chosen, "datastore")}


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

def build_container_model(
    rich_facts: dict,
    repo_root: str,
    repo_name: str | None = None,
    manifests: dict | None = None,
) -> dict:
    """Deterministically build the C4 container model from repository facts.

    Args:
        rich_facts: output of extract_rich_from_directory
        repo_root:  local directory being analysed (may be a temp clone)
        repo_name:  display name derived from the original URL/path *before*
                    it became a temp dir (e.g. "spring-boot-realworld-example-app").
                    Used as the naming fallback when the manifest has no project_name
                    and the repo_root basename is a generated temp dir name.
        manifests:  output of parse_all_manifests(repo_root) — pre-computed so
                    the caller does not repeat the manifest walk.  If None the
                    naming fallback is still available via repo_name.

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

    # Group every runtime file by its nearest manifest directory,
    # skipping test / build-tooling modules.
    units: dict[str, dict] = {}
    for mid, f in facts.items():
        fp = f.get("file", "")
        mdir = _nearest_manifest_dir(fp, manifest_dirs, repo_root_abs)
        if _is_non_deployable(mdir, repo_root_abs):
            continue
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

    # Flatten manifest dependencies for manifest-coordinate scanning
    all_manifest_deps: list[str] = []
    if manifests:
        for parsed in manifests.values():
            all_manifest_deps.extend(parsed.get("dependencies") or [])

    # ── A2: datastores ────────────────────────────────────────────────────────
    db_from_classes = any(
        c.get("is_db_model")
        for f in facts.values()
        for c in f.get("classes", [])
    )
    db_found = _scan_db_engines(all_imports, manifest_deps=all_manifest_deps)
    if not db_found and db_from_classes:
        # ORM usage detected but driver not in catalog → generic node
        db_found["Relational Database"] = ("Relational Database", "datastore")
    # Collapse multiple relational engines to one node (avoids N×M edge hairball)
    db_found = _consolidate_datastores(db_found)

    datastore_nodes = [
        {"id": _slug(lbl), "label": lbl, "kind": kind, "tech": lbl, "description": ""}
        for lbl, (_, kind) in db_found.items()
    ]

    # ── A3: external systems ─────────────────────────────────────────────────
    svc_found = _scan_services(all_imports, manifest_deps=all_manifest_deps)
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

        # Naming priority:
        #   1. manifest project_name for this manifest dir
        #   2. repo_name (from URL/path before any temp-clone renaming)
        #   3. dir_name (current behaviour — may be a temp dir on deployed path)
        display_name = dir_name
        if manifests and mdir in manifests:
            mn = (manifests[mdir].get("project_name") or "").strip()
            if mn:
                display_name = mn
        if display_name == dir_name and repo_name and rel in (".", ""):
            # Only use repo_name as fallback for the root unit (single-module repos)
            display_name = repo_name

        label = _container_label(display_name, kind)
        nid = _slug(display_name) or _slug(label)
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

    # ── A3b: defensive container cap ─────────────────────────────────────────
    # If fixes 1 & 2 still leave too many nodes (pathological monorepo), keep
    # the 8 most architecturally relevant ones (most files + routes first).
    _CONTAINER_CAP = 8
    if len(container_nodes) > _CONTAINER_CAP:
        container_nodes.sort(
            key=lambda c: (c["_has_routes"], len(units.get(c["_mdir"], {}).get("files", []))),
            reverse=True,
        )
        container_nodes = container_nodes[:_CONTAINER_CAP]

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
