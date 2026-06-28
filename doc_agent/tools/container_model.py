"""
Deterministic container-level topology builder (v3 grounded-hybrid redesign).

Public API:
  build_candidate_model(rich_facts, repo_root, repo_name, manifests, orchestration) -> dict
      Strict deployment-boundary gate with evidence fusion: language-agnostic
      discovery from Dockerfiles + compose + k8s + deployable manifests.
      Returns a candidate model (NODES ONLY — no relationships) carrying per-unit
      evidence for the LLM grounded architect and private fields for the
      communication-graph stage.

  discover_orchestration(repo_root) -> dict
      Parse docker-compose (root + one level deep) / k8s (recursive, by kind:)
      / Procfile to find orchestrated deploy units.  k8s workloads carry
      env_refs for the communication-graph stage.

  merge_orchestration(model, orchestration) -> dict
      Augment the candidate model with compose/k8s NODES (containers, datastores).
      NODES ONLY — no relationships.  No-op when no compose/k8s/Procfile exists.

  infer_communication_graph(model, rich_facts, orchestration) -> dict
      Dedicated edge-inference stage that runs after all nodes are discovered.
      Fuses k8s env-var refs, compose depends_on/links, code-import infra
      ownership, and web→service structural edges into one coherent graph.

  infer_entrypoint(model) -> dict
      Infer and wire the user-facing entrypoint after the communication graph
      is built.  Adds 'user' actor if absent and wires actor→entrypoint edge.
      Upgrades 'operator' → 'user' when a web-facing container is present.

  assign_architecture_layers(model) -> dict
      Tag every container node with a deterministic 'layer' field
      (presentation | gateway | application | worker | data | messaging).
      Call after enforce_c4_levels, before render.

  validate_model(model) -> dict
      Score the model and return deterministic findings (fail/warn rules)
      before rendering. Returns {score, findings, passed}.

  apply_grounding(candidate_model, llm_model) -> dict
      Discard any LLM-returned node id absent from the candidate set.

  apply_enrichment(model, enrichment) -> dict
      Merge text-only enrichment (system_purpose, descriptions, edge_labels).

  enforce_c4_levels(model) -> dict
      Remove nodes whose 'kind' is not a valid C4 kind.

  build_container_model(...)
      Kept for backward-compatibility; delegates to build_candidate_model.

Node kind taxonomy (complete, fixed set):
  person · web_app · service · worker · datastore · cache · queue · external
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Admission is a deterministic boolean: declared deployment boundary AND not auxiliary.
# Code-structure signals (routes, frameworks) are NOT used for admission —
# they are classification signals that belong to _classify_unit, not the gate.

from doc_agent.tools.architecture_model import runtime_facts, _is_entry, _NONRUNTIME_DIR
from doc_agent.tools.import_graph import detect_frameworks
from doc_agent.tools.language_detector import SKIP_DIRS

# ── UI vs backend framework sets ──────────────────────────────────────────────
_UI_FRAMEWORKS = {
    "react", "nextjs", "vue", "angular", "svelte", "sveltekit", "remix",
}
_FULLSTACK_FRAMEWORKS = {"nextjs", "remix", "sveltekit"}
_BACKEND_FRAMEWORKS = {
    "express", "fastify", "nestjs", "spring", "aspnetcore",
    "fastapi", "flask", "django", "starlette", "tornado",
}
_LONG_RUNNING_IMPORTS = {
    # Queue consumers / schedulers
    "kafka", "rabbitmq", "amqp", "celery", "bullmq", "sidekiq",
    "org.springframework.kafka", "org.springframework.amqp",
    "activemq", "org.apache.activemq",
    # Scheduler / cron libs
    "apscheduler", "schedule", "quartz", "spring-batch",
    # Server / UI frameworks (subset — for non-route workers)
    "asyncio", "aiohttp", "tornado",
}

# ── Build manifest file names / globs ─────────────────────────────────────────
_MANIFEST_NAMES = frozenset({
    "package.json", "pom.xml", "build.gradle", "build.gradle.kts",
    "go.mod", "pyproject.toml", "setup.py", "Gemfile",
})
_MANIFEST_GLOBS = ("*.csproj", "*.fsproj")

# ── DB engine catalog: import-token -> (display label, kind) ─────────────────
_DB_ENGINES: dict[str, tuple[str, str]] = {
    "mongoose":          ("MongoDB",           "datastore"),
    "mongodb":           ("MongoDB",           "datastore"),
    "pymongo":           ("MongoDB",           "datastore"),
    "motor":             ("MongoDB",           "datastore"),
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
    "redis":             ("Redis",             "cache"),
    "ioredis":           ("Redis",             "cache"),
    "memcached":         ("Memcached",         "cache"),
    "cassandra-driver":  ("Cassandra",         "datastore"),
    "cassandra":         ("Cassandra",         "datastore"),
    "elasticsearch":     ("Elasticsearch",     "datastore"),
    "influxdb":          ("InfluxDB",          "datastore"),
    "@aws-sdk/client-dynamodb": ("DynamoDB",   "datastore"),
    "aws-sdk/client-dynamodb":  ("DynamoDB",   "datastore"),
    "pynamodb":          ("DynamoDB",          "datastore"),
    "neo4j":             ("Neo4j",             "datastore"),
    "org.neo4j":         ("Neo4j",             "datastore"),
    "clickhouse":        ("ClickHouse",        "datastore"),
    "clickhouse-driver": ("ClickHouse",        "datastore"),
    "clickhouse-connect":("ClickHouse",        "datastore"),
    "couchbase":         ("Couchbase",         "datastore"),
    "com.couchbase":     ("Couchbase",         "datastore"),
    "hazelcast":         ("Hazelcast",         "cache"),
    "com.hazelcast":     ("Hazelcast",         "cache"),
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
    "kafka-python":      ("Kafka",             "queue"),
    "confluent-kafka":   ("Kafka",             "queue"),
    "amqplib":           ("RabbitMQ",          "queue"),
    "bullmq":            ("Redis Queue",       "queue"),
    "pika":              ("RabbitMQ",          "queue"),
    "celery":            ("Celery Queue",      "queue"),
    "org.springframework.kafka": ("Kafka",     "queue"),
    "org.springframework.amqp":  ("RabbitMQ",  "queue"),
    "com.rabbitmq":      ("RabbitMQ",          "queue"),
    "aio-pika":          ("RabbitMQ",          "queue"),
    "nats":              ("NATS",              "queue"),
    "io.nats":           ("NATS",              "queue"),
    "pulsar":            ("Pulsar",            "queue"),
    "org.apache.pulsar": ("Pulsar",            "queue"),
    "@aws-sdk/client-sns": ("Amazon SNS",      "queue"),
    "aws-sdk/client-sns":  ("Amazon SNS",      "queue"),
    "@google-cloud/pubsub": ("Cloud Pub/Sub",  "queue"),
    "google-cloud-pubsub":  ("Cloud Pub/Sub",  "queue"),
    "com.google.cloud.pubsub": ("Cloud Pub/Sub", "queue"),
    # ActiveMQ / JMS
    "activemq":                      ("ActiveMQ",  "queue"),
    "org.apache.activemq":           ("ActiveMQ",  "queue"),
    "org.apache.activemq.artemis":   ("ActiveMQ Artemis", "queue"),
    "artemis":                       ("ActiveMQ Artemis", "queue"),
    "javax.jms":                     ("JMS Broker", "queue"),
    "jakarta.jms":                   ("JMS Broker", "queue"),
    "org.springframework.jms":       ("JMS Broker", "queue"),
}

# ── Third-party service SDK catalog ───────────────────────────────────────────
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
    "eureka":                              ("Eureka",    "registry", "registers with"),
    "spring-cloud-starter-netflix-eureka": ("Eureka",    "registry", "registers with"),
    "org.springframework.cloud.netflix.eureka": ("Eureka", "registry", "registers with"),
    "consul":                              ("Consul",    "registry", "registers with"),
    "spring-cloud-starter-consul":         ("Consul",    "registry", "registers with"),
    "zookeeper":                           ("Zookeeper", "registry", "registers with"),
    "org.apache.zookeeper":                ("Zookeeper", "registry", "registers with"),
    "@azure":            ("Azure",        "cloud",   "uses"),
    "azure-storage-blob": ("Azure",       "cloud",   "uses"),
    "com.azure":         ("Azure",        "cloud",   "uses"),
    "auth0":             ("Auth0",        "auth",    "authenticates via"),
    "@auth0":            ("Auth0",        "auth",    "authenticates via"),
    "okta":              ("Okta",         "auth",    "authenticates via"),
    "keycloak":          ("Keycloak",     "auth",    "authenticates via"),
    "org.keycloak":      ("Keycloak",     "auth",    "authenticates via"),
}

# ── Manifest-coordinate fragment catalog ──────────────────────────────────────
_MANIFEST_DB_FRAGMENTS: list[tuple[str, str, str]] = [
    ("sqlite",         "SQLite",            "datastore"),
    ("xerial",         "SQLite",            "datastore"),
    ("postgresql",     "PostgreSQL",        "datastore"),
    ("npgsql",         "PostgreSQL",        "datastore"),
    ("pg",             "PostgreSQL",        "datastore"),
    ("mysql",          "MySQL",             "datastore"),
    ("sqlserver",      "SQL Server",        "datastore"),
    ("mssql",          "SQL Server",        "datastore"),
    ("tedious",        "SQL Server",        "datastore"),
    ("mongodb",        "MongoDB",           "datastore"),
    ("mongoose",       "MongoDB",           "datastore"),
    ("redis",          "Redis",             "cache"),
    ("ioredis",        "Redis",             "cache"),
    ("kafka",          "Kafka",             "queue"),
    ("rabbitmq",       "RabbitMQ",          "queue"),
    ("amqp",           "RabbitMQ",          "queue"),
    ("mybatis",        "Relational Database", "datastore"),
    ("hibernate",      "Relational Database", "datastore"),
    # Spring Data NoSQL variants MUST precede the generic spring-data fragment so
    # they are not mislabeled "Relational Database" (list is matched in order).
    ("data-mongodb",     "MongoDB",         "datastore"),
    ("data-redis",       "Redis",           "cache"),
    ("data-elasticsearch", "Elasticsearch", "datastore"),
    ("data-cassandra",   "Cassandra",       "datastore"),
    ("data-couchbase",   "Couchbase",       "datastore"),
    ("data-neo4j",       "Neo4j",           "datastore"),
    ("spring-data",    "Relational Database", "datastore"),
    ("spring.data",    "Relational Database", "datastore"),
    ("dynamodb",       "DynamoDB",          "datastore"),
    ("neo4j",          "Neo4j",             "datastore"),
    ("clickhouse",     "ClickHouse",        "datastore"),
    ("couchbase",      "Couchbase",         "datastore"),
    ("cassandra",      "Cassandra",         "datastore"),
    ("hazelcast",      "Hazelcast",         "cache"),
    ("nats",           "NATS",              "queue"),
    ("pulsar",         "Pulsar",            "queue"),
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
    ("elasticsearch",  "Elasticsearch",     "datastore"),
    ("activemq",       "ActiveMQ",          "queue"),
    ("artemis",        "ActiveMQ Artemis",  "queue"),
    ("javax.jms",      "JMS Broker",        "queue"),
    ("jakarta.jms",    "JMS Broker",        "queue"),
    ("spring-jms",     "JMS Broker",        "queue"),
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
    ("eureka",         "Eureka",      "registry", "registers with"),
    ("consul",         "Consul",      "registry", "registers with"),
    ("zookeeper",      "Zookeeper",   "registry", "registers with"),
    ("azure",          "Azure",       "cloud",   "uses"),
    ("auth0",          "Auth0",       "auth",    "authenticates via"),
    ("okta",           "Okta",        "auth",    "authenticates via"),
    ("keycloak",       "Keycloak",    "auth",    "authenticates via"),
]

# ── Auxiliary Dockerfile suffix patterns (operational, not architectural) ──────
_AUX_DOCKERFILE_SUFFIXES = re.compile(
    r"\.(migrat\w*|seed\w*|init\w*|test\w*|dev\w*|ci\w*|tool\w*|admin\w*|local\w*)$",
    re.I,
)

# ── Auxiliary compose service names ────────────────────────────────────────────
_AUX_COMPOSE_SERVICES = re.compile(
    r"^(migrat\w*|seed\w*|init\w*|test\w*|e2e\w*|tool\w*|adminer\w*|mailhog\w*|"
    r"pgadmin\w*|redis-?commander\w*|setup\w*|bootstrap\w*)$",
    re.I,
)

# ── Extended non-runtime path segments (beyond _NONRUNTIME_DIR) ────────────────
_AUX_PATH_EXTRA = re.compile(
    r"(^|/)(ops|deploy|infra|docs|documentation|maintenance|devops|"
    r"kubernetes|k8s|helm|charts?|ansible|terraform)(/|$)", re.I,
)

# ── Non-deployable test/tooling module segments ────────────────────────────────
_NON_DEPLOYABLE_SEGMENTS = frozenset({
    "test", "tests", "test-suite", "testsuite", "integration-test",
    "bom", "tools", "buildsrc",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Deterministic node id: lowercase, non-alphanumeric runs → underscore."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "node"


def _has_dockerfile(dir_path: str) -> bool:
    """True if a plain Dockerfile (NOT a suffixed operational variant) exists directly in dir_path."""
    d = Path(dir_path)
    return (d / "Dockerfile").exists()


def _has_any_dockerfile(dir_path: str) -> bool:
    """True if any Dockerfile (including Dockerfile.*) exists in dir_path."""
    d = Path(dir_path)
    if (d / "Dockerfile").exists():
        return True
    return any(d.glob("Dockerfile.*"))


def _dockerfile_is_auxiliary(dir_path: str) -> bool:
    """True if the only Dockerfiles in dir_path are operational-suffix variants."""
    d = Path(dir_path)
    if (d / "Dockerfile").exists():
        return False  # plain Dockerfile = architectural
    for df in d.glob("Dockerfile.*"):
        suffix = "." + df.name.split(".", 1)[1] if "." in df.name else ""
        if not _AUX_DOCKERFILE_SUFFIXES.search(suffix):
            return False  # non-auxiliary suffixed Dockerfile found
    return True  # only auxiliary variants


def _skip_path(parts: tuple) -> bool:
    return any(p in SKIP_DIRS for p in parts)


def _find_dockerfile_dirs(repo_root: str) -> list[str]:
    """Return absolute paths of all dirs containing a plain Dockerfile, respecting SKIP_DIRS."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "Dockerfile" in filenames:
            result.append(str(Path(dirpath).resolve()))
    return result


def _find_manifests(repo_root: str) -> list[str]:
    root = Path(repo_root)
    manifests = []
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            continue
        rel_parts = candidate.relative_to(root).parts
        if _skip_path(rel_parts[:-1]):
            continue
        if candidate.name in _MANIFEST_NAMES:
            manifests.append(str(candidate.resolve()))
        elif any(candidate.match(g) for g in _MANIFEST_GLOBS):
            manifests.append(str(candidate.resolve()))
    return manifests


def _nearest_manifest_dir(file_path: str, manifest_dirs: list[str], repo_root_abs: str) -> str:
    try:
        fpath = Path(file_path).resolve()
    except Exception:
        return repo_root_abs
    best, best_depth = repo_root_abs, len(Path(repo_root_abs).parts)
    for mdir in manifest_dirs:
        mdirp = Path(mdir)
        try:
            fpath.relative_to(mdirp)
            depth = len(mdirp.parts)
            if depth > best_depth:
                best, best_depth = str(mdirp), depth
        except ValueError:
            continue
    return best


def _classify_unit(frameworks: list, has_routes: bool, has_entry: bool) -> str:
    fw_set = set(frameworks)
    if fw_set & _FULLSTACK_FRAMEWORKS:
        return "web_app"
    if fw_set & _UI_FRAMEWORKS and not (fw_set & _BACKEND_FRAMEWORKS):
        return "web_app"
    if (fw_set & _BACKEND_FRAMEWORKS) or has_routes:
        return "service"
    if has_entry:
        return "worker"
    return "service"


_WORKER_NAME_RE = re.compile(
    r"(load[-_]?gen|benchmark|stress[-_]?test|[-_]worker|worker[-_]|consumer|"
    r"scheduler|cron|batch|sidekiq|celery)",
    re.I,
)

_ENTRYPOINT_NAME_RE = re.compile(
    r"(frontend|api[-_]?gateway|gateway|bff|web[-_]?app|web[-_]?ui|spa|edge[-_]?proxy)",
    re.I,
)

_GATEWAY_NAME_RE = re.compile(
    r"(api[-_]?gateway|gateway|bff|edge[-_]?proxy|ingress)",
    re.I,
)
_PRESENTATION_NAME_RE = re.compile(
    r"(frontend|web[-_]?app|web[-_]?ui|webui|\bui\b|spa|portal|dashboard)",
    re.I,
)


def _classify_by_name(name: str, k8s_kind: str | None = None) -> str:
    """Classify a unit by name / k8s kind when code facts are unavailable."""
    if k8s_kind in ("Job", "CronJob"):
        return "worker"
    return "worker" if _WORKER_NAME_RE.search(name or "") else "service"


def _pick_entrypoint(containers: list[dict]) -> "dict | None":
    """Pick the user-facing entrypoint container by name/kind priority."""
    _PRIORITY = ["frontend", "api-gateway", "apigateway", "gateway", "bff",
                 "web-app", "webapp", "web-ui", "webui", "spa", "edge"]

    def _score(c: dict) -> int:
        text = (c.get("label", "") + " " + c.get("id", "")).lower()
        for i, kw in enumerate(_PRIORITY):
            if kw in text:
                return i
        if c.get("kind") == "web_app":
            return 20
        if _ENTRYPOINT_NAME_RE.search(text):
            return 15
        return 99

    candidates = [
        c for c in containers
        if _ENTRYPOINT_NAME_RE.search((c.get("label", "") + " " + c.get("id", "")).lower())
        or c.get("kind") == "web_app"
    ]
    if not candidates:
        return None
    return min(candidates, key=_score)


def _pick_primary_service(model: dict) -> "dict | None":
    """Pick the business-service spine anchor (the node the persistence hangs off).

    Deterministic priority — the application-layer container that owns persistence,
    so the spine actor→entry→[gateway]→service→datastore lands on a real backend:
      1. an `application`-layer service with a reads/writes edge to a datastore
      2. else the highest out-degree application-layer service
      3. else the highest out-degree non-entry container
      4. else the first container
    Returns None only when there are no containers.
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    if not containers:
        return None

    db_ids = {d["id"] for d in cont.get("databases", [])}
    rels = list(cont.get("relationships") or [])
    out_deg: dict[str, int] = {}
    writes_db: set[str] = set()
    for r in rels:
        f, t = r.get("from", ""), r.get("to", "")
        out_deg[f] = out_deg.get(f, 0) + 1
        if t in db_ids:
            writes_db.add(f)

    by_id = {c["id"]: c for c in containers}
    entry = _pick_entrypoint(containers)
    entry_id = entry["id"] if entry else None

    def _is_app(c: dict) -> bool:
        return _node_layer(c) == "application"

    # 1. application service that owns a datastore
    owners = [c for c in containers if _is_app(c) and c["id"] in writes_db]
    if owners:
        return max(owners, key=lambda c: (out_deg.get(c["id"], 0), c["id"]))

    # 2. highest out-degree application service
    apps = [c for c in containers if _is_app(c)]
    if apps:
        return max(apps, key=lambda c: (out_deg.get(c["id"], 0), c["id"]))

    # 3. highest out-degree non-entry container
    non_entry = [c for c in containers if c["id"] != entry_id]
    if non_entry:
        return max(non_entry, key=lambda c: (out_deg.get(c["id"], 0), c["id"]))

    # 4. fall back to the first container
    return containers[0]


def _is_non_deployable(mdir: str, repo_root_abs: str) -> bool:
    try:
        rel = os.path.relpath(mdir, repo_root_abs).replace("\\", "/")
        parts = {p.lower() for p in rel.split("/")}
        return bool(parts & _NON_DEPLOYABLE_SEGMENTS)
    except Exception:
        return False


def _is_auxiliary_path(dir_path: str, repo_root_abs: str) -> bool:
    """True if this directory path matches non-runtime / operational conventions."""
    try:
        rel = os.path.relpath(dir_path, repo_root_abs).replace("\\", "/")
        rel_slash = "/" + rel + "/"
        if _NONRUNTIME_DIR.search(rel_slash):
            return True
        if _AUX_PATH_EXTRA.search(rel_slash):
            return True
    except Exception:
        pass
    return False


def _manifest_matches(fragment: str, lowered_deps: list[str]) -> bool:
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
    found: dict[str, tuple[str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind) in _DB_ENGINES.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind)
    if manifest_deps:
        mdeps = [d.lower() for d in manifest_deps]
        for fragment, label, kind in _MANIFEST_DB_FRAGMENTS:
            if label in found:
                continue
            if _manifest_matches(fragment.lower(), mdeps):
                found[label] = (label, kind)
    return found


def _scan_services(
    imports_flat: list[str],
    manifest_deps: list[str] | None = None,
) -> dict[str, tuple[str, str, str]]:
    found: dict[str, tuple[str, str, str]] = {}
    lowered = [i.lower() for i in imports_flat]
    for token, (label, kind, verb) in _SERVICE_SDKS.items():
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + "/") or imp.startswith(tl + ".") for imp in lowered):
            if label not in found:
                found[label] = (label, kind, verb)
    if manifest_deps:
        mdeps = [d.lower() for d in manifest_deps]
        for fragment, label, kind, verb in _MANIFEST_SVC_FRAGMENTS:
            if label in found:
                continue
            if _manifest_matches(fragment.lower(), mdeps):
                found[label] = (label, kind, verb)
    return found


def _consolidate_datastores(db_found: dict) -> dict:
    """Collapse multiple relational engines to a single node."""
    _RELATIONAL = frozenset({
        "PostgreSQL", "MySQL", "SQL Server", "Oracle",
        "H2 Database", "Relational Database",
    })
    relational = {lbl: v for lbl, v in db_found.items() if lbl in _RELATIONAL}
    non_relational = {lbl: v for lbl, v in db_found.items() if lbl not in _RELATIONAL}
    if not relational:
        return db_found
    concrete = [lbl for lbl in relational if lbl not in ("Relational Database", "H2 Database")]
    chosen = concrete[0] if len(concrete) == 1 else "Relational Database"
    return {**non_relational, chosen: (chosen, "datastore")}


def _consolidate_queues(db_found: dict) -> dict:
    """Drop the generic JMS Broker when a concrete broker exists.

    JMS (javax.jms / jakarta.jms) is an API surface over a concrete broker
    (ActiveMQ, Kafka, RabbitMQ, etc.) — not a separate system node.
    Also merges ActiveMQ Artemis → ActiveMQ when both appear (they are the
    same product line).
    """
    _CONCRETE_BROKERS = frozenset({"Kafka", "RabbitMQ", "ActiveMQ", "ActiveMQ Artemis",
                                   "Redis Queue", "Celery Queue", "Amazon SQS",
                                   "NATS", "Pulsar", "Amazon SNS", "Cloud Pub/Sub"})
    has_concrete = any(lbl in _CONCRETE_BROKERS for lbl in db_found)

    result = {}
    for lbl, v in db_found.items():
        if lbl == "JMS Broker" and has_concrete:
            continue  # drop generic JMS when a concrete broker is present
        result[lbl] = v

    # Merge ActiveMQ Artemis into ActiveMQ (same product line, reduce noise)
    if "ActiveMQ" in result and "ActiveMQ Artemis" in result:
        del result["ActiveMQ Artemis"]

    return result


def _container_label(dir_name: str, kind: str) -> str:
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


# ── Orchestration discovery ───────────────────────────────────────────────────

_K8S_WORKLOAD_KINDS = frozenset({"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"})
_K8S_FILE_CAP = 400  # max YAML files scanned for k8s workloads

_ENV_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")


def _extract_k8s_env_refs(workload_doc: dict) -> list[str]:
    """Extract slugified host references from k8s workload env vars.

    Walks spec.template.spec.containers[*].env[*], skips valueFrom entries,
    strips schemes + ports, reduces cluster DNS to first label, returns deduped
    slug list.  These become the service-graph edge targets in infer_communication_graph.
    """
    refs: list[str] = []
    seen: set[str] = set()
    spec = (workload_doc.get("spec") or {})
    template_spec = (spec.get("template") or {}).get("spec") or {}
    for container in (template_spec.get("containers") or []):
        for env_entry in (container.get("env") or []):
            if not isinstance(env_entry, dict):
                continue
            if "valueFrom" in env_entry:
                continue
            val = str(env_entry.get("value") or "").strip()
            if not val:
                continue
            # strip scheme
            val = _ENV_SCHEME_RE.sub("", val)
            # take substring before first : or /
            host = re.split(r"[:/]", val)[0]
            if not host or len(host) < 2:
                continue
            # collapse cluster DNS: cartservice.ns.svc.cluster.local -> cartservice
            host = host.split(".")[0]
            slug = _slug(host)
            if slug and slug not in seen:
                seen.add(slug)
                refs.append(slug)
    return refs


def discover_orchestration(repo_root: str) -> dict:
    """Parse docker-compose / k8s / Procfile to find orchestrated deploy units.

    Returns:
      {
        "compose_services": [{"name": str, "image": str|None, "build_dir": str|None,
                               "is_auxiliary": bool, "profiles": [str],
                               "depends_on": list, "links": list}],
        "procfile_processes": [{"name": str, "command": str}],
        "has_k8s": bool,
        "k8s_workloads": [{"name": str, "kind": str, "env_refs": [str]}],
        "k8s_service_names": [str],
      }
    """
    root = Path(repo_root)
    result: dict = {
        "compose_services": [], "procfile_processes": [],
        "has_k8s": False, "k8s_service_names": [],
    }

    # ── docker-compose (root + one level deep) ────────────────────────────────
    compose_files: list[Path] = (
        list(root.glob("docker-compose*.yml"))
        + list(root.glob("docker-compose*.yaml"))
    )
    # one level deep: infra/, deploy/, etc.
    for child in root.iterdir():
        if child.is_dir() and child.name not in SKIP_DIRS:
            compose_files += list(child.glob("docker-compose*.yml"))
            compose_files += list(child.glob("docker-compose*.yaml"))

    seen_compose: set[str] = set()
    for cf in compose_files:
        cf_key = str(cf.resolve())
        if cf_key in seen_compose:
            continue
        seen_compose.add(cf_key)
        try:
            import yaml as _yaml
            data = _yaml.safe_load(cf.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:
            continue
        services = data.get("services") or {}
        for svc_name, svc_def in (services.items() if isinstance(services, dict) else []):
            if not isinstance(svc_def, dict):
                continue
            image = svc_def.get("image")
            build = svc_def.get("build")
            build_dir = None
            if isinstance(build, str):
                build_dir = str((cf.parent / build).resolve())
            elif isinstance(build, dict):
                ctx = build.get("context", ".")
                build_dir = str((cf.parent / ctx).resolve())
            profiles = svc_def.get("profiles") or []
            is_aux = bool(
                _AUX_COMPOSE_SERVICES.match(svc_name)
                or (profiles and all(p.lower() not in ("", "default", "production", "prod") for p in profiles))
            )
            result["compose_services"].append({
                "name": svc_name,
                "image": image,
                "build_dir": build_dir,
                "is_auxiliary": is_aux,
                "profiles": list(profiles),
                "depends_on": svc_def.get("depends_on") or [],
                "links": svc_def.get("links") or [],
            })

    # ── Procfile ──────────────────────────────────────────────────────────────
    procfile = root / "Procfile"
    if procfile.exists():
        for line in procfile.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            name, _, command = line.partition(":")
            result["procfile_processes"].append({"name": name.strip(), "command": command.strip()})

    # ── Kubernetes: recursive scan by kind: field ─────────────────────────────
    k8s_workloads: list[dict] = []
    scanned_k8s = 0
    capped = False

    try:
        import yaml as _yaml_k8s
    except ImportError:
        _yaml_k8s = None

    if _yaml_k8s:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            if capped:
                break
            # prune SKIP_DIRS in-place
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                    continue
                if scanned_k8s >= _K8S_FILE_CAP:
                    import logging as _log_cap
                    _log_cap.warning(
                        "discover_orchestration: k8s YAML scan capped at %d files", _K8S_FILE_CAP
                    )
                    capped = True
                    break
                scanned_k8s += 1
                fpath = Path(dirpath) / fname
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    docs = list(_yaml_k8s.safe_load_all(text))
                except Exception:
                    continue
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    k8s_kind = doc.get("kind", "")
                    if k8s_kind in _K8S_WORKLOAD_KINDS:
                        wl_name = (doc.get("metadata") or {}).get("name", "")
                        if wl_name:
                            result["has_k8s"] = True
                            env_refs = _extract_k8s_env_refs(doc)
                            k8s_workloads.append({"name": wl_name, "kind": k8s_kind, "env_refs": env_refs})
                    elif k8s_kind == "Service":
                        svc_name = (doc.get("metadata") or {}).get("name", "")
                        if svc_name:
                            result["k8s_service_names"].append(svc_name)

    result["k8s_workloads"] = k8s_workloads
    return result


# ── Auxiliary candidate filter ────────────────────────────────────────────────

def _unit_has_runtime_substance(udata: dict) -> bool:
    """True if this deploy unit has routes OR long-running framework/consumer imports."""
    if udata.get("routes"):
        return True
    imports_lower = [i.lower() for i in udata.get("imports", [])]
    for token in _LONG_RUNNING_IMPORTS:
        tl = token.lower()
        if any(imp == tl or imp.startswith(tl + ".") or imp.startswith(tl + "/") for imp in imports_lower):
            return True
    fws = detect_frameworks(udata.get("files", []))
    if set(fws) & (_BACKEND_FRAMEWORKS | _UI_FRAMEWORKS | _FULLSTACK_FRAMEWORKS):
        return True
    return False


def _is_runnable_by_convention(
    mdir: str, udata: dict, manifests: dict | None,
    manifest_dirs: list | None = None,
) -> bool:
    """True if a unit is an independently-runnable app *by convention*: it owns a
    build manifest at THIS dir AND has an entrypoint, HTTP routes, or a
    server/UI/fullstack framework. Deliberately conservative — an internal library
    subdir that happens to carry a manifest but has no entry/routes/framework is
    NOT runnable, so a true monolith with internal packages is never over-split."""
    # Manifest ownership: prefer the parsed manifests dict when supplied, but fall
    # back to the filesystem-derived manifest_dirs so callers that don't pre-parse
    # manifests (e.g. tests, lightweight callers) still resolve ownership correctly.
    owns_manifest = bool((manifests or {}).get(mdir)) or (mdir in set(manifest_dirs or []))
    if not owns_manifest:
        return False
    if udata.get("has_entry") or udata.get("routes"):
        return True
    fws = detect_frameworks(udata.get("files", []))
    return bool(set(fws) & (_BACKEND_FRAMEWORKS | _UI_FRAMEWORKS | _FULLSTACK_FRAMEWORKS))


def _multi_app_corroborated(rich_facts: dict) -> bool:
    """Structural second opinion on whether a repo is genuinely multi-app: >=2
    import-graph components each independently owning an entrypoint or routes.
    One dominant component owning all routes/entries => monolith with internal
    packages. rich_facts['components'] is the LIST from compute_components."""
    comps = rich_facts.get("components") or []
    runnable = [c for c in comps
                if c.get("has_routes") or c.get("has_main_entry") or c.get("has_ui")]
    return len(runnable) >= 2


def _mark_auxiliary(candidates: list[dict], repo_root_abs: str) -> None:
    """Mark each candidate is_auxiliary in-place.

    A candidate is auxiliary when ANY of:
    - path matches non-runtime / operational convention
    - only auxiliary-suffix Dockerfiles present (e.g. Dockerfile.migrate, Dockerfile.dev)
    - no runtime substance AND admitted only via a suffixed/auxiliary Dockerfile
      (plain Dockerfile is a declared deployment boundary — skip substance check)
    """
    for c in candidates:
        mdir = c["mdir"]
        # Path-based heuristic
        if _is_auxiliary_path(mdir, repo_root_abs):
            c["is_auxiliary"] = True
            continue
        # Dockerfile-suffix heuristic: if admitted via Dockerfile, check suffix
        if c.get("via_dockerfile") and _dockerfile_is_auxiliary(mdir):
            c["is_auxiliary"] = True
            continue
        # Substance heuristic: ONLY for dirs that have NO plain Dockerfile.
        # A plain Dockerfile is itself a declared deployment boundary — even if the
        # extractor produced no facts (e.g. a Go service), the dir is not auxiliary.
        if c.get("via_dockerfile") and not c.get("via_manifest"):
            if not _has_dockerfile(mdir):  # only suffixed Dockerfiles → check substance
                if not _unit_has_runtime_substance(c["udata"]):
                    c["is_auxiliary"] = True
                    continue
        c.setdefault("is_auxiliary", False)


# ── Per-container infra ownership ─────────────────────────────────────────────

def _build_ownership_edges(
    container_units: dict[str, dict],   # mdir -> udata
    container_id_map: dict[str, str],   # mdir -> node_id
    db_found: dict,                     # label -> (label, kind)
    svc_found: dict,                    # label -> (label, kind, verb)
    ext_verb: dict[str, str],           # node_id -> verb
    single_container: bool,
) -> list[dict]:
    """Build relationships where each container only links to infra IT imports.

    For single-container repos, all infra links to that one container.
    """
    rels: list[dict] = []
    seen: set[tuple] = set()

    def _add(from_id: str, to_id: str, label: str) -> None:
        k = (from_id, to_id)
        if k not in seen:
            seen.add(k)
            rels.append({"from": from_id, "to": to_id, "label": label})

    if single_container:
        # Single container: wire all infra to the one container
        cid = next(iter(container_id_map.values()))
        for label, (_, kind) in db_found.items():
            ds_id = _slug(label)
            verb = "reads/writes" if kind in ("datastore", "cache") else "publishes/consumes"
            _add(cid, ds_id, verb)
        for label, (_, _, verb) in svc_found.items():
            ext_id = _slug(label)
            _add(cid, ext_id, verb)
    else:
        # Multi-container: ownership edges only (this container's imports use the infra)
        for mdir, udata in container_units.items():
            cid = container_id_map.get(mdir)
            if not cid:
                continue
            unit_imports = udata.get("imports", [])
            c_db = _scan_db_engines(unit_imports)
            c_svc = _scan_services(unit_imports)

            for label in c_db:
                if label in db_found:
                    ds_id = _slug(label)
                    kind = db_found[label][1]
                    verb = "reads/writes" if kind in ("datastore", "cache") else "publishes/consumes"
                    _add(cid, ds_id, verb)
            for label in c_svc:
                if label in svc_found:
                    ext_id = _slug(svc_found[label][0])
                    verb = svc_found[label][2]
                    _add(cid, ext_id, verb)

    return rels


# ── Hard-budget simplifier ────────────────────────────────────────────────────

# High safety caps — readability is handled upstream by
# consolidate_containers_for_abstraction (folds large module sets into domain reps),
# so these caps only act as a last-resort guard against truly pathological models.
_CONTAINER_CAP = 64
_TOTAL_NODES_CAP = 96


def _apply_hard_budget(model: dict) -> dict:
    """Dedup edges and apply last-resort node caps.

    These caps only act as a last-resort guard against truly pathological models
    so renderers don't blow up. Dropped nodes (if any) are logged (no silent truncation).
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    databases = cont.get("databases", [])
    externals = cont.get("external_services", [])
    rels = cont.get("relationships", [])

    # Build edge-count index for prioritization (used only if caps are hit)
    edge_count: dict[str, int] = {}
    for r in rels:
        edge_count[r.get("from", "")] = edge_count.get(r.get("from", ""), 0) + 1
        edge_count[r.get("to", "")] = edge_count.get(r.get("to", ""), 0) + 1

    dropped: list[str] = []

    # Last-resort container cap (only triggers for pathological models)
    if len(containers) > _CONTAINER_CAP:
        containers.sort(key=lambda c: edge_count.get(c["id"], 0), reverse=True)
        import logging as _logging
        _logging.warning(
            "_apply_hard_budget: dropping %d containers over safety cap %d",
            len(containers) - _CONTAINER_CAP, _CONTAINER_CAP,
        )
        dropped += [c["label"] for c in containers[_CONTAINER_CAP:]]
        containers = containers[:_CONTAINER_CAP]

    kept_ids = {c["id"] for c in containers}

    # Remove rels referencing dropped containers
    rels = [r for r in rels if r.get("from") in kept_ids or r.get("to") in kept_ids]

    # Last-resort external cap
    total = len(containers) + len(databases)
    if total + len(externals) > _TOTAL_NODES_CAP:
        budget_ext = max(0, _TOTAL_NODES_CAP - total)
        externals.sort(key=lambda e: edge_count.get(e["id"], 0), reverse=True)
        import logging as _logging
        _logging.warning(
            "_apply_hard_budget: dropping %d externals over safety cap",
            len(externals) - budget_ext,
        )
        dropped += [e["label"] for e in externals[budget_ext:]]
        ext_ids_keep = {e["id"] for e in externals[:budget_ext]}
        externals = externals[:budget_ext]
        rels = [r for r in rels if r.get("to") not in ({e["id"] for e in externals} - ext_ids_keep)]

    # Purge relationships that reference removed node ids
    all_node_ids = kept_ids | {d["id"] for d in databases} | {e["id"] for e in externals}
    ctx_actor_ids = {a["id"] for a in model.get("context", {}).get("actors", [])}
    all_node_ids |= ctx_actor_ids
    rels = [
        r for r in rels
        if r.get("from") in all_node_ids and r.get("to") in all_node_ids
    ]

    # Dedup relationships
    seen_rels: set[tuple] = set()
    deduped_rels = []
    for r in rels:
        k = (r.get("from", ""), r.get("to", ""))
        if k not in seen_rels and k[0] != k[1]:
            seen_rels.add(k)
            deduped_rels.append(r)

    cont["containers"] = containers
    cont["databases"] = databases
    cont["external_services"] = externals
    cont["relationships"] = deduped_rels
    model["containers"] = cont
    model["_dropped_nodes"] = dropped
    return model


# ── Image-to-datastore classifier for compose/k8s ────────────────────────────

_IMAGE_DB_PATTERNS: list[tuple[str, str, str]] = [
    ("postgres",  "PostgreSQL", "datastore"),
    ("mysql",     "MySQL",      "datastore"),
    ("mariadb",   "MariaDB",    "datastore"),
    ("mongo",     "MongoDB",    "datastore"),
    ("redis",     "Redis",      "cache"),
    ("cassandra", "Cassandra",  "datastore"),
    ("elastic",   "Elasticsearch", "datastore"),
    ("opensearch","OpenSearch", "datastore"),
    ("kafka",     "Kafka",      "queue"),
    ("rabbitmq",  "RabbitMQ",   "queue"),
    ("zookeeper", "Zookeeper",  "datastore"),
    ("mssql",     "SQL Server", "datastore"),
    ("sqlserver", "SQL Server", "datastore"),
    ("sqlite",    "SQLite",     "datastore"),
    ("memcached", "Memcached",  "cache"),
    ("nats",      "NATS",       "queue"),
    ("pulsar",    "Pulsar",     "queue"),
    ("dynamodb",  "DynamoDB",   "datastore"),
    ("neo4j",     "Neo4j",      "datastore"),
    ("clickhouse","ClickHouse", "datastore"),
    ("couchbase", "Couchbase",  "datastore"),
    ("hazelcast", "Hazelcast",  "cache"),
]


def _image_to_datastore(image: str) -> tuple[str, str] | None:
    """Map a docker image name to (label, kind) if it's a well-known datastore."""
    img_low = (image or "").lower().split(":")[0].split("/")[-1]
    for pattern, label, kind in _IMAGE_DB_PATTERNS:
        if pattern in img_low:
            return label, kind
    return None


def merge_orchestration(model: dict, orchestration: dict) -> dict:
    """Augment candidate model with orchestration topology (compose/k8s).

    Treats orchestration as a high-confidence supplement to architectural
    discovery — never replaces it. For repos with no compose/k8s/Procfile,
    this is a deterministic no-op.

    Adds:
    - Container nodes for compose services / k8s workloads not already present.
      If the image matches a known datastore, routes to databases instead.
    - Relationships from compose depends_on / env service references.
    """
    import logging as _log
    cont = model.get("containers", {})
    ctx  = model.get("context", {})

    existing_ids: set[str] = {
        _slug(c["id"]) for c in cont.get("containers", [])
    } | {
        _slug(d["id"]) for d in cont.get("databases", [])
    }

    new_containers: list[dict] = []
    new_databases: list[dict] = []
    new_rels: list[dict] = []

    # ── Compose services (NODES ONLY — edges in infer_communication_graph) ───────
    for svc in orchestration.get("compose_services", []):
        if svc.get("is_auxiliary"):
            continue
        name = svc.get("name", "")
        if not name:
            continue
        nid = _slug(name)
        if nid in existing_ids:
            continue
        image = svc.get("image") or ""
        ds = _image_to_datastore(image)
        if ds:
            label, kind = ds
            if _slug(label) not in existing_ids:
                existing_ids.add(_slug(label))
                new_databases.append({
                    "id": _slug(label), "label": label, "kind": kind,
                    "tech": label, "description": "",
                })
        else:
            existing_ids.add(nid)
            kind = _classify_by_name(name)
            label = _container_label(name, kind)
            new_containers.append({
                "id": nid, "label": label, "kind": kind,
                "tech": "", "description": "",
            })

    # ── k8s workloads (NODES ONLY — edges in infer_communication_graph) ─────────
    for wl in orchestration.get("k8s_workloads", []):
        name = wl.get("name", "")
        if not name:
            continue
        nid = _slug(name)
        if nid in existing_ids:
            continue
        existing_ids.add(nid)
        kind = _classify_by_name(name, wl.get("kind"))
        label = _container_label(name, kind)
        new_containers.append({
            "id": nid, "label": label, "kind": kind,
            "tech": "", "description": "",
        })

    if new_containers or new_databases:
        cont["containers"] = cont.get("containers", []) + new_containers
        cont["databases"]  = cont.get("databases", []) + new_databases
        model["containers"] = cont

        # Extend candidate ids so apply_grounding doesn't strip the new nodes
        cids = set(model.get("_candidate_ids") or [])
        cids.update(c["id"] for c in new_containers)
        cids.update(d["id"] for d in new_databases)
        model["_candidate_ids"] = list(cids)

        _log.debug(
            "merge_orchestration: +%d containers, +%d databases (nodes only)",
            len(new_containers), len(new_databases),
        )

    model = _apply_hard_budget(model)
    return model


# ── Communication graph inference ────────────────────────────────────────────


def infer_communication_graph(model: dict, rich_facts: dict, orchestration: dict) -> dict:
    """Dedicated edge-inference stage — runs after all nodes are discovered.

    Fuses four deterministic sources into one coherent edge set:
      1. k8s env-var references (the primary inter-service signal for Go/polyglot repos)
      2. compose depends_on / links
      3. code-import infra ownership (databases, queues, external SDKs)
      4. web_app → service structural edges

    Only adds edges when BOTH endpoints already exist in the model — never invents
    nodes.  Consumes and cleans up the private _container_units / _container_id_map
    fields stored by build_candidate_model.
    """
    import logging as _log
    cont = model.get("containers", {})

    # Build the set of all known node slugs (fixed at entry — no new nodes here)
    all_container_ids = {c["id"] for c in cont.get("containers", [])}
    all_db_ids = {d["id"] for d in cont.get("databases", [])}
    all_ext_ids = {e["id"] for e in cont.get("external_services", [])}
    all_node_ids = (
        all_container_ids | all_db_ids | all_ext_ids
        | {a["id"] for a in model.get("context", {}).get("actors", [])}
    )

    new_rels: list[dict] = []
    seen: set[tuple] = set()

    def _add_edge(from_id: str, to_id: str, label: str) -> None:
        if from_id == to_id:
            return
        if from_id not in all_node_ids or to_id not in all_node_ids:
            return
        k = (from_id, to_id)
        if k not in seen:
            seen.add(k)
            new_rels.append({"from": from_id, "to": to_id, "label": label})

    orch = orchestration or {}

    # ── 1. k8s env-var references ─────────────────────────────────────────────
    k8s_svc_slugs = {_slug(n) for n in (orch.get("k8s_service_names") or [])}
    for wl in orch.get("k8s_workloads", []):
        wl_name = wl.get("name", "")
        if not wl_name:
            continue
        from_id = _slug(wl_name)
        for ref in (wl.get("env_refs") or []):
            if ref in all_db_ids:
                _add_edge(from_id, ref, "reads/writes")
            elif ref in all_container_ids:
                _add_edge(from_id, ref, "calls")
            elif ref in k8s_svc_slugs:
                # Service name that also corresponds to a container slug
                if ref in all_container_ids:
                    _add_edge(from_id, ref, "calls")

    # ── 2. compose depends_on / links ─────────────────────────────────────────
    for svc in orch.get("compose_services", []):
        if svc.get("is_auxiliary"):
            continue
        from_id = _slug(svc.get("name", ""))
        if not from_id:
            continue

        deps: list[str] = []
        raw_depends = svc.get("depends_on") or []
        if isinstance(raw_depends, dict):
            deps.extend(raw_depends.keys())
        elif isinstance(raw_depends, list):
            deps.extend(raw_depends)
        for lnk in (svc.get("links") or []):
            deps.append(lnk.split(":")[0] if ":" in lnk else lnk)

        for dep in deps:
            to_id = _slug(dep)
            if to_id in all_db_ids:
                _add_edge(from_id, to_id, "reads/writes")
            else:
                _add_edge(from_id, to_id, "calls")

    # ── 3. code-import infra ownership ────────────────────────────────────────
    container_units = model.pop("_container_units", {})
    container_id_map = model.pop("_container_id_map", {})
    db_found = model.pop("_db_found", {})
    svc_found = model.pop("_svc_found", {})
    ext_verb = model.pop("_ext_verb", {})
    single_container = model.pop("_single_container", False)

    if container_units and container_id_map:
        ownership_rels = _build_ownership_edges(
            container_units=container_units,
            container_id_map=container_id_map,
            db_found=db_found,
            svc_found=svc_found,
            ext_verb=ext_verb,
            single_container=single_container,
        )
        for r in ownership_rels:
            _add_edge(r["from"], r["to"], r["label"])

    # ── 4. web_app → backend front-door (single spine link, NOT a mesh) ───────
    # Connecting every web_app to every service builds a hairball that later edge
    # thinning shreds, destroying the narrative path. Instead wire each web_app to
    # ONE front door: the gateway if present, else the primary business service.
    # Remaining services keep their real edges (Sources 1-3) and the spine pass.
    containers = cont.get("containers", [])
    web_apps = [c for c in containers if c.get("kind") == "web_app"]
    if web_apps:
        gateway = next((c for c in containers if _node_layer(c) == "gateway"), None)
        front = gateway or _pick_primary_service(model)
        if front is not None:
            for wa in web_apps:
                _add_edge(wa["id"], front["id"], "calls API")

    # Merge edges (dedup against any pre-existing rels)
    existing_rels = list(cont.get("relationships", []))
    existing_seen = {(r.get("from", ""), r.get("to", "")) for r in existing_rels}
    for r in new_rels:
        if (r["from"], r["to"]) not in existing_seen:
            existing_rels.append(r)
    cont["relationships"] = existing_rels
    model["containers"] = cont

    _log.debug(
        "infer_communication_graph: +%d edges (k8s env, compose, imports, web→svc)",
        len(new_rels),
    )
    return _apply_hard_budget(model)


def infer_entrypoint(model: dict) -> dict:
    """Infer and wire the user-facing entrypoint after communication graph is built.

    Called after infer_communication_graph so all containers/edges exist.
    - If no actor exists and a web-facing container is present: adds 'user' actor.
    - Wires actor → entrypoint edge if not already present.
    - Updates _candidate_ids so apply_grounding preserves the actor.
    """
    ctx = model.get("context", {})
    cont = model.get("containers", {})

    actors = list(ctx.get("actors", []))
    containers = cont.get("containers", [])
    ctx_rels = list(ctx.get("relationships", []))

    if not containers:
        return model

    # If no actor yet, try to add one via entrypoint name detection
    if not actors:
        entrypoint = _pick_entrypoint(containers)
        if entrypoint:
            actor = {"id": "user", "label": "User", "kind": "person", "description": "Client caller"}
            actors = [actor]
            ctx["actors"] = actors
            cids = set(model.get("_candidate_ids") or [])
            cids.add("user")
            model["_candidate_ids"] = list(cids)
    elif len(actors) == 1 and actors[0].get("id") == "operator":
        # Upgrade Operator → User when a web-facing entrypoint is present.
        # A CLI Operator is correct only for pure batch/CLI repos with no web container.
        entrypoint = _pick_entrypoint(containers)
        if entrypoint:
            actors[0] = {"id": "user", "label": "User", "kind": "person", "description": "Client caller"}
            ctx["actors"] = actors
            for rel in ctx_rels:
                if rel.get("from") == "operator":
                    rel["from"] = "user"
            cids = set(model.get("_candidate_ids") or [])
            cids.discard("operator")
            cids.add("user")
            model["_candidate_ids"] = list(cids)

    # Wire actor → entrypoint if not already present
    if actors:
        actor_id = actors[0]["id"]
        already_wired = any(r.get("from") == actor_id for r in ctx_rels)
        if not already_wired:
            entrypoint = _pick_entrypoint(containers)
            target = entrypoint if entrypoint else containers[0]
            ctx_rels.append({"from": actor_id, "to": target["id"], "label": "uses"})

    ctx["relationships"] = ctx_rels
    model["context"] = ctx
    return model


# ── Architecture backbone synthesis ──────────────────────────────────────────


def synthesize_architecture_backbone(model: dict) -> dict:
    """Guarantee a connected architectural decomposition by attaching orphan containers.

    Runs after infer_communication_graph + infer_entrypoint so all real edges and
    the actor→entrypoint edge already exist.  Only fills gaps: containers with zero
    real edges (orphans) are attached to the primary flow along their architectural
    layer.  Non-orphans are left untouched — real topology is never overwritten.

    Layer order used for edge synthesis (top → bottom):
        presentation → gateway → application → worker
    Datastores and queues are never backbone targets (owned by Source-3 / renderer).
    """
    import logging as _log_bb

    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    if len(containers) < 2:
        return model

    ctx = model.get("context", {})

    # Build the set of all known node ids for the endpoint-existence guard
    container_ids = {c["id"] for c in containers}
    db_ids        = {d["id"] for d in cont.get("databases", [])}
    ext_ids       = {e["id"] for e in cont.get("external_services", [])}
    actor_ids     = {a["id"] for a in ctx.get("actors", [])}
    valid_ids     = container_ids | db_ids | ext_ids | actor_ids

    # Build degree counts and existing edge set from ALL current edges
    all_rels = list(ctx.get("relationships") or []) + list(cont.get("relationships") or [])
    out_deg: dict[str, int] = {}
    in_deg:  dict[str, int] = {}
    existing_pairs: set[tuple] = set()
    for r in all_rels:
        f, t = r.get("from", ""), r.get("to", "")
        out_deg[f] = out_deg.get(f, 0) + 1
        in_deg[t]  = in_deg.get(t, 0) + 1
        existing_pairs.add((f, t))

    rels = cont.setdefault("relationships", [])
    added: list[dict] = []

    def _add(a: str, b: str, label: str) -> None:
        if a == b:
            return
        if a not in valid_ids or b not in valid_ids:
            return
        if (a, b) in existing_pairs:
            return
        existing_pairs.add((a, b))
        e = {"from": a, "to": b, "label": label}
        rels.append(e)
        added.append(e)
        out_deg[a] = out_deg.get(a, 0) + 1
        in_deg[b]  = in_deg.get(b, 0) + 1

    # Compute layer for each container using the pure helper (no persisted field needed)
    layer: dict[str, str] = {c["id"]: _node_layer(c) for c in containers}

    def _ids_in(L: str) -> list[str]:
        return [c["id"] for c in containers if layer.get(c["id"]) == L]

    # Spine anchors
    ep = _pick_entrypoint(containers)
    ep_layer = layer.get(ep["id"]) if ep else None
    P = (ep["id"] if ep and ep_layer in ("presentation", "gateway") else None)
    if P is None:
        pres = _ids_in("presentation")
        P = pres[0] if pres else None

    G = _ids_in("gateway")[0] if _ids_in("gateway") else None
    fan = G or P

    # Fallback fan source for backend-only repos with no front door
    if fan is None:
        apps = _ids_in("application") or [c["id"] for c in containers]
        if apps:
            fan = sorted(apps, key=lambda i: (-out_deg.get(i, 0), i))[0]

    # Spine chain: presentation → gateway
    if P and G:
        _add(P, G, "routes to")

    def _is_orphan(cid: str) -> bool:
        return out_deg.get(cid, 0) == 0 and in_deg.get(cid, 0) == 0

    first_app = _ids_in("application")[0] if _ids_in("application") else None
    first_actor = sorted(actor_ids)[0] if actor_ids else None

    for c in containers:
        cid = c["id"]
        if not _is_orphan(cid) or cid in (P, G):
            continue
        L = layer.get(cid, "application")
        if L == "application":
            if fan:
                _add(fan, cid, "calls")
        elif L == "gateway":
            if P:
                _add(P, cid, "routes to")
        elif L == "presentation":
            if first_actor:
                _add(first_actor, cid, "uses")
            elif fan:
                _add(fan, cid, "serves")
        elif L == "worker":
            src = fan or first_app
            if src:
                _add(src, cid, "triggers")
        # data / messaging: skip — owned by Source-3 / renderer orphan filtering

    _log_bb.debug("synthesize_architecture_backbone: +%d backbone edges", len(added))
    model["containers"] = cont
    return _apply_hard_budget(model)


# ── Narrative spine + connectivity guarantee ─────────────────────────────────


def enforce_narrative_spine(model: dict) -> dict:
    """Guarantee the primary request path exists, and mark it immune to thinning.

    Builds the consecutive spine actor → entry → [gateway] → primary_service →
    datastore(s), adding ONLY the links that are missing. Every spine pair is
    recorded in model['_spine_edges'] so reduce_edges_for_readability and
    curate_significant_edges never drop the narrative path. Never invents nodes.
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    ctx = model.get("context", {})
    if not containers:
        return model

    container_ids = {c["id"] for c in containers}
    db_ids = {d["id"] for d in cont.get("databases", [])}
    actor_ids = [a["id"] for a in ctx.get("actors", [])]

    cont_rels = cont.setdefault("relationships", [])
    ctx_rels = ctx.setdefault("relationships", [])
    existing = {(r.get("from"), r.get("to")) for r in cont_rels + ctx_rels}
    spine: set = set(model.get("_spine_edges") or set())

    def _link(a: str, b: str, label: str, into: list) -> None:
        if not a or not b or a == b:
            return
        if (a, b) not in existing and (b, a) not in existing:
            into.append({"from": a, "to": b, "label": label})
            existing.add((a, b))
        spine.add((a, b))

    entry = _pick_entrypoint(containers)
    entry_id = entry["id"] if entry else None
    gateway = next((c for c in containers if _node_layer(c) == "gateway"), None)
    gateway_id = gateway["id"] if gateway else None
    primary = _pick_primary_service(model)
    primary_id = primary["id"] if primary else None

    # actor → entry (ingress) lives in context.relationships
    if actor_ids and entry_id:
        # already wired by infer_entrypoint in the common case; keep it spine-protected
        if any(r.get("to") == entry_id for r in ctx_rels):
            spine.add((actor_ids[0], entry_id))
        else:
            _link(actor_ids[0], entry_id, "uses", ctx_rels)

    # entry → [gateway] → primary_service
    upstream = entry_id
    if gateway_id and gateway_id != upstream:
        _link(upstream, gateway_id, "routes to", cont_rels)
        upstream = gateway_id
    if primary_id and upstream and primary_id != upstream:
        _link(upstream, primary_id, "calls", cont_rels)
        upstream = primary_id

    # primary_service → datastore(s) it owns (else any connected datastore)
    if primary_id:
        owned = [t for (f, t) in existing if f == primary_id and t in db_ids]
        targets = owned or [
            t for (f, t) in existing if t in db_ids and f in container_ids
        ]
        for ds in targets:
            spine.add((primary_id, ds))

    cont["relationships"] = cont_rels
    ctx["relationships"] = ctx_rels
    model["containers"] = cont
    model["context"] = ctx
    model["_spine_edges"] = spine
    return model


def enforce_connectivity(model: dict) -> dict:
    """Connectivity validator + repair — run LAST, after edge thinning.

    Implements the architecture rules so no subsystem ever floats:
      R2: every container has >=1 edge.
      R3: every datastore/cache/queue connects to an application container.
      R4: a user -> ... -> datastore path exists.
      R5: no isolated clusters (every container reachable from the entry).
    Repairs by attaching ORPHANS to existing spine nodes; never invents nodes.
    Consumes the private _spine_edges marker so it never reaches the renderer.
    """
    import logging as _log_conn
    log = _log_conn.getLogger(__name__)

    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    ctx = model.get("context", {})
    model.pop("_spine_edges", None)
    if not containers:
        return model

    container_ids = {c["id"] for c in containers}
    db_ids = {d["id"] for d in cont.get("databases", [])}
    actor_ids = {a["id"] for a in ctx.get("actors", [])}

    cont_rels = cont.setdefault("relationships", [])
    ctx_rels = ctx.setdefault("relationships", [])

    def _all_pairs() -> set:
        return {(r.get("from"), r.get("to")) for r in cont_rels + ctx_rels}

    pairs = _all_pairs()
    touched = {n for p in pairs for n in p}

    entry = _pick_entrypoint(containers)
    entry_id = entry["id"] if entry else containers[0]["id"]
    primary = _pick_primary_service(model)
    primary_id = primary["id"] if primary else entry_id

    def _add(a: str, b: str, label: str) -> None:
        if not a or not b or a == b or (a, b) in pairs or (b, a) in pairs:
            return
        cont_rels.append({"from": a, "to": b, "label": label})
        pairs.add((a, b))
        touched.add(a)
        touched.add(b)
        log.info("enforce_connectivity: attached %s -> %s (%s)", a, b, label)

    # R2: containers with zero edges attach to the primary service (or entry).
    for c in containers:
        cid = c["id"]
        if cid in touched:
            continue
        if cid == primary_id:
            _add(entry_id, cid, "calls")
        else:
            src = primary_id if primary_id != cid else entry_id
            _add(src, cid, "calls")

    # R3: orphan datastores/caches/queues attach FROM the primary service.
    for d in cont.get("databases", []):
        did = d["id"]
        if did in touched:
            continue
        verb = "publishes/consumes" if d.get("kind") == "queue" else "reads/writes"
        _add(primary_id, did, verb)

    # R5: any container not reachable from the entry attaches to the entry/primary.
    fwd: dict[str, set] = {}
    for (f, t) in pairs:
        fwd.setdefault(f, set()).add(t)
    reachable = set()
    stack = [entry_id]
    while stack:
        n = stack.pop()
        if n in reachable:
            continue
        reachable.add(n)
        stack.extend(fwd.get(n, ()))
    for c in containers:
        cid = c["id"]
        if cid not in reachable and cid != entry_id:
            _add(primary_id if primary_id in reachable else entry_id, cid, "calls")

    # R4: ensure a user -> ... -> datastore path exists.
    if actor_ids and db_ids:
        fwd = {}
        for (f, t) in pairs:
            fwd.setdefault(f, set()).add(t)
        reach = set()
        stack = list(actor_ids)
        while stack:
            n = stack.pop()
            if n in reach:
                continue
            reach.add(n)
            stack.extend(fwd.get(n, ()))
        if not (reach & db_ids):
            any_db = sorted(db_ids)[0]
            _add(primary_id, any_db, "reads/writes")

    cont["relationships"] = cont_rels
    ctx["relationships"] = ctx_rels
    model["containers"] = cont
    model["context"] = ctx
    return model


# ── Architectural layer assignment ───────────────────────────────────────────


def _node_layer(node: dict) -> str:
    """Assign a deterministic architectural layer to a container node.

    Uses kind first (datastore/cache/queue/worker/web_app are unambiguous), then
    name-regex matching for the service-kind split along the interaction path.
    Returns one of: presentation | gateway | application | worker | data | messaging
    """
    kind = node.get("kind", "service")
    text = (node.get("id", "") + " " + node.get("label", "")).lower()
    if kind in ("datastore", "cache"):
        return "data"
    if kind == "queue":
        return "messaging"
    if kind == "worker":
        return "worker"
    if kind == "web_app":
        return "presentation"
    # kind == "service" (or unknown): split by role along the interaction path
    if _GATEWAY_NAME_RE.search(text):
        return "gateway"
    if _PRESENTATION_NAME_RE.search(text):
        return "presentation"
    return "application"


def assign_architecture_layers(model: dict) -> dict:
    """Tag every container node with a deterministic 'layer' field.

    Called as the last model transform before rendering so the renderer can split
    the monolithic 'Services' tier into 'Clients', 'API Gateway', and 'Services'
    subgraphs along the primary interaction path.

    Only container nodes are tagged; databases keep their kind-based tier in the
    renderer.  The field is derived from stable id + kind — it survives any prior
    apply_grounding / apply_enrichment / enforce_c4_levels calls.
    """
    cont = model.get("containers", {})
    for c in cont.get("containers", []):
        c["layer"] = _node_layer(c)
    model["containers"] = cont
    return model


# ── Architectural validation (before render) ──────────────────────────────────

def validate_model(model: dict) -> dict:
    """Score the model and return deterministic findings.

    Returns:
      {
        "score": {actor_count, container_count, datastore_count,
                  relationship_count, external_count, connected_external_count,
                  orphan_count, orphan_datastore_count, avg_edges_per_container,
                  discovered_unit_count},
        "findings": [{"level": "fail"|"warn", "message": str}],
        "passed": bool,   # False if any finding is "fail"
      }
    """
    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    containers   = cont.get("containers", [])
    databases    = cont.get("databases", [])
    all_rels     = list(ctx.get("relationships") or []) + list(cont.get("relationships") or [])
    actors       = ctx.get("actors", [])
    ext_systems  = list(ctx.get("external_systems") or []) + list(cont.get("external_services") or [])
    discovered_unit_count = model.get("_discovered_unit_count", 0)
    # Units represented in the diagram, crediting domain reps for the modules they
    # fold in (set by consolidate_containers_for_abstraction). Falls back to the raw
    # container count when no consolidation happened.
    represented_unit_count = model.get("_represented_unit_count", 0) or len(containers)

    rel_endpoints: set[str] = set()
    for r in all_rels:
        rel_endpoints.add(_slug(r.get("from", "")))
        rel_endpoints.add(_slug(r.get("to", "")))

    ext_ids = {_slug(e["id"]) for e in ext_systems}
    connected_ext = ext_ids & rel_endpoints
    orphan_ext    = ext_ids - connected_ext

    db_ids = {_slug(d["id"]) for d in databases}
    connected_db = db_ids & rel_endpoints
    orphan_db    = db_ids - connected_db

    n_containers   = len(containers)
    n_databases    = len(databases)
    n_rels         = len(all_rels)
    n_actors       = len(actors)
    n_externals    = len(ext_systems)
    n_conn_ext     = len(connected_ext)
    n_orphan       = len(orphan_ext)
    n_conn_db      = len(connected_db)
    n_orphan_db    = len(orphan_db)
    n_cap          = n_containers + n_databases
    avg_edges      = (n_rels / n_cap) if n_cap > 0 else 0.0
    # edges-per-container-only (excludes datastores from denominator — used for dense-graph rule)
    avg_edges_cont = (n_rels / n_containers) if n_containers > 0 else 0.0

    score = {
        "actor_count":                  n_actors,
        "container_count":              n_containers,
        "datastore_count":              n_databases,
        "relationship_count":           n_rels,
        "external_count":               n_externals,
        "connected_external_count":     n_conn_ext,
        "orphan_count":                 n_orphan,
        "orphan_datastore_count":       n_orphan_db,
        "avg_edges_per_container":      round(avg_edges, 2),
        "avg_edges_per_container_only": round(avg_edges_cont, 2),
        "discovered_unit_count":        discovered_unit_count,
        "represented_unit_count":       represented_unit_count,
    }

    findings: list[dict] = []

    # ── Rule 1: multi-container with no edges (fail) ──────────────────────────
    if n_containers > 1 and n_rels == 0:
        findings.append({
            "level": "fail",
            "message": f"{n_containers} containers found but 0 relationships — diagram will be empty",
        })

    # ── Rule 2: under-discovery — fusion found many units but few represented ──
    # Judged on REPRESENTED units (containers + modules folded into domain reps), not
    # the raw rendered count: deliberate consolidation legitimately lowers the box
    # count while still representing every discovered unit. A genuine drop (no
    # consolidation) leaves represented == n_containers, so it still trips this rule.
    if discovered_unit_count >= 5 and represented_unit_count <= 2:
        findings.append({
            "level": "fail",
            "message": (
                f"evidence fusion discovered {discovered_unit_count} deployable units "
                f"but only {represented_unit_count} represented — likely under-discovery"
            ),
        })
    elif discovered_unit_count >= 5 and represented_unit_count < 0.5 * discovered_unit_count:
        findings.append({
            "level": "fail",
            "message": (
                f"only {represented_unit_count}/{discovered_unit_count} discovered units represented "
                f"— pipeline may be dropping containers"
            ),
        })

    # ── Rule 3: datastore connectivity ───────────────────────────────────────
    if n_databases > 2 and n_databases > 0 and n_conn_db / n_databases < 0.5:
        findings.append({
            "level": "fail",
            "message": (
                f"{n_databases} datastores but only {n_conn_db} connected "
                f"— inference likely broken ({n_orphan_db} orphan datastores)"
            ),
        })
    elif n_databases > 0 and n_conn_db == 0:
        findings.append({
            "level": "warn",
            "message": f"{n_databases} datastore(s) but none connected — may be orphaned",
        })

    # ── Rule 4: all externals are orphans (warn) ──────────────────────────────
    if n_externals > 0 and n_conn_ext == 0:
        findings.append({
            "level": "warn",
            "message": f"{n_externals} external(s) found but none are connected — all will be filtered as orphans",
        })

    # ── Rule 5: many containers but no actor (warn) ───────────────────────────
    if n_containers > 3 and n_actors == 0:
        findings.append({
            "level": "warn",
            "message": f"{n_containers} containers but no actor — diagram has no entry point",
        })

    # ── Rule 6: sparse edges (warn) ───────────────────────────────────────────
    if avg_edges < 0.5 and n_cap > 1:
        findings.append({
            "level": "warn",
            "message": f"avg edges per container/datastore is {avg_edges:.2f} — architecture may be incomplete",
        })

    # ── Rule 7: dense-graph completeness (fail for large under-connected models) ─
    _DENSE_MIN_CONTAINERS  = 8
    _DENSE_MIN_AVG_EDGES   = 0.75
    if n_containers >= _DENSE_MIN_CONTAINERS and avg_edges_cont < _DENSE_MIN_AVG_EDGES:
        findings.append({
            "level": "fail",
            "message": (
                f"{n_containers} containers but only {n_rels} edges "
                f"(avg {avg_edges_cont:.2f} per container, expected ≥{_DENSE_MIN_AVG_EDGES}) "
                f"— service graph likely incomplete"
            ),
        })

    return {
        "score":    score,
        "findings": findings,
        "passed":   not any(f["level"] == "fail" for f in findings),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def build_candidate_model(
    rich_facts: dict,
    repo_root: str,
    repo_name: str | None = None,
    manifests: dict | None = None,
    orchestration: dict | None = None,
) -> dict:
    """Build the C4 candidate model — NODES ONLY, no relationships.

    Admission (v5 — deterministic boolean): a unit is admitted iff it is a declared
    deployment boundary (plain Dockerfile, compose service, k8s workload, or deployable
    manifest) AND not auxiliary.  No weighted scoring.

    After admission: auxiliary deploy-unit filter removes migration runners,
    CI images, and operational tooling from the architectural node-set.

    Returns the same dict shape renderers expect, with per-unit evidence preserved
    for apply_grounding() and private _container_units / _db_found / etc. fields
    for infer_communication_graph.  All relationship construction is deferred to
    infer_communication_graph + infer_entrypoint.
    """
    import logging as _log
    facts = runtime_facts(rich_facts, repo_root)
    repo_root_abs = str(Path(repo_root).resolve())
    orch = orchestration or {}

    # Self-sufficiency: the multi-app corroboration gate reads rich_facts["components"]
    # (normally populated upstream by the extractor). If a caller hands us raw facts
    # without them, compute them here so the admission relaxation still works.
    if not rich_facts.get("components"):
        try:
            from doc_agent.tools.component_clusters import compute_components
            from doc_agent.tools.import_graph import build_import_graph
            _files = rich_facts.get("files") or []
            _ig = rich_facts.get("import_graph") or build_import_graph(_files, repo_root)
            _comp = compute_components(_ig, _files, repo_root)
            rich_facts = {**rich_facts, "components": _comp["components"]}
        except Exception:
            _log.debug("build_candidate_model: component fallback failed", exc_info=True)

    # ── Pass 1: group files by nearest manifest dir ───────────────────────────
    manifest_files = _find_manifests(repo_root)
    manifest_dirs = sorted(
        {str(Path(m).parent.resolve()) for m in manifest_files},
        key=lambda d: len(d),
    )

    units: dict[str, dict] = {}
    for mid, f in facts.items():
        fp = f.get("file", "")
        mdir = _nearest_manifest_dir(fp, manifest_dirs, repo_root_abs)
        if _is_non_deployable(mdir, repo_root_abs):
            continue
        if mdir not in units:
            units[mdir] = {"files": [], "imports": [], "routes": [], "has_entry": False}
        units[mdir]["files"].append(f)
        units[mdir]["imports"].extend(f.get("imports", []))
        units[mdir]["routes"].extend(f.get("routes", []))
        if _is_entry(mid, f):
            units[mdir]["has_entry"] = True

    # ── NEW: add Dockerfile dirs not captured by code extraction ─────────────
    # This makes discovery language-agnostic: a Go/Ruby service dir with a
    # Dockerfile becomes a candidate even if the extractor produced no facts.
    for df_dir in _find_dockerfile_dirs(repo_root):
        if df_dir not in units and not _is_non_deployable(df_dir, repo_root_abs):
            units[df_dir] = {"files": [], "imports": [], "routes": [], "has_entry": False}

    # ── NEW: add compose build.context dirs not yet in units ─────────────────
    for svc in orch.get("compose_services", []):
        if svc.get("is_auxiliary"):
            continue
        build_dir = svc.get("build_dir")
        if build_dir and Path(build_dir).is_dir():
            bd_abs = str(Path(build_dir).resolve())
            if bd_abs not in units and not _is_non_deployable(bd_abs, repo_root_abs):
                units[bd_abs] = {"files": [], "imports": [], "routes": [], "has_entry": False}

    if not units:
        units[repo_root_abs] = {"files": list(facts.values()), "imports": [], "routes": [], "has_entry": False}

    # ── Build compose dir lookup and k8s slug lookup (for evidence scoring) ───
    compose_build_dirs: set[str] = set()
    for svc in orch.get("compose_services", []):
        if not svc.get("is_auxiliary") and svc.get("build_dir"):
            bd = svc["build_dir"]
            if Path(bd).is_dir():
                compose_build_dirs.add(str(Path(bd).resolve()))

    k8s_slugs: set[str] = {
        _slug(wl["name"]) for wl in orch.get("k8s_workloads", []) if wl.get("name")
    }

    # ── Pass 2: confidence-scored admission ───────────────────────────────────
    # Single-manifest / no-manifest repos: use all units (same as v3 behaviour).
    # Multi-manifest repos: replace binary gate with confidence scoring so that
    # Dockerfile-only services (e.g. all-Go microservices) are never silently dropped.
    if len(manifest_dirs) <= 1:
        candidates = [
            {
                "mdir": d, "udata": units[d],
                "via_manifest": False,
                "via_dockerfile": _has_any_dockerfile(d),
                "is_auxiliary": False,
                "evidence": {
                    "dockerfile": _has_dockerfile(d),
                    "manifest_deployable": False,
                    "compose": d in compose_build_dirs,
                    "k8s": _slug(Path(d).name) in k8s_slugs,
                    "entry": units[d].get("has_entry", False),
                    "auxiliary": False,
                    "facts": bool(units[d].get("files")),
                },
            }
            for d in units.keys()
        ]
    else:
        candidates = []
        for mdir, udata in units.items():
            m_info = (manifests or {}).get(mdir, {})
            via_manifest = m_info.get("deployable", False)
            via_dockerfile = _has_dockerfile(mdir)
            via_compose = mdir in compose_build_dirs
            dir_slug = _slug(Path(mdir).name)
            via_k8s = dir_slug in k8s_slugs
            has_entry = udata.get("has_entry", False)
            has_facts = bool(udata.get("files"))

            candidates.append({
                "mdir": mdir,
                "udata": udata,
                "via_manifest": via_manifest,
                "via_dockerfile": _has_any_dockerfile(mdir),
                "is_auxiliary": False,
                "evidence": {
                    "dockerfile": via_dockerfile,
                    "manifest_deployable": via_manifest,
                    "compose": via_compose,
                    "k8s": via_k8s,
                    "entry": has_entry,
                    "auxiliary": False,  # set by _mark_auxiliary below
                    "facts": has_facts,
                    "runnable_by_convention": _is_runnable_by_convention(mdir, udata, manifests, manifest_dirs),
                },
            })

    # ── Auxiliary filter: marks is_auxiliary on each candidate in-place ───────
    _mark_auxiliary(candidates, repo_root_abs)

    # ── Apply admission gate ──────────────────────────────────────────────────
    # A unit is admitted iff it is a DECLARED deployment boundary (plain Dockerfile,
    # compose service, k8s workload, or deployable manifest) AND not auxiliary.
    # Relaxation: when fewer than 2 declared boundaries exist but the import-graph
    # component structure independently shows >=2 runnable apps, ALSO admit units
    # that are runnable-by-convention (own manifest + entrypoint/routes/framework).
    # This rescues genuine multi-app monorepos that ship no Dockerfile/compose, while
    # a true monolith (one dominant component) stays exactly one container.
    if len(manifest_dirs) <= 1:
        tier1 = [c for c in candidates if not c["is_auxiliary"]]
    else:
        for c in candidates:
            c["evidence"]["auxiliary"] = c["is_auxiliary"]

        def _is_declared(c: dict) -> bool:
            ev = c["evidence"]
            return bool(ev["dockerfile"] or ev["compose"] or ev["k8s"] or ev["manifest_deployable"])

        declared = [c for c in candidates if _is_declared(c) and not c["is_auxiliary"]]
        tier1 = list(declared)

        if len(declared) < 2 and _multi_app_corroborated(rich_facts):
            seen = {c["mdir"] for c in tier1}
            conventional = [
                c for c in candidates
                if not c["is_auxiliary"]
                and c["evidence"].get("runnable_by_convention")
                and c["mdir"] not in seen
            ]
            for c in conventional:
                _log.debug("build_candidate_model: admitted by convention %s", Path(c["mdir"]).name)
            tier1 += conventional

        for c in candidates:
            if c not in tier1:
                ev = c["evidence"]
                _log.debug(
                    "build_candidate_model: rejected %s (declared=%s, aux=%s, evidence=%s)",
                    Path(c["mdir"]).name, _is_declared(c), c["is_auxiliary"], ev,
                )

    # Track discovered count for validate_model Rule 4 (under-discovery)
    discovered_unit_count = len(tier1)

    # is_multi: 2+ tier1 deploy units
    is_multi = len(tier1) >= 2
    if not is_multi:
        # Collapse to one container (whole repo)
        container_dirs = [repo_root_abs]
        if repo_root_abs not in units:
            combined: dict = {"files": [], "imports": [], "routes": [], "has_entry": False}
            for u in units.values():
                combined["files"].extend(u["files"])
                combined["imports"].extend(u["imports"])
                combined["routes"].extend(u["routes"])
                combined["has_entry"] = combined["has_entry"] or u["has_entry"]
            units[repo_root_abs] = combined
        container_units: dict[str, dict] = {repo_root_abs: units[repo_root_abs]}
    else:
        container_dirs = [c["mdir"] for c in tier1]
        container_units = {c["mdir"]: c["udata"] for c in tier1}

    # ── Pass 3: re-attribute library/domain files into container dirs ─────────
    reattributed: dict[str, dict] = {cd: {"files": [], "imports": [], "routes": [], "has_entry": False}
                                     for cd in container_dirs}
    for mid, f in facts.items():
        fp = f.get("file", "")
        best, best_depth = repo_root_abs, len(Path(repo_root_abs).parts)
        for cd in container_dirs:
            cdp = Path(cd)
            try:
                Path(fp).resolve().relative_to(cdp)
                depth = len(cdp.parts)
                if depth > best_depth:
                    best, best_depth = cd, depth
            except (ValueError, Exception):
                continue
        if best not in reattributed:
            best = container_dirs[0]
        reattributed[best]["files"].append(f)
        reattributed[best]["imports"].extend(f.get("imports", []))
        reattributed[best]["routes"].extend(f.get("routes", []))
        if _is_entry(mid, f):
            reattributed[best]["has_entry"] = True
    container_units = reattributed

    # ── Flatten all imports for whole-repo infra scanning ─────────────────────
    all_imports: list[str] = [imp for u in container_units.values() for imp in u["imports"]]
    all_manifest_deps: list[str] = []
    if manifests:
        for parsed in manifests.values():
            all_manifest_deps.extend(parsed.get("dependencies") or [])

    # ── Datastores ────────────────────────────────────────────────────────────
    db_from_classes = any(
        c.get("is_db_model")
        for f in facts.values()
        for c in f.get("classes", [])
    )
    db_found = _scan_db_engines(all_imports, manifest_deps=all_manifest_deps)
    if not db_found and db_from_classes:
        db_found["Relational Database"] = ("Relational Database", "datastore")
    db_found = _consolidate_datastores(db_found)
    db_found = _consolidate_queues(db_found)

    datastore_nodes = [
        {"id": _slug(lbl), "label": lbl, "kind": kind, "tech": lbl, "description": ""}
        for lbl, (_, kind) in db_found.items()
    ]

    # ── External systems ──────────────────────────────────────────────────────
    svc_found = _scan_services(all_imports, manifest_deps=all_manifest_deps)
    external_nodes = []
    ext_verb: dict[str, str] = {}
    for label, (lbl, _kind, verb) in svc_found.items():
        nid = _slug(lbl)
        external_nodes.append(
            {"id": nid, "label": lbl, "kind": "external", "tech": _kind, "description": ""}
        )
        ext_verb[nid] = verb

    # ── Container nodes ───────────────────────────────────────────────────────
    container_nodes = []
    container_id_map: dict[str, str] = {}
    for mdir, udata in container_units.items():
        fws = detect_frameworks(udata["files"])
        has_routes = bool(udata["routes"])
        has_entry = udata["has_entry"]
        kind = _classify_unit(fws, has_routes, has_entry)
        # Factless dirs (Go/Ruby services, etc.) fall back to name-based classification
        if not udata.get("files") and kind == "service":
            kind = _classify_by_name(Path(mdir).name)

        rel = os.path.relpath(mdir, repo_root_abs).replace("\\", "/")
        dir_name = Path(rel).name if rel not in (".", "") else Path(repo_root_abs).name

        display_name = dir_name
        if manifests and mdir in manifests:
            mn = (manifests[mdir].get("project_name") or "").strip()
            if mn:
                display_name = mn
        if len(container_units) == 1 and repo_name:
            display_name = repo_name
        elif display_name == dir_name and repo_name and rel in (".", ""):
            display_name = repo_name

        label = _container_label(display_name, kind)
        nid = _slug(display_name) or _slug(label)
        tech = ", ".join(fws) if fws else ""

        container_id_map[mdir] = nid
        container_nodes.append({
            "id": nid,
            "label": label,
            "kind": kind,
            "tech": tech,
            "description": "",
            "_has_routes": has_routes,
            "_mdir": mdir,
        })

    # ── Actors (detected from code facts; edges wired by infer_entrypoint) ──────
    any_routes = any(c["_has_routes"] for c in container_nodes) or any(
        f.get("routes") for f in facts.values()
    )
    any_cli = any(u["has_entry"] and not u["routes"] for u in container_units.values())

    actors: list[dict] = []
    if any_routes:
        actors.append({"id": "user", "label": "User", "kind": "person", "description": "Client caller"})
    elif any_cli:
        actors.append({"id": "operator", "label": "Operator", "kind": "person", "description": "CLI operator"})

    # ── Strip internal-only fields ────────────────────────────────────────────
    clean_containers = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in container_nodes
    ]

    # Store candidate ids for apply_grounding
    candidate_ids = {c["id"] for c in clean_containers}
    infra_ids = {d["id"] for d in datastore_nodes} | {e["id"] for e in external_nodes}
    actor_ids = {a["id"] for a in actors}
    all_candidate_ids = candidate_ids | infra_ids | actor_ids

    # Per-container source path (relative to repo root) — consumed by assign_domains
    # for path-based domain grouping of monorepos, then irrelevant to the renderer.
    container_paths: dict[str, str] = {}
    for mdir, nid in container_id_map.items():
        try:
            container_paths[nid] = os.path.relpath(mdir, repo_root_abs).replace("\\", "/")
        except Exception:
            container_paths[nid] = ""

    model = {
        "context": {
            "system_name": "",
            "system_description": "",
            "architecture_style": "",
            "actors": actors,
            "external_systems": [
                {"id": e["id"], "label": e["label"], "kind": e["kind"], "description": e["description"]}
                for e in external_nodes
            ],
            "relationships": [],  # edges built by infer_communication_graph
        },
        "containers": {
            "system_label": "",
            "containers": clean_containers,
            "databases": datastore_nodes,
            "external_services": [
                {"id": e["id"], "label": e["label"], "kind": e["kind"], "description": e["description"]}
                for e in external_nodes
            ],
            "relationships": [],  # edges built by infer_communication_graph
        },
        "_candidate_ids": list(all_candidate_ids),
        "_discovered_unit_count": discovered_unit_count,
        "_container_paths": container_paths,
        # Private fields consumed by infer_communication_graph; cleaned up there
        "_container_units": container_units,
        "_container_id_map": container_id_map,
        "_db_found": db_found,
        "_svc_found": {lbl: v for lbl, v in svc_found.items()},
        "_ext_verb": ext_verb,
        "_single_container": (len(container_units) == 1),
    }

    return _apply_hard_budget(model)


# ── apply_grounding ───────────────────────────────────────────────────────────

def apply_grounding(candidate_model: dict, llm_model: dict) -> dict:
    """Discard any LLM-returned node id absent from the candidate set.

    The LLM may classify/label/fold candidates but must never INVENT new nodes.
    This gate enforces that guarantee: any id not in _candidate_ids is discarded
    along with its relationships.
    """
    candidate_ids = set(candidate_model.get("_candidate_ids") or [])
    if not candidate_ids:
        return llm_model  # no grounding info — pass through

    ctx  = llm_model.get("context", {}) or {}
    cont = llm_model.get("containers", {}) or {}

    def _filter_by_id(nodes: list) -> list:
        return [n for n in (nodes or []) if n.get("id") in candidate_ids]

    ctx["actors"]            = _filter_by_id(ctx.get("actors", []))
    ctx["external_systems"]  = _filter_by_id(ctx.get("external_systems", []))
    cont["containers"]       = _filter_by_id(cont.get("containers", []))
    cont["databases"]        = _filter_by_id(cont.get("databases", []))
    cont["external_services"] = _filter_by_id(cont.get("external_services", []))

    kept_ids = (
        {n["id"] for n in ctx.get("actors", [])}
        | {n["id"] for n in ctx.get("external_systems", [])}
        | {n["id"] for n in cont.get("containers", [])}
        | {n["id"] for n in cont.get("databases", [])}
        | {n["id"] for n in cont.get("external_services", [])}
    )

    def _filter_rels(rels: list) -> list:
        return [r for r in (rels or []) if r.get("from") in kept_ids and r.get("to") in kept_ids]

    ctx["relationships"]  = _filter_rels(ctx.get("relationships", []))
    cont["relationships"] = _filter_rels(cont.get("relationships", []))

    llm_model["context"]    = ctx
    llm_model["containers"] = cont
    return llm_model


# ── enforce_c4_levels ─────────────────────────────────────────────────────────

_ALLOWED_C4_KINDS = frozenset({
    "person", "web_app", "service", "worker",
    "datastore", "cache", "queue", "external",
})


def enforce_c4_levels(model: dict) -> dict:
    """Remove any node whose 'kind' is not in the C4 container-diagram allowed set."""
    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    removed_ids: set[str] = set()

    def _filter_nodes(nodes: list) -> list:
        kept, dropped = [], []
        for n in (nodes or []):
            if n.get("kind") in _ALLOWED_C4_KINDS:
                kept.append(n)
            else:
                dropped.append(n.get("id", ""))
        removed_ids.update(dropped)
        return kept

    ctx["actors"]              = _filter_nodes(ctx.get("actors", []))
    ctx["external_systems"]    = _filter_nodes(ctx.get("external_systems", []))
    cont["containers"]         = _filter_nodes(cont.get("containers", []))
    cont["databases"]          = _filter_nodes(cont.get("databases", []))
    cont["external_services"]  = _filter_nodes(cont.get("external_services", []))

    def _filter_rels(rels: list) -> list:
        return [
            r for r in (rels or [])
            if r.get("from") not in removed_ids and r.get("to") not in removed_ids
        ]

    ctx["relationships"]  = _filter_rels(ctx.get("relationships", []))
    cont["relationships"] = _filter_rels(cont.get("relationships", []))

    model["context"]    = ctx
    model["containers"] = cont
    return model


# ── apply_enrichment ──────────────────────────────────────────────────────────

def apply_enrichment(model: dict, enrichment: dict) -> dict:
    """Merge LLM text enrichment into the model (text-only, no structure)."""
    system_purpose = (
        enrichment.get("system_purpose")
        or enrichment.get("system_name")
        or ""
    ).strip()
    descriptions = enrichment.get("descriptions") or {}
    edge_labels  = enrichment.get("edge_labels") or {}
    labels       = enrichment.get("labels") or {}

    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    if system_purpose:
        ctx["system_name"]   = system_purpose
        cont["system_label"] = system_purpose

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

    def _apply_label(nodes: list) -> None:
        for node in (nodes or []):
            nid = node.get("id", "")
            if nid in labels and labels[nid].strip():
                node["label"] = labels[nid].strip()

    _apply_desc(ctx.get("actors", []))
    _apply_desc(ctx.get("external_systems", []))
    _apply_desc(cont.get("containers", []))
    _apply_desc(cont.get("databases", []))
    _apply_desc(cont.get("external_services", []))

    _apply_label(ctx.get("actors", []))
    _apply_label(ctx.get("external_systems", []))
    _apply_label(cont.get("containers", []))
    _apply_label(cont.get("databases", []))
    _apply_label(cont.get("external_services", []))

    def _apply_edge_labels(rels: list) -> None:
        for rel in (rels or []):
            key = f"{rel.get('from', '')}__{rel.get('to', '')}"
            if key in edge_labels:
                rel["label"] = edge_labels[key]

    _apply_edge_labels(ctx.get("relationships", []))
    _apply_edge_labels(cont.get("relationships", []))

    # Merge LLM-assigned domain groups onto container nodes only
    groups = enrichment.get("groups") or {}
    container_ids = {n["id"] for n in cont.get("containers", [])}
    for node in cont.get("containers", []):
        nid = node.get("id", "")
        if nid in groups and groups[nid].strip():
            node["group"] = groups[nid].strip()

    model["context"]    = ctx
    model["containers"] = cont
    return model


# ── assign_domains ────────────────────────────────────────────────────────────

# Conventional monorepo *roots* whose immediate children are independent units
# (microservices, packages) — units directly under these stay standalone. A folder
# NOT in this set (modules/, plugins/, features/, integrations/, …) is treated as a
# named domain whose sibling children belong together and can be consolidated.
_GENERIC_ROOT_DIRS = frozenset({
    "src", "source", "sources", "app", "apps", "lib", "libs", "pkg", "pkgs",
    "packages", "services", "service", "cmd", "internal", "projects", "repos",
})


def _path_domain(rel_path: str) -> "str | None":
    """Derive a domain name from a unit's repo-relative path, or None if standalone.

    The domain is the TOPMOST non-generic ancestor folder — i.e. skip leading generic
    monorepo roots (src/, packages/, …) then take the next segment. This groups every
    unit nested under a named domain folder, no matter how deep:
        src/modules/FancyZones/FancyZonesEditor → "modules"
        src/modules/launcher/Plugins/Calc       → "modules"
        src/common/Common.UI                    → "common"
    A unit sitting *directly* under generic roots (or at the repo top) is standalone,
    so genuine sibling services keep their own boxes:
        services/cart  → None   (sibling of other independent services)
        src/runner     → None   (direct child of a generic root)
        runner         → None   (top-level unit)
    """
    segs = [s for s in (rel_path or "").replace("\\", "/").split("/") if s and s != "."]
    if len(segs) < 2:
        return None  # top-level unit → standalone
    ancestors = segs[:-1]  # directories containing the unit dir
    i = 0
    while i < len(ancestors) and ancestors[i].lower() in _GENERIC_ROOT_DIRS:
        i += 1
    if i >= len(ancestors):
        return None  # unit sits directly under generic root(s) → standalone
    return ancestors[i]


def assign_domains(model: dict) -> dict:
    """Ensure every container has a 'group' domain; fill gaps deterministically.

    Priority:
    1. LLM-assigned 'group' (set by apply_enrichment) — kept as-is.
    2. Path-based domain: the named folder grouping sibling units (src/modules/* →
       "Modules"), used when it yields ≥2 groups with at least one multi-member group.
    3. Top-level folder prefix when it yields ≥2 distinct groups across containers.
    4. Architectural layer name (from _node_layer) as stable fallback.

    Datastores/queues with no 'group' inherit the domain of their most-connected
    container peer (via relationships). Shared datastores → "Shared Data".
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    databases  = cont.get("databases", [])
    rels       = cont.get("relationships", [])
    paths      = model.get("_container_paths") or {}

    if not containers:
        return model

    # Step 1: collect already-assigned groups
    def _folder_prefix(nid: str) -> str:
        parts = nid.replace("-", "_").split("_")
        return parts[0] if parts else nid

    # Step 2: try folder-prefix grouping for ungrouped containers
    # Only use prefix grouping when prefixes are meaningfully distinct (2-6 groups).
    # If every container has a unique prefix (flat microservices), fall back to layer.
    ungrouped = [c for c in containers if not c.get("group")]
    if ungrouped:
        # Step 2a: path-based domains. Prefer this over slug prefixes — a real folder
        # layout is a far stronger domain signal than a name prefix coincidence. Use it
        # only when it actually clusters siblings (≥2 groups, ≥1 with >1 member).
        path_groups: dict[str, str] = {}
        for c in ungrouped:
            dom = _path_domain(paths.get(c["id"], ""))
            path_groups[c["id"]] = (dom or c["id"]).replace("-", " ").replace("_", " ").title()
        distinct = set(path_groups.values())
        members_per_group: dict[str, int] = {}
        for g in path_groups.values():
            members_per_group[g] = members_per_group.get(g, 0) + 1
        use_path = (
            len(distinct) >= 2
            and any(n >= 2 for n in members_per_group.values())
        )

        all_prefixes = {_folder_prefix(c["id"]) for c in containers}
        use_prefix = 2 <= len(all_prefixes) <= 6

        for c in ungrouped:
            if use_path:
                c["group"] = path_groups[c["id"]]
            elif use_prefix:
                c["group"] = _folder_prefix(c["id"]).replace("_", " ").title()
            else:
                c["group"] = _node_layer(c).replace("_", " ").title()

        # Degeneracy guard: if the deterministic fallback collapsed every
        # container into a single group (e.g. flat microservices → one
        # "Application" box), clear all groups so the renderer falls back to
        # structured tier mode instead of one giant domain subgraph.
        if len({c.get("group", "") for c in containers}) < 2:
            import logging as _log_dom
            _log_dom.getLogger(__name__).info(
                "assign_domains: degenerate grouping (<2 domains); clearing groups for tier-mode fallback"
            )
            for c in containers:
                c.pop("group", None)

    # Step 3: datastores/queues inherit domain of their primary container peer
    db_ids = {db["id"] for db in databases}
    container_ids = {c["id"] for c in containers}
    # Build outbound edge index: db_id → list of container peers
    db_peers: dict[str, list[str]] = {db["id"]: [] for db in databases}
    for rel in rels:
        frm, to = rel.get("from", ""), rel.get("to", "")
        if to in db_ids and frm in container_ids:
            db_peers[to].append(frm)
        elif frm in db_ids and to in container_ids:
            db_peers[frm].append(to)

    id_to_group = {c["id"]: c.get("group", "") for c in containers}
    for db in databases:
        if db.get("group"):
            continue
        peers = db_peers.get(db["id"], [])
        peer_groups = [id_to_group[p] for p in peers if p in id_to_group and id_to_group[p]]
        if len(set(peer_groups)) == 1:
            db["group"] = peer_groups[0]
        elif peer_groups:
            db["group"] = "Shared Data"
        # no peers → leave unset (will be placed in tier by renderer)

    model["containers"] = cont
    return model


# ── consolidate_containers_for_abstraction ────────────────────────────────────

# Readable container window (mirrors the fidelity scorer's _score_hld window). When
# a model exceeds the upper bound it reads as a hairball; consolidation pulls it back
# into the window by folding sibling modules of a domain into one representative.
_ABSTRACTION_WINDOW_LO = 3
_ABSTRACTION_WINDOW_HI = 12

def consolidate_containers_for_abstraction(model: dict) -> dict:
    """Collapse multi-member domain groups into one representative container.

    C4 permits showing a group of closely-related containers as a single container.
    On large monorepos (dozens of modules under src/modules, plugins/, …) rendering
    one box per module is an unreadable hairball that scores ~0.3 on abstraction.
    This deterministic pass folds the members of each domain group into a single
    representative node when the model is over the readable window — turning "44
    module boxes" into one "Modules" domain box.

    Fires only when ALL hold (otherwise a no-op):
      - container_count > _ABSTRACTION_WINDOW_HI
      - ≥2 distinct domain groups exist
      - at least one group has ≥2 foldable members
      - the result keeps ≥_ABSTRACTION_WINDOW_LO containers (never over-collapses)

    Only the actual ingress face is protected from folding: the single entrypoint
    (so the actor→entry spine survives) and any gateway nodes (the front door). Every
    other grouped container is foldable — including presentation/web_app modules,
    because in a GUI-app monorepo (e.g. WinUI desktop) every module is presentation,
    so protecting the whole layer would defeat consolidation entirely. Edges (context
    + container), the protected spine markers, and _candidate_ids are all remapped to
    the representative ids; self-loops and duplicates are dropped. Records
    _represented_unit_count so the scorer credits the folded units as represented
    (abstraction must not be punished by the coverage axis).
    """
    import logging as _log_cons
    log = _log_cons.getLogger(__name__)

    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    if len(containers) <= _ABSTRACTION_WINDOW_HI:
        return model

    # Protect only the real ingress face: the single entrypoint + any gateways. These
    # anchor the actor→entry→… spine and must stay individually visible.
    entry = _pick_entrypoint(containers)
    protected_ids: set[str] = {entry["id"]} if entry else set()
    for c in containers:
        if (c.get("layer") or _node_layer(c)) == "gateway":
            protected_ids.add(c["id"])

    # Partition into foldable grouped members vs. standalone (protected/ungrouped).
    groups: dict[str, list[dict]] = {}
    standalone: list[dict] = []
    for c in containers:
        grp = (c.get("group") or "").strip()
        if grp and c["id"] not in protected_ids:
            groups.setdefault(grp, []).append(c)
        else:
            standalone.append(c)

    foldable_groups = {g: members for g, members in groups.items() if len(members) >= 2}
    # Single-member foldable groups stay standalone (nothing to collapse).
    for g, members in groups.items():
        if len(members) < 2:
            standalone.extend(members)

    if len(foldable_groups) < 1:
        return model

    # Projected count after folding each multi-member group to one rep.
    projected = len(standalone) + len(foldable_groups)
    distinct_result_groups = len(foldable_groups) + len({
        (c.get("group") or c["id"]) for c in standalone
    })
    if projected >= len(containers):
        return model  # no reduction
    if projected < _ABSTRACTION_WINDOW_LO:
        return model  # would over-collapse below the readable floor
    if distinct_result_groups < 2:
        return model  # collapses everything into one box — not an improvement

    # Build representatives + the member→rep id remap.
    _KIND_PRIORITY = ["service", "web_app", "worker"]  # rep kind, deterministic
    existing_ids = {c["id"] for c in containers}
    id_remap: dict[str, str] = {}
    rep_nodes: list[dict] = []
    folded_total = 0

    for grp, members in sorted(foldable_groups.items()):
        rep_id = _slug(grp)
        # Avoid colliding with a surviving standalone node id.
        base_id = rep_id
        n = 2
        standalone_ids = {c["id"] for c in standalone}
        while rep_id in standalone_ids:
            rep_id = f"{base_id}_{n}"
            n += 1
        kinds = [m.get("kind", "service") for m in members]
        rep_kind = next((k for k in _KIND_PRIORITY if k in kinds), sorted(kinds)[0])
        techs = []
        for m in members:
            for t in (m.get("tech") or "").split(","):
                t = t.strip()
                if t and t not in techs:
                    techs.append(t)
        member_labels = [m.get("label", m["id"]) for m in members]
        shown = ", ".join(member_labels[:5]) + ("…" if len(member_labels) > 5 else "")
        rep = {
            "id": rep_id,
            "label": grp,
            "kind": rep_kind,
            "tech": ", ".join(techs[:4]),
            "description": f"{len(members)} modules: {shown}",
            "group": grp,
        }
        rep["layer"] = _node_layer(rep)
        rep_nodes.append(rep)
        for m in members:
            id_remap[m["id"]] = rep_id
        folded_total += len(members)

    # New container set: standalone (unchanged) + representatives.
    new_containers = standalone + rep_nodes
    cont["containers"] = new_containers

    # Remap edges in both relationship lists; drop self-loops, dedupe.
    def _remap_rels(rels: list) -> list:
        seen: set[tuple] = set()
        out: list[dict] = []
        for r in (rels or []):
            f = id_remap.get(r.get("from", ""), r.get("from", ""))
            t = id_remap.get(r.get("to", ""), r.get("to", ""))
            if f == t:
                continue
            if (f, t) in seen:
                continue
            seen.add((f, t))
            nr = dict(r)
            nr["from"], nr["to"] = f, t
            out.append(nr)
        return out

    cont["relationships"] = _remap_rels(cont.get("relationships", []))
    ctx = model.get("context", {})
    ctx["relationships"] = _remap_rels(ctx.get("relationships", []))
    model["context"] = ctx

    # Remap the protected spine markers so reduce/curate still shield the spine.
    spine = model.get("_spine_edges")
    if spine:
        model["_spine_edges"] = {
            (id_remap.get(a, a), id_remap.get(b, b))
            for (a, b) in spine
            if id_remap.get(a, a) != id_remap.get(b, b)
        }

    # Keep grounding ids consistent (rep ids are legitimate, member ids gone).
    cand = set(model.get("_candidate_ids") or [])
    cand -= set(id_remap.keys())
    cand |= {r["id"] for r in rep_nodes}
    model["_candidate_ids"] = list(cand)

    # The folded units are still REPRESENTED (via their domain rep) — record the
    # pre-consolidation container count so coverage credits them and abstraction
    # isn't punished for the very thing it rewards.
    model["_represented_unit_count"] = max(
        len(containers), model.get("_represented_unit_count", 0)
    )

    model["containers"] = cont
    log.info(
        "consolidate_containers_for_abstraction: %d containers → %d "
        "(%d folded into %d domain rep(s))",
        len(containers), len(new_containers), folded_total, len(rep_nodes),
    )
    return model


# ── drop_operational_noise ────────────────────────────────────────────────────

import re as _re

_NOISE_PATTERNS = _re.compile(
    r"load[-_ ]?gen|loadtest|benchmark|stress|locust\b|^k6$|gatling|jmeter"
    r"|prometheus|grafana|jaeger|zipkin|opentelemetry|otel[-_ ]?collector"
    r"|fluentd|fluent[-_ ]?bit|kibana|metrics?[-_ ]?(collector|exporter)|telemetry"
    r"|eureka|consul|zookeeper|config[-_ ]?server",
    _re.IGNORECASE,
)


def drop_operational_noise(model: dict) -> dict:
    """Remove load generators, observability sidecars, and registry agents.

    Safety guards: only when >3 containers; never drops below 2 containers.
    Removes the node's edges too. Logs every drop (no silent truncation).
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])

    if len(containers) <= 3:
        return model

    def _is_noise(node: dict) -> bool:
        label = node.get("label", "")
        nid   = node.get("id", "")
        return bool(_NOISE_PATTERNS.search(label) or _NOISE_PATTERNS.search(nid))

    noise_ids: set[str] = set()
    kept: list[dict] = []
    for c in containers:
        if _is_noise(c):
            noise_ids.add(c["id"])
        else:
            kept.append(c)

    # Never drop below 2
    if len(kept) < 2:
        return model

    if noise_ids:
        import logging as _log_noise
        _log_noise.getLogger(__name__).info(
            "drop_operational_noise: removed %d nodes: %s", len(noise_ids), sorted(noise_ids)
        )
        cont["containers"] = kept
        # Remove edges involving dropped nodes
        cont["relationships"] = [
            r for r in (cont.get("relationships") or [])
            if r.get("from") not in noise_ids and r.get("to") not in noise_ids
        ]
        ctx = model.get("context", {})
        ctx["relationships"] = [
            r for r in (ctx.get("relationships") or [])
            if r.get("from") not in noise_ids and r.get("to") not in noise_ids
        ]
        model["context"] = ctx
        model["containers"] = cont

    return model


# ── reduce_edges_for_readability ──────────────────────────────────────────────

def reduce_edges_for_readability(model: dict) -> dict:
    """Thin container↔container edges for diagram readability.

    1. Bidirectional collapse: keep one direction when (u,v) and (v,u) both exist.
    2. Transitive reduction: remove (u,v) when v is reachable from u via a path
       of length ≥2 through other container nodes, unless removal would orphan a node.

    Actor→container and container→datastore/external edges are never touched.
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    rels = cont.get("relationships", [])

    if not containers or not rels:
        return model

    container_ids: set[str] = {c["id"] for c in containers}
    spine: set = set(model.get("_spine_edges") or set())

    # Split into container↔container and other (preserve) edges
    cc_rels: list[dict] = []
    other_rels: list[dict] = []
    for r in rels:
        frm, to = r.get("from", ""), r.get("to", "")
        if frm in container_ids and to in container_ids:
            cc_rels.append(r)
        else:
            other_rels.append(r)

    if not cc_rels:
        return model

    # Step 1: deduplicate and bidirectional collapse
    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in cc_rels:
        frm, to = r.get("from", ""), r.get("to", "")
        if (frm, to) in seen_pairs or (to, frm) in seen_pairs:
            continue
        seen_pairs.add((frm, to))
        deduped.append(r)

    # Step 2: build adjacency for BFS (directed, using deduped edges)
    adj: dict[str, set[str]] = {c["id"]: set() for c in containers}
    for r in deduped:
        adj[r["from"]].add(r["to"])
        adj[r["to"]].add(r["from"])  # treat as undirected for reachability check

    # Build directed adjacency for transitive reduction
    directed_adj: dict[str, set[str]] = {c["id"]: set() for c in containers}
    for r in deduped:
        directed_adj[r["from"]].add(r["to"])

    def _reachable_via_2plus(src: str, dst: str, exclude_direct: bool = True) -> bool:
        """BFS: can we reach dst from src in ≥2 hops (not via direct edge)?"""
        visited = {src}
        queue = list(directed_adj[src] - ({dst} if exclude_direct else set()))
        while queue:
            node = queue.pop(0)
            if node == dst:
                return True
            if node not in visited:
                visited.add(node)
                queue.extend(n for n in directed_adj[node] if n not in visited)
        return False

    def _degree(nid: str, edge_set: list[dict]) -> int:
        return sum(1 for r in edge_set if r.get("from") == nid or r.get("to") == nid)

    # Step 3: transitive reduction
    reduced: list[dict] = []
    for r in deduped:
        frm, to = r["from"], r["to"]
        # Never prune the protected narrative spine.
        if (frm, to) in spine:
            reduced.append(r)
            continue
        if _reachable_via_2plus(frm, to, exclude_direct=True):
            # Check that removing this edge won't orphan either endpoint
            remaining = [x for x in deduped if x is not r]
            if _degree(frm, remaining) >= 1 and _degree(to, remaining) >= 1:
                import logging as _log_reduce
                _log_reduce.getLogger(__name__).debug("reduce_edges: dropped transitive edge %s→%s", frm, to)
                continue
        reduced.append(r)

    cont["relationships"] = reduced + other_rels
    model["containers"] = cont
    return model


# ── curate_significant_edges ──────────────────────────────────────────────────

def curate_significant_edges(model: dict, cap: int = 4) -> dict:
    """Cap per-source container→container fan-out to keep the diagram readable.

    Runs AFTER reduce_edges_for_readability (bidirectional collapse + transitive
    reduction). Operates ONLY on container↔container edges. Ingress edges
    (actor→container, in context.relationships) and data/external edges
    (container→datastore/cache/queue, container↔external) are NEVER capped or
    dropped — they are preserved verbatim.

    For each source container with more than ``cap`` outbound container→container
    edges, drop the least-significant edges — but ONLY those whose target stays
    reachable from the ingress roots without that edge (i.e. genuinely redundant
    arrows). An edge whose removal would orphan its target is always kept, so a
    pure star (every leaf reached only via this source) is left intact. This means
    a source may retain more than ``cap`` edges when the extra ones are structurally
    load-bearing; connectivity always wins over the cap.

    Least-significant-first ordering (drop these first):
      1. target total degree, LOWER first   (drop arrows to peripheral nodes last
         only if redundant; hubs with alternative paths shed redundant arrows first)
      2. cross-domain before intra-domain    (cross-domain spaghetti goes first)
      3. target id alphabetical (reversed)   (stable, deterministic tie-break)

    Logs every dropped edge. No silent truncation. Mutates and returns model.
    """
    import logging as _log_curate
    log = _log_curate.getLogger(__name__)

    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    rels = cont.get("relationships", [])
    if not containers or not rels or cap < 1:
        return model

    container_ids: set[str] = {c["id"] for c in containers}
    group_of = {c["id"]: c.get("group", "") for c in containers}
    spine: set = set(model.get("_spine_edges") or set())

    # Split into container↔container and other (preserved verbatim) edges
    cc_rels: list[dict] = []
    other_rels: list[dict] = []
    for r in rels:
        frm, to = r.get("from", ""), r.get("to", "")
        if frm in container_ids and to in container_ids:
            cc_rels.append(r)
        else:
            other_rels.append(r)
    if not cc_rels:
        return model

    # Degree computed ONCE over the cc set (undirected), for significance ranking.
    degree: dict[str, int] = {cid: 0 for cid in container_ids}
    for r in cc_rels:
        degree[r["from"]] = degree.get(r["from"], 0) + 1
        degree[r["to"]] = degree.get(r["to"], 0) + 1

    # Ingress roots: entrypoint + actor-targeted containers (reachability seed).
    entry = _pick_entrypoint(containers)
    ctx_rels = model.get("context", {}).get("relationships") or []
    ingress_targets = {r.get("to") for r in ctx_rels if r.get("to") in container_ids}
    roots: set[str] = set()
    if entry:
        roots.add(entry["id"])
    roots |= ingress_targets
    if not roots:
        roots = {r["from"] for r in cc_rels} or set(container_ids)

    def _target_reachable_without(edge: dict, kept: list[dict]) -> bool:
        """Is edge['to'] reachable from roots using kept edges minus this one?"""
        dst = edge["to"]
        fwd: dict[str, set[str]] = {}
        for r in kept:
            if r is edge:
                continue
            fwd.setdefault(r["from"], set()).add(r["to"])
        seen = set(roots)
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            for m in fwd.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return dst in seen

    # Least-significant-first: drop redundant cross-domain arrows to hubs first.
    def _drop_key(r: dict) -> tuple:
        tgt = r["to"]
        cross_domain = group_of.get(r["from"], "") != group_of.get(tgt, "")
        # higher degree + cross-domain = dropped earlier; id reversed for stability
        return (-degree.get(tgt, 0), 0 if cross_domain else 1, tgt)

    by_source: dict[str, list[dict]] = {}
    for r in cc_rels:
        by_source.setdefault(r["from"], []).append(r)

    kept_cc: list[dict] = list(cc_rels)
    dropped: list[dict] = []
    for src in sorted(by_source):
        fan = [e for e in kept_cc if e["from"] == src]
        if len(fan) <= cap:
            continue
        for e in sorted(fan, key=_drop_key):
            if len([x for x in kept_cc if x["from"] == src]) <= cap:
                break
            if (e["from"], e["to"]) in spine:
                continue  # never drop the protected narrative spine
            if _target_reachable_without(e, kept_cc):
                kept_cc.remove(e)
                dropped.append(e)

    if not dropped:
        return model

    for e in dropped:
        log.info("curate_significant_edges: dropped redundant %s→%s (fan-out cap=%d)",
                 e["from"], e["to"], cap)

    cont["relationships"] = kept_cc + other_rels
    model["containers"] = cont
    return model


# ── assign_container_roles ────────────────────────────────────────────────────

def assign_container_roles(model: dict) -> dict:
    """Persist a deterministic node['role'] so diagram intent is explicit.

    role is derived from existing signals only:
      - 'entrypoint'  : the single _pick_entrypoint container (ingress face)
      - otherwise the architectural layer (gateway/presentation/application/worker)

    Additive only; never removes or overrides 'layer' or 'group'.
    """
    cont = model.get("containers", {})
    containers = cont.get("containers", [])
    if not containers:
        return model
    entry = _pick_entrypoint(containers)
    entry_id = entry["id"] if entry else None
    for c in containers:
        if c["id"] == entry_id:
            c["role"] = "entrypoint"
        else:
            c["role"] = c.get("layer") or _node_layer(c)
    model["containers"] = cont
    return model


# ── Backward-compatibility alias ──────────────────────────────────────────────

def build_container_model(
    rich_facts: dict,
    repo_root: str,
    repo_name: str | None = None,
    manifests: dict | None = None,
    orchestration: dict | None = None,
) -> dict:
    """Alias for build_candidate_model (backward compatibility)."""
    return build_candidate_model(
        rich_facts, repo_root, repo_name=repo_name,
        manifests=manifests, orchestration=orchestration,
    )

