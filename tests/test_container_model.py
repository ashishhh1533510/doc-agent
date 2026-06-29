"""
Deterministic container_model tests — no pytest, no LLM, no network.

Tests build_container_model() and apply_enrichment() with synthetic rich_facts
for three real-world archetypes, plus renderer shape assertions for _emit_node.

Run:  ./venv/Scripts/python.exe tests/test_container_model.py
Exit code is non-zero if any case fails.
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.container_model import (
    build_container_model, apply_enrichment,
    infer_communication_graph, infer_entrypoint,
    synthesize_architecture_backbone,
)
from doc_agent.tools.output import render_c4_combined, _emit_node


# ── tiny harness (same style as test_extractor_hld_signals.py) ────────────────
_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_facts(files: list[dict], import_graph: dict | None = None) -> dict:
    """Wrap a file list into a minimal RichFacts dict."""
    return {
        "files": files,
        "import_graph": import_graph or {},
        "primary_language": None,
        "framework": None,
        "frameworks": [],
        "languages": [],
    }


def _kinds(model: dict) -> dict[str, list[str]]:
    """Return {kind: [ids]} across all node lists."""
    out: dict[str, list[str]] = {}
    for lst in (
        model["context"].get("actors", []),
        model["containers"].get("containers", []),
        model["containers"].get("databases", []),
        model["context"].get("external_systems", []),
        model["containers"].get("external_services", []),
    ):
        for node in (lst or []):
            out.setdefault(node.get("kind", "?"), []).append(node["id"])
    return out


def _labels(model: dict) -> set[str]:
    """All node labels across the whole model."""
    labels = set()
    for lst in (
        model["context"].get("actors", []),
        model["containers"].get("containers", []),
        model["containers"].get("databases", []),
        model["context"].get("external_systems", []),
        model["containers"].get("external_services", []),
    ):
        for node in (lst or []):
            labels.add(node.get("label", ""))
    return labels


def _no_capability_labels(labels: set[str]) -> bool:
    """True when no label looks like a business capability (a sanity guard)."""
    bad_words = {"management", "handling", "processing", "service layer"}
    return all(not any(bw in lbl.lower() for bw in bad_words) for lbl in labels)


# ══════════════════════════════════════════════════════════════════════════════
# Archetype 1: Express + Mongoose  ->  service + MongoDB datastore + person
# ══════════════════════════════════════════════════════════════════════════════

def test_express_mongoose_archetype():
    """NodeGoat style: one package.json, Express routes, Mongoose model."""
    with tempfile.TemporaryDirectory() as repo_root:
        # Create a package.json to mark the deployable root
        Path(repo_root, "package.json").write_text('{"name":"nodegoat"}')

        files = [
            {
                "file": os.path.join(repo_root, "app.js"),
                "language": "javascript",
                "imports": ["express", "mongoose"],
                "routes": [{"method": "GET", "path": "/", "handler": "home"}],
                "classes": [],
                "functions": [{"name": "app", "is_async": False, "signature": "app()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
            {
                "file": os.path.join(repo_root, "models", "user.js"),
                "language": "javascript",
                "imports": ["mongoose"],
                "routes": [],
                "classes": [{"name": "User", "bases": [], "is_db_model": True, "docstring": None,
                              "fields": [], "methods": [], "lineno": 1}],
                "functions": [],
            },
        ]
        facts = _make_facts(files)
        model = build_container_model(facts, repo_root)
        kinds = _kinds(model)

        check("express+mongoose: has person actor",
              bool(kinds.get("person")), str(kinds))
        check("express+mongoose: has service container",
              bool(kinds.get("service")), str(kinds))
        check("express+mongoose: has MongoDB datastore",
              "mongodb" in kinds.get("datastore", []) or
              any("mongodb" in nid for nid in kinds.get("datastore", [])),
              str(kinds))
        check("express+mongoose: zero capability-style labels",
              _no_capability_labels(_labels(model)),
              str(_labels(model)))
        check("express+mongoose: no external nodes",
              not kinds.get("external"), str(kinds))


# ══════════════════════════════════════════════════════════════════════════════
# Archetype 2: React (own pkg) + NestJS (own pkg) + pg
#   ->  web_app + service + PostgreSQL datastore + edge web->service
# ══════════════════════════════════════════════════════════════════════════════

def test_react_nestjs_postgres_archetype():
    """Realworld style: separate manifests for frontend and backend."""
    with tempfile.TemporaryDirectory() as repo_root:
        # Two separate manifests -> two deployable roots
        fe_dir = os.path.join(repo_root, "frontend")
        be_dir = os.path.join(repo_root, "backend")
        os.makedirs(fe_dir); os.makedirs(be_dir)
        Path(fe_dir, "package.json").write_text('{"name":"frontend"}')
        Path(be_dir, "package.json").write_text('{"name":"backend"}')

        files = [
            # Frontend — React SPA
            {
                "file": os.path.join(fe_dir, "src", "index.tsx"),
                "language": "typescript",
                "imports": ["react", "react-dom"],
                "routes": [],
                "classes": [],
                "functions": [{"name": "Root", "is_async": False, "signature": "Root()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
            # Backend — NestJS with pg
            {
                "file": os.path.join(be_dir, "src", "app.controller.ts"),
                "language": "typescript",
                "imports": ["@nestjs/common", "pg"],
                "routes": [{"method": "GET", "path": "/articles", "handler": "getArticles"}],
                "classes": [],
                "functions": [],
            },
        ]
        facts = _make_facts(files)
        from doc_agent.tools.container_model import discover_orchestration
        orch = discover_orchestration(repo_root)
        model = build_container_model(facts, repo_root)
        model = infer_communication_graph(model, facts, orch)
        kinds = _kinds(model)

        check("react+nestjs+pg: has web_app for frontend",
              bool(kinds.get("web_app")), str(kinds))
        check("react+nestjs+pg: has service for backend",
              bool(kinds.get("service")), str(kinds))
        check("react+nestjs+pg: has PostgreSQL datastore",
              any("postgresql" in nid or "postgres" in nid for nid in kinds.get("datastore", [])),
              str(kinds))

        # Verify web_app -> service edge exists
        all_rels = (model["containers"].get("relationships") or []) + \
                   (model["context"].get("relationships") or [])
        web_ids = set(kinds.get("web_app", []))
        svc_ids = set(kinds.get("service", []))
        has_fe_to_be = any(
            r.get("from") in web_ids and r.get("to") in svc_ids
            for r in all_rels
        )
        check("react+nestjs+pg: web_app -> service edge present",
              has_fe_to_be, str(all_rels))


# ══════════════════════════════════════════════════════════════════════════════
# Archetype 3: Next.js + Prisma + stripe + googleapis
#   ->  one web_app (fullstack, NOT split) + Database + Stripe + Google APIs externals
# ══════════════════════════════════════════════════════════════════════════════

def test_nextjs_prisma_externals_archetype():
    """cal.com style: fullstack Next.js, Prisma ORM, external SDKs."""
    with tempfile.TemporaryDirectory() as repo_root:
        Path(repo_root, "package.json").write_text('{"name":"cal"}')

        files = [
            {
                "file": os.path.join(repo_root, "pages", "index.tsx"),
                "language": "typescript",
                "imports": ["next", "react", "stripe", "googleapis"],
                "routes": [{"method": "GET", "path": "/", "handler": "Home"}],
                "classes": [],
                "functions": [{"name": "Home", "is_async": False, "signature": "Home()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
            {
                "file": os.path.join(repo_root, "lib", "prisma.ts"),
                "language": "typescript",
                "imports": ["prisma", "@prisma/client"],
                "routes": [],
                "classes": [],
                "functions": [],
            },
        ]
        facts = _make_facts(files)
        model = build_container_model(facts, repo_root)
        kinds = _kinds(model)

        # Next.js is fullstack -> must produce exactly ONE web_app, NOT split
        check("nextjs+prisma: exactly one web_app (not split)",
              len(kinds.get("web_app", [])) == 1, str(kinds))
        check("nextjs+prisma: no separate service container",
              len(kinds.get("service", [])) == 0, str(kinds))
        check("nextjs+prisma: has database datastore from Prisma",
              bool(kinds.get("datastore")), str(kinds))

        ext_labels = _labels(model) - {
            n["label"]
            for lst in (model["containers"].get("containers", []),
                        model["containers"].get("databases", []),
                        model["context"].get("actors", []))
            for n in lst
        }
        check("nextjs+prisma: Stripe appears as external",
              any("stripe" in lbl.lower() for lbl in ext_labels), str(ext_labels))
        check("nextjs+prisma: Google APIs appears as external",
              any("google" in lbl.lower() for lbl in ext_labels), str(ext_labels))


# ══════════════════════════════════════════════════════════════════════════════
# apply_enrichment tests
# ══════════════════════════════════════════════════════════════════════════════

def test_apply_enrichment_fills_text():
    """Enrichment updates system_name/label, descriptions, and edge labels."""
    model = {
        "context": {
            "system_name": "",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "api", "label": "uses"}],
        },
        "containers": {
            "system_label": "",
            "containers": [{"id": "api", "label": "API", "kind": "service", "description": "", "tech": ""}],
            "databases": [{"id": "db", "label": "DB", "kind": "datastore", "description": ""}],
            "external_services": [],
            "relationships": [{"from": "api", "to": "db", "label": "reads/writes"}],
        },
    }
    enrichment = {
        "system_purpose": "Realtime Chat API",
        "descriptions": {
            "user": "Human interacting with the chat interface.",
            "api": "Handles WebSocket connections and message delivery.",
            "db": "Persists messages and user sessions.",
        },
        "edge_labels": {
            "user__api": "connects via WebSocket",
            "api__db": "persists messages",
        },
    }
    result = apply_enrichment(model, enrichment)
    check("enrichment: system_name set",
          result["context"]["system_name"] == "Realtime Chat API")
    check("enrichment: system_label set",
          result["containers"]["system_label"] == "Realtime Chat API")
    check("enrichment: actor description filled",
          result["context"]["actors"][0]["description"] != "")
    check("enrichment: container description filled",
          result["containers"]["containers"][0]["description"] != "")
    check("enrichment: context edge label updated",
          result["context"]["relationships"][0]["label"] == "connects via WebSocket",
          result["context"]["relationships"][0]["label"])
    check("enrichment: container edge label updated",
          result["containers"]["relationships"][0]["label"] == "persists messages",
          result["containers"]["relationships"][0]["label"])


def test_apply_enrichment_ignores_unknown_ids():
    """Enrichment with made-up ids must not crash and must not add nodes."""
    model = {
        "context": {"system_name": "", "actors": [], "external_systems": [], "relationships": []},
        "containers": {"system_label": "", "containers": [], "databases": [], "external_services": [],
                       "relationships": []},
    }
    enrichment = {
        "system_purpose": "Test",
        "descriptions": {"invented_node": "should be ignored"},
        "edge_labels": {"a__b": "phantom"},
    }
    result = apply_enrichment(model, enrichment)
    check("enrichment: no new nodes added by unknown ids",
          not result["containers"]["containers"], str(result["containers"]["containers"]))


# ══════════════════════════════════════════════════════════════════════════════
# _emit_node shape assertions (Change C regression guard)
# ══════════════════════════════════════════════════════════════════════════════

def test_emit_node_shapes():
    """Shape syntax for each kind must match what render_c4_combined emits."""
    # person -> stadium ([" "])
    check("emit: person -> stadium shape",
          '(["' in _emit_node("user", "User", "person"))

    # service/web_app/worker -> rectangle [" "]
    check("emit: service -> rectangle shape",
          '[" ' not in _emit_node("api", "API", "service") and
          '["API"]' in _emit_node("api", "API", "service"),
          _emit_node("api", "API", "service"))

    # datastore -> cylinder [(" ")]
    ds = _emit_node("db", "MongoDB", "datastore")
    check("emit: datastore -> cylinder shape",
          '[("' in ds, ds)

    # cache -> cylinder
    cache = _emit_node("redis", "Redis", "cache")
    check("emit: cache -> cylinder shape",
          '[("' in cache, cache)

    # queue -> parallelogram [/" "/]
    queue = _emit_node("q", "Kafka", "queue")
    check("emit: queue -> parallelogram shape",
          '[/"' in queue, queue)

    # external -> hexagon {{"..."}} + :::ext
    ext = _emit_node("stripe", "Stripe", "external")
    check("emit: external -> hexagon shape (double braces)",
          '{{' in ext and '}}' in ext, ext)
    check("emit: external -> :::ext class applied",
          ":::ext" in ext, ext)


def test_render_c4_cylinder_present():
    """A model with a database node must produce a cylinder shape in flowchart output."""
    model = {
        "context": {
            "system_name": "Test System",
            "actors": [{"id": "user", "label": "User", "kind": "person"}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "api", "label": "uses"}],
        },
        "containers": {
            "system_label": "Test System",
            "containers": [{"id": "api", "label": "API", "kind": "service", "tech": "Express"}],
            "databases": [{"id": "mongodb", "label": "MongoDB", "kind": "datastore"}],
            "external_services": [],
            "relationships": [{"from": "api", "to": "mongodb", "label": "reads/writes"}],
        },
    }
    rendered = render_c4_combined(model)
    check("render: cylinder [( present for datastore (flowchart)",
          "[(" in rendered, rendered[:200])
    check("render: actor stadium ([ present (flowchart)",
          "([" in rendered, rendered[:200])
    check("render: flowchart TD first line",
          rendered.strip().split("\n")[1].startswith("flowchart TD"), rendered[:50])
    check("render: subgraph SYS boundary present",
          'subgraph SYS[' in rendered, rendered[:200])


def test_render_c4_external_hexagon():
    """External nodes must produce a hexagon + :::ext class in flowchart output."""
    model = {
        "context": {
            "system_name": "Test",
            "actors": [],
            "external_systems": [{"id": "stripe", "label": "Stripe", "kind": "external"}],
            "relationships": [{"from": "api", "to": "stripe", "label": "uses"}],
        },
        "containers": {
            "system_label": "Test",
            "containers": [{"id": "api", "label": "API", "kind": "service", "tech": ""}],
            "databases": [],
            "external_services": [],
            "relationships": [],
        },
    }
    rendered = render_c4_combined(model)
    check("render: external -> hexagon {{ + :::ext in output (flowchart)",
          "{{" in rendered and ":::ext" in rendered, rendered[:300])
    check("render: flowchart TD first line (external model)",
          rendered.strip().split("\n")[1].startswith("flowchart TD"), rendered[:50])


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Manifest-grounded naming tests (fix: temp-clone-dir name regression)
# ══════════════════════════════════════════════════════════════════════════════

def test_manifest_name_overrides_temp_dir():
    """Container label must come from pom.xml <name>/<artifactId>, NOT from
    a temp-dir name like 'doc_agent_clone_ws75e72p'."""
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory(prefix="doc_agent_clone_") as repo_root:
        # Simulate a single-module Spring app: pom.xml at root
        pom = Path(repo_root, "pom.xml")
        pom.write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>io.spring</groupId>
  <artifactId>realworld-backend</artifactId>
  <name>Spring Boot Realworld</name>
</project>""")

        files = [
            {
                "file": os.path.join(repo_root, "src", "main", "Application.java"),
                "language": "java",
                "imports": ["org.springframework.boot.SpringApplication"],
                "routes": [{"method": "GET", "path": "/api/articles", "handler": "list"}],
                "classes": [],
                "functions": [{"name": "main", "is_async": False, "signature": "main(String[])",
                                "returns": None, "decorators": [], "routes": [], "calls": [],
                                "docstring": None, "lineno": 1}],
            },
        ]
        facts = _make_facts(files)
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(facts, repo_root, repo_name="spring-realworld", manifests=manifests)
        containers = model["containers"].get("containers", [])

        # Must NOT contain the temp-dir fragment
        temp_dir_name = Path(repo_root).name.lower()
        all_labels = {c["label"].lower() for c in containers}
        check("manifest-name: label does not contain temp-dir name",
              not any(temp_dir_name in lbl for lbl in all_labels),
              str(all_labels))

        # Must use the manifest project name or repo_name
        check("manifest-name: label contains 'realworld' or 'spring'",
              any("realworld" in lbl or "spring" in lbl for lbl in all_labels),
              str(all_labels))


def test_repo_name_fallback_when_no_manifest_name():
    """When pom.xml has no <name>/<artifactId>, repo_name is the fallback."""
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory(prefix="doc_agent_clone_") as repo_root:
        pom = Path(repo_root, "pom.xml")
        # Minimal pom with no name/artifactId so manifest project_name will be empty
        pom.write_text("<?xml version='1.0'?><project></project>")

        files = [
            {
                "file": os.path.join(repo_root, "Main.java"),
                "language": "java",
                "imports": ["org.springframework.boot"],
                "routes": [{"method": "POST", "path": "/users", "handler": "create"}],
                "classes": [],
                "functions": [{"name": "main", "is_async": False, "signature": "main()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
        ]
        facts = _make_facts(files)
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(
            facts, repo_root,
            repo_name="my-awesome-repo",
            manifests=manifests,
        )
        containers = model["containers"].get("containers", [])
        all_labels = {c["label"].lower() for c in containers}
        temp_dir_name = Path(repo_root).name.lower()

        check("repo_name fallback: label does not contain temp-dir name",
              not any(temp_dir_name in lbl for lbl in all_labels),
              str(all_labels))
        check("repo_name fallback: label contains 'my-awesome-repo' or 'my awesome repo'",
              any("my" in lbl and "awesome" in lbl for lbl in all_labels),
              str(all_labels))


# ══════════════════════════════════════════════════════════════════════════════
# Manifest-dep datastore detection tests (fix: MyBatis/SQLite not detected)
# ══════════════════════════════════════════════════════════════════════════════

def test_mybatis_sqlite_from_pom_deps():
    """MyBatis + SQLite declared in pom.xml must produce a datastore node
    even if no DB import appears in source files."""
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory() as repo_root:
        pom = Path(repo_root, "pom.xml")
        pom.write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>realworld</artifactId>
  <dependencies>
    <dependency>
      <groupId>org.mybatis.spring.boot</groupId>
      <artifactId>mybatis-spring-boot-starter</artifactId>
    </dependency>
    <dependency>
      <groupId>org.xerial</groupId>
      <artifactId>sqlite-jdbc</artifactId>
    </dependency>
  </dependencies>
</project>""")

        files = [
            {
                "file": os.path.join(repo_root, "src", "Main.java"),
                "language": "java",
                "imports": ["org.springframework.boot.SpringApplication"],
                "routes": [{"method": "GET", "path": "/api/articles", "handler": "list"}],
                "classes": [],
                "functions": [{"name": "main", "is_async": False, "signature": "main()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
        ]
        facts = _make_facts(files)
        manifests = parse_all_manifests(repo_root)
        from doc_agent.tools.container_model import discover_orchestration
        orch = discover_orchestration(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests)
        model = infer_communication_graph(model, facts, orch)
        dbs = model["containers"].get("databases", [])
        db_labels = {d["label"].lower() for d in dbs}

        check("mybatis+sqlite pom: datastore node exists",
              bool(dbs), str(db_labels))
        check("mybatis+sqlite pom: SQLite or Relational Database in datastore labels",
              any("sqlite" in lbl or "relational" in lbl for lbl in db_labels),
              str(db_labels))

        # Must also have a reads/writes edge from the container to the datastore
        all_rels = model["containers"].get("relationships", [])
        db_ids = {d["id"] for d in dbs}
        has_edge = any(r.get("to") in db_ids for r in all_rels)
        check("mybatis+sqlite pom: reads/writes edge to datastore",
              has_edge, str(all_rels))


def test_java_mapper_annotation_datastore():
    """@Mapper on a Java class -> is_db_model=True -> datastore node appears."""
    with tempfile.TemporaryDirectory() as repo_root:
        Path(repo_root, "pom.xml").write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>myapp</artifactId>
</project>""")

        files = [
            {
                "file": os.path.join(repo_root, "src", "ArticleMapper.java"),
                "language": "java",
                "imports": ["org.apache.ibatis.annotations.Mapper"],
                "routes": [],
                "classes": [{"name": "ArticleMapper", "bases": [], "is_db_model": True,
                              "docstring": None, "fields": [], "methods": [], "lineno": 1}],
                "functions": [],
            },
            {
                "file": os.path.join(repo_root, "src", "ArticleController.java"),
                "language": "java",
                "imports": [],
                "routes": [{"method": "GET", "path": "/api/articles", "handler": "list"}],
                "classes": [],
                "functions": [],
            },
        ]
        facts = _make_facts(files)
        model = build_container_model(facts, repo_root)
        dbs = model["containers"].get("databases", [])

        check("@Mapper class: datastore node emitted via is_db_model",
              bool(dbs), str(dbs))


def test_package_json_dep_datastore():
    """package.json with pg dependency -> PostgreSQL datastore node via manifest scanning."""
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory() as repo_root:
        pj = Path(repo_root, "package.json")
        pj.write_text('{"name":"myapi","dependencies":{"express":"*","pg":"*"}}')

        files = [
            {
                "file": os.path.join(repo_root, "index.js"),
                "language": "javascript",
                "imports": ["express"],   # pg NOT imported in source — only in package.json
                "routes": [{"method": "GET", "path": "/users", "handler": "list"}],
                "classes": [],
                "functions": [{"name": "main", "is_async": False, "signature": "main()", "returns": None,
                                "decorators": [], "routes": [], "calls": [], "docstring": None, "lineno": 1}],
            },
        ]
        facts = _make_facts(files)
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests)
        dbs = model["containers"].get("databases", [])
        db_labels = {d["label"].lower() for d in dbs}

        check("package.json pg dep: PostgreSQL datastore emitted",
              any("postgresql" in lbl or "postgres" in lbl for lbl in db_labels),
              str(db_labels))


def test_consolidate_multiple_relational_to_one():
    """Multiple relational engines (Postgres + MySQL + H2) collapse to one node."""
    from doc_agent.tools.container_model import _consolidate_datastores
    db_found = {
        "PostgreSQL":        ("PostgreSQL",        "datastore"),
        "MySQL":             ("MySQL",             "datastore"),
        "H2 Database":       ("H2 Database",       "datastore"),
        "Relational Database": ("Relational Database", "datastore"),
        "Elasticsearch":     ("Elasticsearch",     "datastore"),
    }
    result = _consolidate_datastores(db_found)
    relational_labels = {"PostgreSQL", "MySQL", "H2 Database", "Relational Database"}
    relational_in_result = [l for l in result if l in relational_labels]
    check("consolidate: exactly one relational node", len(relational_in_result) == 1,
          str(relational_in_result))
    check("consolidate: non-relational Elasticsearch survives",
          "Elasticsearch" in result)


def test_consolidate_single_concrete_engine_keeps_name():
    """Single concrete engine (SQLite) is kept as-is; H2 test DB is dropped."""
    from doc_agent.tools.container_model import _consolidate_datastores
    db_found = {
        "SQLite":    ("SQLite",    "datastore"),
        "H2 Database": ("H2 Database", "datastore"),
        "Relational Database": ("Relational Database", "datastore"),
    }
    # SQLite is NOT in the relational set, so it should survive; H2 + generic get collapsed
    result = _consolidate_datastores(db_found)
    check("consolidate single: SQLite kept", "SQLite" in result, str(list(result.keys())))
    check("consolidate single: H2 removed", "H2 Database" not in result, str(list(result.keys())))


def test_consolidate_non_relational_untouched():
    """Redis cache and Kafka queue survive consolidation unchanged."""
    from doc_agent.tools.container_model import _consolidate_datastores
    db_found = {
        "PostgreSQL": ("PostgreSQL", "datastore"),
        "Redis":      ("Redis",      "cache"),
        "Kafka":      ("Kafka",      "queue"),
    }
    result = _consolidate_datastores(db_found)
    check("consolidate: Redis survives", "Redis" in result)
    check("consolidate: Kafka survives", "Kafka" in result)
    check("consolidate: PostgreSQL kept (single concrete)", "PostgreSQL" in result,
          str(list(result.keys())))


def test_test_suite_module_not_a_container():
    """A pom.xml inside test-suite/ must not produce a container node."""
    import tempfile
    from pathlib import Path
    from doc_agent.tools.container_model import build_container_model

    with tempfile.TemporaryDirectory(prefix="doc_agent_clone_") as repo:
        # root pom
        Path(repo, "pom.xml").write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>my-app</artifactId>
</project>""")
        # test-suite sub-module pom
        ts = Path(repo, "test-suite", "module")
        ts.mkdir(parents=True)
        (ts / "pom.xml").write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>my-app-test-suite</artifactId>
</project>""")
        # minimal Java file in test-suite so extractor has something
        (ts / "Stub.java").write_text("public class Stub {}")

        from doc_agent.tools.manifest_parser import parse_all_manifests
        manifests = parse_all_manifests(repo)
        # Build with an empty rich_facts to isolate the non-deployable filter
        rf = {"files": [], "primary_language": "java", "framework": "", "frameworks": []}
        model = build_container_model(rf, repo, repo_name="my-app", manifests=manifests)
        labels = [c["label"] for c in model["containers"]["containers"]]
        check("test-suite: no container from test-suite dir",
              not any("test" in lbl.lower() and "suite" in lbl.lower() for lbl in labels),
              str(labels))


def _make_model(n_containers=1, n_dbs=1, n_exts=0, n_actors=1):
    containers = [
        {"id": f"svc{i}", "label": f"Service {i}", "kind": "service", "tech": "Python", "description": ""}
        for i in range(n_containers)
    ]
    databases = [
        {"id": f"db{i}", "label": f"DB {i}", "kind": "datastore", "tech": "PostgreSQL", "description": ""}
        for i in range(n_dbs)
    ]
    ext_services = [
        {"id": f"ext{i}", "label": f"External {i}", "kind": "external", "description": ""}
        for i in range(n_exts)
    ]
    actors = [{"id": "user", "label": "User", "kind": "person", "description": ""}] if n_actors > 0 else []
    rels = []
    if actors and containers:
        rels.append({"from": "user", "to": "svc0", "label": "uses"})
    for i in range(n_containers):
        for j in range(n_dbs):
            rels.append({"from": f"svc{i}", "to": f"db{j}", "label": "reads/writes"})
    for i in range(n_exts):
        rels.append({"from": "svc0", "to": f"ext{i}", "label": "calls"})
    return {
        "context": {
            "system_name": "Test System",
            "system_purpose": "For tests",
            "actors": actors,
            "external_systems": [],
            "relationships": [],
        },
        "containers": {
            "system_label": "Test System",
            "containers": containers,
            "databases": databases,
            "external_services": ext_services,
            "relationships": rels,
        },
    }


def test_c4_container_native_syntax():
    """render_c4_container emits flowchart TD with subgraph boundary and shaped nodes."""
    from doc_agent.tools.output import render_c4_container
    from doc_agent.tools.diagram_validator import validate_mermaid
    model = _make_model(n_containers=2, n_dbs=1, n_exts=1, n_actors=1)
    out = render_c4_container(model)
    check("c4_container: flowchart TD first line", out.strip().split("\n")[1].startswith("flowchart TD"), out[:50])
    check("c4_container: subgraph SYS[", "subgraph SYS[" in out)
    check("c4_container: datastore cylinder [(", "[(" in out)
    check("c4_container: actor stadium ([", "([" in out)
    check("c4_container: --> edge", "-->" in out)
    check("c4_container: no C4Container", "C4Container" not in out)
    v = validate_mermaid(out)
    check("c4_container: valid=True", v.get("valid") is True, str(v))
    check("c4_container: type=flowchart", v.get("diagram_type") == "flowchart", str(v))


def test_c4_container_queue_node():
    """Queue kind databases produce parallelogram shape [/ /]."""
    from doc_agent.tools.output import render_c4_container
    model = _make_model()
    model["containers"]["databases"].append(
        {"id": "kafka", "label": "Kafka", "kind": "queue", "tech": "Kafka", "description": ""}
    )
    model["containers"]["relationships"].append({"from": "svc0", "to": "kafka", "label": "publishes"})
    out = render_c4_container(model)
    check("c4_queue: parallelogram [/ shape", "[/" in out)


def test_c4_context_native_syntax():
    """render_c4_context emits flowchart TD with single system box, no subgraph internals."""
    from doc_agent.tools.output import render_c4_context
    from doc_agent.tools.diagram_validator import validate_mermaid
    model = _make_model(n_containers=2, n_dbs=1, n_exts=1, n_actors=1)
    out = render_c4_context(model)
    check("c4_ctx: flowchart TD first line", out.strip().startswith("flowchart TD"), out[:50])
    check("c4_ctx: system box present", "Test System" in out)
    check("c4_ctx: no C4Context", "C4Context" not in out)
    check("c4_ctx: --> edge", "-->" in out)
    v = validate_mermaid(out)
    check("c4_ctx: valid=True", v.get("valid") is True, str(v))
    check("c4_ctx: type=flowchart", v.get("diagram_type") == "flowchart", str(v))


def test_c4_combined_is_container():
    """render_c4_combined emits flowchart TD (same renderer as container view)."""
    from doc_agent.tools.output import render_c4_combined
    from doc_agent.tools.diagram_validator import validate_mermaid
    model = _make_model()
    out = render_c4_combined(model)
    check("combined: flowchart TD first line", out.strip().split("\n")[1].startswith("flowchart TD"), out[:50])
    check("combined: subgraph SYS[", "subgraph SYS[" in out)
    v = validate_mermaid(out)
    check("combined: type=flowchart", v.get("diagram_type") == "flowchart", str(v))


def test_c4_decision_small():
    """Small model (<=8C, <=4E, <=3A) stays under threshold."""
    from doc_agent.tools.c4_views import count_elements, detect_systems
    model = _make_model(n_containers=3, n_dbs=2, n_exts=1, n_actors=1)
    systems = detect_systems(model, [])
    check("decision_small: 1 system", len(systems) == 1, len(systems))
    C, E, A = count_elements(systems[0])
    check("decision_small: C<=8", C <= 8, C)
    check("decision_small: E<=4", E <= 4, E)
    check("decision_small: A<=3", A <= 3, A)


def test_c4_decision_large_escalates():
    """Large model (>8 containers) escalates to context + grouped container diagrams."""
    from doc_agent.tools.c4_views import count_elements, group_containers
    from doc_agent.tools.output import render_c4_context, render_c4_container
    model = _make_model(n_containers=10, n_dbs=2, n_exts=2, n_actors=1)
    C, E, A = count_elements(model)
    check("decision_large: C>8", C > 8, C)
    groups = group_containers(model, max_per=8)
    check("decision_large: >=2 groups", len(groups) >= 2, len(groups))
    for grp in groups:
        gC, _, _ = count_elements(grp)
        check("decision_large: each group<=8", gC <= 8, gC)
    ctx_out = render_c4_context(model)
    check("decision_large: flowchart TD context", ctx_out.strip().startswith("flowchart TD"))
    grp_out = render_c4_container(groups[0])
    check("decision_large: flowchart TD container", grp_out.strip().split("\n")[1].startswith("flowchart TD"))


def test_render_orphan_external_filtered():
    """External with no relationships must be filtered out (not rendered as noise)."""
    from doc_agent.tools.output import render_c4_combined
    model = _make_model(n_containers=1, n_dbs=1, n_exts=0, n_actors=1)
    model["containers"]["external_services"].append(
        {"id": "orphan_ext", "label": "Orphan External", "kind": "external", "description": ""}
    )
    out = render_c4_combined(model)
    check("orphan_ext: label absent from output (filtered)", "Orphan External" not in out, out[:300])
    check("orphan_ext: no orphan hex node", "orphan_ext" not in out, out[:300])


def test_render_connected_external_present():
    """External that participates in a relationship must appear in the diagram."""
    from doc_agent.tools.output import render_c4_combined
    model = _make_model(n_containers=1, n_dbs=0, n_exts=0, n_actors=1)
    model["containers"]["external_services"].append(
        {"id": "payment_gw", "label": "Payment Gateway", "kind": "external", "description": ""}
    )
    model["containers"]["relationships"].append(
        {"from": "svc0", "to": "payment_gw", "label": "processes payments via"}
    )
    out = render_c4_combined(model)
    check("connected_ext: label present in output", "Payment Gateway" in out, out[:400])
    check("connected_ext: hex shape {{ present", "{{" in out, out[:400])


def test_render_disconnected_services_single_diagram():
    """Mutually disconnected services must all appear in ONE combined diagram."""
    from doc_agent.tools.output import render_c4_combined
    model = _make_model(n_containers=3, n_dbs=3, n_exts=0, n_actors=1)
    model["containers"]["relationships"] = [
        {"from": "user", "to": "svc0", "label": "uses"},
        {"from": "svc0", "to": "db0", "label": "reads/writes"},
        {"from": "svc1", "to": "db1", "label": "reads/writes"},
        {"from": "svc2", "to": "db2", "label": "reads/writes"},
    ]
    out = render_c4_combined(model)
    check("disconnected: single flowchart TD", out.strip().split("\n")[1].startswith("flowchart TD"), out[:50])
    for i in range(3):
        check(f"disconnected: Service {i} present", f"Service {i}" in out, out[:500])
        check(f"disconnected: DB {i} present", f"DB {i}" in out, out[:500])


def test_detect_systems_accepts_orchestration_dict():
    """detect_systems must not crash when passed the real discover_orchestration dict shape."""
    from doc_agent.tools.c4_views import detect_systems
    orch = {"compose_services": [{"name": "api"}], "procfile_processes": [], "has_k8s": False}
    model = _make_model(n_containers=2, n_dbs=1)
    systems = detect_systems(model, orch)
    check("detect_systems: dict orchestration no crash", isinstance(systems, list), str(type(systems)))
    check("detect_systems: connected containers -> 1 system", len(systems) == 1, len(systems))


def test_detect_systems_none():
    """detect_systems(model, None) must return [model] without crashing."""
    from doc_agent.tools.c4_views import detect_systems
    model = _make_model(n_containers=2, n_dbs=1)
    systems = detect_systems(model, None)
    check("detect_systems: None orchestration no crash", isinstance(systems, list), str(type(systems)))
    check("detect_systems: None -> 1 system (connected)", len(systems) == 1, len(systems))


def test_detect_systems_shared_db_single():
    """Two services sharing one database = 1 system (connected via shared infra)."""
    from doc_agent.tools.c4_views import detect_systems
    model = {
        "context": {"system_name": "Test", "actors": [], "external_systems": [], "relationships": []},
        "containers": {
            "system_label": "Test",
            "containers": [
                {"id": "svcA", "label": "Service A", "kind": "service", "tech": "", "description": ""},
                {"id": "svcB", "label": "Service B", "kind": "service", "tech": "", "description": ""},
            ],
            "databases": [
                {"id": "db0", "label": "DB", "kind": "datastore", "tech": "PostgreSQL", "description": ""},
            ],
            "external_services": [],
            "relationships": [
                {"from": "svcA", "to": "db0", "label": "reads/writes"},
                {"from": "svcB", "to": "db0", "label": "reads/writes"},
            ],
        },
    }
    systems = detect_systems(model, None)
    check("detect_systems: shared db -> 1 system", len(systems) == 1, len(systems))


def test_detect_systems_splits_disconnected():
    """Two services each with their own db and no cross edges = 2 systems."""
    from doc_agent.tools.c4_views import detect_systems
    model = {
        "context": {"system_name": "Test", "actors": [], "external_systems": [], "relationships": []},
        "containers": {
            "system_label": "Test",
            "containers": [
                {"id": "svcA", "label": "Service A", "kind": "service", "tech": "", "description": ""},
                {"id": "svcB", "label": "Service B", "kind": "service", "tech": "", "description": ""},
            ],
            "databases": [
                {"id": "dbA", "label": "DB A", "kind": "datastore", "tech": "PostgreSQL", "description": ""},
                {"id": "dbB", "label": "DB B", "kind": "datastore", "tech": "MySQL", "description": ""},
            ],
            "external_services": [],
            "relationships": [
                {"from": "svcA", "to": "dbA", "label": "reads/writes"},
                {"from": "svcB", "to": "dbB", "label": "reads/writes"},
            ],
        },
    }
    systems = detect_systems(model, None)
    check("detect_systems: disconnected -> 2 systems", len(systems) == 2, len(systems))
    container_counts = [len(s["containers"]["containers"]) for s in systems]
    check("detect_systems: each sub-model has 1 container", all(c == 1 for c in container_counts), container_counts)


def test_merge_orchestration_adds_compose_services():
    """merge_orchestration adds compose services not already in the model."""
    from doc_agent.tools.container_model import merge_orchestration
    model = _make_model(n_containers=1, n_dbs=0, n_exts=0, n_actors=1)
    orch = {
        "compose_services": [
            {"name": "api",     "image": None,       "build_dir": None, "is_auxiliary": False, "profiles": []},
            {"name": "worker",  "image": None,       "build_dir": None, "is_auxiliary": False, "profiles": []},
            {"name": "redis",   "image": "redis:7",  "build_dir": None, "is_auxiliary": False, "profiles": []},
        ],
        "procfile_processes": [],
        "has_k8s": False,
        "k8s_workloads": [],
    }
    result = merge_orchestration(model, orch)
    cont = result["containers"]
    all_ids = {c["id"] for c in cont.get("containers", [])} | {d["id"] for d in cont.get("databases", [])}
    check("orchestration: api container added", "api" in all_ids, all_ids)
    check("orchestration: worker container added", "worker" in all_ids, all_ids)
    check("orchestration: redis routed to databases", any(d["kind"] == "cache" for d in cont.get("databases", [])), cont.get("databases"))


def test_merge_orchestration_adds_depends_on_edges():
    """infer_communication_graph adds depends_on relationships from compose orchestration."""
    from doc_agent.tools.container_model import merge_orchestration
    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = []
    orch = {
        "compose_services": [
            {"name": "web",  "image": None, "build_dir": None, "is_auxiliary": False, "profiles": [],
             "depends_on": ["api"], "links": []},
            {"name": "api",  "image": None, "build_dir": None, "is_auxiliary": False, "profiles": [],
             "depends_on": [], "links": []},
        ],
        "procfile_processes": [],
        "has_k8s": False,
        "k8s_workloads": [],
        "k8s_service_names": [],
    }
    # merge_orchestration adds nodes; infer_communication_graph adds edges
    result = merge_orchestration(model, orch)
    result = infer_communication_graph(result, _make_facts([]), orch)
    rels = result["containers"].get("relationships", [])
    check("orchestration: depends_on edge web->api added",
          any(r["from"] == "web" and r["to"] == "api" for r in rels), rels)


def test_merge_orchestration_noop_when_empty():
    """merge_orchestration is a no-op for repos with no orchestration evidence."""
    from doc_agent.tools.container_model import merge_orchestration
    model = _make_model(n_containers=2, n_dbs=1, n_exts=0, n_actors=1)
    orig_containers = list(model["containers"]["containers"])
    orch = {"compose_services": [], "procfile_processes": [], "has_k8s": False, "k8s_workloads": []}
    result = merge_orchestration(model, orch)
    check("orchestration noop: same container count",
          len(result["containers"]["containers"]) == len(orig_containers),
          len(result["containers"]["containers"]))


def test_validate_model_fail_multi_container_no_rels():
    """validate_model fails when >1 container exists but 0 relationships."""
    from doc_agent.tools.container_model import validate_model
    model = _make_model(n_containers=2, n_dbs=0, n_exts=0, n_actors=1)
    model["containers"]["relationships"] = []
    model["context"]["relationships"] = []
    report = validate_model(model)
    check("validate: fails on no rels", not report["passed"], report)
    check("validate: fail finding present", any(f["level"] == "fail" for f in report["findings"]), report["findings"])


def test_validate_model_warn_orphan_externals():
    """validate_model warns when all externals are orphans."""
    from doc_agent.tools.container_model import validate_model
    model = _make_model(n_containers=1, n_dbs=0, n_exts=1, n_actors=1)
    model["containers"]["relationships"] = []
    model["context"]["relationships"] = []
    report = validate_model(model)
    check("validate: warn on orphan ext", any(f["level"] == "warn" for f in report["findings"]), report["findings"])


def test_validate_model_passes_healthy():
    """validate_model passes for a well-connected model."""
    from doc_agent.tools.container_model import validate_model
    model = _make_model(n_containers=2, n_dbs=1, n_exts=0, n_actors=1)
    report = validate_model(model)
    check("validate: healthy model passes", report["passed"], report["findings"])
    check("validate: score has container_count", "container_count" in report["score"], report["score"])


def test_render_containers_layered_inside_boundary():
    """Layered Combined HLD: containers + datastores render in tier lanes inside SYS."""
    from doc_agent.tools.output import render_c4_combined
    model = _make_model(n_containers=1, n_dbs=1, n_exts=0, n_actors=1)
    model["containers"]["relationships"] = [
        {"from": "svc0", "to": "db0", "label": "reads/writes"},
        {"from": "user", "to": "svc0", "label": "uses"},
    ]
    out = render_c4_combined(model)
    check("layered: SYS boundary present", "subgraph SYS" in out, out[:600])
    check("layered: Services lane subgraph", "subgraph lane_services" in out, out[:600])
    check("layered: Data Stores lane subgraph", "subgraph lane_data_stores" in out, out[:600])
    check("layered: service node present", "svc0[" in out, out[:600])
    check("layered: datastore node present", "db0[" in out, out[:600])


def test_render_queue_node_lane():
    """A queue node renders in its own Messaging lane inside SYS."""
    from doc_agent.tools.output import render_c4_combined
    model = _make_model(n_containers=1, n_dbs=0, n_exts=0, n_actors=1)
    model["containers"]["databases"].append(
        {"id": "event_bus", "label": "Event Bus", "kind": "queue", "tech": "Kafka", "description": ""}
    )
    model["containers"]["relationships"].append(
        {"from": "svc0", "to": "event_bus", "label": "publishes/consumes"}
    )
    out = render_c4_combined(model)
    check("queue lane: Event Bus present", "Event Bus" in out, out)
    check("queue lane: Messaging lane subgraph", "subgraph lane_messaging" in out, out[:700])


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: evidence fusion + language-agnostic discovery
# ══════════════════════════════════════════════════════════════════════════════

def test_dockerfile_only_dir_admitted_no_facts():
    """A dir with only a Dockerfile (no extracted facts) is admitted via the Dockerfile floor.
    This simulates a Go service where the extractor produces zero facts.
    """
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory() as repo_root:
        # Root has one manifest (multi-manifest triggers confidence scoring)
        svc_a = os.path.join(repo_root, "svc-a")
        svc_go = os.path.join(repo_root, "svc-go")
        os.makedirs(svc_a); os.makedirs(svc_go)
        Path(svc_a, "package.json").write_text('{"name":"svc-a"}')
        Path(svc_go, "Dockerfile").write_text("FROM golang:1.21\nCOPY . .\nRUN go build")
        # Also put a manifest in svc-a to make multi-manifest kick in
        # (need at least 2 manifest dirs to trigger confidence scoring)

        files_a = [
            {
                "file": os.path.join(svc_a, "index.js"),
                "language": "javascript",
                "imports": ["express"],
                "routes": [{"method": "GET", "path": "/", "handler": "home"}],
                "classes": [], "functions": [],
            }
        ]
        facts = _make_facts(files_a)
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests)
        containers = model["containers"]["containers"]
        labels = {c["label"] for c in containers}
        ids = {c["id"] for c in containers}

        check("dockerfile_floor: svc-go container admitted (Dockerfile floor)",
              any("go" in lbl.lower() or "svc_go" in cid for lbl in labels for cid in ids),
              str(labels))
        check("dockerfile_floor: at least 2 containers total",
              len(containers) >= 2, str(labels))


def test_sibling_service_dirs_stay_separate():
    """payment-service and payment-worker must stay as TWO distinct units (no name merge)."""
    with tempfile.TemporaryDirectory() as repo_root:
        svc = os.path.join(repo_root, "payment-service")
        worker = os.path.join(repo_root, "payment-worker")
        os.makedirs(svc); os.makedirs(worker)
        Path(svc, "Dockerfile").write_text("FROM python:3.11")
        Path(worker, "Dockerfile").write_text("FROM python:3.11")

        facts = _make_facts([])  # no extracted facts (simulating Go/unextracted services)
        model = build_container_model(facts, repo_root)
        containers = model["containers"]["containers"]

        check("sibling_dirs: payment-service present",
              any("payment" in c["label"].lower() and "service" in c["label"].lower() for c in containers),
              str([c["label"] for c in containers]))
        check("sibling_dirs: payment-worker present",
              any("worker" in c["label"].lower() for c in containers),
              str([c["label"] for c in containers]))
        check("sibling_dirs: exactly 2 containers",
              len(containers) == 2, str([c["label"] for c in containers]))


def test_k8s_workload_discovery_nested_dir():
    """k8s workloads under a non-standard dir (kubernetes-manifests/) are found."""
    from doc_agent.tools.container_model import discover_orchestration

    with tempfile.TemporaryDirectory() as repo_root:
        k8s_dir = os.path.join(repo_root, "kubernetes-manifests")
        os.makedirs(k8s_dir)
        deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkoutservice
spec:
  replicas: 1
"""
        Path(k8s_dir, "checkoutservice.yaml").write_text(deployment_yaml)

        orch = discover_orchestration(repo_root)
        workload_names = [wl["name"] for wl in orch.get("k8s_workloads", [])]

        check("k8s_nested: checkoutservice found in kubernetes-manifests/",
              "checkoutservice" in workload_names, workload_names)
        check("k8s_nested: has_k8s set to True",
              orch.get("has_k8s") is True, orch.get("has_k8s"))


def test_k8s_statefulset_found():
    """StatefulSet kind is also captured by the recursive k8s scan."""
    from doc_agent.tools.container_model import discover_orchestration

    with tempfile.TemporaryDirectory() as repo_root:
        release_dir = os.path.join(repo_root, "release")
        os.makedirs(release_dir)
        Path(release_dir, "db.yaml").write_text("""
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cartservice-db
""")
        orch = discover_orchestration(repo_root)
        names = [wl["name"] for wl in orch.get("k8s_workloads", [])]
        check("k8s_statefulset: cartservice-db found", "cartservice-db" in names, names)


def test_compose_build_context_fused_with_dockerfile():
    """A compose service whose build.context == a Dockerfile dir fuses to ONE unit."""
    from doc_agent.tools.manifest_parser import parse_all_manifests

    with tempfile.TemporaryDirectory() as repo_root:
        svc_dir = os.path.join(repo_root, "api-service")
        svc_go = os.path.join(repo_root, "go-worker")
        os.makedirs(svc_dir); os.makedirs(svc_go)
        Path(svc_dir, "package.json").write_text('{"name":"api-service"}')
        Path(svc_dir, "Dockerfile").write_text("FROM node:18")
        Path(svc_go, "Dockerfile").write_text("FROM golang:1.21")

        compose = f"""
version: "3"
services:
  api:
    build:
      context: {svc_dir}
    ports:
      - "3000:3000"
  go-worker:
    build:
      context: {svc_go}
"""
        Path(repo_root, "docker-compose.yml").write_text(compose)

        facts = _make_facts([{
            "file": os.path.join(svc_dir, "index.js"),
            "language": "javascript",
            "imports": ["express"],
            "routes": [{"method": "GET", "path": "/", "handler": "home"}],
            "classes": [], "functions": [],
        }])
        manifests = parse_all_manifests(repo_root)
        orch_data = None  # let build_candidate_model call discover_orchestration
        from doc_agent.tools.container_model import discover_orchestration
        orch_data = discover_orchestration(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests, orchestration=orch_data)

        containers = model["containers"]["containers"]
        labels = {c["label"] for c in containers}

        # api-service should appear ONCE (fused, not duplicated)
        api_count = sum(1 for lbl in labels if "api" in lbl.lower() or "service" in lbl.lower())
        # Duplication check: api-service + go-worker = exactly 2 containers (not 3)
        check("compose_fusion: exactly 2 containers (no duplication)",
              len(containers) == 2, str(labels))
        check("compose_fusion: go-worker container present",
              any("worker" in lbl.lower() or "go" in lbl.lower() for lbl in labels), str(labels))


def test_auxiliary_dockerfile_not_admitted():
    """A dir with only Dockerfile.migrate must NOT be admitted (auxiliary, no floor).
    Uses 2 real service dirs to trigger multi-service detection.
    """
    with tempfile.TemporaryDirectory() as repo_root:
        svc_dir = os.path.join(repo_root, "real-service")
        svc2_dir = os.path.join(repo_root, "extra-service")  # 2nd service → multi detection
        aux_dir = os.path.join(repo_root, "db-migrate")
        os.makedirs(svc_dir); os.makedirs(svc2_dir); os.makedirs(aux_dir)
        Path(svc_dir, "Dockerfile").write_text("FROM node:18")
        Path(svc2_dir, "Dockerfile").write_text("FROM python:3.11")
        Path(aux_dir, "Dockerfile.migrate").write_text("FROM flyway/flyway")

        facts = _make_facts([])
        model = build_container_model(facts, repo_root)
        containers = model["containers"]["containers"]
        labels = {c["label"] for c in containers}

        check("aux_dockerfile: db-migrate not admitted",
              not any("migrate" in lbl.lower() for lbl in labels), str(labels))
        check("aux_dockerfile: real-service is present",
              any("real" in lbl.lower() for lbl in labels), str(labels))
        check("aux_dockerfile: exactly 2 containers (not 3)",
              len(containers) == 2, str(labels))


def test_find_dockerfile_dirs_returns_plain_only():
    """_find_dockerfile_dirs finds dirs with plain Dockerfile, not Dockerfile.dev."""
    from doc_agent.tools.container_model import _find_dockerfile_dirs

    with tempfile.TemporaryDirectory() as repo_root:
        svc_a = os.path.join(repo_root, "svc-a")
        svc_b = os.path.join(repo_root, "svc-b")
        os.makedirs(svc_a); os.makedirs(svc_b)
        Path(svc_a, "Dockerfile").write_text("FROM python:3.11")
        Path(svc_b, "Dockerfile.dev").write_text("FROM python:3.11-dev")

        dirs = _find_dockerfile_dirs(repo_root)
        abs_svc_a = str(Path(svc_a).resolve())
        abs_svc_b = str(Path(svc_b).resolve())

        check("find_dockerfile_dirs: svc-a (plain Dockerfile) found",
              abs_svc_a in dirs, dirs)
        check("find_dockerfile_dirs: svc-b (Dockerfile.dev only) NOT found",
              abs_svc_b not in dirs, dirs)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: strengthened validation + orphan datastores
# ══════════════════════════════════════════════════════════════════════════════

def test_validate_model_under_discovery_fail():
    """validate_model fails when discovered_unit_count >= 5 but rendered_containers <= 2."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=2, n_dbs=1, n_exts=0, n_actors=1)
    # Simulate: fusion found 8 units but only 2 containers survived to render
    model["_discovered_unit_count"] = 8

    report = validate_model(model)
    check("under_discovery: fail rule triggered", not report["passed"], report)
    fail_msgs = [f["message"] for f in report["findings"] if f["level"] == "fail"]
    check("under_discovery: fail message mentions discovered count",
          any("8" in m for m in fail_msgs), fail_msgs)


def test_validate_model_datastore_ratio_fail():
    """validate_model fails when >2 datastores and <50% are connected."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=1, n_dbs=0, n_exts=0, n_actors=1)
    # 3 datastores but only 1 connected
    model["containers"]["databases"] = [
        {"id": "db0", "label": "DB 0", "kind": "datastore", "tech": "PostgreSQL", "description": ""},
        {"id": "db1", "label": "DB 1", "kind": "datastore", "tech": "MySQL", "description": ""},
        {"id": "db2", "label": "DB 2", "kind": "datastore", "tech": "MongoDB", "description": ""},
    ]
    model["containers"]["relationships"] = [
        {"from": "svc0", "to": "db0", "label": "reads/writes"},
    ]

    report = validate_model(model)
    check("datastore_ratio: fail when >2 dbs and <50% connected", not report["passed"], report)


def test_validate_model_single_orphan_datastore_warn():
    """1 datastore with 0 connections → warn (not fail, below the >2 threshold)."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=1, n_dbs=0, n_exts=0, n_actors=1)
    model["containers"]["databases"] = [
        {"id": "db0", "label": "Session Cache", "kind": "cache", "tech": "Redis", "description": ""},
    ]
    model["containers"]["relationships"] = []
    model["context"]["relationships"] = []

    report = validate_model(model)
    # 1 orphan datastore should warn, not fail
    check("orphan_db: warn triggered (not fail)", report["passed"], report)
    check("orphan_db: score has orphan_datastore_count",
          "orphan_datastore_count" in report["score"], report["score"])
    check("orphan_db: orphan_datastore_count == 1",
          report["score"]["orphan_datastore_count"] == 1, report["score"])


def test_render_orphan_datastore_filtered():
    """A datastore with no edges must be filtered from the rendered diagram."""
    from doc_agent.tools.output import render_c4_combined

    model = {
        "context": {
            "system_name": "Test",
            "actors": [{"id": "user", "label": "User", "kind": "person"}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "api", "label": "uses"}],
        },
        "containers": {
            "system_label": "Test",
            "containers": [{"id": "api", "label": "API", "kind": "service", "tech": ""}],
            "databases": [
                {"id": "main_db", "label": "Main DB", "kind": "datastore", "tech": "PostgreSQL"},
                {"id": "session_cache", "label": "Session Cache", "kind": "cache", "tech": "Redis"},
            ],
            "external_services": [],
            "relationships": [
                {"from": "api", "to": "main_db", "label": "reads/writes"},
                # session_cache has NO relationships → should be filtered
            ],
        },
    }
    out = render_c4_combined(model)
    check("orphan_db_render: Main DB present (has edge)", "Main DB" in out, out[:600])
    check("orphan_db_render: Session Cache absent (orphan)", "Session Cache" not in out, out[:600])


def test_validate_model_score_has_new_fields():
    """validate_model score dict includes all new Phase 3 fields."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=2, n_dbs=1, n_exts=0, n_actors=1)
    report = validate_model(model)
    score = report["score"]
    for field in ("datastore_count", "orphan_datastore_count", "discovered_unit_count"):
        check(f"validate_score: {field} present", field in score, str(score.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: communication graph inference, worker classification, entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def _make_orch(compose_services=None, k8s_workloads=None, k8s_service_names=None):
    return {
        "compose_services": compose_services or [],
        "procfile_processes": [],
        "has_k8s": bool(k8s_workloads),
        "k8s_workloads": k8s_workloads or [],
        "k8s_service_names": k8s_service_names or [],
    }


def test_comm_graph_k8s_env_refs():
    """k8s env var ADDR → edge from workload to target container."""
    from doc_agent.tools.container_model import merge_orchestration

    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "frontend", "label": "Frontend", "kind": "service", "tech": ""},
        {"id": "cartservice", "label": "Cart Service", "kind": "service", "tech": ""},
    ]
    orch = _make_orch(k8s_workloads=[
        {"name": "frontend", "kind": "Deployment",
         "env_refs": ["cartservice"]},
        {"name": "cartservice", "kind": "Deployment", "env_refs": []},
    ])

    result = infer_communication_graph(model, _make_facts([]), orch)
    rels = result["containers"].get("relationships", [])
    check("k8s_env_edge: frontend → cartservice",
          any(r["from"] == "frontend" and r["to"] == "cartservice" for r in rels), rels)


def test_comm_graph_env_ref_to_datastore():
    """k8s env var pointing at a known datastore → reads/writes edge."""
    from doc_agent.tools.container_model import merge_orchestration

    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "cartservice", "label": "Cart Service", "kind": "service", "tech": ""},
    ]
    model["containers"]["databases"] = [
        {"id": "redis", "label": "Redis", "kind": "cache", "tech": "Redis"},
    ]
    orch = _make_orch(k8s_workloads=[
        {"name": "cartservice", "kind": "Deployment", "env_refs": ["redis"]},
    ])

    result = infer_communication_graph(model, _make_facts([]), orch)
    rels = result["containers"].get("relationships", [])
    check("k8s_env_db_edge: cartservice → redis reads/writes",
          any(r["from"] == "cartservice" and r["to"] == "redis" and "write" in r["label"] for r in rels),
          rels)


def test_comm_graph_sources_fuse():
    """compose depends_on AND web_app→service AND import-based DB all produce edges."""
    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "frontend", "label": "Frontend Web App", "kind": "web_app", "tech": ""},
        {"id": "api", "label": "API Service", "kind": "service", "tech": ""},
    ]
    # Use slug-matching db id so _build_ownership_edges edges resolve
    model["containers"]["databases"] = [
        {"id": "postgresql", "label": "PostgreSQL", "kind": "datastore", "tech": "PostgreSQL"},
    ]
    model["containers"]["relationships"] = []
    # inject private fields that _build_ownership_edges needs
    model["_container_units"] = {
        "frontend_dir": {"files": [], "imports": [], "routes": [], "has_entry": False},
        "api_dir": {"files": [], "imports": ["pg"], "routes": [], "has_entry": False},
    }
    model["_container_id_map"] = {"frontend_dir": "frontend", "api_dir": "api"}
    model["_db_found"] = {"PostgreSQL": ("PostgreSQL", "datastore")}
    model["_svc_found"] = {}
    model["_ext_verb"] = {}
    model["_single_container"] = False

    orch = _make_orch(compose_services=[
        {"name": "frontend", "image": None, "build_dir": None, "is_auxiliary": False,
         "profiles": [], "depends_on": ["api"], "links": []},
        {"name": "api", "image": None, "build_dir": None, "is_auxiliary": False,
         "profiles": [], "depends_on": [], "links": []},
    ])

    result = infer_communication_graph(model, _make_facts([]), orch)
    rels = result["containers"].get("relationships", [])
    has_web_to_svc = any(r["from"] == "frontend" and r["to"] == "api" for r in rels)
    has_db_edge = any(r["to"] == "postgresql" for r in rels)

    check("comm_graph_fuse: web_app → service edge present", has_web_to_svc, rels)
    check("comm_graph_fuse: api → db edge present", has_db_edge, rels)


def test_comm_graph_no_edges_from_build_candidate():
    """build_candidate_model returns 0 relationships — edges deferred to infer stage."""
    with tempfile.TemporaryDirectory() as repo_root:
        svc_a = os.path.join(repo_root, "svc-a")
        svc_b = os.path.join(repo_root, "svc-b")
        os.makedirs(svc_a); os.makedirs(svc_b)
        Path(svc_a, "Dockerfile").write_text("FROM node:18")
        Path(svc_b, "Dockerfile").write_text("FROM python:3.11")

        facts = _make_facts([])
        model = build_container_model(facts, repo_root)
        ctx_rels = model["context"].get("relationships", [])
        cont_rels = model["containers"].get("relationships", [])
        check("no_edges_from_build: 0 context relationships",
              len(ctx_rels) == 0, ctx_rels)
        check("no_edges_from_build: 0 container relationships",
              len(cont_rels) == 0, cont_rels)


def test_worker_classification_by_name():
    """_classify_by_name correctly identifies workers vs services by directory name."""
    from doc_agent.tools.container_model import _classify_by_name

    check("classify_by_name: loadgenerator → worker",
          _classify_by_name("loadgenerator") == "worker", _classify_by_name("loadgenerator"))
    check("classify_by_name: load-gen → worker",
          _classify_by_name("load-gen") == "worker", _classify_by_name("load-gen"))
    check("classify_by_name: email-worker → worker",
          _classify_by_name("email-worker") == "worker", _classify_by_name("email-worker"))
    check("classify_by_name: cartservice → service",
          _classify_by_name("cartservice") == "service", _classify_by_name("cartservice"))
    check("classify_by_name: k8s Job → worker",
          _classify_by_name("db-migration", "Job") == "worker",
          _classify_by_name("db-migration", "Job"))
    check("classify_by_name: k8s CronJob → worker",
          _classify_by_name("cleanup", "CronJob") == "worker",
          _classify_by_name("cleanup", "CronJob"))


def test_worker_label_in_merge_orchestration():
    """loadgenerator workload gets kind=worker and label ending in 'Worker'."""
    from doc_agent.tools.container_model import merge_orchestration

    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = []
    orch = _make_orch(k8s_workloads=[
        {"name": "loadgenerator", "kind": "Deployment", "env_refs": []},
        {"name": "cartservice",   "kind": "Deployment", "env_refs": []},
    ])
    result = merge_orchestration(model, orch)
    containers = result["containers"]["containers"]
    kinds = {c["id"]: c["kind"] for c in containers}
    labels = {c["id"]: c["label"] for c in containers}

    check("worker_label: loadgenerator kind==worker",
          kinds.get("loadgenerator") == "worker", kinds)
    check("worker_label: loadgenerator label ends with Worker",
          "Worker" in labels.get("loadgenerator", ""), labels)
    check("worker_label: cartservice kind==service",
          kinds.get("cartservice") == "service", kinds)


def test_infer_entrypoint_adds_actor_when_absent():
    """infer_entrypoint adds User actor when model has a frontend container but no actor."""
    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "frontend", "label": "Frontend", "kind": "service", "tech": ""},
        {"id": "cartservice", "label": "Cart Service", "kind": "service", "tech": ""},
    ]
    model["context"]["actors"] = []
    model["context"]["relationships"] = []

    result = infer_entrypoint(model)
    actors = result["context"].get("actors", [])
    ctx_rels = result["context"].get("relationships", [])

    check("entrypoint: user actor added", any(a["id"] == "user" for a in actors), actors)
    check("entrypoint: user → frontend edge added",
          any(r["from"] == "user" and r["to"] == "frontend" for r in ctx_rels), ctx_rels)


def test_infer_entrypoint_no_actor_for_backend_only():
    """infer_entrypoint does not add an actor when no frontend/gateway container exists."""
    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "cartservice",    "label": "Cart Service",    "kind": "service", "tech": ""},
        {"id": "checkoutservice","label": "Checkout Service","kind": "service", "tech": ""},
    ]
    model["context"]["actors"] = []
    model["context"]["relationships"] = []

    result = infer_entrypoint(model)
    actors = result["context"].get("actors", [])
    check("entrypoint_backend_only: no actor added", len(actors) == 0, actors)


def test_validate_model_dense_graph_fail():
    """validate_model fails when >=8 containers have <0.75 avg edges per container."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=10, n_dbs=0, n_exts=0, n_actors=1)
    # Only 2 edges for 10 containers → avg=0.2 < 0.75 → fail
    model["containers"]["relationships"] = [
        {"from": "svc0", "to": "svc1", "label": "calls"},
        {"from": "svc1", "to": "svc2", "label": "calls"},
    ]
    model["context"]["relationships"] = [{"from": "user", "to": "svc0", "label": "uses"}]

    report = validate_model(model)
    check("dense_graph_fail: rule 7 triggered", not report["passed"], report)
    fail_msgs = [f["message"] for f in report["findings"] if f["level"] == "fail"]
    check("dense_graph_fail: message mentions edge count",
          any("10 containers" in m or "service graph" in m for m in fail_msgs), fail_msgs)


def test_validate_model_dense_graph_pass():
    """validate_model does NOT fail rule 7 when edges are sufficient."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=10, n_dbs=0, n_exts=0, n_actors=1)
    # 10 containers with 10 edges → avg=1.0 > 0.75 → pass rule 7
    rels = [{"from": "user", "to": "svc0", "label": "uses"}]
    for i in range(9):
        rels.append({"from": f"svc{i}", "to": f"svc{i+1}", "label": "calls"})
    model["containers"]["relationships"] = rels[1:]
    model["context"]["relationships"] = [rels[0]]

    report = validate_model(model)
    rule7_fail = any("service graph" in f["message"] for f in report["findings"] if f["level"] == "fail")
    check("dense_graph_pass: rule 7 not triggered", not rule7_fail, report["findings"])


def test_validate_model_score_has_avg_edges_per_container_only():
    """validate_model score includes avg_edges_per_container_only field."""
    from doc_agent.tools.container_model import validate_model

    model = _make_model(n_containers=2, n_dbs=1, n_exts=0, n_actors=1)
    report = validate_model(model)
    check("score: avg_edges_per_container_only present",
          "avg_edges_per_container_only" in report["score"], str(report["score"].keys()))


def test_discover_orchestration_captures_env_refs():
    """discover_orchestration captures env_refs from k8s workload env vars."""
    from doc_agent.tools.container_model import discover_orchestration

    with tempfile.TemporaryDirectory() as repo_root:
        k8s_dir = os.path.join(repo_root, "kubernetes-manifests")
        os.makedirs(k8s_dir)
        deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  template:
    spec:
      containers:
      - name: frontend
        image: frontend:latest
        env:
        - name: CART_SERVICE_ADDR
          value: "cartservice:7070"
        - name: PRODUCT_CATALOG_SERVICE_ADDR
          value: "productcatalogservice:3550"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: frontend-secret
              key: key
"""
        Path(k8s_dir, "frontend.yaml").write_text(deployment_yaml)

        orch = discover_orchestration(repo_root)
        workloads = orch.get("k8s_workloads", [])
        frontend_wl = next((w for w in workloads if w["name"] == "frontend"), None)

        check("env_refs: frontend workload found", frontend_wl is not None, workloads)
        if frontend_wl:
            env_refs = frontend_wl.get("env_refs", [])
            check("env_refs: cartservice captured", "cartservice" in env_refs, env_refs)
            check("env_refs: productcatalogservice captured",
                  "productcatalogservice" in env_refs, env_refs)
            check("env_refs: valueFrom secret not captured",
                  "frontend-secret" not in env_refs and "key" not in env_refs, env_refs)


# ══════════════════════════════════════════════════════════════════════════════
# Part A: multi-doc k8s YAML fix
# ══════════════════════════════════════════════════════════════════════════════

def test_k8s_multidoc_manifest_discovers_all_workloads():
    """A single YAML file with multiple ---‑separated Deployments must yield all workloads."""
    from doc_agent.tools.container_model import discover_orchestration

    with tempfile.TemporaryDirectory() as repo_root:
        k8s_dir = os.path.join(repo_root, "release")
        os.makedirs(k8s_dir)
        # Three Deployments in one file, first one with *_ADDR env refs
        combined = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  template:
    spec:
      containers:
      - name: frontend
        env:
        - name: CART_SERVICE_ADDR
          value: "cartservice:7070"
        - name: PRODUCT_CATALOG_SERVICE_ADDR
          value: "productcatalogservice:3550"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cartservice
spec:
  template:
    spec:
      containers:
      - name: cartservice
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: productcatalogservice
spec:
  template:
    spec:
      containers:
      - name: productcatalogservice
"""
        Path(k8s_dir, "kubernetes-manifests.yaml").write_text(combined)

        orch = discover_orchestration(repo_root)
        names = [wl["name"] for wl in orch.get("k8s_workloads", [])]

        check("multidoc: frontend discovered", "frontend" in names, names)
        check("multidoc: cartservice discovered", "cartservice" in names, names)
        check("multidoc: productcatalogservice discovered", "productcatalogservice" in names, names)
        check("multidoc: all 3 workloads found", len(names) == 3, names)

        frontend_wl = next((w for w in orch["k8s_workloads"] if w["name"] == "frontend"), None)
        if frontend_wl:
            refs = frontend_wl.get("env_refs", [])
            check("multidoc: cartservice in env_refs", "cartservice" in refs, refs)
            check("multidoc: productcatalogservice in env_refs", "productcatalogservice" in refs, refs)


def test_comm_graph_k8s_multidoc_wires_services():
    """After multi-doc fix, env-refs from a combined manifest produce real inter-service edges."""
    from doc_agent.tools.container_model import merge_orchestration

    model = _make_model(n_containers=0, n_dbs=0, n_exts=0, n_actors=0)
    model["containers"]["containers"] = [
        {"id": "frontend",              "label": "Frontend",               "kind": "service", "tech": ""},
        {"id": "cartservice",           "label": "Cart Service",           "kind": "service", "tech": ""},
        {"id": "productcatalogservice", "label": "Product Catalog Service","kind": "service", "tech": ""},
    ]
    # Simulate what discover_orchestration now returns from a multi-doc manifest
    orch = _make_orch(k8s_workloads=[
        {"name": "frontend", "kind": "Deployment",
         "env_refs": ["cartservice", "productcatalogservice"]},
        {"name": "cartservice",           "kind": "Deployment", "env_refs": []},
        {"name": "productcatalogservice", "kind": "Deployment", "env_refs": []},
    ])
    result = infer_communication_graph(model, _make_facts([]), orch)
    rels = result["containers"].get("relationships", [])

    check("multidoc_edges: frontend → cartservice",
          any(r["from"] == "frontend" and r["to"] == "cartservice" for r in rels), rels)
    check("multidoc_edges: frontend → productcatalogservice",
          any(r["from"] == "frontend" and r["to"] == "productcatalogservice" for r in rels), rels)


# ══════════════════════════════════════════════════════════════════════════════
# Part B: synthesize_architecture_backbone
# ══════════════════════════════════════════════════════════════════════════════

def _connected(model: dict) -> bool:
    """Return True if all containers form a single weakly-connected component
    (via undirected traversal of all context + container relationships)."""
    from doc_agent.tools.output import _slug
    containers = model["containers"].get("containers", [])
    if len(containers) <= 1:
        return True
    cont_ids = {c["id"] for c in containers}
    all_rels = (
        list(model.get("context", {}).get("relationships") or [])
        + list(model.get("containers", {}).get("relationships") or [])
    )
    # Build undirected adjacency over container ids only
    adj: dict[str, set] = {cid: set() for cid in cont_ids}
    for r in all_rels:
        f, t = r.get("from", ""), r.get("to", "")
        if f in cont_ids and t in cont_ids:
            adj[f].add(t)
            adj[t].add(f)
    start = next(iter(cont_ids))
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited == cont_ids


def _empty_model(containers, actor=None):
    """Build a minimal model with the given container dicts and no edges."""
    actors = [actor] if actor else []
    return {
        "context": {
            "system_name": "Test",
            "actors": actors,
            "external_systems": [],
            "relationships": [],
        },
        "containers": {
            "system_label": "Test",
            "containers": containers,
            "databases": [],
            "external_services": [],
            "relationships": [],
        },
    }


def test_backbone_connects_orphan_services():
    """Frontend + N orphan services with no edges: backbone must wire them all."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "frontend",              "label": "Frontend",               "kind": "web_app",  "tech": ""},
        {"id": "cartservice",           "label": "Cart Service",           "kind": "service",  "tech": ""},
        {"id": "checkoutservice",       "label": "Checkout Service",       "kind": "service",  "tech": ""},
        {"id": "productcatalogservice", "label": "Product Catalog Service","kind": "service",  "tech": ""},
    ]
    model = _empty_model(containers,
                         actor={"id": "user", "label": "User", "kind": "person", "description": ""})
    model["context"]["relationships"] = [{"from": "user", "to": "frontend", "label": "uses"}]

    result = synthesize_architecture_backbone(model)
    rels = result["containers"].get("relationships", [])
    for svc in ("cartservice", "checkoutservice", "productcatalogservice"):
        has_inbound = any(r.get("to") == svc for r in rels)
        check(f"backbone_connects: {svc} has inbound edge", has_inbound, rels)

    check("backbone_connects: diagram is connected", _connected(result))


def test_backbone_preserves_real_edges_and_fills_only_orphans():
    """Real frontier→cartservice edge is untouched; only the orphan gets a new edge."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "frontend",    "label": "Frontend",    "kind": "web_app", "tech": ""},
        {"id": "cartservice", "label": "Cart Service", "kind": "service", "tech": ""},
        {"id": "checkoutservice","label": "Checkout","kind": "service", "tech": ""},
    ]
    # cartservice already has a real edge; checkoutservice is an orphan
    model = _empty_model(containers)
    model["containers"]["relationships"] = [
        {"from": "frontend", "to": "cartservice", "label": "calls API"},
    ]

    result = synthesize_architecture_backbone(model)
    rels = result["containers"].get("relationships", [])

    # Real edge must appear exactly once (not duplicated) with original label
    real_edges = [r for r in rels if r["from"] == "frontend" and r["to"] == "cartservice"]
    check("backbone_preserve: real edge count == 1", len(real_edges) == 1, real_edges)
    check("backbone_preserve: real edge label unchanged",
          real_edges[0]["label"] == "calls API", real_edges[0])

    # Orphan gets an inbound edge
    check("backbone_preserve: orphan checkoutservice now connected",
          any(r["to"] == "checkoutservice" for r in rels), rels)


def test_backbone_no_fan_to_already_connected():
    """Services that already have real inbound edges are NOT targeted by the backbone."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "frontend", "label": "Frontend",    "kind": "web_app", "tech": ""},
        {"id": "svcA",     "label": "Service A",   "kind": "service", "tech": ""},
        {"id": "svcB",     "label": "Service B",   "kind": "service", "tech": ""},
        {"id": "svcC",     "label": "Service C",   "kind": "service", "tech": ""},
        {"id": "svcD",     "label": "Service D",   "kind": "service", "tech": ""},
    ]
    # svcA and svcB already have real inbound edges; svcC and svcD are orphans
    model = _empty_model(containers)
    model["containers"]["relationships"] = [
        {"from": "frontend", "to": "svcA", "label": "calls"},
        {"from": "frontend", "to": "svcB", "label": "calls"},
    ]
    original_count = len(model["containers"]["relationships"])

    result = synthesize_architecture_backbone(model)
    rels = result["containers"].get("relationships", [])

    synthetic = [r for r in rels if (r["from"], r["to"]) not in
                 {("frontend", "svcA"), ("frontend", "svcB")}]
    check("backbone_no_redundant: only 2 orphans get synthetic edges",
          len(synthetic) == 2, synthetic)
    check("backbone_no_redundant: svcC connected",
          any(r["to"] == "svcC" for r in synthetic), synthetic)
    check("backbone_no_redundant: svcD connected",
          any(r["to"] == "svcD" for r in synthetic), synthetic)


def test_backbone_backend_only_mesh_uses_fallback_root():
    """No presentation/gateway, no actor: backbone still connects all services."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "orderservice",   "label": "Order Service",   "kind": "service", "tech": ""},
        {"id": "paymentservice", "label": "Payment Service", "kind": "service", "tech": ""},
        {"id": "emailservice",   "label": "Email Service",   "kind": "service", "tech": ""},
    ]
    model = _empty_model(containers, actor=None)
    result = synthesize_architecture_backbone(model)

    check("backbone_fallback: diagram is connected", _connected(result))
    # No actors should have been invented
    actors_after = result["context"].get("actors", [])
    check("backbone_fallback: no actors invented", len(actors_after) == 0, actors_after)


def test_backbone_single_container_noop():
    """Single container repo: backbone adds nothing."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    model = _empty_model([{"id": "api", "label": "API", "kind": "service", "tech": ""}])
    result = synthesize_architecture_backbone(model)
    rels = result["containers"].get("relationships", [])
    check("backbone_noop: zero edges added", len(rels) == 0, rels)


def test_backbone_attaches_worker():
    """An orphan worker container gets a 'triggers' edge from the fan source."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "frontend",     "label": "Frontend",          "kind": "web_app", "tech": ""},
        {"id": "api",          "label": "API",               "kind": "service", "tech": ""},
        {"id": "loadgenerator","label": "Load Generator Worker","kind": "worker","tech": ""},
    ]
    model = _empty_model(containers)
    model["containers"]["relationships"] = [
        {"from": "frontend", "to": "api", "label": "calls API"},
    ]
    result = synthesize_architecture_backbone(model)
    rels = result["containers"].get("relationships", [])
    check("backbone_worker: loadgenerator has inbound edge",
          any(r["to"] == "loadgenerator" for r in rels), rels)
    worker_edges = [r for r in rels if r["to"] == "loadgenerator"]
    check("backbone_worker: edge label is 'triggers'",
          all(r["label"] == "triggers" for r in worker_edges), worker_edges)


def test_backbone_never_invents_nodes():
    """Node id sets must be identical before and after backbone synthesis."""
    from doc_agent.tools.container_model import synthesize_architecture_backbone

    containers = [
        {"id": "frontend",    "label": "Frontend",    "kind": "web_app", "tech": ""},
        {"id": "cartservice", "label": "Cart Service", "kind": "service", "tech": ""},
        {"id": "orderservice","label": "Order Service","kind": "service", "tech": ""},
    ]
    actor = {"id": "user", "label": "User", "kind": "person", "description": ""}
    model = _empty_model(containers, actor=actor)

    cont_ids_before = {c["id"] for c in model["containers"]["containers"]}
    db_ids_before   = {d["id"] for d in model["containers"]["databases"]}
    actor_ids_before = {a["id"] for a in model["context"]["actors"]}

    result = synthesize_architecture_backbone(model)

    cont_ids_after  = {c["id"] for c in result["containers"]["containers"]}
    db_ids_after    = {d["id"] for d in result["containers"]["databases"]}
    actor_ids_after = {a["id"] for a in result["context"]["actors"]}

    check("backbone_no_new_nodes: containers unchanged", cont_ids_before == cont_ids_after,
          f"before={cont_ids_before} after={cont_ids_after}")
    check("backbone_no_new_nodes: databases unchanged", db_ids_before == db_ids_after)
    check("backbone_no_new_nodes: actors unchanged", actor_ids_before == actor_ids_after)
    rels = result["containers"].get("relationships", [])
    check("backbone_no_new_nodes: at least 1 edge added (not a noop)", len(rels) > 0, rels)


def _model_with_edges(containers, databases=None, rels=None, ctx_rels=None):
    """Build a model with explicit edges."""
    return {
        "context": {
            "system_name": "Test",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": ctx_rels or [],
        },
        "containers": {
            "system_label": "Test",
            "containers": containers,
            "databases": databases or [],
            "external_services": [],
            "relationships": rels or [],
        },
    }


def test_apply_enrichment_assigns_groups():
    """Groups from enrichment land on container nodes; unknown ids are ignored."""
    from doc_agent.tools.container_model import apply_enrichment

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app"},
        {"id": "checkout", "label": "Checkout", "kind": "service"},
    ]
    model = _model_with_edges(containers)
    enrichment = {
        "system_name": "Test System",
        "labels": {},
        "descriptions": {},
        "edge_labels": {},
        "groups": {
            "frontend": "Storefront",
            "checkout": "Ordering",
            "ghost_id": "Ghost Domain",   # unknown id — must be ignored
        },
    }
    result = apply_enrichment(model, enrichment)
    c_map = {c["id"]: c for c in result["containers"]["containers"]}
    check("enrichment_groups: frontend group set", c_map["frontend"].get("group") == "Storefront",
          c_map["frontend"])
    check("enrichment_groups: checkout group set", c_map["checkout"].get("group") == "Ordering",
          c_map["checkout"])
    check("enrichment_groups: ghost_id not added as container", "ghost_id" not in c_map)


def test_assign_domains_fallback_fills_all():
    """Every container ends with a 'group'; datastore inherits owner domain."""
    from doc_agent.tools.container_model import assign_domains

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "layer": "presentation"},
        {"id": "api",      "label": "API",      "kind": "service",  "layer": "application"},
    ]
    databases = [{"id": "db", "label": "DB", "kind": "datastore"}]
    rels = [
        {"from": "frontend", "to": "api",    "label": "calls"},
        {"from": "api",      "to": "db",     "label": "reads"},
    ]
    model = _model_with_edges(containers, databases=databases, rels=rels)
    result = assign_domains(model)
    c_map = {c["id"]: c for c in result["containers"]["containers"]}
    db_map = {d["id"]: d for d in result["containers"]["databases"]}

    check("assign_domains: frontend has group", bool(c_map["frontend"].get("group")),
          c_map["frontend"])
    check("assign_domains: api has group", bool(c_map["api"].get("group")),
          c_map["api"])
    check("assign_domains: db inherits group from api",
          db_map["db"].get("group") == c_map["api"].get("group"), db_map["db"])


def test_reduce_edges_transitive():
    """frontend→checkout, frontend→cart, checkout→cart ⇒ frontend→cart dropped (transitive)."""
    from doc_agent.tools.container_model import reduce_edges_for_readability

    containers = [
        {"id": "frontend",  "label": "Frontend",  "kind": "web_app"},
        {"id": "checkout",  "label": "Checkout",  "kind": "service"},
        {"id": "cart",      "label": "Cart",       "kind": "service"},
    ]
    rels = [
        {"from": "frontend", "to": "checkout", "label": "calls"},
        {"from": "frontend", "to": "cart",     "label": "calls"},
        {"from": "checkout", "to": "cart",     "label": "calls"},
    ]
    model = _model_with_edges(containers, rels=rels)
    result = reduce_edges_for_readability(model)
    result_rels = result["containers"]["relationships"]
    pairs = {(r["from"], r["to"]) for r in result_rels}
    check("transitive: frontend→cart removed", ("frontend", "cart") not in pairs, pairs)
    check("transitive: frontend→checkout kept", ("frontend", "checkout") in pairs, pairs)
    check("transitive: checkout→cart kept", ("checkout", "cart") in pairs, pairs)
    # Graph must stay connected (all nodes reachable)
    check("transitive: graph still connected", _connected(result))


def test_reduce_edges_bidirectional_collapse():
    """(a, b) and (b, a) collapse to one edge."""
    from doc_agent.tools.container_model import reduce_edges_for_readability

    containers = [
        {"id": "a", "label": "A", "kind": "service"},
        {"id": "b", "label": "B", "kind": "service"},
    ]
    rels = [
        {"from": "a", "to": "b", "label": "calls"},
        {"from": "b", "to": "a", "label": "responds"},
    ]
    model = _model_with_edges(containers, rels=rels)
    result = reduce_edges_for_readability(model)
    result_rels = result["containers"]["relationships"]
    check("bidir_collapse: exactly one edge remains", len(result_rels) == 1, result_rels)


def test_drop_operational_noise():
    """loadgenerator and prometheus removed; real services kept. No-op when ≤3 containers."""
    from doc_agent.tools.container_model import drop_operational_noise

    def _make_containers(names):
        return [{"id": n, "label": n, "kind": "service"} for n in names]

    # Normal case: >3 containers
    model = _model_with_edges(
        _make_containers(["frontend", "checkout", "cartservice", "loadgenerator", "prometheus"]),
        rels=[
            {"from": "loadgenerator", "to": "frontend", "label": "generates load"},
            {"from": "frontend", "to": "checkout", "label": "calls"},
        ],
    )
    result = drop_operational_noise(model)
    kept_ids = {c["id"] for c in result["containers"]["containers"]}
    check("drop_noise: loadgenerator removed", "loadgenerator" not in kept_ids, kept_ids)
    check("drop_noise: prometheus removed", "prometheus" not in kept_ids, kept_ids)
    check("drop_noise: frontend kept", "frontend" in kept_ids, kept_ids)
    check("drop_noise: checkout kept", "checkout" in kept_ids, kept_ids)
    # Edges involving dropped nodes must also be removed
    remaining_rels = result["containers"]["relationships"]
    noise_in_rels = any(
        r.get("from") in ("loadgenerator", "prometheus") or r.get("to") in ("loadgenerator", "prometheus")
        for r in remaining_rels
    )
    check("drop_noise: edges to noise removed", not noise_in_rels, remaining_rels)

    # No-op when ≤3 containers
    small_model = _model_with_edges(
        _make_containers(["frontend", "loadgenerator", "api"]),
    )
    small_result = drop_operational_noise(small_model)
    small_kept = {c["id"] for c in small_result["containers"]["containers"]}
    check("drop_noise: noop when <=3 containers", "loadgenerator" in small_kept, small_kept)


def test_render_group_field_ignored_layered():
    """Even with 'group' set, lanes are by architecture tier — never domain subgraphs."""
    from doc_agent.tools.output import render_c4_combined

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "layer": "presentation", "group": "Storefront"},
        {"id": "checkout", "label": "Checkout", "kind": "service", "layer": "application", "group": "Ordering"},
        {"id": "cart",     "label": "Cart",     "kind": "service", "layer": "application", "group": "Ordering"},
    ]
    databases = [
        {"id": "db", "label": "DB", "kind": "datastore", "group": "Ordering"},
    ]
    rels = [
        {"from": "frontend", "to": "checkout", "label": "calls"},
        {"from": "checkout", "to": "db",       "label": "reads"},
    ]
    model = {
        "context": {
            "system_name": "Demo",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "frontend", "label": "uses"}],
        },
        "containers": {
            "system_label": "Demo",
            "containers": containers,
            "databases": databases,
            "external_services": [],
            "relationships": rels,
        },
    }
    output = render_c4_combined(model)
    check("group_ignored: no Storefront subgraph", "Storefront" not in output, output[:300])
    check("group_ignored: no Ordering subgraph", "Ordering" not in output, output[:300])
    check("group_ignored: tier lanes present (Clients/Services)",
          "subgraph lane_clients" in output and "subgraph lane_services" in output, output[:400])
    check("group_ignored: frontend node present", "frontend[" in output)
    check("group_ignored: checkout node present", "checkout[" in output)
    check("group_ignored: db node present", "db[" in output)
    check("group_ignored: flowchart header present", "flowchart TD" in output)


def test_render_no_group_layered():
    """Group-less model renders tier lanes by layer."""
    from doc_agent.tools.output import render_c4_combined

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "layer": "presentation"},
        {"id": "api",      "label": "API",       "kind": "service",  "layer": "application"},
    ]
    databases = [{"id": "db", "label": "DB", "kind": "datastore"}]
    rels = [
        {"from": "frontend", "to": "api", "label": "calls"},
        {"from": "api",      "to": "db",  "label": "reads"},
    ]
    model = {
        "context": {
            "system_name": "Demo",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "frontend", "label": "uses"}],
        },
        "containers": {
            "system_label": "Demo",
            "containers": containers,
            "databases": databases,
            "external_services": [],
            "relationships": rels,
        },
    }
    output = render_c4_combined(model)
    check("no_group_layered: Clients lane present (presentation web_app)",
          "subgraph lane_clients" in output, output[:400])
    check("no_group_layered: Services lane present (application service)",
          "subgraph lane_services" in output, output[:400])
    check("no_group_layered: frontend + api nodes present",
          "frontend[" in output and "api[" in output, output[:400])


def test_combined_layered_grouping():
    """Combined contract: one SYS boundary with tier lanes inside, containers/datastores
    inside SYS, actors and external systems OUTSIDE the boundary, relationships resolve."""
    from doc_agent.tools.output import render_c4_combined
    containers = [
        {"id": "web", "label": "Web App", "kind": "web_app", "layer": "presentation", "group": "Frontend"},
        {"id": "api", "label": "API", "kind": "service", "layer": "application", "group": "Backend"},
    ]
    databases = [{"id": "pg", "label": "Postgres", "kind": "datastore", "group": "Backend"}]
    model = {
        "context": {
            "system_name": "Demo",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [{"id": "stripe", "label": "Stripe", "kind": "external", "description": ""}],
            "relationships": [{"from": "user", "to": "web", "label": "uses"}],
        },
        "containers": {
            "system_label": "Demo",
            "containers": containers,
            "databases": databases,
            "external_services": [],
            "relationships": [
                {"from": "web", "to": "api", "label": "calls"},
                {"from": "api", "to": "pg", "label": "reads/writes"},
                {"from": "api", "to": "stripe", "label": "pays via"},
            ],
        },
    }
    out = render_c4_combined(model)
    lines = out.splitlines()
    # One SYS boundary wrapping tier lanes.
    check("layered_contract: SYS boundary present", "subgraph SYS" in out, out[:400])
    check("layered_contract: tier lanes inside", "subgraph lane_" in out, out[:400])
    # Containers + datastore are inside SYS (between 'subgraph SYS' and its LAST 'end',
    # since nested lane subgraphs each emit their own 'end').
    sys_start = next(i for i, ln in enumerate(lines) if "subgraph SYS" in ln)
    sys_end = max(i for i in range(sys_start + 1, len(lines)) if lines[i].strip() == "end")
    inside = "\n".join(lines[sys_start:sys_end])
    check("layered_contract: web inside SYS", "web[" in inside, inside)
    check("layered_contract: api inside SYS", "api[" in inside, inside)
    check("layered_contract: pg datastore inside SYS", "pg[" in inside, inside)
    # Actor + external are OUTSIDE SYS.
    before = "\n".join(lines[:sys_start])
    after = "\n".join(lines[sys_end + 1:])
    check("layered_contract: user actor outside SYS", "user(" in before, before)
    check("layered_contract: stripe external outside SYS", "stripe" in after, after)
    # Relationships resolve.
    check("layered_contract: api->pg edge", "api -->" in out and "pg" in out, out)
    check("layered_contract: api->stripe edge present", "stripe" in after and "-->" in out, out)


def test_backbone_then_thinning_microservices():
    """End-to-end: backbone + noise drop + thinning on a microservices-demo-shaped model."""
    from doc_agent.tools.container_model import (
        synthesize_architecture_backbone, drop_operational_noise,
        assign_domains, reduce_edges_for_readability,
    )
    from doc_agent.tools.output import render_c4_combined

    # Synthetic microservices-demo shape (14 containers)
    svc_names = [
        "frontend", "cartservice", "checkoutservice", "recommendationservice",
        "productcatalogservice", "currencyservice", "paymentservice",
        "shippingservice", "emailservice", "adservice",
        "loadgenerator", "redis", "postgresql", "kafka",
    ]
    containers_raw = []
    databases_raw = []
    for name in svc_names:
        if name in ("redis", "postgresql"):
            databases_raw.append({"id": name, "label": name.title(), "kind": "datastore"})
        elif name == "kafka":
            databases_raw.append({"id": name, "label": "Kafka", "kind": "queue"})
        else:
            containers_raw.append({
                "id": name, "label": name.title(), "kind": "web_app" if name == "frontend" else "service",
                "layer": "presentation" if name == "frontend" else "application",
            })

    # Full mesh: frontend→all services, checkout→many services
    rels = []
    services = [c["id"] for c in containers_raw if c["id"] != "frontend"]
    for svc in services:
        rels.append({"from": "frontend", "to": svc, "label": "calls"})
    for svc in ("cartservice", "currencyservice", "paymentservice", "shippingservice", "emailservice"):
        rels.append({"from": "checkoutservice", "to": svc, "label": "calls"})
    rels.append({"from": "cartservice", "to": "redis", "label": "reads"})
    rels.append({"from": "checkoutservice", "to": "postgresql", "label": "writes"})
    rels.append({"from": "checkoutservice", "to": "kafka", "label": "publishes"})

    model = {
        "context": {
            "system_name": "Microservices Demo",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "frontend", "label": "uses"}],
        },
        "containers": {
            "system_label": "Microservices Demo",
            "containers": containers_raw,
            "databases": databases_raw,
            "external_services": [],
            "relationships": rels,
        },
    }

    model = synthesize_architecture_backbone(model)
    model = drop_operational_noise(model)
    model = assign_domains(model)
    model = reduce_edges_for_readability(model)

    cont_ids = {c["id"] for c in model["containers"]["containers"]}
    final_rels = model["containers"]["relationships"]
    output = render_c4_combined(model)

    check("ms_e2e: loadgenerator dropped", "loadgenerator" not in cont_ids, cont_ids)
    check("ms_e2e: frontend kept", "frontend" in cont_ids, cont_ids)
    original_edge_count = len(services) + 6  # frontend fan + checkout fan + infra
    check("ms_e2e: edge count thinned",
          len(final_rels) < original_edge_count,
          f"edges={len(final_rels)}, original_approx={original_edge_count}")
    check("ms_e2e: output is non-empty", len(output) > 100)
    check("ms_e2e: flowchart header present", "flowchart TD" in output)
    # Count domain subgraphs in output — should be ≤6
    import re as _re_test
    domain_count = len(_re_test.findall(r'\bsubgraph\b', output))
    check("ms_e2e: ≤8 subgraphs total (≤6 domains + SYS + maybe shared data)",
          domain_count <= 8, f"subgraph count={domain_count}\n{output[:600]}")


def test_curate_fanout_cap_drops_excess():
    """Redundant fan edges are dropped down to cap; load-bearing edges survive.

    frontend → a,b,c,d,e,f (6). 'a' also calls c,d,e,f, so frontend→c/d/e/f are
    redundant (reachable via frontend→a→). cap=4 ⇒ two redundant arrows dropped.
    """
    from doc_agent.tools.container_model import curate_significant_edges

    containers = [{"id": "frontend", "label": "Frontend", "kind": "web_app", "group": "A"}]
    for n in ("a", "b", "c", "d", "e", "f"):
        containers.append({"id": n, "label": n.upper(), "kind": "service", "group": "A"})
    rels = [{"from": "frontend", "to": n, "label": "calls"} for n in ("a", "b", "c", "d", "e", "f")]
    rels += [{"from": "a", "to": n, "label": "calls"} for n in ("c", "d", "e", "f")]
    model = _model_with_edges(containers, rels=rels,
                              ctx_rels=[{"from": "user", "to": "frontend", "label": "uses"}])
    result = curate_significant_edges(model, cap=4)
    fan = [r["to"] for r in result["containers"]["relationships"] if r["from"] == "frontend"]
    check("curate_cap: frontend fan reduced to cap", len(fan) == 4, fan)
    check("curate_cap: load-bearing frontend→a kept", "a" in fan, fan)
    check("curate_cap: load-bearing frontend→b kept", "b" in fan, fan)
    check("curate_cap: graph still connected", _connected(result))


def test_curate_preserves_ingress_and_data_edges():
    """Ingress (actor→container) and data (container→datastore) edges are never capped."""
    from doc_agent.tools.container_model import curate_significant_edges

    containers = [{"id": "frontend", "label": "Frontend", "kind": "web_app", "group": "A"}]
    for i in range(1, 7):
        containers.append({"id": f"s{i}", "label": f"S{i}", "kind": "service", "group": "A"})
    databases = [{"id": "db", "label": "DB", "kind": "datastore"}]
    rels = [{"from": "frontend", "to": f"s{i}", "label": "calls"} for i in range(1, 7)]
    rels.append({"from": "s1", "to": "db", "label": "reads"})
    ctx_rels = [{"from": "user", "to": "frontend", "label": "uses"}]
    model = _model_with_edges(containers, databases=databases, rels=rels, ctx_rels=ctx_rels)
    result = curate_significant_edges(model, cap=4)
    pairs = {(r["from"], r["to"]) for r in result["containers"]["relationships"]}
    check("curate_preserve: data edge s1→db kept", ("s1", "db") in pairs, pairs)
    check("curate_preserve: ingress edge untouched",
          ("user", "frontend") in {(r["from"], r["to"]) for r in result["context"]["relationships"]})


def test_curate_no_orphan_guard():
    """Capping never orphans a node reachable only via a low-ranked edge."""
    from doc_agent.tools.container_model import curate_significant_edges

    # frontend fans to 5 high-degree hubs; 'leaf' is reachable ONLY via frontend→leaf.
    containers = [{"id": "frontend", "label": "Frontend", "kind": "web_app", "group": "A"}]
    for i in range(1, 6):
        containers.append({"id": f"h{i}", "label": f"H{i}", "kind": "service", "group": "A"})
    containers.append({"id": "leaf", "label": "Leaf", "kind": "service", "group": "A"})
    rels = [{"from": "frontend", "to": f"h{i}", "label": "calls"} for i in range(1, 6)]
    rels.append({"from": "frontend", "to": "leaf", "label": "calls"})
    # boost hub degrees so leaf would be dropped by a naive cap
    for i in range(1, 6):
        rels.append({"from": f"h{i}", "to": "h1" if i != 1 else "h2", "label": "x"})
    model = _model_with_edges(containers, rels=rels,
                              ctx_rels=[{"from": "user", "to": "frontend", "label": "uses"}])
    result = curate_significant_edges(model, cap=4)
    check("curate_orphan: graph stays connected", _connected(result))


def test_curate_noop_under_cap():
    """Source with ≤cap edges ⇒ edge set unchanged."""
    from doc_agent.tools.container_model import curate_significant_edges

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "group": "A"},
        {"id": "a", "label": "A", "kind": "service", "group": "A"},
        {"id": "b", "label": "B", "kind": "service", "group": "A"},
    ]
    rels = [
        {"from": "frontend", "to": "a", "label": "calls"},
        {"from": "frontend", "to": "b", "label": "calls"},
    ]
    model = _model_with_edges(containers, rels=rels)
    before = list(model["containers"]["relationships"])
    result = curate_significant_edges(model, cap=4)
    check("curate_noop: edges unchanged",
          {(r["from"], r["to"]) for r in result["containers"]["relationships"]}
          == {(r["from"], r["to"]) for r in before})


def test_curate_no_entrypoint_backend_only():
    """No entrypoint match ⇒ no exception, no orphan, deterministic."""
    from doc_agent.tools.container_model import curate_significant_edges

    containers = [{"id": f"svc{i}", "label": f"Svc{i}", "kind": "service", "group": "A"}
                  for i in range(7)]
    rels = [{"from": "svc0", "to": f"svc{i}", "label": "calls"} for i in range(1, 7)]
    model = _model_with_edges(containers, rels=rels, ctx_rels=[])
    result = curate_significant_edges(model, cap=4)
    check("curate_noentry: no crash and connected-or-reduced",
          len(result["containers"]["relationships"]) >= 4)


def test_curate_determinism():
    """Running twice on a shuffled edge copy yields identical kept set."""
    from doc_agent.tools.container_model import curate_significant_edges

    def _build():
        containers = [{"id": "frontend", "label": "Frontend", "kind": "web_app", "group": "A"}]
        for i in range(1, 8):
            containers.append({"id": f"t{i}", "label": f"T{i}", "kind": "service", "group": "A"})
        rels = [{"from": "frontend", "to": f"t{i}", "label": "calls"} for i in range(1, 8)]
        rels += [{"from": "t1", "to": "t2", "label": "x"}, {"from": "t3", "to": "t4", "label": "x"}]
        return _model_with_edges(containers, rels=rels)

    r1 = curate_significant_edges(_build(), cap=4)
    m2 = _build()
    m2["containers"]["relationships"] = list(reversed(m2["containers"]["relationships"]))
    r2 = curate_significant_edges(m2, cap=4)
    p1 = {(r["from"], r["to"]) for r in r1["containers"]["relationships"]}
    p2 = {(r["from"], r["to"]) for r in r2["containers"]["relationships"]}
    check("curate_determinism: identical kept set", p1 == p2, f"{p1}\n{p2}")


def test_assign_container_roles():
    """Entrypoint marked 'entrypoint'; others derive role from layer."""
    from doc_agent.tools.container_model import assign_container_roles

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "layer": "presentation"},
        {"id": "worker1", "label": "Worker", "kind": "worker", "layer": "worker"},
    ]
    model = _model_with_edges(containers)
    result = assign_container_roles(model)
    c_map = {c["id"]: c for c in result["containers"]["containers"]}
    check("roles: frontend is entrypoint", c_map["frontend"].get("role") == "entrypoint", c_map["frontend"])
    check("roles: worker role from layer", c_map["worker1"].get("role") == "worker", c_map["worker1"])


def test_assign_domains_degenerate_clears_groups():
    """Flat microservices with no LLM groups ⇒ groups cleared (tier-mode fallback)."""
    from doc_agent.tools.container_model import assign_domains

    # All same layer 'application' and unique prefixes ⇒ fallback would make 1 group.
    containers = [{"id": f"svc{i}", "label": f"Svc{i}", "kind": "service", "layer": "application"}
                  for i in range(8)]
    model = _model_with_edges(containers)
    result = assign_domains(model)
    groups = [c.get("group") for c in result["containers"]["containers"]]
    check("degenerate: all groups cleared", all(not g for g in groups), groups)


def test_render_subgraph_id_never_collides_with_node_id():
    """A node id equal to a domain/tier slug must not collide with the subgraph id.

    Regression: domain "Db" slugged to "db" collided with container node "db",
    producing `subgraph db[...]` + `db[...]` → Mermaid 'Syntax error in text'.
    """
    from doc_agent.tools.output import render_c4_combined
    import re as _re_t

    containers = [
        {"id": "frontend", "label": "Frontend", "kind": "web_app", "layer": "presentation", "group": "Storefront"},
        {"id": "db", "label": "Database Access", "kind": "service", "layer": "application", "group": "Db"},
    ]
    model = {
        "context": {
            "system_name": "Demo",
            "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
            "external_systems": [],
            "relationships": [{"from": "user", "to": "frontend", "label": "uses"}],
        },
        "containers": {
            "system_label": "Demo",
            "containers": containers,
            "databases": [],
            "external_services": [],
            "relationships": [{"from": "frontend", "to": "db", "label": "accesses"}],
        },
    }
    output = render_c4_combined(model)
    subgraph_ids = set(_re_t.findall(r'subgraph\s+(\S+)\[', output))
    node_ids = set(_re_t.findall(r'^\s+([A-Za-z0-9_]+)(?:\[|\(|\{)', output, _re_t.MULTILINE))
    node_ids -= subgraph_ids
    collisions = subgraph_ids & node_ids
    check("subgraph_collision: no subgraph id equals a node id", not collisions,
          f"collisions={collisions}\n{output}")
    check("subgraph_collision: db node still present", '    db[' in output or 'db["' in output, output)


def test_runnable_by_convention_helper():
    """_is_runnable_by_convention: own manifest + entry/routes/framework => runnable."""
    from doc_agent.tools.container_model import _is_runnable_by_convention
    md = "/repo/services/orders"
    manifests = {md: {"name": "orders"}}
    check("conv_helper: manifest + entry => True",
          _is_runnable_by_convention(md, {"has_entry": True, "files": []}, manifests) is True)
    check("conv_helper: manifest + routes => True",
          _is_runnable_by_convention(md, {"routes": [{"path": "/x"}], "files": []}, manifests) is True)
    check("conv_helper: no manifest => False",
          _is_runnable_by_convention(md, {"has_entry": True, "files": []}, {}) is False)
    check("conv_helper: manifest but no entry/routes/framework => False",
          _is_runnable_by_convention(md, {"files": []}, manifests) is False)


def test_multi_app_corroborated_helper():
    """_multi_app_corroborated: >=2 components owning routes/entry => True."""
    from doc_agent.tools.container_model import _multi_app_corroborated
    check("corrob: two runnable components => True",
          _multi_app_corroborated({"components": [{"has_routes": True}, {"has_main_entry": True}]}) is True)
    check("corrob: one runnable component => False",
          _multi_app_corroborated({"components": [{"has_routes": True}, {"has_routes": False}]}) is False)
    check("corrob: no components => False", _multi_app_corroborated({}) is False)


def test_multi_app_monorepo_no_docker():
    """Two runnable apps (own manifest + routes), NO Dockerfile/compose, corroborated
    by >=2 components => admitted as 2 containers (not collapsed to a false monolith)."""
    from doc_agent.tools.container_model import build_container_model
    from doc_agent.tools.manifest_parser import parse_all_manifests
    with tempfile.TemporaryDirectory() as repo_root:
        a = os.path.join(repo_root, "orders")
        b = os.path.join(repo_root, "billing")
        os.makedirs(a); os.makedirs(b)
        Path(a, "package.json").write_text('{"name":"orders"}')
        Path(b, "package.json").write_text('{"name":"billing"}')
        facts = _make_facts([
            {"file": os.path.join(a, "index.js"), "language": "javascript",
             "imports": ["express"], "routes": [{"method": "GET", "path": "/orders", "handler": "h"}],
             "classes": [], "functions": []},
            {"file": os.path.join(b, "index.js"), "language": "javascript",
             "imports": ["express"], "routes": [{"method": "GET", "path": "/billing", "handler": "h"}],
             "classes": [], "functions": []},
        ])
        # Structural corroboration: two independently-runnable components.
        facts["components"] = [
            {"has_routes": True, "has_main_entry": True},
            {"has_routes": True, "has_main_entry": True},
        ]
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests, orchestration={})
        containers = model["containers"]["containers"]
        check("monorepo_no_docker: 2 containers (not collapsed)", len(containers) == 2,
              [c["label"] for c in containers])
        check("monorepo_no_docker: not single-container", model.get("_single_container") is not True)


def test_monolith_one_manifest_stays_one_container():
    """A single-manifest repo with internal packages stays exactly ONE container."""
    from doc_agent.tools.container_model import build_container_model
    from doc_agent.tools.manifest_parser import parse_all_manifests
    with tempfile.TemporaryDirectory() as repo_root:
        Path(repo_root, "package.json").write_text('{"name":"app"}')
        os.makedirs(os.path.join(repo_root, "src", "api"))
        os.makedirs(os.path.join(repo_root, "src", "services"))
        facts = _make_facts([
            {"file": os.path.join(repo_root, "src", "api", "routes.js"), "language": "javascript",
             "imports": ["express"], "routes": [{"method": "GET", "path": "/", "handler": "h"}],
             "classes": [], "functions": []},
            {"file": os.path.join(repo_root, "src", "services", "svc.js"), "language": "javascript",
             "imports": [], "routes": [], "classes": [], "functions": []},
        ])
        manifests = parse_all_manifests(repo_root)
        model = build_container_model(facts, repo_root, manifests=manifests, orchestration={})
        check("monolith_one_manifest: exactly 1 container",
              len(model["containers"]["containers"]) == 1,
              [c["label"] for c in model["containers"]["containers"]])


def test_spring_data_nosql_not_relational():
    """Spring Data Mongo/Elasticsearch deps map to the right engine, not 'Relational Database'."""
    from doc_agent.tools.container_model import _scan_db_engines
    mongo = _scan_db_engines([], manifest_deps=["spring-boot-starter-data-mongodb"])
    check("spring_data: mongo => MongoDB", "MongoDB" in mongo, mongo)
    check("spring_data: mongo not relational", "Relational Database" not in mongo, mongo)
    es = _scan_db_engines([], manifest_deps=["spring-boot-starter-data-elasticsearch"])
    check("spring_data: es => Elasticsearch", "Elasticsearch" in es, es)
    check("spring_data: es not relational", "Relational Database" not in es, es)


def test_infra_catalog_additions():
    """New datastores/queues/externals are recognized."""
    from doc_agent.tools.container_model import _scan_db_engines, _scan_services, _image_to_datastore
    db = _scan_db_engines(["@aws-sdk/client-dynamodb", "neo4j", "nats", "hazelcast"])
    check("infra: DynamoDB", "DynamoDB" in db, db)
    check("infra: Neo4j", "Neo4j" in db, db)
    check("infra: NATS queue", db.get("NATS") == ("NATS", "queue"), db)
    check("infra: Hazelcast cache", db.get("Hazelcast") == ("Hazelcast", "cache"), db)
    svc = _scan_services(["@azure/storage-blob"])
    check("infra: Azure external", "Azure" in svc, svc)
    check("infra: clickhouse image", _image_to_datastore("clickhouse:23") == ("ClickHouse", "datastore"),
          _image_to_datastore("clickhouse:23"))


def test_mermaid_valid_special_char_ids():
    """Container ids with @ / : ( ) must not leak Mermaid-breaking chars into ids."""
    from doc_agent.tools.output import render_c4_combined
    import re as _re_t
    containers = [
        {"id": "@scope/pkg", "label": "Scoped Pkg", "kind": "service"},
        {"id": "a.b:core", "label": "Core", "kind": "service"},
        {"id": "(x)", "label": "X Service", "kind": "service"},
    ]
    model = _model_with_edges(
        containers,
        rels=[{"from": "@scope/pkg", "to": "a.b:core", "label": "calls"},
              {"from": "a.b:core", "to": "(x)", "label": "calls"}],
        ctx_rels=[{"from": "user", "to": "@scope/pkg", "label": "uses"}],
    )
    out = render_c4_combined(model)
    check("special_ids: flowchart header", "flowchart TD" in out)
    # The raw special-char ids must be slugged away; their safe slugs must appear.
    for raw in ("@scope/pkg", "a.b:core", "(x)"):
        check(f"special_ids: raw id '{raw}' not emitted", raw not in out, out)
    for slug in ("scope_pkg", "a_b_core"):
        check(f"special_ids: slug '{slug}' present", slug in out, out)
    # And the edges must resolve between the slugged ids.
    check("special_ids: edge scope_pkg-->a_b_core present", "scope_pkg -->" in out, out)


def test_narrative_spine_survives_thinning():
    """The actor→entry→primary→datastore spine must survive reduce+curate even when
    incidental service→service edges create alternate 2-hop paths (the OpenMRS bug)."""
    from doc_agent.tools.container_model import (
        enforce_narrative_spine, reduce_edges_for_readability, curate_significant_edges,
    )
    containers = [
        {"id": "web", "label": "Web App", "kind": "web_app", "layer": "presentation"},
        {"id": "core", "label": "Core API", "kind": "service", "layer": "application"},
        {"id": "loki", "label": "Loki", "kind": "service", "layer": "application"},
        {"id": "lokiinit", "label": "Loki Init", "kind": "service", "layer": "application"},
        {"id": "alloy", "label": "Alloy", "kind": "service", "layer": "application"},
    ]
    databases = [{"id": "pg", "label": "Postgres", "kind": "datastore"}]
    # web→core is the front-door spine link; core→pg ownership; plus incidental noise
    # edges that previously caused transitive reduction to strip web's direct edges.
    rels = [
        {"from": "web", "to": "core", "label": "calls API"},
        {"from": "core", "to": "pg", "label": "reads/writes"},
        {"from": "loki", "to": "lokiinit", "label": "calls"},
        {"from": "loki", "to": "alloy", "label": "calls"},
    ]
    model = {
        "context": {"system_name": "OpenMRS", "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
                    "external_systems": [], "relationships": [{"from": "user", "to": "web", "label": "uses"}]},
        "containers": {"system_label": "OpenMRS", "containers": containers, "databases": databases,
                       "external_services": [], "relationships": rels},
    }
    model = enforce_narrative_spine(model)
    model = reduce_edges_for_readability(model)
    model = curate_significant_edges(model)
    pairs = {(r["from"], r["to"]) for r in model["containers"]["relationships"]}
    check("spine: web→core survives", ("web", "core") in pairs, pairs)
    check("spine: core→pg survives", ("core", "pg") in pairs, pairs)
    # The private spine marker is still present until enforce_connectivity consumes it.
    check("spine: _spine_edges recorded", bool(model.get("_spine_edges")), model.get("_spine_edges"))


def test_enforce_connectivity_no_floating_nodes():
    """Every container/datastore must end with >=1 edge; user→datastore path must exist."""
    from doc_agent.tools.container_model import enforce_connectivity
    containers = [
        {"id": "web", "label": "Web App", "kind": "web_app", "layer": "presentation"},
        {"id": "api", "label": "API", "kind": "service", "layer": "application"},
        {"id": "floating", "label": "Floating Svc", "kind": "service", "layer": "application"},
    ]
    databases = [
        {"id": "pg", "label": "Postgres", "kind": "datastore"},
        {"id": "orphandb", "label": "Orphan DB", "kind": "datastore"},
    ]
    rels = [{"from": "web", "to": "api", "label": "calls"}, {"from": "api", "to": "pg", "label": "reads/writes"}]
    model = {
        "context": {"system_name": "Demo", "actors": [{"id": "user", "label": "User", "kind": "person", "description": ""}],
                    "external_systems": [], "relationships": [{"from": "user", "to": "web", "label": "uses"}]},
        "containers": {"system_label": "Demo", "containers": containers, "databases": databases,
                       "external_services": [], "relationships": rels, "_spine_edges": set()},
    }
    model["_spine_edges"] = set()
    model = enforce_connectivity(model)
    all_rels = model["containers"]["relationships"] + model["context"]["relationships"]
    touched = {n for r in all_rels for n in (r["from"], r["to"])}
    check("conn: floating service attached", "floating" in touched, touched)
    check("conn: orphan datastore attached", "orphandb" in touched, touched)
    check("conn: no container left floating",
          all(c["id"] in touched for c in containers), touched)
    check("conn: _spine_edges consumed (not leaked)", "_spine_edges" not in model, list(model.keys()))


def test_primary_service_owns_datastore():
    """_pick_primary_service prefers the application service that writes a datastore."""
    from doc_agent.tools.container_model import _pick_primary_service
    model = {
        "containers": {
            "containers": [
                {"id": "web", "label": "Web", "kind": "web_app", "layer": "presentation"},
                {"id": "auth", "label": "Auth", "kind": "service", "layer": "application"},
                {"id": "core", "label": "Core", "kind": "service", "layer": "application"},
            ],
            "databases": [{"id": "pg", "label": "PG", "kind": "datastore"}],
            "relationships": [{"from": "core", "to": "pg", "label": "reads/writes"}],
        },
    }
    primary = _pick_primary_service(model)
    check("primary: picks the datastore owner (core)", primary and primary["id"] == "core", primary)


# ── consolidate_containers_for_abstraction + path domains ─────────────────────

def _many_app_containers(n, group=None, layer="application", kind="service"):
    out = []
    for i in range(n):
        c = {"id": f"svc{i:02d}", "label": f"Svc{i:02d}", "kind": kind, "layer": layer}
        if group is not None:
            c["group"] = group(i) if callable(group) else group
        out.append(c)
    return out


def test_consolidate_noop_under_window():
    """≤12 containers: never consolidates (already readable)."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = _many_app_containers(6, group=lambda i: f"Dom{i % 3}")
    model = _model_with_edges(conts)
    result = consolidate_containers_for_abstraction(model)
    check("consolidate: noop under window", len(result["containers"]["containers"]) == 6,
          len(result["containers"]["containers"]))
    check("consolidate: no represented marker when noop",
          "_represented_unit_count" not in result, result.get("_represented_unit_count"))


def test_consolidate_folds_domains_into_reps():
    """24 app containers in 4 domains → 4 representative containers."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = _many_app_containers(24, group=lambda i: f"Domain {i % 4}")
    model = _model_with_edges(conts)
    result = consolidate_containers_for_abstraction(model)
    out = result["containers"]["containers"]
    groups = {c["group"] for c in out}
    check("consolidate: 24→4 reps", len(out) == 4, len(out))
    check("consolidate: 4 distinct domains", len(groups) == 4, groups)
    check("consolidate: represented credits pre-fold count",
          result.get("_represented_unit_count") == 24, result.get("_represented_unit_count"))
    check("consolidate: rep description lists members",
          all("modules:" in c.get("description", "") for c in out), [c.get("description") for c in out])


def test_consolidate_remaps_edges_and_drops_selfloops():
    """Intra-domain edges become self-loops on the rep and are dropped; cross-domain kept."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = _many_app_containers(24, group=lambda i: f"Domain {i % 4}")
    # svc00 & svc04 are both Domain 0; svc01 is Domain 1 → cross-domain edge survives
    rels = [
        {"from": "svc00", "to": "svc04", "label": "calls"},   # intra Domain 0 → self-loop
        {"from": "svc00", "to": "svc01", "label": "calls"},   # Domain0 → Domain1 (cross)
    ]
    model = _model_with_edges(conts, rels=rels)
    result = consolidate_containers_for_abstraction(model)
    pairs = {(r["from"], r["to"]) for r in result["containers"]["relationships"]}
    check("consolidate: no self-loops after fold", all(f != t for f, t in pairs), pairs)
    check("consolidate: cross-domain edge preserved (remapped)",
          ("domain_0", "domain_1") in pairs, pairs)


def test_consolidate_keeps_presentation_standalone():
    """A web_app/presentation entrypoint is never folded into a domain rep."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = _many_app_containers(20, group="Plugins")
    conts.append({"id": "frontend", "label": "Frontend", "kind": "web_app",
                  "layer": "presentation", "group": "Plugins"})
    model = _model_with_edges(conts)
    result = consolidate_containers_for_abstraction(model)
    ids = {c["id"] for c in result["containers"]["containers"]}
    check("consolidate: frontend stays standalone", "frontend" in ids, ids)
    check("consolidate: presentation not folded", any(c["id"] == "frontend"
          and c.get("kind") == "web_app" for c in result["containers"]["containers"]), ids)


def test_consolidate_folds_presentation_modules():
    """GUI-app monorepo: '*UI' modules classify as presentation but must still fold;
    only the actual entrypoint stays standalone (regression: whole presentation layer
    was wrongly protected → nothing folded on PowerToys)."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = []
    for i in range(16):  # all classify as presentation via the '... UI' name
        conts.append({"id": f"mod{i:02d}_ui", "label": f"Module {i:02d} UI",
                      "kind": "service", "layer": "presentation", "group": "Modules"})
    for i in range(3):
        conts.append({"id": f"set{i}", "label": f"Setting {i}", "kind": "service",
                      "layer": "presentation", "group": "Settings"})
    # a real entrypoint
    conts.append({"id": "frontend", "label": "Frontend", "kind": "web_app",
                  "layer": "presentation", "group": "Modules"})
    model = _model_with_edges(conts)
    result = consolidate_containers_for_abstraction(model)
    out = {c["id"]: c for c in result["containers"]["containers"]}
    check("consolidate: presentation modules folded (not 20 boxes)",
          len(out) <= 4, sorted(out))
    check("consolidate: entrypoint frontend stays standalone", "frontend" in out, sorted(out))
    check("consolidate: a 'Modules' rep exists",
          any(c.get("group") == "Modules" and "modules:" in c.get("description", "")
              for c in result["containers"]["containers"]), sorted(out))


def test_consolidate_skips_single_giant_group():
    """One domain of N siblings would collapse to 1 box (<floor) → left untouched."""
    from doc_agent.tools.container_model import consolidate_containers_for_abstraction
    conts = _many_app_containers(20, group="OneBig")
    model = _model_with_edges(conts)
    result = consolidate_containers_for_abstraction(model)
    check("consolidate: single giant group not over-collapsed",
          len(result["containers"]["containers"]) == 20,
          len(result["containers"]["containers"]))


def test_path_domain_helper():
    """_path_domain = topmost non-generic ancestor; groups deep-nested domain folders,
    keeps generic-root siblings standalone."""
    from doc_agent.tools.container_model import _path_domain
    check("path: src/modules/x → modules", _path_domain("src/modules/fancyzones") == "modules",
          _path_domain("src/modules/fancyzones"))
    # deep nesting (the real PowerToys shape) still groups under the domain folder
    check("path: src/modules/A/B/proj → modules (deep)",
          _path_domain("src/modules/launcher/Plugins/Calculator") == "modules",
          _path_domain("src/modules/launcher/Plugins/Calculator"))
    check("path: src/common/Common.UI → common", _path_domain("src/common/Common.UI") == "common",
          _path_domain("src/common/Common.UI"))
    check("path: services/cart → standalone", _path_domain("services/cart") is None,
          _path_domain("services/cart"))
    check("path: src/services/cart → standalone (both generic)",
          _path_domain("src/services/cart") is None, _path_domain("src/services/cart"))
    check("path: src/runner → standalone (generic root)", _path_domain("src/runner") is None,
          _path_domain("src/runner"))
    check("path: top-level unit → standalone", _path_domain("runner") is None,
          _path_domain("runner"))


def test_assign_domains_uses_path_grouping():
    """Path-based domains beat slug prefixes for monorepo layouts."""
    from doc_agent.tools.container_model import assign_domains
    conts = [
        {"id": "fancyzones", "label": "FancyZones", "kind": "service", "layer": "application"},
        {"id": "awake",      "label": "Awake",      "kind": "service", "layer": "application"},
        {"id": "runner",     "label": "Runner",     "kind": "service", "layer": "application"},
    ]
    model = _model_with_edges(conts)
    model["_container_paths"] = {
        "fancyzones": "src/modules/fancyzones",
        "awake":      "src/modules/awake",
        "runner":     "src/runner",
    }
    result = assign_domains(model)
    g = {c["id"]: c.get("group") for c in result["containers"]["containers"]}
    check("path-domains: modules grouped together", g["fancyzones"] == g["awake"] == "Modules", g)
    check("path-domains: runner standalone", g["runner"] == "Runner", g)


def test_score_hld_credits_consolidated_coverage():
    """End-to-end: a consolidated model scores abstraction AND coverage high."""
    from doc_agent.tools.container_model import (
        consolidate_containers_for_abstraction, validate_model,
    )
    from doc_agent.evaluation.fidelity_scorer import _score_hld
    conts = _many_app_containers(24, group=lambda i: f"Domain {i % 4}")
    rels = [{"from": f"svc{i:02d}", "to": f"svc{(i + 1) % 24:02d}", "label": "calls"}
            for i in range(24)]
    model = _model_with_edges(conts, rels=rels)
    model["_discovered_unit_count"] = 24
    model = consolidate_containers_for_abstraction(model)
    vr = validate_model(model)
    sc = _score_hld({}, model, vr, [])
    bd = sc["breakdown"]
    check("score: abstraction high after consolidation", bd["abstraction"] >= 85, bd)
    check("score: coverage NOT punished for abstracting", bd["coverage"] >= 85, bd)
    check("score: validity not failed by consolidation", bd["validity"] >= 85, bd)


def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        try:
            case()
        except Exception as e:
            _FAILURES.append(f"{case.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail[:200]}")


if __name__ == "__main__":
    raise SystemExit(main())
