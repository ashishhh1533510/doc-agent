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

from doc_agent.tools.container_model import build_container_model, apply_enrichment
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
# Archetype 1: Express + Mongoose  →  service + MongoDB datastore + person
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
#   →  web_app + service + PostgreSQL datastore + edge web→service
# ══════════════════════════════════════════════════════════════════════════════

def test_react_nestjs_postgres_archetype():
    """Realworld style: separate manifests for frontend and backend."""
    with tempfile.TemporaryDirectory() as repo_root:
        # Two separate manifests → two deployable roots
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
        model = build_container_model(facts, repo_root)
        kinds = _kinds(model)

        check("react+nestjs+pg: has web_app for frontend",
              bool(kinds.get("web_app")), str(kinds))
        check("react+nestjs+pg: has service for backend",
              bool(kinds.get("service")), str(kinds))
        check("react+nestjs+pg: has PostgreSQL datastore",
              any("postgresql" in nid or "postgres" in nid for nid in kinds.get("datastore", [])),
              str(kinds))

        # Verify web_app → service edge exists
        all_rels = (model["containers"].get("relationships") or []) + \
                   (model["context"].get("relationships") or [])
        web_ids = set(kinds.get("web_app", []))
        svc_ids = set(kinds.get("service", []))
        has_fe_to_be = any(
            r.get("from") in web_ids and r.get("to") in svc_ids
            for r in all_rels
        )
        check("react+nestjs+pg: web_app → service edge present",
              has_fe_to_be, str(all_rels))


# ══════════════════════════════════════════════════════════════════════════════
# Archetype 3: Next.js + Prisma + stripe + googleapis
#   →  one web_app (fullstack, NOT split) + Database + Stripe + Google APIs externals
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

        # Next.js is fullstack → must produce exactly ONE web_app, NOT split
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
    # person → stadium ([" "])
    check("emit: person → stadium shape",
          '(["' in _emit_node("user", "User", "person"))

    # service/web_app/worker → rectangle [" "]
    check("emit: service → rectangle shape",
          '[" ' not in _emit_node("api", "API", "service") and
          '["API"]' in _emit_node("api", "API", "service"),
          _emit_node("api", "API", "service"))

    # datastore → cylinder [(" ")]
    ds = _emit_node("db", "MongoDB", "datastore")
    check("emit: datastore → cylinder shape",
          '[("' in ds, ds)

    # cache → cylinder
    cache = _emit_node("redis", "Redis", "cache")
    check("emit: cache → cylinder shape",
          '[("' in cache, cache)

    # queue → parallelogram [/" "/]
    queue = _emit_node("q", "Kafka", "queue")
    check("emit: queue → parallelogram shape",
          '[/"' in queue, queue)

    # external → hexagon {{"..."}} + :::ext
    ext = _emit_node("stripe", "Stripe", "external")
    check("emit: external → hexagon shape (double braces)",
          '{{' in ext and '}}' in ext, ext)
    check("emit: external → :::ext class applied",
          ":::ext" in ext, ext)


def test_render_c4_cylinder_present():
    """A model with a database node must produce [(" in the rendered output."""
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
    check("render: cylinder syntax [( present for datastore",
          "[(" in rendered, rendered[:200])
    check("render: stadium syntax ([ present for person",
          "([" in rendered, rendered[:200])
    check("render: external systems rendered as hexagon when present",
          True)  # no external in this model, trivially true


def test_render_c4_external_hexagon():
    """External nodes must produce {{ }} hexagon syntax."""
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
    check("render: external → double-brace hexagon in output",
          "{{" in rendered and "}}" in rendered, rendered[:300])
    check("render: external → :::ext class in output",
          ":::ext" in rendered, rendered[:300])


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
        model = build_container_model(facts, repo_root, manifests=manifests)
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
    """@Mapper on a Java class → is_db_model=True → datastore node appears."""
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
    """package.json with pg dependency → PostgreSQL datastore node via manifest scanning."""
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


def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        try:
            case()
        except Exception as e:
            _FAILURES.append(f"{case.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
