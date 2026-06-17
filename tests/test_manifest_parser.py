"""
Tests for doc_agent/tools/manifest_parser.py — no network, no LLM.

Covers parse_manifest() and parse_all_manifests() for all supported formats.

Run:  ./venv/Scripts/python.exe tests/test_manifest_parser.py
Exit code is non-zero if any check fails.
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.manifest_parser import parse_manifest, parse_all_manifests


# ── tiny harness ──────────────────────────────────────────────────────────────
_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ══════════════════════════════════════════════════════════════════════════════
# pom.xml
# ══════════════════════════════════════════════════════════════════════════════

def test_pom_full():
    """Parse a pom.xml with <name>, <artifactId>, and <dependencies>."""
    with tempfile.TemporaryDirectory() as d:
        pom = Path(d, "pom.xml")
        pom.write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>spring-realworld</artifactId>
  <name>Spring Boot Realworld Example</name>
  <dependencies>
    <dependency>
      <groupId>org.mybatis.spring.boot</groupId>
      <artifactId>mybatis-spring-boot-starter</artifactId>
    </dependency>
    <dependency>
      <groupId>org.xerial</groupId>
      <artifactId>sqlite-jdbc</artifactId>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
  </dependencies>
</project>""")
        result = parse_manifest(str(pom))
        check("pom: language is java", result["language"] == "java", result["language"])
        check("pom: project_name from <name>",
              "Spring Boot Realworld" in result["project_name"], result["project_name"])
        check("pom: mybatis dep captured",
              any("mybatis" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))
        check("pom: sqlite-jdbc dep captured",
              any("sqlite" in d.lower() or "xerial" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))
        check("pom: spring-boot-starter-web dep captured",
              any("spring-boot-starter-web" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))


def test_pom_fallback_to_artifactid():
    """When <name> is absent, use <artifactId>."""
    with tempfile.TemporaryDirectory() as d:
        pom = Path(d, "pom.xml")
        pom.write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>my-service</artifactId>
</project>""")
        result = parse_manifest(str(pom))
        check("pom fallback: project_name = artifactId",
              result["project_name"] == "my-service", result["project_name"])


def test_pom_malformed_returns_empty():
    """A broken pom.xml must not raise."""
    with tempfile.TemporaryDirectory() as d:
        pom = Path(d, "pom.xml")
        pom.write_text("<not valid xml<<")
        result = parse_manifest(str(pom))
        check("pom malformed: returns dict", isinstance(result, dict))
        check("pom malformed: project_name is str", isinstance(result.get("project_name", ""), str))


# ══════════════════════════════════════════════════════════════════════════════
# package.json
# ══════════════════════════════════════════════════════════════════════════════

def test_package_json_full():
    """Parse a package.json with name + dependencies."""
    with tempfile.TemporaryDirectory() as d:
        pj = Path(d, "package.json")
        pj.write_text('{"name":"my-node-api","dependencies":{"express":"*","pg":"*"},"devDependencies":{"jest":"*"}}')
        result = parse_manifest(str(pj))
        check("pkg.json: language is javascript", result["language"] == "javascript")
        check("pkg.json: project_name", result["project_name"] == "my-node-api", result["project_name"])
        check("pkg.json: express dep captured",
              "express" in result["dependencies"], str(result["dependencies"]))
        check("pkg.json: pg dep captured",
              "pg" in result["dependencies"], str(result["dependencies"]))
        check("pkg.json: jest devDep captured",
              "jest" in result["dependencies"], str(result["dependencies"]))


def test_package_json_empty_name():
    """package.json without 'name' key returns empty string, not None."""
    with tempfile.TemporaryDirectory() as d:
        pj = Path(d, "package.json")
        pj.write_text('{"dependencies":{}}')
        result = parse_manifest(str(pj))
        check("pkg.json no name: project_name is str",
              isinstance(result["project_name"], str))


# ══════════════════════════════════════════════════════════════════════════════
# pyproject.toml
# ══════════════════════════════════════════════════════════════════════════════

def test_pyproject_pep517():
    """Parse a PEP-517 pyproject.toml with [project] section."""
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d, "pyproject.toml")
        pp.write_text("""[project]
name = "my-python-app"
dependencies = ["fastapi>=0.90", "sqlalchemy", "redis"]
""")
        result = parse_manifest(str(pp))
        check("pyproject pep517: project_name",
              result["project_name"] == "my-python-app", result["project_name"])
        check("pyproject pep517: fastapi dep",
              any("fastapi" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))
        check("pyproject pep517: sqlalchemy dep",
              any("sqlalchemy" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))


def test_pyproject_poetry():
    """Parse a Poetry pyproject.toml with [tool.poetry] section."""
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d, "pyproject.toml")
        pp.write_text("""[tool.poetry]
name = "poetry-app"

[tool.poetry.dependencies]
python = "^3.10"
flask = "*"
psycopg2 = "*"
""")
        result = parse_manifest(str(pp))
        check("pyproject poetry: project_name",
              result["project_name"] == "poetry-app", result["project_name"])
        check("pyproject poetry: flask dep",
              any("flask" in d.lower() for d in result["dependencies"]),
              str(result["dependencies"]))


# ══════════════════════════════════════════════════════════════════════════════
# *.csproj
# ══════════════════════════════════════════════════════════════════════════════

def test_csproj():
    """Parse a .csproj file."""
    with tempfile.TemporaryDirectory() as d:
        csp = Path(d, "MyApp.csproj")
        csp.write_text("""<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" Version="8.0.0" />
    <PackageReference Include="Stripe.net" Version="43.0.0" />
  </ItemGroup>
</Project>""")
        result = parse_manifest(str(csp))
        check("csproj: language is csharp", result["language"] == "csharp", result["language"])
        check("csproj: project_name = file stem",
              result["project_name"] == "MyApp", result["project_name"])
        check("csproj: EFCore dep captured",
              any("entityframeworkcore" in dep.lower() for dep in result["dependencies"]),
              str(result["dependencies"]))
        check("csproj: Stripe dep captured",
              any("stripe" in dep.lower() for dep in result["dependencies"]),
              str(result["dependencies"]))


# ══════════════════════════════════════════════════════════════════════════════
# go.mod
# ══════════════════════════════════════════════════════════════════════════════

def test_go_mod():
    """Parse a go.mod file."""
    with tempfile.TemporaryDirectory() as d:
        gm = Path(d, "go.mod")
        gm.write_text("""module github.com/myorg/my-go-service

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
    go.mongodb.org/mongo-driver v1.13.1
)
""")
        result = parse_manifest(str(gm))
        check("go.mod: language is go", result["language"] == "go", result["language"])
        check("go.mod: project_name = repo basename",
              result["project_name"] == "my-go-service", result["project_name"])
        check("go.mod: gin dep captured",
              any("gin" in dep.lower() for dep in result["dependencies"]),
              str(result["dependencies"]))
        check("go.mod: pq (postgres) dep captured",
              any("pq" in dep.lower() for dep in result["dependencies"]),
              str(result["dependencies"]))


# ══════════════════════════════════════════════════════════════════════════════
# parse_all_manifests
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_all_manifests_single_pom():
    """Single pom.xml at repo root → one entry keyed by repo_root abs path."""
    with tempfile.TemporaryDirectory() as repo_root:
        pom = Path(repo_root, "pom.xml")
        pom.write_text("""<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <artifactId>realworld</artifactId>
  <dependencies>
    <dependency>
      <groupId>org.xerial</groupId><artifactId>sqlite-jdbc</artifactId>
    </dependency>
  </dependencies>
</project>""")

        result = parse_all_manifests(repo_root)
        check("parse_all: single pom → one entry", len(result) == 1, str(list(result.keys())))
        parsed = list(result.values())[0]
        check("parse_all: project_name from pom",
              parsed["project_name"] == "realworld", parsed["project_name"])
        check("parse_all: sqlite-jdbc dep present",
              any("sqlite" in dep.lower() or "xerial" in dep.lower() for dep in parsed["dependencies"]),
              str(parsed["dependencies"]))


def test_parse_all_manifests_multi_module():
    """Two package.json files (frontend + backend) → two entries."""
    with tempfile.TemporaryDirectory() as repo_root:
        fe = Path(repo_root, "frontend")
        be = Path(repo_root, "backend")
        fe.mkdir(); be.mkdir()
        (fe / "package.json").write_text('{"name":"frontend","dependencies":{"react":"*"}}')
        (be / "package.json").write_text('{"name":"backend","dependencies":{"express":"*","pg":"*"}}')

        result = parse_all_manifests(repo_root)
        check("parse_all multi: two entries", len(result) == 2, str(list(result.keys())))
        names = {v["project_name"] for v in result.values()}
        check("parse_all multi: frontend name", "frontend" in names, str(names))
        check("parse_all multi: backend name", "backend" in names, str(names))


def test_parse_all_manifests_skips_node_modules():
    """Manifests inside node_modules / .git must be ignored."""
    with tempfile.TemporaryDirectory() as repo_root:
        (Path(repo_root, "package.json")).write_text('{"name":"root"}')
        nm = Path(repo_root, "node_modules", "some-pkg")
        nm.mkdir(parents=True)
        (nm / "package.json").write_text('{"name":"should-be-ignored"}')

        result = parse_all_manifests(repo_root)
        names = {v["project_name"] for v in result.values()}
        check("parse_all skip: node_modules not in results",
              "should-be-ignored" not in names, str(names))
        check("parse_all skip: root still present",
              "root" in names, str(names))


# ══════════════════════════════════════════════════════════════════════════════

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
