"""
Manifest parser — extract project name + dependency list from build manifests.

Supports:
  pom.xml           -> Maven (Java)
  package.json      -> Node.js
  pyproject.toml    -> Python (PEP 517 / Poetry)
  *.csproj *.fsproj -> .NET
  go.mod            -> Go
  build.gradle(.kts) -> Gradle (Java/Kotlin)

All parsing is best-effort and exception-safe: a malformed manifest returns
empty strings / empty lists rather than crashing the run. The caller
(build_container_model) picks the best available signal.

Public API:
  parse_manifest(path: str) -> {"project_name": str, "dependencies": [str], "language": str}
  parse_all_manifests(repo_root: str) -> {abs_dir: parsed_dict}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from doc_agent.tools.language_detector import SKIP_DIRS


# ── manifest file names covered by this parser ────────────────────────────────
_MANIFEST_NAMES = frozenset({
    "package.json", "pom.xml", "build.gradle", "build.gradle.kts",
    "go.mod", "pyproject.toml", "setup.py", "requirements.txt", "Gemfile",
})
_MANIFEST_GLOBS = ("*.csproj", "*.fsproj")


def _skip_path(parts: tuple) -> bool:
    return any(p in SKIP_DIRS for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-format parsers
# ─────────────────────────────────────────────────────────────────────────────

def _pom_is_deployable(path: Path) -> bool:
    """True if this pom.xml has a Spring Boot plugin or war packaging (bootable artifact)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "spring-boot-maven-plugin" in text:
            return True
        if "<packaging>war</packaging>" in text:
            return True
        return False
    except Exception:
        return False


def _gradle_is_deployable(path: Path) -> bool:
    """True if this build.gradle(.kts) has a Spring Boot or application plugin (bootable).

    Ignores 'apply false' version-pin lines (BOM / platform catalog usage) — a root
    build.gradle that declares 'id ... apply false' is only a version lock, not an
    applied plugin, so it must NOT be treated as a bootable entry point.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # bootJar / bootRun task references always mean the boot plugin is active
        if "bootJar" in text or "bootRun" in text:
            return True
        for line in text.splitlines():
            stripped = line.strip()
            if "apply false" in stripped:
                continue  # version pin — not an applied plugin, skip
            if "org.springframework.boot" in stripped:
                return True
        # Gradle 'application' plugin marks an executable distribution
        for line in text.splitlines():
            stripped = line.strip()
            if "apply false" in stripped:
                continue
            if re.search(r"""['"]application['"]""", stripped) or "id('application')" in stripped or 'id("application")' in stripped:
                return True
        return False
    except Exception:
        return False


def _package_json_is_deployable(path: Path) -> bool:
    """True if this package.json has a start script, bin field, or server-side framework dep."""
    _SERVER_DEPS = {
        "express", "fastify", "nestjs", "@nestjs/core", "koa", "hapi", "@hapi/hapi",
        "next", "remix", "@remix-run/node", "sveltekit", "@sveltejs/kit",
        "nuxt", "@nuxt/kit",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        scripts = data.get("scripts") or {}
        if any(k in scripts for k in ("start", "serve", "dev")):
            return True
        if data.get("bin"):
            return True
        all_deps = set((data.get("dependencies") or {}).keys()) | set((data.get("devDependencies") or {}).keys())
        if all_deps & _SERVER_DEPS:
            return True
        return False
    except Exception:
        return False


def _pyproject_is_deployable(path: Path) -> bool:
    """True if this pyproject.toml/setup.py has scripts or a web-server dep."""
    _SERVER_DEPS = {
        "fastapi", "flask", "django", "uvicorn", "gunicorn", "starlette",
        "tornado", "aiohttp", "quart", "sanic",
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "[project.scripts]" in text or "console_scripts" in text or "[tool.poetry.scripts]" in text:
            return True
        for dep in _SERVER_DEPS:
            if re.search(rf'(?i)["\']?{re.escape(dep)}["\']?\s*[=\[]', text):
                return True
        # Also check sibling requirements.txt
        req = path.parent / "requirements.txt"
        if req.exists():
            req_text = req.read_text(encoding="utf-8", errors="replace").lower()
            if any(dep in req_text for dep in _SERVER_DEPS):
                return True
        return False
    except Exception:
        return False


_CSPROJ_TEST_MARKERS = (
    "<istestproject>true",          # explicit MSBuild test flag
    "microsoft.net.test.sdk",       # the .NET test host
    'include="mstest"', "mstest.testframework",
    "xunit", "nunit", "microsoft.testplatform",
)


def _csproj_is_test(text_lower: str, path: Path) -> bool:
    """True if this .csproj is a test project (unit/UI/integration/fuzz/benchmark).

    Test runners are built as `Exe` (so the test host can launch them) but are NOT
    architectural deployment units — admitting them puts 'FancyZones UI Tests' boxes
    in the diagram instead of the real modules. Detected via the MSBuild test flag,
    a test-framework package reference, or the conventional project-name suffix.
    """
    if any(m in text_lower for m in _CSPROJ_TEST_MARKERS):
        return True
    stem = path.stem.lower()
    return stem.endswith((
        "tests", "test", "unittests", "uitests", "integrationtests",
        "fuzztests", "benchmarks", "benchmark",
    ))


def _csproj_is_deployable(path: Path) -> bool:
    """True if this .csproj targets an executable or web/worker SDK.

    Excludes test projects: PowerToys-style WinUI/WPF apps ship as `WinExe`, while
    MSTest/xUnit test runners also build as `Exe` — so without the test exclusion the
    gate admits test harnesses and (before the WinExe fix) dropped the real modules.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tl = text.lower()
        if _csproj_is_test(tl, path):
            return False
        # WinExe = Windows GUI app (the real desktop modules); Exe = console app.
        if "<outputtype>exe" in tl or "<outputtype>winexe" in tl:
            return True
        if "Microsoft.NET.Sdk.Web" in text or "Microsoft.NET.Sdk.Worker" in text:
            return True
        return False
    except Exception:
        return False


def _parse_pom(path: Path) -> dict:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(path))
        root = tree.getroot()
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        def _find(node, tag: str):
            """Find child element, trying Maven namespace then bare tag."""
            el = node.find(f"m:{tag}", ns)
            if el is None:
                el = node.find(tag)
            return el

        def _text(tag: str) -> str:
            el = _find(root, tag)
            return (el.text or "").strip() if el is not None else ""

        name = _text("name") or _text("artifactId")
        deps = []
        dep_nodes = root.findall(".//m:dependency", ns)
        if not dep_nodes:
            dep_nodes = root.findall(".//dependency")
        for d in dep_nodes:
            gid = _find(d, "groupId")
            aid = _find(d, "artifactId")
            if gid is not None and aid is not None:
                g = (gid.text or "").strip()
                a = (aid.text or "").strip()
                if g and a:
                    deps.append(f"{g}:{a}")
        deployable = _pom_is_deployable(path)
        return {"project_name": name, "dependencies": deps, "language": "java", "deployable": deployable}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "java", "deployable": False}


def _parse_package_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        name = (data.get("name") or "").strip()
        deps = list(data.get("dependencies", {}).keys()) + list(data.get("devDependencies", {}).keys())
        deployable = _package_json_is_deployable(path)
        return {"project_name": name, "dependencies": deps, "language": "javascript", "deployable": deployable}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "javascript", "deployable": False}


def _parse_pyproject(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Try to extract [project] name or [tool.poetry] name
        name = ""
        m = re.search(r'^\[project\].*?^name\s*=\s*["\']([^"\']+)', text, re.MULTILINE | re.DOTALL)
        if not m:
            m = re.search(r'^\[tool\.poetry\].*?^name\s*=\s*["\']([^"\']+)', text, re.MULTILINE | re.DOTALL)
        if m:
            name = m.group(1).strip()

        # Collect dependencies
        deps = []
        # PEP-517 style: dependencies = ["foo>=1.0", ...]
        dep_m = re.search(r'^dependencies\s*=\s*\[(.*?)\]', text, re.MULTILINE | re.DOTALL)
        if dep_m:
            for tok in re.findall(r'["\']([A-Za-z0-9_\-\.]+)', dep_m.group(1)):
                deps.append(tok)
        # Poetry style: [tool.poetry.dependencies] foo = "..."
        poetry_sec = re.search(r'\[tool\.poetry\.dependencies\](.*?)(?=^\[|\Z)', text, re.MULTILINE | re.DOTALL)
        if poetry_sec:
            for tok in re.findall(r'^([A-Za-z0-9_\-\.]+)\s*=', poetry_sec.group(1), re.MULTILINE):
                if tok not in ("python",):
                    deps.append(tok)

        # Also read sibling requirements.txt
        req = path.parent / "requirements.txt"
        if req.exists():
            for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = re.split(r"[>=<!;#\s]", line)[0]
                    if pkg:
                        deps.append(pkg)

        deployable = _pyproject_is_deployable(path)
        return {"project_name": name, "dependencies": list(dict.fromkeys(deps)), "language": "python", "deployable": deployable}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "python", "deployable": False}


def _parse_requirements_txt(path: Path) -> dict:
    _SERVER_DEPS = {"fastapi", "flask", "django", "uvicorn", "gunicorn", "starlette", "tornado", "aiohttp"}
    try:
        deps = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = re.split(r"[>=<!;#\s]", line)[0]
                if pkg:
                    deps.append(pkg)
        deployable = any(d.lower() in _SERVER_DEPS for d in deps)
        return {"project_name": "", "dependencies": deps, "language": "python", "deployable": deployable}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "python", "deployable": False}


def _parse_csproj(path: Path) -> dict:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(path))
        root = tree.getroot()
        name = path.stem   # .csproj filename is the project name
        deps = []
        for ref in root.iter("PackageReference"):
            include = ref.get("Include") or ref.get("include") or ""
            if include:
                deps.append(include.strip())
        deployable = _csproj_is_deployable(path)
        return {"project_name": name, "dependencies": deps, "language": "csharp", "deployable": deployable}
    except Exception:
        return {"project_name": path.stem, "dependencies": [], "language": "csharp", "deployable": False}


def _parse_go_mod(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        name = ""
        m = re.search(r'^module\s+(\S+)', text, re.MULTILINE)
        if m:
            name = m.group(1).rsplit("/", 1)[-1]
        deps = re.findall(r'^\s+(\S+)\s+v[\d.]+', text, re.MULTILINE)
        # go.mod deployability is determined by whether any file in the module has
        # package main; we set a neutral True here and let _has_entry (from
        # architecture_model) act as the gate in container_model's pass 2.
        return {"project_name": name, "dependencies": deps, "language": "go", "deployable": True}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "go", "deployable": False}


def _parse_gradle(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Try sibling settings.gradle / settings.gradle.kts for rootProject.name
        name = ""
        for settings_name in ("settings.gradle", "settings.gradle.kts"):
            sfile = path.parent / settings_name
            if sfile.exists():
                sm = re.search(r'rootProject\.name\s*[=:]\s*["\']([^"\']+)', sfile.read_text(encoding="utf-8", errors="replace"))
                if sm:
                    name = sm.group(1).strip()
                    break
        if not name:
            # fallback: directory name
            name = path.parent.name

        deps = []
        # Groovy DSL: implementation 'group:artifact:version' or implementation("group:artifact:version")
        for m in re.finditer(r"""(?:implementation|api|compile|runtimeOnly|testImplementation)\s*[\("']+([^"'\)]+)""", text):
            coord = m.group(1).strip().strip("\"'")
            if ":" in coord:
                parts = coord.split(":")
                if len(parts) >= 2:
                    deps.append(f"{parts[0]}:{parts[1]}")

        deployable = _gradle_is_deployable(path)
        return {"project_name": name, "dependencies": deps, "language": "java", "deployable": deployable}
    except Exception:
        return {"project_name": "", "dependencies": [], "language": "java", "deployable": False}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

def parse_manifest(path: str) -> dict:
    """Parse one manifest file.

    Returns {"project_name": str, "dependencies": [str], "language": str}.
    All values are best-effort; may be empty strings / empty lists on failure.
    """
    p = Path(path)
    name_lower = p.name.lower()
    if name_lower == "pom.xml":
        return _parse_pom(p)
    if name_lower == "package.json":
        return _parse_package_json(p)
    if name_lower == "pyproject.toml":
        return _parse_pyproject(p)
    if name_lower in ("requirements.txt",):
        return _parse_requirements_txt(p)
    if name_lower in ("build.gradle", "build.gradle.kts"):
        return _parse_gradle(p)
    if name_lower == "go.mod":
        return _parse_go_mod(p)
    if p.suffix.lower() in (".csproj", ".fsproj"):
        return _parse_csproj(p)
    # Unknown manifest type — return empty (not deployable)
    return {"project_name": "", "dependencies": [], "language": "unknown", "deployable": False}


def parse_all_manifests(repo_root: str) -> dict[str, dict]:
    """Walk repo_root and parse every build manifest found.

    Returns {absolute_manifest_dir: parsed_dict}.
    Skips SKIP_DIRS and build/dist/generated folders.
    When multiple manifests share the same directory (e.g. build.gradle +
    settings.gradle), the one with the richer result is kept.
    """
    root = Path(repo_root)
    result: dict[str, dict] = {}

    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir():
            continue
        rel_parts = candidate.relative_to(root).parts
        if _skip_path(rel_parts[:-1]):
            continue
        name_lower = candidate.name.lower()
        is_manifest = (
            name_lower in {n.lower() for n in _MANIFEST_NAMES}
            or any(candidate.match(g) for g in _MANIFEST_GLOBS)
        )
        if not is_manifest:
            continue

        dir_abs = str(candidate.parent.resolve())
        parsed = parse_manifest(str(candidate))

        if dir_abs not in result:
            result[dir_abs] = parsed
        else:
            # Keep the richer of the two results for the same directory
            existing = result[dir_abs]
            # deployable: True wins (either manifest signals bootable)
            is_deployable = existing.get("deployable", False) or parsed.get("deployable", False)
            if len(parsed.get("dependencies", [])) > len(existing.get("dependencies", [])):
                # Merge: prefer the one with more deps; keep the non-empty name
                result[dir_abs] = {
                    "project_name": existing.get("project_name") or parsed.get("project_name", ""),
                    "dependencies": parsed["dependencies"],
                    "language": parsed.get("language") or existing.get("language", ""),
                    "deployable": is_deployable,
                }
            else:
                if not existing.get("project_name") and parsed.get("project_name"):
                    result[dir_abs]["project_name"] = parsed["project_name"]
                result[dir_abs]["deployable"] = is_deployable

    return result
