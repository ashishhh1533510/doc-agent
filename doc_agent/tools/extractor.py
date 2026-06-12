"""
Extractor facade — the language-agnostic entry point every pipeline imports.

Looks at each file's extension and dispatches to the right per-language
extractor (extractors/registry.py). Every extractor returns the same FileFacts
shape, so all downstream pipelines (README, QA, HLD, LLD) keep working
unchanged. The Python logic now lives in extractors/python_extractor.py.
"""

from pathlib import Path

from doc_agent.tools.extractors.registry import get_extractor, EXTENSION_TO_LANGUAGE
from doc_agent.tools.language_detector import SKIP_DIRS, SUPPORTED_LANGUAGES

# Re-export the Python source parser under its old name for any back-compat use.
from doc_agent.tools.extractors.python_extractor import (
    extract_from_python_source as extract_from_source,
)


class UnsupportedLanguageError(Exception):
    """Raised when an input has no files in any supported language."""


# Generated / minified files we never want to parse.
_SKIP_FILE_SUFFIXES = (".min.js", ".d.ts", ".designer.cs", ".g.cs")


def _is_skippable_file(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(name.endswith(suf) for suf in _SKIP_FILE_SUFFIXES)


def extract_from_file(path) -> dict:
    """Extract FileFacts from one file, dispatching by extension."""
    path = Path(path)
    extractor = get_extractor(path.suffix)
    if extractor is None:
        raise UnsupportedLanguageError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}."
        )
    return extractor(str(path))


def extract_from_directory(path) -> list[dict]:
    """
    Walk a directory and extract FileFacts from every supported source file.

    Skips junk/build folders and generated files. A file that fails to parse is
    recorded as an error entry instead of crashing the run. Raises
    UnsupportedLanguageError if the directory has no supported source at all.
    """
    path = Path(path)
    results = []
    found_supported = False
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTENSION_TO_LANGUAGE:
            continue
        if _is_skippable_file(f):
            continue
        found_supported = True
        try:
            results.append(extract_from_file(f))
        except SyntaxError as e:
            results.append({"file": str(f), "error": f"SyntaxError: {e}"})
        except Exception as e:  # tree-sitter never raises, but be defensive
            results.append({"file": str(f), "error": f"{type(e).__name__}: {e}"})
    if not found_supported:
        raise UnsupportedLanguageError(
            f"No supported source files found. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}."
        )
    return results


# ---------------------------------------------------------------------------
# Rich extraction — UNCHANGED for now; Step 6 reworks the import graph.
# ---------------------------------------------------------------------------

_EXT_LANGUAGE = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".cs": "csharp", ".go": "go",
}
_RICH_SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "dist", "build"}


def _count_languages(path: Path) -> dict:
    """Count source files per language under path, skipping generated folders."""
    counts: dict[str, int] = {}
    for f in path.rglob("*"):
        if f.is_file() and not any(p in _RICH_SKIP_DIRS for p in f.parts):
            lang = _EXT_LANGUAGE.get(f.suffix.lower())
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
    return counts


def _detect_framework(files: list[dict]) -> str:
    """Infer the primary web framework from import names across all files."""
    all_imports = " ".join(
        imp for entry in files for imp in entry.get("imports", [])
    ).lower()
    for framework in ("fastapi", "flask", "django", "starlette", "tornado", "express"):
        if framework in all_imports:
            return framework
    return "unknown"


def extract_rich_from_directory(path) -> dict:
    """RichFacts: per-file details plus import graph, language counts, framework."""
    path = Path(path)
    files = extract_from_directory(path)

    pkg_name = path.name
    import_graph: dict[str, list[str]] = {}
    for entry in files:
        if "error" in entry:
            continue
        try:
            rel = Path(entry["file"]).relative_to(path)
            mod = str(rel).replace("\\", "/").replace("/", ".").removesuffix(".py")
        except ValueError:
            continue
        internal = [i for i in entry.get("imports", []) if i.startswith(pkg_name)]
        if internal:
            import_graph[mod] = internal

    languages = _count_languages(path)
    primary = max(languages, key=lambda k: languages[k]) if languages else "unknown"
    framework = _detect_framework(files)

    return {
        "primary_language": primary,
        "languages": languages,
        "framework": framework,
        "files": files,
        "import_graph": import_graph,
    }


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    mode = sys.argv[2] if len(sys.argv) > 2 else "basic"
    if mode == "rich":
        print(json.dumps(extract_rich_from_directory(target), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(extract_from_directory(target), indent=2, ensure_ascii=False))
