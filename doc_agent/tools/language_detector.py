"""
Language detection for the documentation agent.

Single source of truth for which languages/extensions are supported, and for
mapping a file or a directory to language(s). Consulted by the extractor facade
(tools/extractor.py), input_resolver (single-file validation), and the API
error path that returns HTTP 400 for unsupported repos.

Keep SUPPORTED_EXTENSIONS in sync with extractors/registry.EXTENSION_TO_LANGUAGE.
"""

from pathlib import Path

# The only extensions the agent can extract facts from.
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".cs": "csharp",
    ".java": "java",
}

# Sorted, de-duplicated language names — handy for user-facing error messages.
SUPPORTED_LANGUAGES = sorted(set(SUPPORTED_EXTENSIONS.values()))

# Directories that never hold hand-written source worth documenting. Load-bearing:
# generated/build output here would skew language counts and component clustering.
SKIP_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", "node_modules", ".idea",
    ".vscode", "dist", "build", ".next", "bin", "obj", "target", ".gradle",
    "coverage", ".pytest_cache", ".mypy_cache",
}


def language_for_file(path) -> str | None:
    """Return the canonical language name for a file, or None if unsupported."""
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())


def detect_languages(path) -> dict:
    """
    Count supported source files per language under a directory.

    Returns:
        {
            "dominant": "<language>" | None,   # most common supported language
            "languages": {lang: count, ...},
            "total_files": int,
            "supported_languages": [lang, ...],
        }
    """
    root = Path(path)
    counts: dict[str, int] = {}
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        lang = SUPPORTED_EXTENSIONS.get(f.suffix.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return {"dominant": None, "languages": {}, "total_files": 0,
                "supported_languages": []}

    dominant = max(counts, key=counts.get)
    return {
        "dominant": dominant,
        "languages": counts,
        "total_files": sum(counts.values()),
        "supported_languages": list(counts.keys()),
    }
