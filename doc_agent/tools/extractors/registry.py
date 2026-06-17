"""
Extractor registry — the single place that maps a file extension to the
function that extracts FileFacts from that kind of file.

The facade (tools/extractor.py) consults this instead of hard-coding ".py",
so adding a language later means editing only this table plus its module.
Keep EXTENSION_TO_LANGUAGE in sync with language_detector.SUPPORTED_EXTENSIONS.
"""

import importlib

# File extension -> canonical language name.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".cs": "csharp",
    ".java": "java",
}

# Language -> (module under this package, per-file extractor function name).
# All of these functions return the same FileFacts dict shape.
_EXTRACTORS = {
    "python": ("python_extractor", "extract_from_python_file"),
    "typescript": ("ts_extractor", "extract_from_typescript_file"),
    "javascript": ("ts_extractor", "extract_from_typescript_file"),
    "csharp": ("csharp_extractor", "extract_from_csharp_file"),
    "java": ("java_extractor", "extract_from_java_file"),
}


def get_extractor(ext: str):
    """
    Return the per-file extractor callable for a file extension, or None if the
    extension is unsupported. Imports the extractor module lazily so unrelated
    languages' import errors never block the supported path.
    """
    lang = EXTENSION_TO_LANGUAGE.get(ext.lower())
    if lang is None:
        return None
    mod_name, fn_name = _EXTRACTORS[lang]
    module = importlib.import_module(f"doc_agent.tools.extractors.{mod_name}")
    return getattr(module, fn_name)
