"""
Scan-and-extract tool for the documentation agent.

Reads Python source code and pulls out its structure (functions, classes,
signatures, docstrings) using the built-in `ast` module. This produces the
"ground truth" facts that later agents use to generate and verify docs.
No LLM is involved here -- this is pure, deterministic parsing.
"""

import ast
import re
from pathlib import Path
import sys


def _format_arguments(args: ast.arguments) -> str:
    """
    Turn a function's argument node into a readable parameter string.

    Example output: 'name: str, count: int = 1, *args, **kwargs'
    Handles positional args, default values, *args, keyword-only args,
    and **kwargs so the signature we record matches the real source.
    """
    parts = []

    # posonlyargs are parameters before a '/' in the signature (rare).
    posonly = getattr(args, "posonlyargs", [])
    positional = posonly + args.args

    # Defaults line up with the END of the positional args. If a function has
    # 3 params but only 1 default, that default belongs to the LAST param.
    default_offset = len(positional) - len(args.defaults)

    for i, arg in enumerate(positional):
        piece = arg.arg  # the parameter's name
        # arg.annotation is the type hint (e.g. the 'str' in 'name: str'), if any.
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        # If this param falls in the range that has defaults, attach its default.
        if i >= default_offset:
            piece += f" = {ast.unparse(args.defaults[i - default_offset])}"
        parts.append(piece)
        # After the last positional-only arg, add the '/' separator.
        if posonly and arg is posonly[-1]:
            parts.append("/")

    # '*args' (vararg) OR a bare '*' marking the start of keyword-only args.
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    # Keyword-only args have their own defaults list (kw_defaults).
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        parts.append(piece)

    # '**kwargs' (kwarg) comes last.
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


# HTTP methods recognised in route decorators (@app.get, @router.post, etc.)
_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
_ROUTE_RE = re.compile(
    r'[\w.]+\.(' + '|'.join(_ROUTE_METHODS) + r')\(["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _describe_function(node) -> dict:
    """
    Extract the documentable facts from a single function node.

    Returns its name, whether it's async, a full readable signature, its
    return type annotation, any decorators, extracted HTTP routes, a call
    graph (top-level attribute calls made inside the body), its docstring,
    and its line number.
    """
    decorators = [ast.unparse(d) for d in node.decorator_list]

    # Task 1: Route extraction — parse @app.get("/path") style decorators.
    routes = []
    for raw in decorators:
        m = _ROUTE_RE.match(raw)
        if m:
            routes.append({
                "method": m.group(1).upper(),
                "path": m.group(2),
                "handler": node.name,
                "lineno": node.lineno,
            })

    # Task 3: Call graph — collect attribute-style calls inside the body
    # (e.g. self._agent.run, self.db.add). Cap at 10 to avoid noise.
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            calls.append(f"{ast.unparse(n.func.value)}.{n.func.attr}")
    calls = list(dict.fromkeys(calls))[:10]  # deduplicate, preserve order

    return {
        "name": node.name,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "signature": f"{node.name}({_format_arguments(node.args)})",
        # node.returns is the '-> str' part of the signature, if present.
        "returns": ast.unparse(node.returns) if node.returns else None,
        # decorator_list holds things like @app.get("/health").
        "decorators": decorators,
        "routes": routes,
        "calls": calls,
        "docstring": ast.get_docstring(node),
        "lineno": node.lineno,
    }


# ORM base class names used to detect database model classes.
_DB_BASES = {"Base", "Model", "Document", "db.Model", "DeclarativeBase", "SQLModel"}


def _describe_class(node: ast.ClassDef) -> dict:
    """
    Extract the documentable facts from a class node.

    Returns its name, base classes, whether it is a DB/ORM model, docstring,
    extracted fields (annotated and plain assignments), and the list of its
    methods (each described with the same shape as a standalone function).
    """
    # Walk only this class's own body so methods stay grouped under their class.
    methods = [
        _describe_function(item)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    # Task 2: Field extraction — class-body assignments and annotated assignments.
    fields = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            fields.append({
                "name": item.target.id,
                "type": ast.unparse(item.annotation),
                "kind": "annotated",
            })
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("__"):
                    fields.append({
                        "name": t.id,
                        "type": None,
                        "kind": "assign",
                    })

    # Task 4: DB model detection — check base class names and __tablename__.
    bases_list = [ast.unparse(b) for b in node.bases]
    is_db_model = any(b in _DB_BASES for b in bases_list)
    if not is_db_model:
        is_db_model = any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "__tablename__"
                for t in item.targets
            )
            for item in node.body
        )

    return {
        "name": node.name,
        "bases": bases_list,
        "is_db_model": is_db_model,
        "docstring": ast.get_docstring(node),
        "fields": fields,
        "methods": methods,
        "lineno": node.lineno,
    }


# Standard-library module names, used to filter import noise out of diagrams.
_STDLIB = set(sys.stdlib_module_names)


def _extract_imports(tree: ast.Module) -> list[str]:
    """
    Collect the NON-standard-library modules this file imports, as dotted names.

    Standard-library imports (json, os, sys, ast, pathlib, asyncio, ...) are
    filtered out -- they're pure noise in an architecture diagram. What remains
    is the project's own modules plus meaningful third-party libraries: the real
    dependencies worth showing.
    """
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level  # leading dots for relative imports
            imports.append(prefix + (node.module or ""))
    seen, unique = set(), []
    for imp in imports:
        top = imp.lstrip(".").split(".")[0]  # top-level package name
        if not imp or top in _STDLIB:
            continue  # drop standard-library noise
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)
    return unique


def extract_from_source(source: str, filename: str = "<unknown>") -> dict:
    """
    Parse a string of Python source and return its documented structure.

    Looks only at the TOP level of the module (we iterate tree.body rather
    than ast.walk) so top-level functions and classes stay cleanly separated,
    instead of flattening every nested helper into one big list.

    Routes are collected from top-level functions and returned as a flat
    file-level list so callers can see all HTTP endpoints in one place.
    """
    tree = ast.parse(source, filename=filename)
    functions, classes, routes = [], [], []
    for node in tree.body:  # top-level statements only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = _describe_function(node)
            functions.append(func)
            routes.extend(func.get("routes", []))
        elif isinstance(node, ast.ClassDef):
            classes.append(_describe_class(node))
    return {
        "file": filename,
        "module_docstring": ast.get_docstring(tree),
        "imports": _extract_imports(tree),
        "routes": routes,
        "functions": functions,
        "classes": classes,
    }


def extract_from_file(path) -> dict:
    """Read a single .py file from disk and extract its structure."""
    path = Path(path)
    return extract_from_source(path.read_text(encoding="utf-8"), filename=str(path))


def extract_from_directory(path) -> list[dict]:
    """
    Walk a directory tree and extract every .py file inside it.

    Skips junk folders (.venv, caches, .git, node_modules). If a file has a
    syntax error, it's recorded as an error entry instead of crashing the run.
    """
    path = Path(path)
    skip = {".venv", "__pycache__", ".git", "node_modules"}
    results = []
    for py_file in sorted(path.rglob("*.py")):  # rglob = recursive search
        if any(part in skip for part in py_file.parts):
            continue
        try:
            results.append(extract_from_file(py_file))
        except SyntaxError as e:
            results.append({"file": str(py_file), "error": f"SyntaxError: {e}"})
    return results


# ---------------------------------------------------------------------------
# Task 6: Language detection helpers
# ---------------------------------------------------------------------------

_EXT_LANGUAGE = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".cs": "csharp", ".go": "go",
}
_SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", "dist", "build"}


def _count_languages(path: Path) -> dict:
    """Count source files per language under path, skipping generated folders."""
    counts: dict[str, int] = {}
    for f in path.rglob("*"):
        if f.is_file() and not any(p in _SKIP_DIRS for p in f.parts):
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


# ---------------------------------------------------------------------------
# Task 5: Rich extraction — adds import graph, language info, framework
# ---------------------------------------------------------------------------

def extract_rich_from_directory(path) -> dict:
    """
    Enhanced extraction returning RichFacts: all per-file details plus an
    inter-module import graph, language counts, and detected framework.

    Used by the new HLD/LLD pipelines. extract_from_directory() is left
    unchanged so all existing pipeline/QA code keeps working.
    """
    path = Path(path)
    files = extract_from_directory(path)

    # Build inter-module import graph: {module_name: [internal_imports]}
    pkg_name = path.name
    import_graph: dict[str, list[str]] = {}
    for entry in files:
        if "error" in entry:
            continue
        try:
            rel = Path(entry["file"]).relative_to(path)
            # e.g. api/app.py -> api.app
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


# This block runs only when you execute the file directly, letting you test the
# extractor from the terminal: `python -m doc_agent.tools.extractor doc_agent`
if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    mode = sys.argv[2] if len(sys.argv) > 2 else "basic"
    if mode == "rich":
        print(json.dumps(extract_rich_from_directory(target), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(extract_from_directory(target), indent=2, ensure_ascii=False))
