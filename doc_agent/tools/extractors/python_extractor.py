"""
Python fact extractor.

Parses Python source with the built-in `ast` module and returns FileFacts —
the deterministic "ground truth" (functions, classes, signatures, routes,
imports, docstrings) that downstream agents document and verify. No LLM here.

This module holds the Python-specific logic. tools/extractor.py is the
language-agnostic facade that dispatches to it (and the other languages) by
file extension. Logic here is unchanged from the original extractor.py; only
the entry-point names and the new language/namespace keys differ.
"""

import ast
import re
import sys
from pathlib import Path

from doc_agent.tools.language_detector import SKIP_DIRS


def _format_arguments(args: ast.arguments) -> str:
    """Turn a function's argument node into a readable parameter string."""
    parts = []
    posonly = getattr(args, "posonlyargs", [])
    positional = posonly + args.args
    default_offset = len(positional) - len(args.defaults)

    for i, arg in enumerate(positional):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if i >= default_offset:
            piece += f" = {ast.unparse(args.defaults[i - default_offset])}"
        parts.append(piece)
        if posonly and arg is posonly[-1]:
            parts.append("/")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        parts.append(piece)

    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


_ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
_ROUTE_RE = re.compile(
    r'[\w.]+\.(' + '|'.join(_ROUTE_METHODS) + r')\(["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def _describe_function(node) -> dict:
    """Extract the documentable facts from a single function node."""
    decorators = [ast.unparse(d) for d in node.decorator_list]

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

    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            calls.append(f"{ast.unparse(n.func.value)}.{n.func.attr}")
    calls = list(dict.fromkeys(calls))[:10]

    return {
        "name": node.name,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "signature": f"{node.name}({_format_arguments(node.args)})",
        "returns": ast.unparse(node.returns) if node.returns else None,
        "decorators": decorators,
        "routes": routes,
        "calls": calls,
        "docstring": ast.get_docstring(node),
        "lineno": node.lineno,
    }


_DB_BASES = {"Base", "Model", "Document", "db.Model", "DeclarativeBase", "SQLModel"}


def _describe_class(node: ast.ClassDef) -> dict:
    """Extract the documentable facts from a class node."""
    methods = [
        _describe_function(item)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

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
                    fields.append({"name": t.id, "type": None, "kind": "assign"})

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


_STDLIB = set(sys.stdlib_module_names)


def _extract_imports(tree: ast.Module) -> list[str]:
    """Collect the NON-standard-library modules this file imports."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            imports.append(prefix + (node.module or ""))
    seen, unique = set(), []
    for imp in imports:
        top = imp.lstrip(".").split(".")[0]
        if not imp or top in _STDLIB:
            continue
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)
    return unique


def extract_from_python_source(source: str, filename: str = "<unknown>") -> dict:
    """Parse Python source text and return its FileFacts."""
    tree = ast.parse(source, filename=filename)
    functions, classes, routes = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = _describe_function(node)
            functions.append(func)
            routes.extend(func.get("routes", []))
        elif isinstance(node, ast.ClassDef):
            classes.append(_describe_class(node))
    return {
        "file": filename,
        "language": "python",      # NEW: FileFacts now records its language
        "namespace": None,         # NEW: Python has no namespace (module path serves)
        "module_docstring": ast.get_docstring(tree),
        "imports": _extract_imports(tree),
        "routes": routes,
        "functions": functions,
        "classes": classes,
    }


def extract_from_python_file(file_path) -> dict:
    """Read a single .py file from disk and extract its FileFacts."""
    path = Path(file_path)
    return extract_from_python_source(
        path.read_text(encoding="utf-8"), filename=str(path)
    )


def extract_from_python_directory(directory) -> list[dict]:
    """Walk a directory and extract every .py file (skips junk folders)."""
    path = Path(directory)
    results = []
    for py_file in sorted(path.rglob("*.py")):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        try:
            results.append(extract_from_python_file(py_file))
        except SyntaxError as e:
            results.append({"file": str(py_file), "error": f"SyntaxError: {e}"})
    return results
