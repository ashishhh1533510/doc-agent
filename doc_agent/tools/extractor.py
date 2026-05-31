"""
Scan-and-extract tool for the documentation agent.

Reads Python source code and pulls out its structure (functions, classes,
signatures, docstrings) using the built-in `ast` module. This produces the
"ground truth" facts that later agents use to generate and verify docs.
No LLM is involved here -- this is pure, deterministic parsing.
"""

import ast
from pathlib import Path


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


def _describe_function(node) -> dict:
    """
    Extract the documentable facts from a single function node.

    Returns its name, whether it's async, a full readable signature, its
    return type annotation, any decorators, its docstring, and its line number.
    """
    return {
        "name": node.name,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "signature": f"{node.name}({_format_arguments(node.args)})",
        # node.returns is the '-> str' part of the signature, if present.
        "returns": ast.unparse(node.returns) if node.returns else None,
        # decorator_list holds things like @app.get("/health").
        "decorators": [ast.unparse(d) for d in node.decorator_list],
        "docstring": ast.get_docstring(node),
        "lineno": node.lineno,
    }


def _describe_class(node: ast.ClassDef) -> dict:
    """
    Extract the documentable facts from a class node.

    Returns its name, base classes, docstring, and the list of its methods
    (each method described with the same shape as a standalone function).
    """
    # Walk only this class's own body so methods stay grouped under their class.
    methods = [
        _describe_function(item)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        "name": node.name,
        # bases are the parent classes, e.g. the 'BaseModel' in 'class X(BaseModel)'.
        "bases": [ast.unparse(b) for b in node.bases],
        "docstring": ast.get_docstring(node),
        "methods": methods,
        "lineno": node.lineno,
    }


def extract_from_source(source: str, filename: str = "<unknown>") -> dict:
    """
    Parse a string of Python source and return its documented structure.

    Looks only at the TOP level of the module (we iterate tree.body rather
    than ast.walk) so top-level functions and classes stay cleanly separated,
    instead of flattening every nested helper into one big list.
    """
    tree = ast.parse(source, filename=filename)
    functions, classes = [], []
    for node in tree.body:  # top-level statements only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_describe_function(node))
        elif isinstance(node, ast.ClassDef):
            classes.append(_describe_class(node))
    return {
        "file": filename,
        "module_docstring": ast.get_docstring(tree),
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


# This block runs only when you execute the file directly, letting you test the
# extractor from the terminal: `python -m doc_agent.extractor doc_agent`
if __name__ == "__main__":
    import json
    import sys

    # Use the path given on the command line, or default to the doc_agent package.
    target = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    print(json.dumps(extract_from_directory(target), indent=2, ensure_ascii=False))