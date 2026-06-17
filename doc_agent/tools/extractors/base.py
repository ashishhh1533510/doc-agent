"""
Shared tree-sitter helpers — the ONLY module that imports tree_sitter directly.

Isolates parser/query access so version churn (the parse + query API changed
across tree-sitter 0.22-0.25) stays contained here. Verified against
tree-sitter 0.25.x with the official per-grammar packages
(tree-sitter-python / -javascript / -typescript / -c-sharp / -java).

Every language extractor builds on these helpers and never touches tree_sitter.
"""

from functools import lru_cache
import tree_sitter as ts


def _grammar_loaders() -> dict:
    """Map grammar name -> the per-package C-language loader function.

    Imported lazily so a missing optional grammar can't break import of the
    whole extractors package. typescript ships two grammars (.ts vs .tsx);
    the javascript grammar covers .js/.jsx.
    """
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_c_sharp
    import tree_sitter_java
    return {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
        "tsx": tree_sitter_typescript.language_tsx,
        "csharp": tree_sitter_c_sharp.language,
        "java": tree_sitter_java.language,
    }


@lru_cache(maxsize=None)
def get_language(grammar: str) -> "ts.Language":
    """Return a cached tree-sitter Language for a grammar name."""
    loaders = _grammar_loaders()
    if grammar not in loaders:
        raise ValueError(
            f"No tree-sitter grammar registered for {grammar!r}; "
            f"known: {sorted(loaders)}"
        )
    return ts.Language(loaders[grammar]())


@lru_cache(maxsize=None)
def get_parser(grammar: str) -> "ts.Parser":
    """Return a cached tree-sitter Parser configured for a grammar."""
    return ts.Parser(get_language(grammar))


def parse(grammar: str, source: str) -> "ts.Tree":
    """Parse source text into a tree-sitter Tree (never raises on bad syntax)."""
    return get_parser(grammar).parse(source.encode("utf-8"))


@lru_cache(maxsize=None)
def _compiled_query(grammar: str, query_source: str) -> "ts.Query":
    """Compile and cache a query per (grammar, query text)."""
    return ts.Query(get_language(grammar), query_source)


def query_captures(grammar: str, query_source: str, node) -> dict:
    """
    Run a query rooted at `node`, returning {capture_name: [nodes]}.

    Wraps the 0.25 QueryCursor API so extractors never import tree_sitter.
    Returns an empty dict when nothing matches.
    """
    cursor = ts.QueryCursor(_compiled_query(grammar, query_source))
    return cursor.captures(node)


def node_text(node) -> str:
    """Decode a node's source span to a string ('' for a missing node)."""
    if node is None:
        return ""
    return node.text.decode("utf-8", "replace")


def child_text(node, field_name: str) -> str:
    """Text of a named child field (e.g. 'name', 'return_type'), or '' if absent."""
    return node_text(node.child_by_field_name(field_name)) if node else ""


def walk(node):
    """Yield every node in the subtree, pre-order (cheap manual stack)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def has_errors(tree) -> bool:
    """True if tree-sitter flagged any syntax error (best-effort facts still emit)."""
    return tree.root_node.has_error


def make_route(method: str, path: str, handler: str, lineno: int) -> dict:
    """Build the canonical route dict used across all languages and pipelines."""
    return {
        "method": method.upper(),
        "path": path,
        "handler": handler,
        "lineno": lineno,
    }


def clean_doc(raw: str, markers: tuple = ("///", "/**", "*/", "*", "//", "#")) -> str | None:
    """
    Normalize a raw doc/comment span into clean text.

    Strips common comment markers line-by-line (C#'s ///, JSDoc/Javadoc /** */,
    Python #), drops blank edges, and rejoins. Returns None if nothing remains.
    """
    if not raw:
        return None
    out = []
    for line in raw.splitlines():
        s = line.strip()
        for m in markers:
            if s.startswith(m):
                s = s[len(m):].strip()
        out.append(s)
    text = "\n".join(out).strip()
    return text or None
