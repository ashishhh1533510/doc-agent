"""
C# / .NET fact extractor (tree-sitter).

Returns the SAME FileFacts shape as python_extractor, so the facade and every
downstream pipeline treat C# exactly like Python. Covers:
  - namespace (block and file-scoped) + `using` directives
  - classes / records / structs / interfaces (base_list, properties + fields,
    attributes as decorators)
  - routes: ASP.NET attributes [HttpGet]/[HttpPost]/... joined to a class-level
    [Route(...)] template (with [controller]/[action] token substitution), plus
    minimal-API app.MapGet("/x", ...)
  - DB models: classes deriving DbContext, holding DbSet<T>, or marked
    [Table]/[Entity]/[Keyless] -> is_db_model
  - /// <summary> XML doc comments

Tree-sitter never raises on bad syntax, so we always emit best-effort facts.
"""

import re
from pathlib import Path

from doc_agent.tools.extractors import base


_HTTP_ATTRS = {
    "HttpGet": "GET", "HttpPost": "POST", "HttpPut": "PUT",
    "HttpDelete": "DELETE", "HttpPatch": "PATCH", "HttpHead": "HEAD",
    "HttpOptions": "OPTIONS",
}
_MAP_METHODS = {
    "MapGet": "GET", "MapPost": "POST", "MapPut": "PUT",
    "MapDelete": "DELETE", "MapPatch": "PATCH",
}
_DB_ATTRS = {"Table", "Keyless", "Entity"}

_CLASS_TYPES = {
    "class_declaration", "record_declaration", "struct_declaration",
    "interface_declaration", "record_struct_declaration",
}
_METHOD_TYPES = {"method_declaration", "constructor_declaration", "local_function_statement"}

_TAG_RE = re.compile(r"<[^>]+>")


def _lineno(node) -> int:
    """1-indexed start line (tree-sitter rows are 0-indexed; match Python)."""
    return node.start_point[0] + 1


def _child(node, type_name):
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _dedupe(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _strip_quotes(text: str) -> str:
    s = text.strip()
    if s.startswith("@"):
        s = s[1:]
    if s.startswith("$"):
        s = s[1:]
    return s.strip('"')


def _is_async(node) -> bool:
    return any(c.type == "modifier" and base.node_text(c) == "async" for c in node.children)


def _attributes(node):
    """All `attribute` nodes from the node's attribute_list children."""
    out = []
    for c in node.children:
        if c.type == "attribute_list":
            for a in c.children:
                if a.type == "attribute":
                    out.append(a)
    return out


def _attr_info(attr):
    """(name, first_string_arg) for an attribute node, e.g. ('HttpGet', '{id}')."""
    name_node = (attr.child_by_field_name("name")
                 or _child(attr, "identifier")
                 or _child(attr, "qualified_name")
                 or _child(attr, "generic_name"))
    name = base.node_text(name_node).split(".")[-1] if name_node else ""
    arg = ""
    args = _child(attr, "attribute_argument_list")
    if args is not None:
        for ar in args.named_children:
            sl = ar if ar.type.endswith("string_literal") else (
                _child(ar, "string_literal") or _child(ar, "verbatim_string_literal"))
            if sl is not None:
                arg = _strip_quotes(base.node_text(sl))
                break
    return name, arg


def _first_string_arg(args) -> str:
    if args is None:
        return ""
    for a in args.named_children:
        sl = a if a.type.endswith("string_literal") else _child(a, "string_literal")
        if sl is not None:
            return _strip_quotes(base.node_text(sl))
    return ""


def _controller_token(class_name: str) -> str:
    return class_name[:-10] if class_name.endswith("Controller") else class_name


def _join_route(prefix, path) -> str:
    a = (prefix or "").strip("/")
    b = (path or "").strip("/")
    return "/" + "/".join(p for p in (a, b) if p)


def _clean_xmldoc(parts) -> str | None:
    """Join consecutive /// lines, strip the markers and XML tags, collapse space."""
    if not parts:
        return None
    lines = []
    for block in parts:
        for line in block.splitlines():
            t = line.strip()
            if t.startswith("///"):
                t = t[3:].strip()
            elif t.startswith("//"):
                t = t[2:].strip()
            lines.append(t)
    joined = _TAG_RE.sub(" ", " ".join(lines))
    return " ".join(joined.split()) or None


def _signature(name, node) -> str:
    params = node.child_by_field_name("parameters")
    ptext = base.node_text(params) if params is not None else "()"
    return f"{name}{' '.join(ptext.split())}"


def _returns(node):
    r = node.child_by_field_name("returns")
    return base.node_text(r) if r is not None else None


def _calls(node):
    """Member-style invocations inside the body (e.g. _db.Users.FindAsync), max 10."""
    out = []
    body = node.child_by_field_name("body") or node
    for n in base.walk(body):
        if n.type == "invocation_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "member_access_expression":
                out.append(base.node_text(fn))
    return _dedupe(out)[:10]


def _describe_method(node, controller_prefix, doc) -> dict:
    is_ctor = node.type == "constructor_declaration"
    name = base.node_text(node.child_by_field_name("name") or _child(node, "identifier"))
    routes = []
    for attr in _attributes(node):
        aname, aarg = _attr_info(attr)
        if aname in _HTTP_ATTRS:
            # ASP.NET replaces the [action] token in a route template with the
            # method name (the [controller] token was already substituted above).
            path = _join_route(controller_prefix, aarg).replace("[action]", name)
            routes.append(base.make_route(_HTTP_ATTRS[aname], path, name, _lineno(node)))
    return {
        "name": name,
        "is_async": _is_async(node),
        "signature": _signature(name, node),
        "returns": None if is_ctor else _returns(node),
        "decorators": [base.node_text(a) for a in _attributes(node)],
        "routes": routes,
        "calls": _calls(node),
        "docstring": _clean_xmldoc(doc),
        "lineno": _lineno(node),
    }


def _describe_field(node):
    if node.type == "property_declaration":
        return {
            "name": base.node_text(node.child_by_field_name("name")),
            "type": base.node_text(node.child_by_field_name("type")),
            "kind": "property",
        }
    vd = _child(node, "variable_declaration")
    if vd is None:
        return None
    typ_node = vd.child_by_field_name("type")
    decl = _child(vd, "variable_declarator")
    name_node = _child(decl, "identifier") if decl is not None else None
    return {
        "name": base.node_text(name_node),
        "type": base.node_text(typ_node) if typ_node is not None else None,
        "kind": "field",
    }


def _describe_class(node, doc) -> dict:
    name = base.node_text(node.child_by_field_name("name") or _child(node, "identifier"))
    bases = []
    bl = _child(node, "base_list")
    if bl is not None:
        bases = [base.node_text(c) for c in bl.named_children]

    attrs = _attributes(node)
    attr_names = [_attr_info(a)[0] for a in attrs]
    controller_prefix = None
    for a in attrs:
        an, av = _attr_info(a)
        if an == "Route":
            controller_prefix = av.replace("[controller]", _controller_token(name))

    is_db_model = (
        any(b.split("<")[0].split(".")[-1].strip() == "DbContext" for b in bases)
        or any(an in _DB_ATTRS for an in attr_names)
    )

    methods, fields = [], []
    body = node.child_by_field_name("body")
    if body is not None:
        pending = []
        for child in body.named_children:
            if child.type == "comment":
                tx = base.node_text(child)
                if tx.lstrip().startswith("///"):
                    pending.append(tx)
                continue
            if child.type in _METHOD_TYPES:
                methods.append(_describe_method(child, controller_prefix, pending))
            elif child.type in ("field_declaration", "property_declaration"):
                fld = _describe_field(child)
                if fld is not None:
                    fields.append(fld)
                    if (fld.get("type") or "").startswith("DbSet"):
                        is_db_model = True
            pending = []

    return {
        "name": name,
        "bases": bases,
        "is_db_model": is_db_model,
        "docstring": _clean_xmldoc(doc),
        "fields": fields,
        "methods": methods,
        "lineno": _lineno(node),
    }


def _using_name(node):
    for c in node.children:
        if c.type in ("identifier", "qualified_name"):
            return base.node_text(c)
    return ""


def _minimal_api_routes(node):
    """app.MapGet('/x', ...) style routes from a top-level statement."""
    routes = []
    for n in base.walk(node):
        if n.type == "invocation_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "member_access_expression":
                meth = base.node_text(fn.child_by_field_name("name") or fn).split(".")[-1]
                if meth in _MAP_METHODS:
                    path = _first_string_arg(n.child_by_field_name("arguments"))
                    if path:
                        routes.append(base.make_route(
                            _MAP_METHODS[meth], path, "minimal_api", _lineno(n)))
    return routes


def extract_from_csharp_source(source: str, filename: str) -> dict:
    """Parse C# source text and return its FileFacts."""
    tree = base.parse("csharp", source)
    state = {"namespace": None}
    imports, functions, classes, routes = [], [], [], []

    def scan(container):
        pending = []
        for node in container.named_children:
            t = node.type
            if t == "comment":
                tx = base.node_text(node)
                if tx.lstrip().startswith("///"):
                    pending.append(tx)
                continue
            if t == "using_directive":
                nm = _using_name(node)
                if nm:
                    imports.append(nm)
            elif t in ("namespace_declaration", "file_scoped_namespace_declaration"):
                ns = base.node_text(node.child_by_field_name("name"))
                if state["namespace"] is None:
                    state["namespace"] = ns
                body = node.child_by_field_name("body")
                if body is not None:
                    scan(body)
            elif t in _CLASS_TYPES:
                classes.append(_describe_class(node, pending))
            elif t == "global_statement":
                routes.extend(_minimal_api_routes(node))
            pending = []

    scan(tree.root_node)
    for c in classes:
        for m in c["methods"]:
            routes.extend(m.get("routes", []))

    return {
        "file": filename,
        "language": "csharp",
        "namespace": state["namespace"],
        "module_docstring": None,
        "imports": _dedupe(imports),
        "routes": routes,
        "functions": functions,
        "classes": classes,
    }


def extract_from_csharp_file(file_path) -> dict:
    """Read a single .cs file and extract its FileFacts."""
    path = Path(file_path)
    source = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_csharp_source(source, str(path))
