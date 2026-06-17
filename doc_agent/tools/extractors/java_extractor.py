"""
Java fact extractor (tree-sitter).

Returns the SAME FileFacts shape as python_extractor, so the facade and every
downstream pipeline treat Java exactly like Python. Covers:
  - package (-> namespace) + import declarations
  - classes / interfaces / enums / records (extends/implements bases, fields,
    annotations as decorators)
  - routes: Spring @GetMapping/@PostMapping/... joined to a class-level
    @RequestMapping prefix
  - DB models: JPA @Entity/@Table/@MappedSuperclass/@Embeddable -> is_db_model
  - /** ... */ Javadoc comments

Tree-sitter never raises on bad syntax, so we always emit best-effort facts.
"""

from pathlib import Path

from doc_agent.tools.extractors import base


_HTTP_MAPPINGS = {
    "GetMapping": "GET", "PostMapping": "POST", "PutMapping": "PUT",
    "DeleteMapping": "DELETE", "PatchMapping": "PATCH",
}
_DB_ANNOS = {
    "Entity", "Table", "Document", "Embeddable", "MappedSuperclass",
    # MyBatis: @Mapper marks a persistence interface
    "Mapper",
    # Spring Data: @Repository marks a data-access bean
    "Repository",
}
# JAX-RS (javax.ws.rs / jakarta.ws.rs) HTTP verb annotations.
_JAXRS_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}

_CLASS_TYPES = {
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
}
_METHOD_TYPES = {
    "method_declaration", "constructor_declaration", "compact_constructor_declaration",
}


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


def _annotations(node):
    """Annotation nodes (@Foo / @Foo(...)) from a declaration's modifiers."""
    mods = _child(node, "modifiers")
    if mods is None:
        return []
    return [c for c in mods.children if c.type in ("annotation", "marker_annotation")]


def _anno_name(a) -> str:
    return base.node_text(_child(a, "identifier"))


def _anno_string(a) -> str:
    """First string-literal value inside an annotation (handles value=... pairs)."""
    al = _child(a, "annotation_argument_list")
    if al is None:
        return ""
    for n in base.walk(al):
        if n.type == "string_literal":
            frag = _child(n, "string_fragment")
            return base.node_text(frag) if frag is not None else base.node_text(n).strip('"')
    return ""


def _join_route(prefix, path) -> str:
    a = (prefix or "").strip("/")
    b = (path or "").strip("/")
    return "/" + "/".join(p for p in (a, b) if p)


def _clean_javadoc(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("/**"):
        s = s[3:]
    elif s.startswith("/*"):
        s = s[2:]
    if s.endswith("*/"):
        s = s[:-2]
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if t.startswith("*"):
            t = t[1:].strip()
        lines.append(t)
    return "\n".join(lines).strip() or None


def _signature(name, node) -> str:
    params = node.child_by_field_name("parameters")
    ptext = base.node_text(params) if params is not None else "()"
    return f"{name}{' '.join(ptext.split())}"


def _calls(node):
    """Member-style method invocations inside the body (e.g. repo.save), max 10."""
    out = []
    body = node.child_by_field_name("body") or node
    for n in base.walk(body):
        if n.type == "method_invocation":
            obj = n.child_by_field_name("object")
            nm = n.child_by_field_name("name")
            if obj is not None and nm is not None:
                out.append(f"{' '.join(base.node_text(obj).split())}.{base.node_text(nm)}")
    return _dedupe(out)[:10]


def _bases(node):
    """extends (superclass) + implements (super_interfaces) + interface extends."""
    out = []
    for c in node.children:
        if c.type == "superclass":
            out.extend(base.node_text(t) for t in c.named_children)
        elif c.type in ("super_interfaces", "extends_interfaces"):
            for t in c.named_children:
                if t.type == "type_list":
                    out.extend(base.node_text(x) for x in t.named_children)
                else:
                    out.append(base.node_text(t))
    return out


def _describe_method(node, controller_prefix, doc) -> dict:
    is_ctor = node.type in ("constructor_declaration", "compact_constructor_declaration")
    name = base.node_text(node.child_by_field_name("name") or _child(node, "identifier"))
    routes = []
    for a in _annotations(node):
        an = _anno_name(a)
        if an in _HTTP_MAPPINGS:
            routes.append(base.make_route(
                _HTTP_MAPPINGS[an], _join_route(controller_prefix, _anno_string(a)),
                name, _lineno(node)))
    # JAX-RS: collect @GET/@POST/... and optional @Path from this method's annotations.
    jaxrs_verb, method_path = None, None
    for a in _annotations(node):
        an = _anno_name(a)
        if an in _JAXRS_METHODS:
            jaxrs_verb = an
        elif an == "Path":
            method_path = _anno_string(a)
    if jaxrs_verb:
        routes.append(base.make_route(
            jaxrs_verb, _join_route(controller_prefix, method_path), name, _lineno(node)))
    rtype = node.child_by_field_name("type")
    return {
        "name": name,
        "is_async": False,  # Java has no async/await keyword
        "signature": _signature(name, node),
        "returns": None if is_ctor or rtype is None else base.node_text(rtype),
        "decorators": [base.node_text(a) for a in _annotations(node)],
        "routes": routes,
        "calls": _calls(node),
        "docstring": _clean_javadoc(doc),
        "lineno": _lineno(node),
    }


def _describe_fields(node):
    """A field_declaration can declare several names; return one dict each."""
    typ_node = node.child_by_field_name("type")
    typ = base.node_text(typ_node) if typ_node is not None else None
    out = []
    for c in node.children:
        if c.type == "variable_declarator":
            out.append({
                "name": base.node_text(_child(c, "identifier")),
                "type": typ,
                "kind": "field",
            })
    return out


def _describe_class(node, doc) -> dict:
    name = base.node_text(node.child_by_field_name("name") or _child(node, "identifier"))
    bases = _bases(node)
    annos = _annotations(node)
    anno_names = [_anno_name(a) for a in annos]
    is_db_model = any(an in _DB_ANNOS for an in anno_names)

    controller_prefix = None
    for a in annos:
        if _anno_name(a) in ("RequestMapping", "Path"):
            controller_prefix = _anno_string(a)

    body = (node.child_by_field_name("body")
            or _child(node, "class_body")
            or _child(node, "interface_body")
            or _child(node, "enum_body")
            or _child(node, "annotation_type_body"))

    methods, fields = [], []
    if body is not None:
        pending = None
        for child in body.named_children:
            if child.type == "block_comment" and base.node_text(child).lstrip().startswith("/**"):
                pending = base.node_text(child)
                continue
            if child.type in ("line_comment", "block_comment"):
                continue
            if child.type in _METHOD_TYPES:
                methods.append(_describe_method(child, controller_prefix, pending))
            elif child.type == "field_declaration":
                fields.extend(_describe_fields(child))
            pending = None

    return {
        "name": name,
        "bases": bases,
        "is_db_model": is_db_model,
        "docstring": _clean_javadoc(doc),
        "fields": fields,
        "methods": methods,
        "lineno": _lineno(node),
    }


def _import_name(node) -> str:
    txt = base.node_text(node).strip()
    if txt.startswith("import"):
        txt = txt[len("import"):].strip()
    if txt.startswith("static "):
        txt = txt[len("static "):].strip()
    return txt.rstrip(";").strip()


def extract_from_java_source(source: str, filename: str) -> dict:
    """Parse Java source text and return its FileFacts."""
    tree = base.parse("java", source)
    namespace = None
    imports, functions, classes, routes = [], [], [], []

    pending = None
    for node in tree.root_node.named_children:
        t = node.type
        if t == "block_comment" and base.node_text(node).lstrip().startswith("/**"):
            pending = base.node_text(node)
            continue
        if t in ("line_comment", "block_comment"):
            continue
        if t == "package_declaration":
            si = _child(node, "scoped_identifier") or _child(node, "identifier")
            namespace = base.node_text(si) if si is not None else None
        elif t == "import_declaration":
            imports.append(_import_name(node))
        elif t in _CLASS_TYPES:
            classes.append(_describe_class(node, pending))
        pending = None

    for c in classes:
        for m in c["methods"]:
            routes.extend(m.get("routes", []))

    return {
        "file": filename,
        "language": "java",
        "namespace": namespace,
        "module_docstring": None,
        "imports": _dedupe(imports),
        "routes": routes,
        "functions": functions,
        "classes": classes,
    }


def extract_from_java_file(file_path) -> dict:
    """Read a single .java file and extract its FileFacts."""
    path = Path(file_path)
    source = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_java_source(source, str(path))
