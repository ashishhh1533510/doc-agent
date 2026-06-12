"""
TypeScript / JavaScript fact extractor (tree-sitter).

Returns the SAME FileFacts shape as python_extractor, so the facade and every
downstream pipeline treat TS/JS exactly like Python. Covers:
  - ES-module imports (+ `require(...)`)
  - function declarations and arrow functions (with typed signatures)
  - classes / interfaces (extends / implements, fields, decorators)
  - routes: NestJS (@Controller prefix + @Get/@Post/...) and Express/Fastify
    (app.get('/x', ...))
  - DB models: TypeORM @Entity (and @Schema/@Table) -> is_db_model
  - JSDoc /** ... */ doc comments

Grammar by extension: .ts -> typescript, .tsx -> tsx, .js/.jsx -> javascript.
Tree-sitter never raises on bad syntax, so we always emit best-effort facts.
"""

from pathlib import Path

from doc_agent.tools.extractors import base


# Decorator names that mark an HTTP route handler (NestJS).
_HTTP_DECORATORS = {"Get", "Post", "Put", "Patch", "Delete", "Head", "Options", "All"}
# Express/Fastify router methods: app.get('/x', ...), router.post(...), etc.
_EXPRESS_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "all"}
# Class decorators that mark an ORM/DB model.
_DB_DECORATORS = {"Entity", "Schema", "Table", "ViewEntity"}

_CLASS_TYPES = {"class_declaration", "abstract_class_declaration"}
_METHOD_TYPES = {"method_definition", "method_signature"}
_FIELD_TYPES = {"public_field_definition", "field_definition", "property_signature"}
_FUNC_DECL_TYPES = {"function_declaration", "generator_function_declaration"}
_ARROW_TYPES = {"arrow_function", "function_expression", "function"}

_GRAMMAR_BY_SUFFIX = {
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
}


# ---------------------------------------------------------------------------
# Small node helpers
# ---------------------------------------------------------------------------

def _lineno(node) -> int:
    """1-indexed start line (tree-sitter rows are 0-indexed; match Python)."""
    return node.start_point[0] + 1


def _child(node, type_name):
    """First direct child of a given type, or None."""
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _name_of(node) -> str:
    """Best-effort declared name (identifier / type_identifier / property_identifier)."""
    for t in ("name", ):
        n = node.child_by_field_name(t)
        if n is not None:
            return base.node_text(n)
    for t in ("type_identifier", "identifier", "property_identifier"):
        n = _child(node, t)
        if n is not None:
            return base.node_text(n)
    return ""


def _is_async(node) -> bool:
    return any(c.type == "async" for c in node.children)


def _is_jsdoc(node) -> bool:
    return node.type == "comment" and base.node_text(node).lstrip().startswith("/**")


def _clean_jsdoc(raw: str | None) -> str | None:
    """Strip /** */ and leading-* from a JSDoc block; None if empty."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("/**"):
        s = s[3:]
    elif s.startswith("/*"):
        s = s[2:]
    elif s.startswith("//"):
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


def _dedupe(items: list) -> list:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _first_string_arg(args_node) -> str:
    """Text of the first string-literal argument (quotes stripped), or ''."""
    if args_node is None:
        return ""
    for c in args_node.named_children:
        if c.type in ("string", "template_string"):
            frag = _child(c, "string_fragment")
            return base.node_text(frag) if frag else base.node_text(c).strip("'\"`")
    return ""


def _decorator_info(dec_node):
    """(name, first_string_arg) for a decorator node (@Name or @Name(args))."""
    inner = None
    for c in dec_node.children:
        if c.type in ("call_expression", "identifier", "member_expression"):
            inner = c
            break
    if inner is None:
        return "", ""
    if inner.type == "call_expression":
        fn = inner.child_by_field_name("function")
        name = base.node_text(fn).split(".")[-1] if fn else ""
        return name, _first_string_arg(inner.child_by_field_name("arguments"))
    return base.node_text(inner).split(".")[-1], ""


def _join_route(prefix: str | None, path: str | None) -> str:
    """Join a controller prefix and a method path into a single leading-slash route."""
    a = (prefix or "").strip("/")
    b = (path or "").strip("/")
    return "/" + "/".join(p for p in (a, b) if p)


# ---------------------------------------------------------------------------
# Function / class description
# ---------------------------------------------------------------------------

def _signature(name: str, fn_node) -> str:
    """`name(params)` using the verbatim formal-parameters text (keeps types)."""
    params = _child(fn_node, "formal_parameters")
    ptext = base.node_text(params) if params is not None else "()"
    return f"{name}{' '.join(ptext.split())}"


def _return_type(fn_node) -> str | None:
    """Return-type annotation text (the type_annotation after the parameters)."""
    seen_params = False
    for c in fn_node.children:
        if c.type == "formal_parameters":
            seen_params = True
            continue
        if seen_params and c.type == "type_annotation":
            return base.node_text(c).lstrip(":").strip() or None
    return None


def _calls(fn_node) -> list:
    """Member-style calls inside a function body (e.g. this.db.fetch), deduped, max 10."""
    out = []
    for n in base.walk(fn_node):
        if n.type == "call_expression":
            callee = n.child_by_field_name("function")
            if callee is not None and callee.type == "member_expression":
                out.append(base.node_text(callee))
    return _dedupe(out)[:10]


def _describe_function(name, fn_node, decorators, doc, controller_prefix=None) -> dict:
    """FileFacts function dict (same shape as python_extractor)."""
    routes = []
    for dec in decorators:
        dname, dpath = _decorator_info(dec)
        if dname in _HTTP_DECORATORS:
            routes.append(base.make_route(
                dname.upper(),
                _join_route(controller_prefix, dpath),
                name,
                _lineno(fn_node),
            ))
    return {
        "name": name,
        "is_async": _is_async(fn_node),
        "signature": _signature(name, fn_node),
        "returns": _return_type(fn_node),
        "decorators": [base.node_text(d) for d in decorators],
        "routes": routes,
        "calls": _calls(fn_node),
        "docstring": _clean_jsdoc(doc),
        "lineno": _lineno(fn_node),
    }


def _describe_field(node) -> dict:
    name_node = _child(node, "property_identifier") or _child(node, "identifier")
    ta = _child(node, "type_annotation")
    typ = base.node_text(ta).lstrip(":").strip() if ta is not None else None
    return {"name": base.node_text(name_node), "type": typ, "kind": "field"}


def _class_bases(class_node) -> list:
    """extends + implements (and interface `extends`) base names."""
    bases = []
    for c in class_node.children:
        if c.type == "class_heritage":
            for clause in c.children:
                if clause.type in ("extends_clause", "implements_clause"):
                    bases.extend(base.node_text(t) for t in clause.named_children)
        elif c.type in ("extends_clause", "implements_clause", "extends_type_clause"):
            bases.extend(base.node_text(t) for t in c.named_children)
    return bases


def _describe_class(name, class_node, decorators, doc) -> dict:
    """FileFacts class dict (same shape as python_extractor)."""
    dec_names = [_decorator_info(d)[0] for d in decorators]
    is_db_model = any(dn in _DB_DECORATORS for dn in dec_names)

    controller_prefix = None
    for d in decorators:
        dn, dpath = _decorator_info(d)
        if dn == "Controller":
            controller_prefix = dpath

    body = (_child(class_node, "class_body")
            or _child(class_node, "interface_body")
            or _child(class_node, "object_type"))

    methods, fields = [], []
    if body is not None:
        pending_decorators, pending_doc = [], None
        for child in body.named_children:
            if child.type == "decorator":
                pending_decorators.append(child)
                continue
            if _is_jsdoc(child):
                pending_doc = base.node_text(child)
                continue
            if child.type == "comment":
                continue
            if child.type in _METHOD_TYPES:
                own = [c for c in child.children if c.type == "decorator"]
                mname = base.node_text(
                    _child(child, "property_identifier") or _child(child, "identifier")
                )
                methods.append(_describe_function(
                    mname, child, pending_decorators + own, pending_doc, controller_prefix
                ))
            elif child.type in _FIELD_TYPES:
                fields.append(_describe_field(child))
            pending_decorators, pending_doc = [], None

    return {
        "name": name,
        "bases": _class_bases(class_node),
        "is_db_model": is_db_model,
        "docstring": _clean_jsdoc(doc),
        "fields": fields,
        "methods": methods,
        "lineno": _lineno(class_node),
    }


def _import_sources(node) -> list:
    """Module specifier(s) of an import_statement (e.g. '@nestjs/common', './x')."""
    src = _child(node, "string")
    if src is None:
        return []
    frag = _child(src, "string_fragment")
    return [base.node_text(frag) if frag else base.node_text(src).strip("'\"`")]


def _express_routes(node) -> list:
    """app.get('/x', ...) / router.post(...) style routes from an expression_statement."""
    expr = node.named_children[0] if node.named_children else None
    if expr is None or expr.type != "call_expression":
        return []
    callee = expr.child_by_field_name("function")
    if callee is None or callee.type != "member_expression":
        return []
    prop = callee.child_by_field_name("property")
    method = base.node_text(prop).lower()
    if method not in _EXPRESS_METHODS:
        return []
    path = _first_string_arg(expr.child_by_field_name("arguments"))
    if not path:
        return []
    obj = callee.child_by_field_name("object")
    return [base.make_route(method.upper(), path, base.node_text(obj), _lineno(node))]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def extract_from_typescript_source(source: str, filename: str, grammar: str = "typescript") -> dict:
    """Parse TS/JS source text and return its FileFacts."""
    tree = base.parse(grammar, source)
    root = tree.root_node

    imports, functions, classes, routes = [], [], [], []

    def handle_declaration(decl, decorators, doc):
        if decl.type in _CLASS_TYPES or decl.type == "interface_declaration":
            cls = _describe_class(_name_of(decl), decl, decorators, doc)
            classes.append(cls)
            for m in cls["methods"]:
                routes.extend(m.get("routes", []))
        elif decl.type in _FUNC_DECL_TYPES:
            fn = _describe_function(_name_of(decl), decl, decorators, doc)
            functions.append(fn)
            routes.extend(fn.get("routes", []))
        elif decl.type in ("lexical_declaration", "variable_declaration"):
            for vd in decl.named_children:
                if vd.type != "variable_declarator":
                    continue
                nm = base.node_text(vd.child_by_field_name("name") or _child(vd, "identifier"))
                val = vd.child_by_field_name("value")
                if val is None:
                    continue
                if val.type in _ARROW_TYPES:
                    fn = _describe_function(nm, val, decorators, doc)
                    functions.append(fn)
                    routes.extend(fn.get("routes", []))
                elif val.type == "call_expression":
                    callee = val.child_by_field_name("function")
                    if callee is not None and base.node_text(callee) == "require":
                        arg = _first_string_arg(val.child_by_field_name("arguments"))
                        if arg:
                            imports.append(arg)

    pending_doc, pending_decorators = None, []
    for node in root.named_children:
        t = node.type
        if t == "comment":
            if _is_jsdoc(node):
                pending_doc = base.node_text(node)
            continue
        if t == "decorator":
            pending_decorators.append(node)
            continue

        if t == "import_statement":
            imports.extend(_import_sources(node))
        elif t == "export_statement":
            exp_decs = [c for c in node.children if c.type == "decorator"]
            decl = next(
                (c for c in node.children
                 if c.type in _CLASS_TYPES
                 or c.type in _FUNC_DECL_TYPES
                 or c.type in ("lexical_declaration", "variable_declaration", "interface_declaration")),
                None,
            )
            if decl is not None:
                handle_declaration(decl, pending_decorators + exp_decs, pending_doc)
        elif t in _CLASS_TYPES or t == "interface_declaration":
            handle_declaration(node, list(pending_decorators), pending_doc)
        elif t in _FUNC_DECL_TYPES:
            handle_declaration(node, list(pending_decorators), pending_doc)
        elif t in ("lexical_declaration", "variable_declaration"):
            handle_declaration(node, list(pending_decorators), pending_doc)
        elif t == "expression_statement":
            routes.extend(_express_routes(node))

        pending_doc, pending_decorators = None, []

    return {
        "file": filename,
        "language": "javascript" if grammar == "javascript" else "typescript",
        "namespace": None,
        "module_docstring": None,
        "imports": _dedupe(imports),
        "routes": routes,
        "functions": functions,
        "classes": classes,
    }


def extract_from_typescript_file(file_path) -> dict:
    """Read a single TS/JS file and extract its FileFacts (grammar chosen by extension)."""
    path = Path(file_path)
    grammar = _GRAMMAR_BY_SUFFIX.get(path.suffix.lower(), "typescript")
    source = path.read_text(encoding="utf-8", errors="replace")
    return extract_from_typescript_source(source, str(path), grammar)
