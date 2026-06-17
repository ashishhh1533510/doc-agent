"""
Cross-file import-graph resolution + framework detection (all languages).

extract_rich_from_directory() calls these to turn the per-file `imports`
(raw module specifiers) into a real dependency graph:
    {module_id: [internal module_ids it depends on]}
where module_id = repo-relative POSIX path with the source extension stripped.

External imports stay out of the graph (they're kept per-file for framework
detection). Resolution is language-specific:
  - Python : dotted modules (with/without the root package prefix) + relative
  - TS/JS  : ./x and @/x path resolution against the repo's file set
  - C#     : using -> namespace -> every file declaring that namespace (coarse;
             clustering tolerates the over-linking)
  - Java   : import -> fully-qualified class -> file; wildcard -> whole package
"""

from pathlib import Path

_EXTS = (".py", ".tsx", ".ts", ".jsx", ".js", ".cs", ".java")


def module_id(file_path, repo_root) -> str:
    """Canonical id = repo-relative POSIX path with the source extension stripped."""
    rel = Path(file_path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    for e in _EXTS:
        if rel.endswith(e):
            return rel[:-len(e)]
    return rel


# ---------------- Python ----------------

def _resolve_python(imp, current, index, module_ids):
    if imp.startswith("."):
        level = len(imp) - len(imp.lstrip("."))
        rest = imp[level:]
        parts = current.split("/")
        pkg = parts[:-1]                      # the file's package (its directory)
        base = pkg[: len(pkg) - (level - 1)] if level - 1 <= len(pkg) else []
        target = base + (rest.split(".") if rest else [])
        cand = "/".join(target)
        for c in (cand, cand + "/__init__", "/".join(target[:-1])):
            if c in module_ids:
                return c
        return None
    segs = imp.split(".")
    for k in range(len(segs), 0, -1):         # longest dotted prefix wins
        c = ".".join(segs[:k])
        if c in index:
            return index[c]
    return None


# ---------------- TypeScript / JavaScript ----------------

def _resolve_ts(imp, current, ts_ids):
    if not (imp.startswith(".") or imp.startswith("@/")):
        return None                            # bare import = node_modules (external)
    cur_dir = current.rsplit("/", 1)[0] if "/" in current else ""
    if imp.startswith("@/"):
        target = "src/" + imp[2:]
    else:
        parts = [p for p in cur_dir.split("/") if p]
        for seg in imp.split("/"):
            if seg in (".", ""):
                continue
            if seg == "..":
                parts = parts[:-1]
            else:
                parts.append(seg)
        target = "/".join(parts)
    for c in (target, target + "/index"):
        if c in ts_ids:
            return c
    return None


# ---------------- C# ----------------

def _resolve_csharp(imp, ns_index):
    return list(ns_index.get(imp, []))         # using -> all files in that namespace


# ---------------- Java ----------------

def _resolve_java(imp, pkg_index, fqcn_index):
    if imp.endswith(".*"):
        return list(pkg_index.get(imp[:-2], []))
    if imp in fqcn_index:
        return [fqcn_index[imp]]
    pkg = imp.rsplit(".", 1)[0]                 # maybe a member of a class we indexed
    return list(pkg_index.get(pkg, [])) if pkg in pkg_index else []


def build_import_graph(files, repo_root) -> dict:
    """Resolve every file's imports into internal module_id edges."""
    repo_root = Path(repo_root)
    rootname = repo_root.resolve().name
    mod, by_lang = {}, {}
    for f in files:
        if f.get("error"):
            continue
        mod[f["file"]] = module_id(f["file"], repo_root)
        by_lang.setdefault(f.get("language"), []).append(f)
    module_ids = set(mod.values())

    # Python dotted index: both "core.llm" and "<root>.core.llm" -> module_id
    py_index = {}
    for f in by_lang.get("python", []):
        m = mod[f["file"]]
        parts = m.split("/")
        dotted = "/".join(parts[:-1]).replace("/", ".") if parts[-1] == "__init__" else m.replace("/", ".")
        if dotted:
            py_index[dotted] = m
            py_index[f"{rootname}.{dotted}"] = m

    ts_ids = {mod[f["file"]] for f in by_lang.get("typescript", []) + by_lang.get("javascript", [])}

    cs_ns = {}
    for f in by_lang.get("csharp", []):
        if f.get("namespace"):
            cs_ns.setdefault(f["namespace"], []).append(mod[f["file"]])

    java_pkg, java_fqcn = {}, {}
    for f in by_lang.get("java", []):
        ns, m = f.get("namespace"), mod[f["file"]]
        if ns:
            java_pkg.setdefault(ns, []).append(m)
            for c in f.get("classes", []):
                java_fqcn[f"{ns}.{c['name']}"] = m

    graph = {}
    for f in files:
        if f.get("error"):
            continue
        m, lang = mod[f["file"]], f.get("language")
        deps = set()
        for imp in f.get("imports", []):
            if lang == "python":
                r = _resolve_python(imp, m, py_index, module_ids)
                if r:
                    deps.add(r)
            elif lang in ("typescript", "javascript"):
                r = _resolve_ts(imp, m, ts_ids)
                if r:
                    deps.add(r)
            elif lang == "csharp":
                deps.update(_resolve_csharp(imp, cs_ns))
            elif lang == "java":
                deps.update(_resolve_java(imp, java_pkg, java_fqcn))
        deps.discard(m)
        if deps:
            graph[m] = sorted(deps)
    return graph


# mode "substr": match anywhere (namespace-style names that can't collide).
# mode "module": match a whole JS module specifier (so "RegularExpressions"
# does not match "express").
_FRAMEWORK_SIGNS = [
    # .NET / JVM
    ("aspnetcore", "substr", ("microsoft.aspnetcore",)),
    ("spring",     "substr", ("org.springframework",)),
    # Node backend
    ("nestjs",     "substr", ("@nestjs",)),
    ("express",    "module", ("express",)),
    ("fastify",    "module", ("fastify",)),
    # Fullstack / meta-frameworks (single-deploy, UI+API)
    ("nextjs",     "module", ("next",)),
    ("remix",      "substr", ("@remix-run",)),
    ("sveltekit",  "substr", ("@sveltejs/kit",)),
    # UI frameworks
    ("react",      "module", ("react", "react-dom")),
    ("vue",        "module", ("vue",)),
    ("angular",    "substr", ("@angular/core",)),
    ("svelte",     "module", ("svelte",)),
    # Python backend
    ("fastapi",    "substr", ("fastapi",)),
    ("flask",      "substr", ("flask",)),
    ("django",     "substr", ("django",)),
    ("starlette",  "substr", ("starlette",)),
    ("tornado",    "substr", ("tornado",)),
]


def detect_frameworks(files) -> list:
    """Ordered list of frameworks detected across all files' imports."""
    imports = [imp.lower() for f in files for imp in f.get("imports", [])]
    blob = " ".join(imports)
    found = []
    for name, mode, sigs in _FRAMEWORK_SIGNS:
        if mode == "substr":
            hit = any(s in blob for s in sigs)
        else:  # whole module specifier ("express" or "express/...")
            hit = any(imp == s or imp.startswith(s + "/") for imp in imports for s in sigs)
        if hit:
            found.append(name)
    return found
