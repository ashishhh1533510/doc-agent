"""
Output tool: render generated documentation into different formats and save it.

md / html   -> prose (html is converted from markdown)
json / yaml -> the extracted facts as a structured spec
mermaid     -> C4 combined HLD + LLD diagram text (saved as-is)

All text formats are written as plain text. Deterministic -- no LLM here.
"""
import json
from pathlib import Path

import markdown as _markdown
import yaml


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ```...``` fence the model sometimes adds."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def to_json(data) -> str:
    """Serialize extracted facts to a JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_yaml(data) -> str:
    """Serialize extracted facts to a YAML string (spec/config-style docs)."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def markdown_to_html(text: str) -> str:
    """Convert a markdown document into a standalone HTML page."""
    body = _markdown.markdown(text, extensions=["fenced_code", "tables", "toc"])
    return (
        "<!doctype html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        "<title>Documentation</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def save_text(path, content: str) -> str:
    """Write any text content to a file; return the absolute path written."""
    path = Path(path)
    try:
        # ensure parent directory exists
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())
    except PermissionError:
        # fallback: attempt to write to an outputs/ subfolder within cwd
        fallback_dir = Path("outputs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / path.name
        fallback_path.write_text(content, encoding="utf-8")
        return str(fallback_path.resolve())


def save_json(path, data) -> str:
    """Save a dict/list as pretty-printed JSON; return the absolute path written."""
    return save_text(path, to_json(data))


def slim_facts_for_llm(facts: list, max_tokens: int = 100_000) -> list:
    """
    Produce a token-trimmed copy of the extracted facts for LLM calls only.

    The full facts feed json/yaml (which need completeness and cost 0 LLM calls).
    This slimmed version is what goes to the writer/reviewer/diagrammer so large
    repos stay under Gemini's per-minute input-token limit.

    Trimming, applied progressively until under budget:
      1. public API only (drop _private / __dunder methods and functions)
      2. first line of each docstring, truncated
      3. drop modules with nothing public
      4. if still over budget, drop docstrings, then drop method lists
    """
    import json

    def build(keep_doc: bool, keep_methods: bool) -> list:
        out = []
        for f in facts:
            sclasses = []
            for c in f.get("classes", []):
                entry = {"name": c.get("name", "")}
                if keep_doc:
                    d = (c.get("docstring") or "").split("\n")[0].strip()
                    if d:
                        entry["doc"] = d[:80]
                if keep_methods:
                    methods = c.get("methods", []) or []
                    names = []
                    for m in methods:
                        mn = m.get("name") if isinstance(m, dict) else m
                        if mn and not mn.startswith("_"):
                            names.append(mn)
                    if names:
                        entry["methods"] = names
                sclasses.append(entry)
            sfuncs = []
            for fn in f.get("functions", []) or []:
                fnn = fn.get("name") if isinstance(fn, dict) else fn
                if fnn and not fnn.startswith("_"):
                    sfuncs.append(fnn)
            if not sclasses and not sfuncs and not f.get("module_docstring"):
                continue
            entry = {"file": f.get("file", ""), "functions": sfuncs, "classes": sclasses}
            if f.get("imports"):
                entry["imports"] = f["imports"]
            if keep_doc:
                md = (f.get("module_docstring") or "").split("\n")[0].strip()
                if md:
                    entry["doc"] = md[:100]
            out.append(entry)
        return out

    for kwargs in (
        {"keep_doc": True,  "keep_methods": True},
        {"keep_doc": True,  "keep_methods": False},
        {"keep_doc": False, "keep_methods": False},
    ):
        payload = build(**kwargs)
        if len(json.dumps(payload)) // 4 <= max_tokens:
            return payload
    return payload


def render_c4_combined(model: dict) -> str:
    """Render HLD as a flowchart TD with a named system boundary subgraph.

    Visual hierarchy:
      Actor(s)
          ↓
      [system_purpose boundary]
          capabilities inside
          ↓
      External systems (outside, below)
    """
    ctx  = model.get("context", {})
    cont = model.get("containers", {})

    # Prefer containers.system_label (set to system_purpose by updated architect).
    # Fall back to context.system_name for backward compatibility.
    system_label = (
        cont.get("system_label")
        or ctx.get("system_name")
        or "System"
    )

    lines = ["flowchart TD"]

    # ── Actors ──────────────────────────────────────────────────────────────
    actor_ids = []
    for a in ctx.get("actors", []):
        aid = _slug(a["id"])
        actor_ids.append(aid)
        lines.append(f'    {aid}["{a["label"]}"]')

    lines.append("")

    # ── System boundary (subgraph) with capabilities inside ─────────────────
    safe_label = system_label.replace('"', "'")
    lines.append(f'    subgraph SYS["{safe_label}"]')

    cap_ids = []
    for c in cont.get("containers", []):
        cid = _slug(c["id"])
        cap_ids.append(cid)
        tech = c.get("tech", "")
        label = c["label"] + (f"<br/><small>{tech}</small>" if tech else "")
        lines.append(f'        {cid}["{label}"]')

    for db in cont.get("databases", []):
        did = _slug(db["id"])
        cap_ids.append(did)
        lines.append(f'        {did}[("{db["label"]}")]')

    lines.append("    end")
    lines.append("")

    # ── External systems (outside boundary, styled separately) ───────────────
    ext_ids = set()
    all_externals = (
        list(ctx.get("external_systems", []))
        + list(cont.get("external_services", []))
    )
    seen_ext = set()
    for ext in all_externals:
        eid = _slug(ext["id"])
        if eid not in seen_ext:
            seen_ext.add(eid)
            ext_ids.add(eid)
            lines.append(f'    {eid}["{ext["label"]}"]:::ext')

    lines.append("")
    lines.append("    classDef ext fill:#f5f5f5,stroke:#999,color:#333")
    lines.append("")

    # ── Build declared-ID set for relationship validation ────────────────────
    declared = set(actor_ids) | set(cap_ids) | ext_ids

    # ── Relationships (deduped, validated) ───────────────────────────────────
    seen_rels: set = set()

    def _add_rel(f: str, t: str, label: str) -> None:
        f, t = _slug(f), _slug(t)
        if f in declared and t in declared and (f, t) not in seen_rels:
            seen_rels.add((f, t))
            escaped = label.replace('"', "'")
            lines.append(f'    {f} -->|"{escaped}"| {t}')

    for rel in list(ctx.get("relationships", [])) + list(cont.get("relationships", [])):
        _add_rel(rel["from"], rel["to"], rel.get("label", ""))

    # ── Fallback: guarantee at least one actor → capability edge ─────────────
    # If the relationships block has nothing connecting any actor to any capability,
    # add a plain edge from the first actor to the first capability so the visual
    # hierarchy (actor → system boundary → externals) is always present.
    if actor_ids and cap_ids:
        if not any(
            (a, c) in seen_rels
            for a in actor_ids
            for c in cap_ids
        ):
            _add_rel(actor_ids[0], cap_ids[0], "uses")

    return "\n".join(lines)

def _clean_params(params: str) -> str:
    """Mermaid class members cannot contain [ ] | or = — keep parameter names only."""
    if not params:
        return ""
    names, current, depth = [], "", 0
    for ch in params:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            names.append(current)
            current = ""
            continue
        current += ch
    names.append(current)
    out = []
    for n in names:
        n = n.split(":")[0].split("=")[0].strip()
        if n and n not in ("self", "cls"):
            out.append(n)
    return ", ".join(out)


def _clean_type(typ: str) -> str:
    """Reduce an annotation to a Mermaid-safe name: 'dict[str, bool] | None' -> 'dict'."""
    if not typ:
        return ""
    return typ.split("|")[0].split("=")[0].split("[")[0].strip()




def render_class_diagram(model: dict) -> str:
    """Render a class diagram JSON model as Mermaid classDiagram."""
    _REL = {
        "inheritance": "<|--", "composition": "*--", "aggregation": "o--",
        "dependency": "-->", "realization": "<|..",
    }
    lines = ["classDiagram"]

    for cls in model.get("classes", []):
        fields = cls.get("fields", [])
        methods = cls.get("methods", [])
        if not fields and not methods:
            # Mermaid 10.x throws "Syntax error" on an empty '{ }' body — declare it bare.
            lines.append(f'  class {cls["name"]}')
            continue
        lines.append(f'  class {cls["name"]} {{')
        for f in fields:
            vis = f.get("visibility", "+")
            typ = _clean_type(f.get("type") or "")
            type_suffix = f" : {typ}" if typ else ""
            lines.append(f'    {vis}{f["name"]}{type_suffix}')
        for m in methods:
            vis = m.get("visibility", "+")
            ret = _clean_type(m.get("return_type") or "")
            ret_suffix = f" {ret}" if ret else ""
            params = _clean_params(m.get("params", ""))
            lines.append(f'    {vis}{m["name"]}({params}){ret_suffix}')
        lines.append("  }")


    for rel in model.get("relationships", []):
        rtype = rel.get("type", "dependency")
        arrow = _REL.get(rtype, "-->")
        label = f' : {rel["label"]}' if rel.get("label") else ""
        src, dst = rel["from"], rel["to"]
        if rtype in ("inheritance", "realization"):
            src, dst = dst, src  # Mermaid reads 'Base <|-- Derived'
        lines.append(f'  {src} {arrow} {dst}{label}')




    return "\n".join(lines)


def render_sequence_diagram(model: dict) -> str:
    """Render a sequence diagram JSON model as Mermaid sequenceDiagram."""
    _ARROW = {"sync": "->>", "async": "->>", "return": "-->>"}

    def _short(name: str) -> str:
        """Use only the last dotted segment as the participant alias."""
        return _slug(name.split(".")[-1])

    lines = ["sequenceDiagram"]

    for p in model.get("participants", []):
        alias = _short(p)
        label = p.split(".")[-1]
        lines.append(f'  participant {alias} as {label}')

    for msg in model.get("messages", []):
        arrow = _ARROW.get(msg.get("type", "sync"), "->>")
        label = msg.get("label", "")
        lines.append(f'  {_short(msg["from"])}{arrow}{_short(msg["to"])}: {label}')

    return "\n".join(lines)



def render_component_diagram(model: dict) -> str:
    """Render a component diagram JSON model as Mermaid graph TD."""
    lines = ["graph TD", '  subgraph "Application"']

    for comp in model.get("components", []):
        cid = _slug(comp["id"])
        tech = f" ({comp['tech']})" if comp.get("tech") else ""
        lines.append(f'    {cid}["{comp["label"]}{tech}"]')

    lines.append("  end")

    for dep in model.get("dependencies", []):
        fid = _slug(dep["from"])
        tid = _slug(dep["to"])
        label = f'|"{dep["label"]}"|' if dep.get("label") else ""
        lines.append(f'  {fid} -->{label} {tid}')

    return "\n".join(lines)



def render_dependency_diagram(model: dict) -> str:
    """Render a dependency diagram JSON model as Mermaid graph LR."""
    lines = ["graph LR"]

    internal = [p for p in model.get("packages", []) if p.get("kind") == "internal"]
    external = [p for p in model.get("packages", []) if p.get("kind") != "internal"]

    if internal:
        lines.append('  subgraph "This Repo"')
        for pkg in internal:
            lines.append(f'    {pkg["id"]}["{pkg["label"]}"]')
        lines.append("  end")

    for pkg in external:
        lines.append(f'  {pkg["id"]}["{pkg["label"]}"]')

    for edge in model.get("edges", []):
        label = f'|"{edge["label"]}"|' if edge.get("label") else ""
        lines.append(f'  {edge["from"]} -->{label} {edge["to"]}')

    return "\n".join(lines)


def _slug(name: str) -> str:
    """Convert a display name to a safe Mermaid node ID."""
    return name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
