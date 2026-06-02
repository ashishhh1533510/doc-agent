"""
Output tool: render generated documentation into different formats and save it.

md / html  -> prose (html is converted from markdown)
json / yaml -> the extracted facts as a structured spec
mermaid     -> diagram text (produced by the diagrammer agent, saved as-is)
png / svg   -> architecture image (diagrams library)
dot         -> raw Graphviz DOT source
drawio_xml  -> native draw.io file (rank-based layout, opens directly)

All text formats are written as plain text. Deterministic -- no LLM here.
"""

import json
from pathlib import Path

import markdown as _markdown
import yaml

import os
import shutil


# Ensure Graphviz's `dot` is findable even when the server process didn't
# inherit the user PATH. Point the graphviz library straight at the binary.
def _ensure_graphviz_on_path():
    if shutil.which("dot"):
        return  # already findable, nothing to do
    # common install locations on this machine
    candidates = [
        r"C:\Users\AshishKumar\Downloads\Graphviz-15.0.0-win64\bin",
        r"C:\Program Files\Graphviz\bin",
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "dot.exe")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_graphviz_on_path()


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


def render_architecture_mermaid(model: dict) -> str:
    """
    Deterministically render an architecture model (components, externals, edges)
    into a clean Mermaid flowchart.

    The LLM decides WHAT the architecture is (this dict); this function decides
    HOW to draw it correctly: it groups components into layer subgraphs, adds
    external-system nodes, and draws edges while DROPPING self-loops, removing
    duplicate edges, and ignoring edges that point at unknown nodes. That keeps
    the diagram structurally valid no matter how the model is phrased.
    """
    components = model.get("components", [])
    externals = model.get("externals", [])
    edges = model.get("edges", [])

    valid_ids = {c["id"] for c in components} | {e["id"] for e in externals}
    lines = ["flowchart TB"]

    # Group components into one subgraph per layer (unique, safe subgraph ids).
    layers = {}
    for c in components:
        layers.setdefault(c.get("layer", "Components"), []).append(c)
    for i, (layer, comps) in enumerate(layers.items()):
        lines.append(f'    subgraph sg{i}["{layer.replace(chr(34), chr(39))}"]')
        for c in comps:
            lines.append(f'        {c["id"]}["{c["label"].replace(chr(34), chr(39))}"]')
        lines.append("    end")

    # External systems as rounded/stadium nodes, outside any subgraph.
    for ext in externals:
        lines.append(f'    {ext["id"]}(["{ext["label"].replace(chr(34), chr(39))}"])')

    # Edges: drop self-loops, duplicates, and references to unknown nodes.
    seen = set()
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue                       # no self-loops, no incomplete edges
        if src not in valid_ids or dst not in valid_ids:
            continue                       # no edges to undefined nodes
        if (src, dst) in seen:
            continue                       # no duplicate edges between a pair
        seen.add((src, dst))
        label = str(e.get("label", "")).replace("|", "/").strip()
        lines.append(f"    {src} -->|{label}| {dst}" if label else f"    {src} --> {dst}")

    return "\n".join(lines)


def render_architecture_diagram(model: dict, output_path, fmt: str = "png",
                                title: str = "Architecture", direction: str = "TB") -> str:
    """
    Deterministically render an architecture model into an IMAGE (png/svg) using
    the `diagrams` library, which drives Graphviz under the hood.

    Same model and same contract as render_architecture_mermaid(): the LLM decides
    WHAT (the model dict), this code decides HOW. Components are grouped into one
    Cluster per layer, externals get system icons, and edges are validated the same
    way -- self-loops, duplicates, and edges to unknown nodes are dropped.

    Requires the Graphviz `dot` binary to be installed and on PATH.
    Returns the path to the image file written.
    """
    from diagrams import Diagram, Cluster, Edge
    from diagrams.programming.language import Python
    from diagrams.programming.framework import Fastapi
    from diagrams.generic.storage import Storage
    from diagrams.generic.compute import Rack
    from diagrams.generic.blank import Blank

    components = model.get("components", []) or []
    externals = model.get("externals", []) or []
    edges = model.get("edges", []) or []

    # Same validation as the mermaid renderer.
    valid_ids = {c["id"] for c in components} | {e["id"] for e in externals}
    seen = set()
    clean_edges = []
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue
        if src not in valid_ids or dst not in valid_ids:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean_edges.append(e)

    # `diagrams` appends the format extension itself, so strip it from the name.
    base = str(Path(output_path).with_suffix(""))

    def _icon_for_external(label: str):
        low = label.lower()
        if any(k in low for k in ("faiss", "vector", "store", "db")):
            return Storage
        if "fastapi" in low:
            return Fastapi
        if any(k in low for k in ("api", "llm", "gemini")):
            return Rack
        return Blank

    graph_attr = {
        "fontsize": "20",
        "bgcolor": "white",
        "splines": "ortho",
        "nodesep": "0.6",
        "ranksep": "1.0",
    }

    with Diagram(title, filename=base, outformat=fmt, show=False,
                 direction=direction, graph_attr=graph_attr):
        nodes = {}

        # One Cluster per layer, insertion order preserved.
        layers = {}
        for c in components:
            layers.setdefault(c.get("layer", "Components"), []).append(c)
        for layer_name, comps in layers.items():
            with Cluster(layer_name):
                for c in comps:
                    nodes[c["id"]] = Python(c["label"])

        if externals:
            with Cluster("External Systems"):
                for x in externals:
                    nodes[x["id"]] = _icon_for_external(x["label"])(x["label"])

        for e in clean_edges:
            nodes[e["from"]] >> Edge(label=str(e.get("label", ""))) >> nodes[e["to"]]

    return f"{base}.{fmt}"


def render_typed_architecture(model: dict, output_path, fmt: str = "png",
                              title: str = "Architecture", direction: str = "TB") -> str:
    """
    Render a TYPED architecture model into a semantic-shape image (png/svg) via
    the `diagrams` library (Graphviz under the hood).

    Each node carries a `type` that maps to a distinct shape:
      actor     -> user/person icon (the entry actor)
      framework -> web framework icon
      process   -> compute/logic block
      datastore -> database/file store
      external  -> external API block
      io        -> input/output artifact

    The LLM decides WHAT and WHAT-KIND (the model); this code decides WHICH SHAPE.
    Edges are validated like the other renderers: self-loops, duplicates, and edges
    to unknown nodes are dropped. Requires Graphviz `dot` on PATH.
    Returns the path to the image written.
    """
    from diagrams import Diagram, Cluster, Edge
    from diagrams.onprem.client import User
    from diagrams.generic.database import SQL
    from diagrams.generic.storage import Storage
    from diagrams.generic.compute import Rack
    from diagrams.programming.language import Python
    from diagrams.programming.framework import Fastapi

    shapes = {
        "actor": User, "datastore": SQL, "external": Rack,
        "framework": Fastapi, "process": Python, "io": Storage,
    }
    default = Python

    nodes_in = model.get("nodes", []) or []
    edges = model.get("edges", []) or []
    valid = {n["id"] for n in nodes_in}

    seen = set()
    clean = []
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue
        if src not in valid or dst not in valid:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean.append(e)

    base = str(Path(output_path).with_suffix(""))
    graph_attr = {
        "fontsize": "22", "bgcolor": "white", "splines": "ortho",
        "nodesep": "1.2", "ranksep": "1.3", "pad": "0.5", "compound": "true",
    }
    node_attr = {"fontsize": "13"}

    with Diagram(title, filename=base, outformat=fmt, show=False,
                 direction=direction, graph_attr=graph_attr, node_attr=node_attr):
        objs = {}
        groups = {}
        for n in nodes_in:
            groups.setdefault(n.get("group"), []).append(n)
        for group_name, members in groups.items():
            if group_name:
                with Cluster(group_name):
                    for n in members:
                        objs[n["id"]] = shapes.get(n.get("type"), default)(n["label"])
            else:
                for n in members:
                    objs[n["id"]] = shapes.get(n.get("type"), default)(n["label"])
        for e in clean:
            objs[e["from"]] >> Edge(label=str(e.get("label", ""))) >> objs[e["to"]]

    return f"{base}.{fmt}"


def render_typed_architecture_svg(model: dict, output_path, title: str = "Architecture") -> str:
    """
    Render a typed architecture model into a fully-editable SVG using pure
    Graphviz shapes -- no embedded images. Every shape imports into draw.io
    and Visio as a native editable element.

    Node types map to distinct shapes and colors:
      actor     -> circle  (blue)       -- the entry actor
      framework -> rounded rectangle (green) -- web framework
      process   -> rectangle (yellow)   -- application logic
      datastore -> cylinder (red)       -- storage / vector store
      external  -> dashed rectangle (purple) -- external API
      io        -> note shape (grey)    -- input/output artifact

    Returns the path to the SVG written.
    """
    import graphviz

    nodes_in = model.get("nodes", []) or []
    edges = model.get("edges", []) or []
    valid = {n["id"] for n in nodes_in}

    seen = set()
    clean = []
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue
        if src not in valid or dst not in valid:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean.append(e)

    STYLE = {
        "actor":     {"shape": "circle",    "style": "filled",         "fillcolor": "#DAE8FC", "color": "#6C8EBF"},
        "framework": {"shape": "rectangle", "style": "filled,rounded", "fillcolor": "#D5E8D4", "color": "#82B366"},
        "process":   {"shape": "rectangle", "style": "filled",         "fillcolor": "#FFF2CC", "color": "#D6B656"},
        "datastore": {"shape": "cylinder",  "style": "filled",         "fillcolor": "#F8CECC", "color": "#B85450"},
        "external":  {"shape": "rectangle", "style": "filled,dashed",  "fillcolor": "#E1D5E7", "color": "#9673A6"},
        "io":        {"shape": "note",      "style": "filled",         "fillcolor": "#F5F5F5", "color": "#666666"},
    }
    default_style = STYLE["process"]

    dot = graphviz.Digraph(
        comment=title,
        graph_attr={
            "rankdir": "TB", "splines": "polyline", "nodesep": "0.9",
            "ranksep": "1.2", "fontname": "Arial", "label": title,
            "labelloc": "b", "fontsize": "22", "bgcolor": "white",
            "compound": "true", "pad": "0.5",
        },
        node_attr={"fontname": "Arial", "fontsize": "13"},
        edge_attr={"fontname": "Arial", "fontsize": "11"},
    )

    groups = {}
    for n in nodes_in:
        groups.setdefault(n.get("group"), []).append(n)

    for group_name, members in groups.items():
        if group_name:
            with dot.subgraph(name=f"cluster_{group_name.replace(' ', '_')}") as sub:
                sub.attr(label=group_name, style="filled,rounded",
                         fillcolor="#EEF4FF", color="#AAAAAA",
                         fontname="Arial", fontsize="14", margin="20")
                for n in members:
                    sub.node(n["id"], label=n["label"],
                             **STYLE.get(n.get("type"), default_style))
        else:
            for n in members:
                dot.node(n["id"], label=n["label"],
                         **STYLE.get(n.get("type"), default_style))

    for e in clean:
        dot.edge(e["from"], e["to"], xlabel=str(e.get("label", "")))

    base = str(Path(output_path).with_suffix(""))
    dot.render(base, format="svg", cleanup=True)
    return f"{base}.svg"


def render_typed_architecture_dot(model: dict, output_path, title: str = "Architecture") -> str:
    """
    Export the typed architecture model as raw Graphviz DOT source (.gv file).
    No rendering -- pure text. Import into draw.io via:
      Extras -> Edit Diagram -> switch to Graphviz tab -> paste.
    Also opens directly in VS Code with the Graphviz extension.
    Returns the path to the .gv file written.
    """
    import graphviz

    nodes_in = model.get("nodes", []) or []
    edges = model.get("edges", []) or []
    valid = {n["id"] for n in nodes_in}

    seen = set()
    clean = []
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue
        if src not in valid or dst not in valid:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean.append(e)

    STYLE = {
        "actor":     {"shape": "circle",    "style": "filled",         "fillcolor": "#DAE8FC", "color": "#6C8EBF"},
        "framework": {"shape": "rectangle", "style": "filled,rounded", "fillcolor": "#D5E8D4", "color": "#82B366"},
        "process":   {"shape": "rectangle", "style": "filled",         "fillcolor": "#FFF2CC", "color": "#D6B656"},
        "datastore": {"shape": "cylinder",  "style": "filled",         "fillcolor": "#F8CECC", "color": "#B85450"},
        "external":  {"shape": "rectangle", "style": "filled,dashed",  "fillcolor": "#E1D5E7", "color": "#9673A6"},
        "io":        {"shape": "note",      "style": "filled",         "fillcolor": "#F5F5F5", "color": "#666666"},
    }

    dot = graphviz.Digraph(
        comment=title,
        graph_attr={
            "rankdir": "TB", "splines": "polyline", "nodesep": "0.9",
            "ranksep": "1.2", "fontname": "Arial", "label": title,
            "labelloc": "b", "fontsize": "22", "bgcolor": "white",
            "compound": "true", "pad": "0.5",
        },
        node_attr={"fontname": "Arial", "fontsize": "13"},
        edge_attr={"fontname": "Arial", "fontsize": "11"},
    )

    groups = {}
    for n in nodes_in:
        groups.setdefault(n.get("group"), []).append(n)

    for group_name, members in groups.items():
        if group_name:
            with dot.subgraph(name=f"cluster_{group_name.replace(' ', '_')}") as sub:
                sub.attr(label=group_name, style="filled,rounded",
                         fillcolor="#EEF4FF", color="#AAAAAA",
                         fontname="Arial", fontsize="14", margin="20")
                for n in members:
                    sub.node(n["id"], label=n["label"],
                             **STYLE.get(n.get("type"), STYLE["process"]))
        else:
            for n in members:
                dot.node(n["id"], label=n["label"],
                         **STYLE.get(n.get("type"), STYLE["process"]))

    for e in clean:
        dot.edge(e["from"], e["to"], xlabel=str(e.get("label", "")))

    path = Path(output_path).with_suffix(".gv")
    path.write_text(dot.source, encoding="utf-8")
    return str(path)


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


def _layout_from_intent(model: dict):
    """Compute exact node coordinates from the LLM's layout intent.

    The LLM assigns each node a `rank` (integer band, 0 = first) and a flow
    `direction`. This function turns that intent into clean coordinates: each
    rank becomes an evenly-spaced, centered row (TB) or column (LR). Because the
    code owns every coordinate, overlaps are impossible by construction.
    """
    from collections import defaultdict

    direction = model.get("direction", "TB")
    nodes = model.get("nodes", []) or []

    NODE_W, NODE_H = 200, 70
    H_GAP, V_GAP = 140, 200   # wide gaps give edges their own routing lanes

    ranks = defaultdict(list)
    for n in nodes:
        ranks[n.get("rank", 0)].append(n)
    sorted_ranks = sorted(ranks.keys())

    max_in_rank = max((len(ranks[r]) for r in sorted_ranks), default=1)
    canvas_w = max_in_rank * NODE_W + (max_in_rank - 1) * H_GAP

    pos = {}
    for band, r in enumerate(sorted_ranks):
        row = ranks[r]
        row_w = len(row) * NODE_W + (len(row) - 1) * H_GAP
        start = (canvas_w - row_w) / 2
        for i, n in enumerate(row):
            if direction == "LR":
                x = band * (NODE_W + H_GAP)
                y = start + i * (NODE_H + H_GAP)
            else:  # TB
                x = start + i * (NODE_W + H_GAP)
                y = band * (NODE_H + V_GAP)
            pos[n["id"]] = {"x": x, "y": y, "w": NODE_W, "h": NODE_H}
    return pos


def render_drawio_xml(model: dict, output_path, title: str = "Architecture") -> str:
    """
    Generate a native draw.io XML file (.drawio) from a typed architecture model
    that includes the LLM's layout intent (per-node `rank` + `direction`).

    The LLM decides the structure (which nodes share a rank, grouping, flow
    direction); this code computes exact, evenly-spaced coordinates from that
    intent. No Graphviz, no auto-layout, no overlaps. Opens directly in draw.io.

    Edge connection points are distributed across each node's perimeter so that
    multiple edges on the same node don't stack on one point. Returns the path
    to the .drawio file written.
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    nodes = model.get("nodes", []) or []
    edges = model.get("edges", []) or []
    valid = {n["id"] for n in nodes}

    seen, clean = set(), []
    for e in edges:
        src, dst = e.get("from"), e.get("to")
        if not src or not dst or src == dst:
            continue
        if src not in valid or dst not in valid:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean.append(e)

    pos = _layout_from_intent(model)
    direction = model.get("direction", "TB")

    STYLES = {
        "actor":     "ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=14;fontStyle=1;",
        "framework": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=14;",
        "process":   "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=14;",
        "datastore": "shape=cylinder3;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=14;",
        "external":  "rounded=0;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=14;",
        "io":        "shape=note;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=14;",
    }
    EDGE_STYLE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;"
                  "orthogonalLoop=1;fontSize=11;fontColor=#555555;strokeColor=#888888;"
                  "endArrow=classic;labelBackgroundColor=#ffffff;")
    CLUSTER_STYLE = ("rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F8FF;strokeColor=#9DB8D8;"
                     "verticalAlign=top;align=left;fontSize=14;fontStyle=1;"
                     "spacingLeft=12;spacingTop=8;arcSize=3;")

    root_el = ET.Element("mxGraphModel",
                         dx="1422", dy="762", grid="1", gridSize="10",
                         guides="1", tooltips="1", connect="1", arrows="1",
                         fold="1", page="1", pageScale="1",
                         pageWidth="1654", pageHeight="1169",
                         math="0", shadow="0")
    xr = ET.SubElement(root_el, "root")
    ET.SubElement(xr, "mxCell", id="0")
    ET.SubElement(xr, "mxCell", id="1", parent="0")

    # group background boxes (computed from member bounds)
    groups = defaultdict(list)
    for n in nodes:
        if n.get("group"):
            groups[n["group"]].append(n)
    PAD, HDR = 40, 30
    for g, members in groups.items():
        mp = [pos[m["id"]] for m in members if m["id"] in pos]
        if not mp:
            continue
        x = min(p["x"] for p in mp) - PAD
        y = min(p["y"] for p in mp) - PAD - HDR
        x2 = max(p["x"] + p["w"] for p in mp) + PAD
        y2 = max(p["y"] + p["h"] for p in mp) + PAD
        cell = ET.SubElement(xr, "mxCell",
                             id=f"cluster_{g.replace(' ', '_')}", value=g,
                             style=CLUSTER_STYLE, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry",
                      x=str(round(x)), y=str(round(y)),
                      width=str(round(x2 - x)), height=str(round(y2 - y)),
                      **{"as": "geometry"})

    # nodes at computed coordinates
    for n in nodes:
        p = pos[n["id"]]
        cell = ET.SubElement(xr, "mxCell",
                             id=f"node_{n['id']}", value=n["label"],
                             style=STYLES.get(n.get("type"), STYLES["process"]),
                             vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry",
                      x=str(round(p["x"])), y=str(round(p["y"])),
                      width=str(p["w"]), height=str(p["h"]),
                      **{"as": "geometry"})

    # edges with distributed connection points so they don't stack on one point
    out_count, in_count = defaultdict(int), defaultdict(int)
    for e in clean:
        out_count[e["from"]] += 1
        in_count[e["to"]] += 1
    out_idx, in_idx = defaultdict(int), defaultdict(int)

    for i, e in enumerate(clean):
        src, dst = e["from"], e["to"]
        out_idx[src] += 1
        in_idx[dst] += 1
        frac_out = round(out_idx[src] / (out_count[src] + 1), 3)
        frac_in = round(in_idx[dst] / (in_count[dst] + 1), 3)

        if direction == "LR":
            # flow left->right: exit right side, enter left side
            conn = (f"exitX=1;exitY={frac_out};exitDx=0;exitDy=0;"
                    f"entryX=0;entryY={frac_in};entryDx=0;entryDy=0;")
        else:
            # flow top->bottom: exit bottom, enter top
            conn = (f"exitX={frac_out};exitY=1;exitDx=0;exitDy=0;"
                    f"entryX={frac_in};entryY=0;entryDx=0;entryDy=0;")

        cell = ET.SubElement(xr, "mxCell",
                             id=f"edge_{i}", value=str(e.get("label", "")),
                             style=EDGE_STYLE + conn, edge="1",
                             source=f"node_{src}", target=f"node_{dst}",
                             parent="1")
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

    tree = ET.ElementTree(root_el)
    ET.indent(tree, space="  ")
    path = Path(output_path).with_suffix(".drawio")
    tree.write(str(path), encoding="unicode", xml_declaration=False)
    return str(path)