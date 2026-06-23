"""
Deterministic Mermaid syntax validator.

Checks diagram text for structural errors without calling an LLM.
Returns a dict: { "valid": bool, "errors": [...], "diagram_type": str | None }
"""

import re

# Known diagram type keywords and which LLD/HLD output they correspond to:
#   flowchart  → HLD combined  (render_c4_combined  → flowchart TD)
#   graph      → LLD component (graph TD) + LLD dependency (graph LR)
#   classDiagram    → LLD class
#   sequenceDiagram → LLD sequence
#   C4Context       → HLD context-only
#   C4Container     → HLD container-only
_DIAGRAM_TYPES = {
    "flowchart",
    "graph",
    "classDiagram",
    "sequenceDiagram",
    "C4Context",
    "C4Container",
    "erDiagram",
    "stateDiagram",
}

_ARROW_BY_TYPE = {
    "flowchart":       re.compile(r"-->|==>|~~~|-.->|--\|"),
    "graph":           re.compile(r"-->|==>|~~~|-.->|--\|"),
    "classDiagram":    re.compile(r"<\|--|\.\.>|-->|\*--|o--"),
    "sequenceDiagram": re.compile(r"->>|-->>|->|-->"),
    "C4Context":       re.compile(r"Rel\("),
    "C4Container":     re.compile(r"Rel\("),
}


def _detect_type(text: str) -> str | None:
    """Return the diagram type keyword from the first content token, or None.

    Leading Mermaid directive lines (``%%{init: ...}%%``) and blank lines are
    skipped — they legally precede the diagram-type keyword.
    """
    first = ""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%%"):
            continue
        first = s.split()[0]
        break
    for dt in _DIAGRAM_TYPES:
        if first.startswith(dt):
            return dt
    return None


def _check_braces(text: str) -> list[str]:
    """Check that every opening brace has a matching closing brace."""
    errors = []
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"Unexpected '}}' near position {i}")
                depth = 0
    if depth != 0:
        errors.append(f"Unclosed '{{': {depth} brace(s) never closed")
    return errors


def _check_empty_ids(text: str, diagram_type: str) -> list[str]:
    """Check for empty node labels or missing IDs."""
    errors = []
    if diagram_type in ("flowchart", "graph"):
        for m in re.finditer(r'\b(\w+)\[""\]', text):
            errors.append(f"Empty label on node '{m.group(1)}'")
    if diagram_type in ("C4Context", "C4Container"):
        for m in re.finditer(r'(Person|Container|System|System_Ext|ContainerDb)\(\s*,', text):
            errors.append(f"Missing ID in {m.group(1)}() call")
    if diagram_type == "sequenceDiagram":
        for m in re.finditer(r'^[ \t]*participant\s*$', text, re.MULTILINE):
            errors.append("Empty participant name found")
    if diagram_type == "classDiagram":
        for m in re.finditer(r'class\s+\{', text):
            errors.append("Class block with no name found")
    return errors


def _check_arrows(text: str, diagram_type: str) -> list[str]:
    """Check that at least one valid arrow exists for diagrams with many content lines."""
    if diagram_type not in _ARROW_BY_TYPE:
        return []
    pattern = _ARROW_BY_TYPE[diagram_type]
    skip_prefixes = (
        "title", "participant", "class ", "subgraph", "end",
        "classDef", "%%", "Person", "Container", "System",
        "ContainerDb", "flowchart", "graph", "sequenceDiagram",
        "classDiagram", "C4Context", "C4Container",
    )
    content_lines = [
        l for l in text.splitlines()
        if l.strip() and not any(l.strip().startswith(p) for p in skip_prefixes)
    ]
    if len(content_lines) > 3:
        if not any(pattern.search(l) for l in content_lines):
            return [f"No valid arrows found for diagram type '{diagram_type}'"]
    return []


def _check_duplicate_ids(text: str, diagram_type: str) -> list[str]:
    """Check for duplicate node IDs which cause Mermaid parse errors."""
    seen: dict[str, int] = {}

    if diagram_type in ("C4Context", "C4Container"):
        pattern = re.compile(r'(?:Person|Container|System_Ext|System|ContainerDb)\((\w+),')
    elif diagram_type in ("flowchart", "graph"):
        pattern = re.compile(r'\b([A-Za-z_]\w*)\[')
    elif diagram_type == "sequenceDiagram":
        pattern = re.compile(r'participant\s+(\w+)')
    elif diagram_type == "classDiagram":
        pattern = re.compile(r'class\s+(\w+)\s*\{')
    else:
        return []
    for m in pattern.finditer(text):
        nid = m.group(1)
        seen[nid] = seen.get(nid, 0) + 1
    return [
        f"Duplicate node ID '{nid}' appears {count} times"
        for nid, count in seen.items()
        if count > 1
    ]
def _check_class_members(text: str, diagram_type: str) -> list[str]:
    """Mermaid classDiagram member lines cannot contain [ ] | or =."""
    if diagram_type != "classDiagram":
        return []
    errors, in_class, member_count = [], False, 0
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("class ") and s.endswith("{"):
            in_class, member_count = True, 0
        elif s == "}":
            if in_class and member_count == 0:
                errors.append("Empty class body '{ }' — Mermaid errors on it; declare the class without braces")
            in_class = False
        elif in_class:
            member_count += 1
            if re.search(r"[\[\]|=]", s):
                errors.append(f"Unparseable class member (contains [ ] | or =): '{s}'")

    return errors



def validate_mermaid(text: str) -> dict:
    """
    Run deterministic checks on Mermaid diagram text.

    Returns:
        {
            "valid": bool,
            "errors": list[str],
            "diagram_type": str | None
        }
    """
    if not text or not text.strip():
        return {"valid": False, "errors": ["Empty diagram text"], "diagram_type": None}

    errors: list[str] = []
    diagram_type = _detect_type(text)

    if diagram_type is None:
        errors.append(
            f"Unknown or missing diagram type keyword "
            f"(first token: '{text.strip().split()[0]}')"
        )

    errors += _check_braces(text)

    if diagram_type:
        errors += _check_empty_ids(text, diagram_type)
        errors += _check_arrows(text, diagram_type)
        errors += _check_duplicate_ids(text, diagram_type)
        errors += _check_class_members(text, diagram_type)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "diagram_type": diagram_type,
    }
