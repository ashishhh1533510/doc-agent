"""
Output tool: writes generated documentation to disk.

Markdown documents (READMEs) save as .md; structured specs (like an
OpenAPI/Swagger spec) save as .json. Deterministic -- no LLM here.
"""

import json
from pathlib import Path


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ```markdown ... ``` fence the model sometimes adds."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):     # drop opening ``` / ```markdown
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":      # drop closing ```
        lines = lines[:-1]
    return "\n".join(lines).strip()


def save_markdown(path, content: str) -> str:
    """Save markdown content to a .md file; return the absolute path written."""
    path = Path(path)
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def save_json(path, data) -> str:
    """Save a dict/list as pretty-printed JSON; return the absolute path written."""
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.resolve())