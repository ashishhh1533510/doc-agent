"""
Diagrammer agent: produces a HIGH-LEVEL ARCHITECTURE (HLD) diagram.

Understanding is separated from drawing:
  1. The LLM analyzes the codebase facts and outputs a structured architecture
     MODEL as JSON -- components grouped by layer, external systems, and the
     edges between them. This is the agent's reasoning.
  2. tools.output.render_architecture_mermaid() deterministically turns that
     model into clean, valid Mermaid: no self-loops, no duplicate edges, always
     well-formed.

The LLM decides WHAT the architecture is; code decides HOW to draw it.
"""

import json

from doc_agent.core.llm import build_agent, run_agent
from doc_agent.tools.output import (
    render_architecture_diagram,
    render_architecture_mermaid,
    render_typed_architecture,
    render_typed_architecture_svg,
    render_typed_architecture_dot,
    render_drawio_xml,
    strip_code_fence,
)

INSTRUCTIONS = """You are a senior software architect producing a HIGH-LEVEL DESIGN (HLD) diagram.
You are given a JSON description of a Python codebase: each module's file path, docstring, classes,
functions, and the non-standard modules it imports.

Your job is to produce an ABSTRACT architecture, NOT a module-by-module wiring diagram. Think of how
an architect would whiteboard the system for a new engineer: a handful of meaningful building blocks
and how data flows between them -- not every file.

CRITICAL RULES FOR ABSTRACTION:
- Produce a MAXIMUM of 6-8 components. Fewer is better if the system allows it.
- GROUP modules by RESPONSIBILITY, not by file. Multiple modules that do the same kind of work
  become ONE component. For example, several agent modules become a single "LLM Agents" block;
  several orchestration modules become one "Orchestration" block.
- COMPLETENESS IS MANDATORY: every module in the input must be folded into exactly one component.
  Nothing may be left out. In each component's label, list the modules it contains in parentheses
  so the whole codebase is accounted for. Example label: "LLM Agents (writer, reviewer, qa)".
- Name each component by WHAT IT DOES (its role), not after a single file. Good names: "API Layer",
  "Orchestration", "LLM Agents", "Retrieval / RAG", "Deterministic Tools", "Model Connection".
- Use the `layer` field to place each component in a conceptual tier. Give each component its OWN
  distinct layer value unless two truly belong in the same visual tier.

Reason it through first:
- Skim every module and decide its responsibility from its name, docstring, and imports.
- Cluster modules with the same responsibility into one component.
- Identify the few external systems that matter: the LLM/model API, the vector store, the web
  framework. Ignore utility libraries (json, yaml, markdown, dotenv, pydantic, numpy, etc.).
- Determine the high-level data flow BETWEEN the grouped components, using imports as ground truth.
  An edge ALWAYS connects two DIFFERENT components. NEVER connect a component to itself.

Output JSON in exactly this shape and nothing else:

{
  "components": [
    {"id": "apiLayer", "label": "API Layer (api.app)", "layer": "Entry"}
  ],
  "externals": [
    {"id": "llm", "label": "Gemini LLM API"}
  ],
  "edges": [
    {"from": "apiLayer", "to": "orchestration", "label": "dispatches requests"}
  ]
}

Rules:
- id: short alphanumeric, no spaces or dots, unique across components and externals.
- label: the role name, with the contained modules in parentheses so coverage is visible.
- Each edge connects two DIFFERENT ids that both exist above. One edge per pair maximum.
- Edge label: a short, specific verb phrase describing the data flow ("dispatches requests",
  "delegates generation", "embeds chunks", "retrieves context"). Never "uses".

Output ONLY the JSON object. No prose, no Mermaid, no code fences.codebase: each module's file path, docstring, classes, functions, and the non-standard modules it
imports.

Analyze the system and output a STRUCTURED ARCHITECTURE MODEL as JSON. Do NOT draw a diagram or
write any Mermaid -- output ONLY the JSON model described below.

Reason it through first:
- Identify the logical layers and which modules belong to each (infer roles from names and
  docstrings: API/entry point, orchestration, LLM agents, RAG/retrieval, model connection,
  deterministic tools).
- Identify the few external systems that matter: the LLM/model API, the vector store, the web
  framework. Ignore utility libraries (json, yaml, markdown, dotenv, pydantic, numpy, etc.).
- Determine the real relationships BETWEEN DIFFERENT components, using the imports as ground truth.
  A relationship ALWAYS connects two DIFFERENT components. NEVER connect a component to itself, and
  never describe a component's own internal behaviour as an edge.

Output JSON in exactly this shape and nothing else:

{
  "components": [
    {"id": "apiApp", "label": "api.app", "layer": "API Layer"}
  ],
  "externals": [
    {"id": "llm", "label": "Gemini LLM API"}
  ],
  "edges": [
    {"from": "apiApp", "to": "docPipeline", "label": "generate docs"}
  ]
}

Rules:
- id: short alphanumeric, no spaces or dots, unique across components and externals.
- label: the readable name; use the package path so same-named files stay distinct
  ("workflow.qa" vs "agents.qa").
- Each edge connects two DIFFERENT ids that both exist above. One edge per pair maximum.
- Edge label: a short, specific verb phrase ("generates draft", "fact-checks draft",
  "embeds chunks", "retrieves context", "calls"). Never "uses".

Output ONLY the JSON object. No prose, no Mermaid, no code fences."""

TYPED_INSTRUCTIONS = """You are a senior software architect producing a clean, high-level TECHNICAL
ARCHITECTURE diagram. You are given a JSON description of a Python codebase (each module's path,
docstring, classes, functions, imports).

Analyze the system and output an ARCHITECTURE MODEL as JSON. You decide BOTH the content AND the
visual layout — think like an architect arranging a clean whiteboard diagram. Output ONLY the JSON.

CONTENT RULES:
- Produce 6-10 nodes. Group modules by RESPONSIBILITY, not per-file (e.g. several agent modules
  become ONE "LLM Agents" node). Every module must be represented by some node.
- Always include an "actor" node for the user/developer, and node(s) for the outputs produced.

EACH NODE HAS A TYPE that controls its shape:
- "actor"     : the human/user who initiates a request
- "framework" : the web/API framework entry point
- "process"   : application logic (orchestration, agents, tools, retrieval)
- "datastore" : persisted data (vector store, database, files)
- "external"  : an external service/API the system calls (the LLM provider)
- "io"        : an output artifact (generated docs/diagrams)

LAYOUT INTENT — you decide how the diagram is arranged:
- Set "direction" to "TB" (top-to-bottom) or "LR" (left-to-right). Choose whatever makes the data
  flow read most naturally for THIS system.
- Give every node a "rank": an integer starting at 0. Rank is the band the node sits in along the
  flow direction. Nodes that logically sit side-by-side (siblings doing parallel work, e.g. the
  agents and tools called by the same orchestrator) MUST share the same rank so they line up in a
  row. The flow generally goes: actor (rank 0) -> entry (rank 1) -> core logic (increasing ranks)
  -> external systems / outputs (highest ranks).
- Keep each rank to at most 4 nodes so rows stay readable; if a band has more, split into two ranks.
- Use "group" to place related nodes in a labelled box (e.g. "Application Core"). Nodes in the same
  group should occupy adjacent ranks so the box stays compact.

Output JSON in exactly this shape and nothing else:

{
  "direction": "TB",
  "nodes": [
    {"id": "user", "label": "Developer", "type": "actor", "group": null, "rank": 0},
    {"id": "api", "label": "API Layer", "type": "framework", "group": "Entry Point", "rank": 1},
    {"id": "orch", "label": "Orchestration", "type": "process", "group": "Application Core", "rank": 2}
  ],
  "edges": [
    {"from": "user", "to": "api", "label": "requests"},
    {"from": "api", "to": "orch", "label": "dispatches"}
  ]
}

Rules:
- id: short alphanumeric, unique. label: short (1-3 words). type: one of the six. rank: integer.
- Each edge connects two DIFFERENT ids that both exist. One edge per pair. Short verb labels.

Output ONLY the JSON object. No prose, no code fences."""


class DiagrammerAgent:
    """Produces a high-level architecture diagram: the LLM models it, code renders it."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="Architect")
        self._typed_agent = build_agent(instructions=TYPED_INSTRUCTIONS, name="ArchitectTyped")

    async def _build_typed_model(self, facts) -> dict:
        """Call the LLM ONCE and return the typed architecture model dict.
        All diagram_* methods share this so every format gets identical content."""
        raw = await run_agent(
            self._typed_agent,
            "Codebase facts (JSON):\n\n"
            + json.dumps(facts, indent=2, ensure_ascii=False)
            + "\n\nOutput the typed architecture model JSON now.",
        )
        text = strip_code_fence(raw)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return json.loads(text)

    async def _build_model(self, facts) -> dict:
        """Original mermaid model — kept for the diagram() / diagram_image() path."""
        raw = await run_agent(
            self._agent,
            "Codebase facts (JSON):\n\n"
            + json.dumps(facts, indent=2, ensure_ascii=False)
            + "\n\nOutput the architecture model JSON now.",
        )
        text = strip_code_fence(raw)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return json.loads(text)

    async def diagram(self, facts) -> str:
        """Render to Mermaid text."""
        model = await self._build_model(facts)
        return render_architecture_mermaid(model)

    async def diagram_image(self, facts, output_path, fmt: str = "png") -> str:
        """Render to png/svg image via diagrams library."""
        model = await self._build_model(facts)
        return render_architecture_diagram(
            model, output_path, fmt=fmt, title="Architecture"
        )

    async def diagram_typed(self, facts, output_path, fmt: str = "png") -> str:
        """Typed semantic-shape image (png/svg). One LLM call."""
        model = await self._build_typed_model(facts)
        return render_typed_architecture(
            model, output_path, fmt=fmt, title="Technical Architecture"
        )

    async def diagram_editable(self, facts, output_path) -> str:
        """Editable SVG via pure Graphviz shapes. Same model as all other typed formats."""
        model = await self._build_typed_model(facts)
        return render_typed_architecture_svg(
            model, output_path, title="Technical Architecture"
        )

    async def diagram_dot(self, facts, output_path) -> str:
        """Raw Graphviz DOT source. Same model as all other typed formats."""
        model = await self._build_typed_model(facts)
        return render_typed_architecture_dot(
            model, output_path, title="Technical Architecture"
        )

    async def diagram_drawio(self, facts, output_path) -> str:
        """Native draw.io XML with Graphviz-calculated layout. Same model as all other typed formats."""
        model = await self._build_typed_model(facts)
        return render_drawio_xml(
            model, output_path, title="Technical Architecture"
        )