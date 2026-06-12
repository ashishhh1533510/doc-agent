"""
LLD Diagram Agents: four specialised agents, one per diagram type.

Each agent receives RichFacts + ArchitectureContext and produces a typed JSON model.
Each also has a revise() method for the iteration loop in lld_pipeline.py.
"""

import json

from doc_agent.core.llm import build_agent, run_agent

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_model(text: str) -> dict:
    """Strip code fence and parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"LLD agent returned unparseable JSON: {text[:200]}") from e

_GROUNDING = """GROUNDING (applies to the whole diagram):
RichFacts includes deterministic `components` (id, files, fan_in/out, has_routes,
has_db_models), `component_edges`, and `architecture_signals` from the real import
graph — treat them as ground truth. Every component/participant/package you emit
must map to a given component id or a real class/file in RichFacts; never invent
one. Every dependency/message/edge must trace to component_edges, the import_graph,
or a call graph. Ungrounded nodes or edges will be rejected.

"""


def _base_prompt(rich_facts: dict, arch_context: dict) -> str:
    return (
        _GROUNDING
        + "RichFacts (JSON):\n\n"
        + json.dumps(rich_facts, indent=2, ensure_ascii=False)
        + "\n\nArchitectureContext (JSON):\n\n"
        + json.dumps(arch_context, indent=2, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Class Diagram Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS_CLASS = """You are a senior software architect producing a UML class diagram. You receive
RichFacts (static code analysis) and ArchitectureContext (repo-specific insights).

Produce a JSON object:
{
  "diagram_type": "class",
  "classes": [
    {
      "name": "<ClassName>",
      "fields": [{ "name": "<field>", "type": "<type or null>", "visibility": "+ or -" }],
      "methods": [{ "name": "<method>", "params": "<params>", "return_type": "<type or null>", "visibility": "+ or -" }]
    }
  ],
  "relationships": [
    { "from": "<ClassName>", "to": "<ClassName>", "type": "<inheritance|composition|aggregation|dependency|realization>", "label": "<optional short verb>" }
  ]
}

=== NAMING (CRITICAL) ===
- Use the EXACT class name from RichFacts, with its original capitalization, everywhere:
  in classes[].name AND in relationships[].from / relationships[].to.
  Endpoints must match a declared class name character-for-character.
- Never reference a class in relationships[] that is not declared in classes[].

=== CLASS SELECTION ===
- Include ONLY classes present in RichFacts.classes[]. Never invent classes.
- Pick the architecturally significant classes: orchestrators, services, adapters,
  core domain classes. Aim for 5-12 classes total.
- Request/response DTOs (e.g. Pydantic BaseModel subclasses): include only those that
  connect to a selected core class, and show their shared external base
  (e.g. BaseModel) as ONE class with empty fields/methods so the hierarchy reads clearly.
- A class with no evidence-backed relationship to any other selected class must be
  removed (unless it is the single central class of the system).

=== FIELDS & METHODS (keep boxes compact) ===
- Fields come from RichFacts fields[]; methods from RichFacts methods[].
- params: parameter names ONLY — never type hints, never default values.
  Write "source_dir, from_version", NOT "source_dir: Path, from_version: str = None".
- Field types and return types: one simple name with no brackets, unions, or defaults.
  Write "dict", NOT "dict[str, bool] | None".
- Skip dunder methods except __init__. Show at most 7 methods per class; prefer public ones.
- Visibility: "-" for names starting with underscore, "+" otherwise.

=== RELATIONSHIPS (the heart of the diagram) ===
Extract every relationship the evidence supports:
- inheritance: from bases[] (subclass --> base).
- composition: class A owns an instance of class B — B appears as a field type of A,
  or A instantiates B in __init__ (see calls[]).
- aggregation: A holds a reference to B it does not create (B passed in and stored).
- dependency: A's methods call B's methods or instantiate B transiently (see calls[]).
- realization: A implements an abstract base / protocol.
- Label non-inheritance edges with a short verb: "uses", "creates", "delegates to".
- One edge per (from, to, type) — no duplicates, no self-loops.
- EVERY declared class must appear in at least one relationship.

Respond with ONLY the JSON object, no other text."""

# ---------------------------------------------------------------------------
# Sequence Diagram Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS_SEQUENCE = """You are a software architect. You receive RichFacts (static code analysis)
and ArchitectureContext (repo-specific insights).

Produce a JSON object tracing the SINGLE most important end-to-end workflow:
{
  "diagram_type": "sequence",
  "name": "<workflow name>",
  "participants": ["<name1>", "<name2>", ...],
  "messages": [
    { "from": "<participant>", "to": "<participant>", "label": "<short call description>", "type": "<sync|async|return>" }
  ]
}

Rules:
- Start from the entry point in ArchitectureContext.key_workflows[0].
- Participants must be real components from RichFacts.components or real
  classes/route handlers from RichFacts. Use short names ("RuntimeManager",
  not "doc_agent.core.RuntimeManager").
- Aim for 3-6 participants and 6-15 messages: enough to tell the story, few enough to read.
- Every message's from/to MUST exactly match an entry in participants[].
- Messages must be traceable to routes[] (entry) and calls[] (internal steps).
- message type: "sync" for regular calls, "async" for awaited calls, "return" for responses.
  End the workflow with a "return" message back to the original caller.
- Labels: one short verb phrase on a single line — no quotes, semicolons, or newlines.
- Respond with ONLY the JSON object, no other text."""

# ---------------------------------------------------------------------------
# Component Diagram Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS_COMPONENT = """You are a software architect. You receive RichFacts (static code analysis)
and ArchitectureContext (repo-specific insights).

Produce a JSON object grouping modules into software components:
{
  "diagram_type": "component",
  "components": [
    {
      "id": "<slug>",
      "label": "<Component Name>",
      "tech": "<technology>",
      "layer": "<entry|core|infrastructure|external>",
      "interfaces": ["<interface or method exposed>"]
    }
  ],
  "dependencies": [
    { "from": "<id>", "to": "<id>", "label": "<verb phrase>" }
  ]
}

Rules:
- Use RichFacts.components (the deterministic component list) as your component list. Do not invent new ones.
- Group related modules (same folder or same responsibility) under one component.
- ids: lowercase letters, digits and underscores only (e.g. "api_layer").
- Every dependencies[].from and .to MUST exactly match a declared components[].id.
- No self-dependencies, no duplicate edges. Every component should appear in at
  least one dependency.
- Dependencies must be grounded in the import_graph from RichFacts.
- interfaces[] should list the public routes or methods each component exposes.
- Labels: short verb phrases on a single line — no quotes or semicolons.
- Respond with ONLY the JSON object, no other text.

=== LAYERS ===
Classify every component with one of these layers — the renderer uses it for layout:
- "entry"          : HTTP/CLI entrypoints (FastAPI app, CLI runner, route handlers)
- "core"           : orchestrators, domain logic, pipeline controllers
- "infrastructure" : adapters, LLM/DB/queue clients, internal utility libraries
- "external"       : third-party services or systems outside this repo that this
                     repo calls (external APIs, hosted DBs, SaaS services).
                     Do NOT use "external" for third-party Python libraries — those
                     belong in the dependency diagram, not here."""



# ---------------------------------------------------------------------------
# Dependency Diagram Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS_DEPENDENCY = """You are a software architect. You receive RichFacts (static code analysis)
and ArchitectureContext (repo-specific insights).

Produce a JSON object showing package-level dependencies:
{
  "diagram_type": "dependency",
  "packages": [
    { "id": "<slug>", "label": "<package name>", "kind": "<internal|external>" }
  ],
  "edges": [
    { "from": "<id>", "to": "<id>", "label": "<imports|depends on|calls>" }
  ]
}


Rules:
- Source rule: ONLY use packages that appear in RichFacts.imports[]. Do NOT infer
  from Dockerfile, package.json, requirements.txt, lock files, or any config file.
- Language rule: match the primary language of the repo. For a Python repo, only
  Python package names appear here — never npm/JavaScript packages.
- Internal packages: top-level sub-packages of this repo (from import_graph keys).
- External packages: third-party libraries in imports[] that are architecturally significant
  (framework, LLM client, DB driver, queue client). Exclude stdlib noise (json, os, re, etc.).
- ids: lowercase letters, digits and underscores only (e.g. "core", "agent_framework").
- Every edges[].from and .to MUST exactly match a declared packages[].id.
- No self-loops, no duplicate edges. Every package should appear in at least one edge.
- Edges come directly from import_graph for internal→internal, and from imports[] for internal→external.
- Labels: single line, no quotes or semicolons.
- Respond with ONLY the JSON object, no other text."""

# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class ClassDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_CLASS, name="LLDClass")

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the class diagram JSON now."
        return _parse_model(await run_agent(self._agent, prompt))

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + json.dumps(model, indent=2, ensure_ascii=False)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected class diagram JSON fixing every issue."
        )
        return _parse_model(await run_agent(self._agent, prompt))


class SequenceDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_SEQUENCE, name="LLDSequence")

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the sequence diagram JSON now."
        return _parse_model(await run_agent(self._agent, prompt))

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + json.dumps(model, indent=2, ensure_ascii=False)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected sequence diagram JSON fixing every issue."
        )
        return _parse_model(await run_agent(self._agent, prompt))


class ComponentDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_COMPONENT, name="LLDComponent")

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the component diagram JSON now."
        return _parse_model(await run_agent(self._agent, prompt))

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + json.dumps(model, indent=2, ensure_ascii=False)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected component diagram JSON fixing every issue."
        )
        return _parse_model(await run_agent(self._agent, prompt))


class DependencyDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_DEPENDENCY, name="LLDDependency")

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the dependency diagram JSON now."
        return _parse_model(await run_agent(self._agent, prompt))

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + json.dumps(model, indent=2, ensure_ascii=False)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected dependency diagram JSON fixing every issue."
        )
        return _parse_model(await run_agent(self._agent, prompt))
