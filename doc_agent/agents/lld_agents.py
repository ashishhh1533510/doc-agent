"""
LLD Diagram Agents: four specialised agents, one per diagram type.

Each agent receives RichFacts + ArchitectureContext and produces a typed JSON model.
Each also has a revise() method for the iteration loop in lld_pipeline.py.
"""

from doc_agent.core.llm import build_agent, run_agent_json, compact_json

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
        + compact_json(rich_facts)
        + "\n\nArchitectureContext (JSON):\n\n"
        + compact_json(arch_context)
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

INSTRUCTIONS_CLASS_REFINE = """You are a senior software architect refining ONE focused UML class diagram view.
You receive a CandidateView — a small, pre-selected, already-bounded set of classes
with candidate relationships — plus ArchitectureContext.

Your job is to POLISH this view for READABILITY, NOT to re-scope it.

=== CLASS SELECTION (stay inside the candidate) ===
- Work ONLY within the given classes. NEVER add a class not in the candidate.
- DROP a class if it is: a test/mock/fixture class; a pure data carrier (DTO,
  Request, Response, *Args) with no behaviour; or has no real relationship to any
  other kept class in this view.
- Keep at most 12 classes. If more remain, keep the most architecturally central
  (services, repositories, aggregates, interfaces) and drop leaf data classes.

=== METHODS & FIELDS (HARD CAPS — keep boxes small) ===
- MAXIMUM 5 methods per class. No exceptions. Choose the most important PUBLIC
  methods (the ones other classes call); always keep a constructor/__init__ if present.
- MAXIMUM 5 fields per class. Prefer fields that are themselves other classes
  (dependencies) over primitive value fields.
- params: parameter NAMES only — never types, never defaults.
- type / return_type: ONE simple identifier. NO <>, [], |, =, dots, or "?".
  Write "Task", not "Task<Result<Basket>>"; write "IRepository", not "IRepository<Basket>".

=== RELATIONSHIPS (the value of the diagram) ===
- Every kept class MUST appear in at least one relationship. If it can't, drop it.
- Prefer structural edges (inheritance, realization, composition) over loose
  "dependency" edges. If a class IMPLEMENTS an interface in this view, emit a
  "realization" edge — do not omit it.
- Correct/enrich edges from the evidence; remove edges that aren't real design.
- One edge per (from, to) pair. No self-loops.

=== LABEL ===
- "label": a short human theme name derived from the ACTUAL class names
  (e.g. "Ordering", "Basket Checkout", "Catalog Caching"). NEVER generic like
  "Cluster 1" or "Diagram 2".

Return ONLY this JSON:
{ "diagram_type":"class", "label":"<short name>",
  "classes":[{"name":"","fields":[{"name":"","type":"","visibility":"+"}],
              "methods":[{"name":"","params":"","return_type":"","visibility":"+"}]}],
  "relationships":[{"from":"","to":"","type":"inheritance|composition|aggregation|dependency|realization","label":""}] }
"""


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

INSTRUCTIONS_SEQUENCE_REFINE = """You are a software architect refining ONE pre-traced sequence diagram.
You receive a CandidateSequence — participants and an ORDERED message list already
extracted deterministically from the real call graph — plus ArchitectureContext.

POLISH it, do NOT re-scope:
- Keep the given participants and message ORDER. You MAY drop a redundant message or
  merge two trivial consecutive self-calls, but never invent a participant or message.
- Rewrite each message "label" as a short human verb phrase describing intent
  (e.g. "validate basket", "persist order") — one line, no quotes/semicolons/newlines.
- Set "name" to a concise human workflow title derived from the actual participants
  (e.g. "Add Item to Basket", "Generate Class Diagram"). Never generic.
- Keep message "type" as given (sync/async/return).

Return ONLY this JSON:
{ "diagram_type":"sequence", "name":"<title>",
  "participants":["",...],
  "messages":[{"from":"","to":"","label":"","type":"sync|async|return"}] }
"""


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


INSTRUCTIONS_COMPONENT_REFINE = """You are a senior software architect NAMING a pre-computed UML component model.
The CandidateComponentView STRUCTURE is FROZEN: components (id, layer, stereotype, member
file ids) and classified dependency edges were derived deterministically from the import
graph. You must NOT re-scope.

Each component also carries:
- "capabilities": a count (1-2) of capability SURFACES it exposes — already decided
  deterministically from structural evidence (has a route surface? has persistence?).
  You name these; you NEVER change how many there are.
- "operation_evidence": a sample of raw method/route names — EVIDENCE ONLY, for you to
  read and summarize. NEVER copy an operation name as an interface name.

=== FROZEN — echo verbatim ===
- components[].id, .module, .module_label, .layer, .stereotype, .members.
  Packages (top-level groups) are named deterministically from the repo's real
  project/directory structure — that is intentional and outside your control.
  Never rename, regroup, merge, or reassign a component's module.
- dependencies (from/to/label). Never add, drop, reorder, or relabel an edge.
- Never add, remove, split, or merge a component.
- Never change the NUMBER of interfaces — exactly one name per given capability slot.

=== YOUR JOB — naming only ===
- components[].label: the architectural RESPONSIBILITY this component owns WITHIN its
  module — this is the one name you derive freely. DO NOT derive it from file paths,
  folder names, project names, or framework names (that rule applies to this label only;
  it does NOT apply to the module/package name, which is frozen and intentionally a real
  project name).
  Read operation_evidence to determine what domain actions occur here, then name the
  domain. Apply this test before finalising: "If the implementation framework were
  replaced with a different one, would this label still apply?" If NO, the label
  contains a technology artifact — drop it and re-derive from the operation evidence alone.
  Good labels are domain nouns: "Ordering", "Catalog", "Identity", "Administration",
  "Basket", "Payments", "Notifications", "Inventory", "Reporting".
  Never produce: "Web API", "REST Service", "Blazor App", "MVC Layer", "Spring Controllers",
  or any name whose meaning changes when the framework changes. Never generic ("Component 1").

- components[].interfaces: one capability NAME per given capability slot — a SERVICE
  CONTRACT, never a raw operation. Ask: "would this still make sense if the implementation
  technology changed?" Good: "Order Management", "Authentication", "Catalog Persistence",
  "Basket Operations". Bad — NEVER produce these shapes: a single CRUD verb+noun method
  name ("GetOrders"), an HTTP verb+path ("GET /orders/{id}"), a lifecycle/event callback
  ("OnInitialized", "RefreshBroadcast"), or any literal name copied from
  operation_evidence. Summarize the THEME of the operation_evidence sample for that
  component into one short capability phrase per slot.

Return ONLY this JSON:
{ "diagram_type":"component",
  "components":[{"id":"","label":"","layer":"","stereotype":"","members":[],"interfaces":[""]}],
  "dependencies":[{"from":"","to":"","label":""}] }
"""


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
        self._agent = build_agent(instructions=INSTRUCTIONS_CLASS_REFINE, name="LLDClass")

    async def refine(self, candidate_view: dict, arch_context: dict) -> dict:
        prompt = (
            _GROUNDING
            + "CandidateView (JSON):\n\n"
            + compact_json(candidate_view)
            + "\n\nArchitectureContext (JSON):\n\n"
            + compact_json(arch_context)
            + "\n\nRefine and name this view now."
        )
        out = await run_agent_json(self._agent, prompt)
        out.setdefault("label", candidate_view.get("label", ""))
        return out



class SequenceDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_SEQUENCE, name="LLDSequence")
        self._refiner = build_agent(instructions=INSTRUCTIONS_SEQUENCE_REFINE, name="LLDSequenceRefine")

    async def refine(self, candidate: dict, arch_context: dict) -> dict:
        prompt = (
            _GROUNDING
            + "CandidateSequence (JSON):\n\n" + compact_json(candidate)
            + "\n\nArchitectureContext (JSON):\n\n" + compact_json(arch_context)
            + "\n\nRefine and name this workflow now."
        )
        out = await run_agent_json(self._refiner, prompt)
        out.setdefault("name", candidate.get("name", ""))
        out.setdefault("participants", candidate.get("participants", []))
        out.setdefault("messages", candidate.get("messages", []))
        return out

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the sequence diagram JSON now."
        return await run_agent_json(self._agent, prompt)

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + compact_json(model)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected sequence diagram JSON fixing every issue."
        )
        return await run_agent_json(self._agent, prompt)



class ComponentDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_COMPONENT, name="LLDComponent")
        self._refiner = build_agent(instructions=INSTRUCTIONS_COMPONENT_REFINE, name="LLDComponentRefine")

    async def refine(self, candidate_view: dict, arch_context: dict) -> dict:
        prompt = (
            _GROUNDING
            + "CandidateComponentView (JSON):\n\n"
            + compact_json(candidate_view)
            + "\n\nArchitectureContext (JSON):\n\n"
            + compact_json(arch_context)
            + "\n\nName and refine this component view now."
        )
        out = await run_agent_json(self._refiner, prompt, fallback={"components": []})

        # Re-merge FROZEN structure: the LLM only NAMES; it cannot re-scope/relabel.
        named = {c.get("id"): c for c in out.get("components", [])}
        components = []
        for c in candidate_view.get("components", []):
            nc = named.get(c["id"], {})
            slots = c.get("capabilities", [])
            evidence = {str(e).lower() for e in c.get("operation_evidence", [])}
            ifaces = nc.get("interfaces") or []
            interfaces = slots                          # safe default: deterministic suffix
            if len(ifaces) == len(slots) and slots:
                candidate_names = [str(x).strip() for x in ifaces]
                # reject any name that is just a raw operation echoed back (abstraction-level guard)
                if all(name and name.lower() not in evidence for name in candidate_names):
                    interfaces = candidate_names
            components.append({
                "id": c["id"],
                "label": (nc.get("label") or "").strip() or c["id"],
                "module": c.get("module"),
                "module_label": c.get("module_label"),
                "layer": c["layer"],
                "stereotype": c.get("stereotype", c["layer"]),
                "members": c.get("members", []),
                "has_routes": c.get("has_routes", False),
                "has_db": c.get("has_db", False),
                "owns_entities": c.get("owns_entities", []),
                "interfaces": interfaces,
            })
        dependencies = [{"from": e["from"], "to": e["to"], "label": e.get("label", "requires"),
                         "weight": e.get("weight", 1)}
                        for e in candidate_view.get("edges", [])]
        return {"diagram_type": "component", "components": components, "dependencies": dependencies,
                "packages": candidate_view.get("packages", [])}

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the component diagram JSON now."
        return await run_agent_json(self._agent, prompt)

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + compact_json(model)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected component diagram JSON fixing every issue."
        )
        return await run_agent_json(self._agent, prompt)



class DependencyDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_DEPENDENCY, name="LLDDependency")

    async def analyze(self, rich_facts: dict, arch_context: dict) -> dict:
        prompt = _base_prompt(rich_facts, arch_context) + "\n\nProduce the dependency diagram JSON now."
        return await run_agent_json(self._agent, prompt)

    async def revise(self, rich_facts: dict, arch_context: dict, model: dict, issues: list[str]) -> dict:
        prompt = (
            _base_prompt(rich_facts, arch_context)
            + "\n\nPrevious model:\n" + compact_json(model)
            + "\n\nReviewer issues:\n- " + "\n- ".join(issues)
            + "\n\nReturn a corrected dependency diagram JSON fixing every issue."
        )
        return await run_agent_json(self._agent, prompt)
