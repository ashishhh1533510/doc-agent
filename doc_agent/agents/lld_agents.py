"""
LLD Diagram Agents: four specialised agents, one per diagram type.

Each agent receives RichFacts + ArchitectureContext and produces a typed JSON model.
Each also has a revise() method for the iteration loop in lld_pipeline.py.
"""

from collections import Counter
from doc_agent.integrations.llm_provider import build_agent, run_agent_json, compact_json

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

INSTRUCTIONS_SEQUENCE_REFINE_BATCH = """You are a software architect refining SEVERAL pre-traced sequence diagrams in a
single pass. You receive a JSON array of CandidateSequences — each with participants
and an ORDERED message list already extracted deterministically from the real call
graph — plus one shared ArchitectureContext.

POLISH each one independently, do NOT re-scope:
- Keep each sequence's given participants and message ORDER. You MAY drop a redundant
  message or merge two trivial consecutive self-calls, but NEVER invent a participant
  or message.
- Rewrite each message "label" as a short human verb phrase ("validate basket",
  "persist order") — one line, no quotes/semicolons/newlines.
- Set "name" to a concise human workflow title derived from the actual participants.
  Never generic.
- Keep each message "type" as given (sync/async/return).

CRITICAL OUTPUT CONTRACT:
- Return EXACTLY ONE refined workflow per input sequence, IN THE SAME ORDER.
- The "workflows" array length MUST equal the number of input CandidateSequences.

Return ONLY this JSON:
{ "workflows": [
  { "diagram_type":"sequence", "name":"<title>",
    "participants":["",...],
    "messages":[{"from":"","to":"","label":"","type":"sync|async|return"}] }
] }
"""


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

INSTRUCTIONS_CLASS_REFINE_BATCH = """You are a senior software architect refining SEVERAL focused UML class diagram
views in a single pass. You receive a JSON array of CandidateViews — each one a
small, pre-selected, already-bounded set of classes with candidate relationships —
plus one shared ArchitectureContext.

Apply the SAME per-view rules as for a single view, INDEPENDENTLY to each view:
- Work ONLY within each view's given classes. NEVER add a class not in that view.
- DROP only: test/mock/fixture classes; pure data carriers (DTO/Request/Response/*Args)
  with no behaviour; or a class with no real relationship to any other kept class IN
  THAT view.
- MAXIMUM 5 methods and 5 fields per class (keep a constructor if present).
- params: parameter NAMES only — never types, never defaults.
- type / return_type: ONE simple identifier. NO <>, [], |, =, dots, or "?".
- Every kept class MUST appear in at least one relationship; prefer structural edges
  (inheritance, realization, composition) over loose "dependency" edges.
- One edge per (from, to) pair. No self-loops.
- "label": a short human theme name derived from the ACTUAL class names. Never generic.

CRITICAL OUTPUT CONTRACT:
- Return EXACTLY ONE refined view per input view, IN THE SAME ORDER.
- The "views" array length MUST equal the number of input CandidateViews.

Return ONLY this JSON:
{ "views": [
  { "diagram_type":"class", "label":"<short name>",
    "classes":[{"name":"","fields":[{"name":"","type":"","visibility":"+"}],
                "methods":[{"name":"","params":"","return_type":"","visibility":"+"}]}],
    "relationships":[{"from":"","to":"","type":"inheritance|composition|aggregation|dependency|realization","label":""}] }
] }
"""


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


INSTRUCTIONS_COMPONENT_GROUP = """You are a senior software architect reading raw code clusters extracted from a repository.
Your job is to produce the component diagram a human architect would actually draw — one that
tells someone what this system DOES, not what folders exist in the repo.

You receive:
- "clusters": raw code clusters (id, layer, has_routes, has_db, is_infra, owns_entities, operation_evidence)
- "arch_context": repo-level architectural context

=== YOUR JOB ===
Group the clusters into 4-7 architectural subsystems and name each one.

STEP 1 — GROUP:
- Read ALL clusters together. Ask: which clusters are doing the same architectural job?
- Clusters in the same layer serving the same domain → ONE component
- A set of feature controllers all serving the same domain → ONE component
- A cluster that is just detail of another → absorb it
- The infra/platform cluster (is_infra=true) → keep as ONE separate component
- Only keep a cluster separate if it represents a genuinely DISTINCT subsystem
- You MUST cover every input cluster id in at least one absorbed_ids list
- Target: 4-7 components total

STEP 2 — NAME:
- Each component label must be an architectural concept a human would recognize
- Test: "If the implementation framework were replaced, would this name still apply?"
  If NO → rename it using operation_evidence alone
- Good: "Metadata Management", "Ingestion Pipeline", "Access Control", "Query Engine", "API Gateway"
- Bad: folder names, framework names, "Component 1", anything with Handler/Controller/Endpoint

STEP 3 — INTERFACE:
- One interface per component = the service contract it exposes to the rest of the system
- Good: "Metadata API", "Ingestion Service", "Authentication", "Search Interface"
- Bad: raw operation names, HTTP verb+path, lifecycle callbacks

=== CONSTRAINTS ===
- Every cluster id from the input MUST appear in exactly one absorbed_ids list — no cluster left out, no cluster in two groups
- id: use one of the absorbed cluster ids as the canonical id (the most representative one)
- layer: dominant layer across absorbed clusters
- is_infra: true only if ALL absorbed clusters are infra
- has_routes: true if ANY absorbed cluster has_routes
- has_db: true if ANY absorbed cluster has_db

Return ONLY this JSON:
{
  "grouped_components": [
    {
      "id": "<one of the absorbed cluster ids>",
      "label": "<architectural subsystem name>",
      "layer": "<presentation|application|domain|infrastructure|persistence>",
      "interfaces": ["<one service contract name>"],
      "absorbed_ids": ["<cluster_id_1>", "<cluster_id_2>", ...]
    }
  ]
}
"""


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
        self._batch_agent = build_agent(instructions=INSTRUCTIONS_CLASS_REFINE_BATCH, name="LLDClassBatch")

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

    async def refine_many(self, candidate_views: list, arch_context: dict) -> list | None:
        """Refine ALL views in a single LLM call (one request instead of N).

        Returns a list aligned 1:1 with candidate_views. Any element the model
        returns malformed falls back to that deterministic candidate view (no node
        is ever dropped — the structure is already final, only naming is polished).
        Returns None if the batch call fails or the array shape can't be trusted,
        so the caller can fall back to the per-view path.
        """
        if not candidate_views:
            return []
        prompt = (
            _GROUNDING
            + "CandidateViews (JSON array — refine EACH independently, same order):\n\n"
            + compact_json(candidate_views)
            + "\n\nArchitectureContext (JSON):\n\n"
            + compact_json(arch_context)
            + "\n\nRefine and name all views now."
        )
        out = await run_agent_json(self._batch_agent, prompt, fallback=None)
        if not isinstance(out, dict):
            return None
        refined = out.get("views")
        if not isinstance(refined, list) or len(refined) != len(candidate_views):
            return None
        result = []
        for cand, ref in zip(candidate_views, refined):
            if isinstance(ref, dict) and ref.get("classes"):
                ref.setdefault("label", cand.get("label", ""))
                result.append(ref)
            else:
                result.append(cand)  # deterministic candidate — already renderable
        return result



class SequenceDiagramAgent:
    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS_SEQUENCE, name="LLDSequence")
        self._refiner = build_agent(instructions=INSTRUCTIONS_SEQUENCE_REFINE, name="LLDSequenceRefine")
        self._batch_refiner = build_agent(instructions=INSTRUCTIONS_SEQUENCE_REFINE_BATCH, name="LLDSequenceRefineBatch")

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

    async def refine_many(self, candidates: list, arch_context: dict) -> list | None:
        """Refine ALL workflows in a single LLM call (one request instead of N).

        Returns a list aligned 1:1 with candidates. Any element the model returns
        malformed falls back to that deterministic candidate (participants/messages
        are already final — only labels are polished). Returns None if the batch
        call fails or the array shape can't be trusted, so the caller can fall back
        to the per-candidate path.
        """
        if not candidates:
            return []
        prompt = (
            _GROUNDING
            + "CandidateSequences (JSON array — refine EACH independently, same order):\n\n"
            + compact_json(candidates)
            + "\n\nArchitectureContext (JSON):\n\n" + compact_json(arch_context)
            + "\n\nRefine and name all workflows now."
        )
        out = await run_agent_json(self._batch_refiner, prompt, fallback=None)
        if not isinstance(out, dict):
            return None
        refined = out.get("workflows")
        if not isinstance(refined, list) or len(refined) != len(candidates):
            return None
        result = []
        for cand, ref in zip(candidates, refined):
            if isinstance(ref, dict) and ref.get("messages") and ref.get("participants"):
                ref.setdefault("name", cand.get("name", ""))
                result.append(ref)
            else:
                result.append(cand)  # deterministic candidate — already renderable
        return result

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
        self._agent   = build_agent(instructions=INSTRUCTIONS_COMPONENT,       name="LLDComponent")
        self._refiner = build_agent(instructions=INSTRUCTIONS_COMPONENT_REFINE, name="LLDComponentRefine")
        self._grouper = build_agent(instructions=INSTRUCTIONS_COMPONENT_GROUP,  name="LLDComponentGroup")

    async def refine(self, candidate_view: dict, arch_context: dict) -> dict:
        """Try architectural grouping first; fall back to naming-only if grouping fails."""
        result = await self._group_architecturally(candidate_view, arch_context)
        if result is None:
            result = await self._name_only_fallback(candidate_view, arch_context)
        return result

    @staticmethod
    def _merge_grouped(grouped_components: list, candidate_view: dict) -> dict:
        """Pure Python merge: union structural fields from absorbed clusters, re-derive edges."""
        from doc_agent.tools.component_arch import _capabilities
        by_id = {c["id"]: c for c in candidate_view.get("components", [])}

        cluster_to_group: dict[str, str] = {}
        for g in grouped_components:
            for cid in g.get("absorbed_ids", []):
                cluster_to_group[cid] = g["id"]

        components = []
        for g in grouped_components:
            absorbed = [by_id[cid] for cid in g.get("absorbed_ids", []) if cid in by_id]
            if not absorbed:
                continue

            # union member_files preserving order
            member_files: list = []
            seen_files: set = set()
            for c in absorbed:
                for f in (c.get("member_files") or c.get("members") or []):
                    if f not in seen_files:
                        seen_files.add(f)
                        member_files.append(f)

            owns_entities = sorted({
                e for c in absorbed for e in (c.get("owns_entities") or [])
            })[:8]
            evidence = {str(e).lower() for c in absorbed
                        for e in (c.get("operation_evidence") or [])}

            # layer: use LLM decision if valid, else dominant across absorbed
            layer = (g.get("layer") or "").strip()
            if layer not in ("presentation", "application", "domain", "infrastructure", "persistence"):
                layer_counts = Counter(c.get("layer", "application") for c in absorbed)
                layer = layer_counts.most_common(1)[0][0]

            has_routes = any(c.get("has_routes", False) for c in absorbed)
            has_db     = any(c.get("has_db",     False) for c in absorbed)
            is_infra   = all(c.get("is_infra",   False) for c in absorbed)

            # interface: validate LLM name against evidence, fall back to capability suffix
            cap_slots  = _capabilities(layer, has_routes, has_db)
            interfaces = cap_slots
            ifaces = g.get("interfaces") or []
            if ifaces:
                candidate_name = str(ifaces[0]).strip()
                if candidate_name and candidate_name.lower() not in evidence:
                    interfaces = [candidate_name]

            mod_counts      = Counter(c.get("module")       or "" for c in absorbed)
            modlabel_counts = Counter(c.get("module_label") or "" for c in absorbed)
            dominant_module       = mod_counts.most_common(1)[0][0]      if mod_counts      else ""
            dominant_module_label = modlabel_counts.most_common(1)[0][0] if modlabel_counts else ""

            components.append({
                "id":            g["id"],
                "label":         (g.get("label") or "").strip() or g["id"],
                "module":        dominant_module,
                "module_label":  dominant_module_label,
                "layer":         layer,
                "stereotype":    layer,
                "members":       member_files[:12],
                "member_files":  member_files,
                "member_count":  len(member_files),
                "is_infra":      is_infra,
                "has_routes":    has_routes,
                "has_db":        has_db,
                "owns_entities": owns_entities,
                "interfaces":    interfaces,
            })

        # re-derive edges: map original cluster-level edges through cluster_to_group
        edge_weights: dict[tuple, int] = {}
        for e in candidate_view.get("edges", []):
            src = cluster_to_group.get(e.get("from", ""))
            dst = cluster_to_group.get(e.get("to",   ""))
            if not src or not dst or src == dst:
                continue
            key = (src, dst)
            edge_weights[key] = edge_weights.get(key, 0) + e.get("weight", 1)

        group_ids = {g["id"] for g in grouped_components}
        dependencies = [
            {"from": src, "to": dst, "label": "requires", "weight": w}
            for (src, dst), w in sorted(edge_weights.items(), key=lambda kv: -kv[1])
            if src in group_ids and dst in group_ids
        ]

        return {
            "diagram_type": "component",
            "components":   components,
            "dependencies": dependencies,
            "packages":     candidate_view.get("packages", []),
        }

    async def _group_architecturally(self, candidate_view: dict, arch_context: dict) -> dict | None:
        """One LLM call: group ALL clusters into 4-7 architectural subsystems.
        Returns None if the response is invalid so the caller falls back."""
        clusters = candidate_view.get("components", [])
        if not clusters:
            return None
        all_ids = {c["id"] for c in clusters}

        lean_clusters = [
            {
                "id":                c["id"],
                "layer":             c.get("layer"),
                "has_routes":        c.get("has_routes", False),
                "has_db":            c.get("has_db",     False),
                "is_infra":          c.get("is_infra",   False),
                "owns_entities":     (c.get("owns_entities") or [])[:4],
                "operation_evidence": (c.get("operation_evidence") or [])[:10],
            }
            for c in clusters
        ]
        prompt = (
            "Clusters (JSON — group these into 4-7 architectural subsystems):\n\n"
            + compact_json({"clusters": lean_clusters})
            + "\n\nArchitectureContext (JSON):\n\n"
            + compact_json(arch_context)
            + "\n\nGroup and name the architectural subsystems now."
        )
        out = await run_agent_json(self._grouper, prompt, fallback=None)
        if not isinstance(out, dict):
            return None
        grouped = out.get("grouped_components")
        if not isinstance(grouped, list) or not (3 <= len(grouped) <= 8):
            return None

        # validate: every cluster id covered exactly once, no invented ids
        covered: set = set()
        for g in grouped:
            if not g.get("id") or not isinstance(g.get("absorbed_ids"), list):
                return None
            for cid in g["absorbed_ids"]:
                if cid in covered:
                    return None  # duplicate — cluster assigned to two groups
                covered.add(cid)
        if covered != all_ids:
            return None  # missing or invented cluster ids

        return self._merge_grouped(grouped, candidate_view)

    async def _name_only_fallback(self, candidate_view: dict, arch_context: dict) -> dict:
        """Original batch-naming logic — used when architectural grouping fails."""
        _BATCH_SIZE = 15
        all_components = candidate_view.get("components", [])
        named: dict = {}
        batches = [all_components[i:i + _BATCH_SIZE]
                   for i in range(0, max(1, len(all_components)), _BATCH_SIZE)]
        for batch in batches:
            lean_batch = [
                {
                    "id":                 c["id"],
                    "layer":              c.get("layer"),
                    "module_label":       c.get("module_label"),
                    "capabilities":       c.get("capabilities", []),
                    "members":            c.get("members", [])[:4],
                    "operation_evidence": c.get("operation_evidence", [])[:8],
                }
                for c in batch
            ]
            prompt = (
                _GROUNDING
                + "ComponentBatch (JSON — name each component, return same ids):\n\n"
                + compact_json({"components": lean_batch})
                + "\n\nArchitectureContext (JSON):\n\n"
                + compact_json(arch_context)
                + "\n\nName and refine these components now."
            )
            batch_out = await run_agent_json(self._refiner, prompt, fallback={"components": []})
            for c in batch_out.get("components", []):
                if c.get("id"):
                    named[c["id"]] = c

        components = []
        for c in candidate_view.get("components", []):
            nc = named.get(c["id"], {})
            slots = c.get("capabilities", [])
            evidence = {str(e).lower() for e in c.get("operation_evidence", [])}
            ifaces = nc.get("interfaces") or []
            interfaces = slots
            if len(ifaces) == len(slots) and slots:
                candidate_names = [str(x).strip() for x in ifaces]
                if all(name and name.lower() not in evidence for name in candidate_names):
                    interfaces = candidate_names
            components.append({
                "id":            c["id"],
                "label":         (nc.get("label") or "").strip() or (c.get("label") or "").strip() or c["id"],
                "module":        c.get("module"),
                "module_label":  c.get("module_label"),
                "layer":         c["layer"],
                "stereotype":    c.get("stereotype", c["layer"]),
                "members":       c.get("members", []),
                "member_files":  c.get("member_files", c.get("members", [])),
                "member_count":  c.get("member_count", len(c.get("members", []))),
                "is_infra":      c.get("is_infra", False),
                "has_routes":    c.get("has_routes", False),
                "has_db":        c.get("has_db", False),
                "owns_entities": c.get("owns_entities", []),
                "interfaces":    interfaces,
            })
        dependencies = [
            {"from": e["from"], "to": e["to"], "label": e.get("label", "requires"), "weight": e.get("weight", 1)}
            for e in candidate_view.get("edges", [])
        ]
        return {"diagram_type": "component", "components": components,
                "dependencies": dependencies, "packages": candidate_view.get("packages", [])}

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
