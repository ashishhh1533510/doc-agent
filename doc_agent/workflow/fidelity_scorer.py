"""
Deterministic diagram-fidelity scorer — "how faithful is this diagram to the repo?"

Produces an accuracy score out of 100 with a transparent breakdown, so the user
trusts the diagram on-screen instead of validating it elsewhere. It re-uses the
ground truth the pipeline already extracts (deterministic components, real class
names from FileFacts, the import graph) and HLD's own validate_model() report; it
never calls an LLM.

=== Per-diagram-type axes (first-principles) ===
Each diagram type answers a DIFFERENT architectural question, so each has its OWN
named axes and weights — there is no single one-size-fits-all definition of
"quality". A score of 90 means "excellent execution of THIS diagram's purpose".

  C4 Combined : grounding 25 · coverage 20 · connectivity 25 · abstraction 15 · validity 15
  Component   : grounding 25 · coverage 20 · readability 20 · abstraction 15 · validity 20
  Dependency  : grounding 30 · coverage 30 · connectivity 20 · validity 20
  Class       : grounding 35 · coverage 20 · relationships 20 · readability 10 · validity 15
  Sequence    : grounding 30 · coverage 20 · flow 20 · readability 10 · validity 20

Common axes (grounding/coverage/validity) keep a consistent MEANING across types
(no hallucination / real surface captured / structurally sound) but are MEASURED
against the ground truth appropriate to each model. Type-specific axes
(connectivity, abstraction, readability, relationships, flow) actively PENALIZE the
failure modes unique to that diagram — orphans, hairballs, raw dumps, isolated
classes, an unclosed workflow — so quality, not just grounding, drives the number.

Every axis is structural and repository-agnostic (counts, windows, graph
properties, trace-to-ground-truth) — never name/keyword matching.
"""
import re

from doc_agent.tools.dependency_graph import (
    candidate_external_libs,
    internal_top_segments,
    MAX_INTERNAL_PACKAGES,
    MAX_EXTERNAL_PACKAGES,
)

# Component node budget — mirrors view_planner.MAX_COMPONENTS_SINGLE (=9): a
# component diagram beyond this stops being readable at a glance.
_COMPONENT_NODE_MAX = 9


def _norm(name: str) -> str:
    """Normalize an identifier for grounding comparison.

    Drops leading underscores and lowercases so a rendered ``RateLimiter`` still
    matches a real ``_RateLimiter`` — the LLM routinely strips the private-marker
    underscore and changes case when it relabels nodes, and an exact-match
    grounding check would wrongly score a faithful diagram as a hallucination.
    """
    return (name or "").lstrip("_").lower()


def _grade(score: int) -> str:
    if score >= 85:
        return "High"
    if score >= 70:
        return "Moderate"
    return "Low"


def _window(n: float, lo: float, hi: float, floor: float = 0.0) -> float:
    """Score a count 1.0 inside [lo, hi], decaying linearly to ``floor`` outside.

    The deterministic "right number of nodes" signal: too few (over-collapsed) and
    too many (a raw dump / hairball) both read worse than a count in the readable
    window. Below lo it scales by n/lo; above hi it sheds 1/hi per node over.
    """
    if n < 0:
        n = 0
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return max(floor, n / lo) if lo > 0 else 1.0
    over = n - hi
    span = hi if hi > 0 else 1
    return max(floor, 1.0 - over / span)


def _real_class_names(facts: dict) -> set:
    return {
        c.get("name")
        for f in facts.get("files", []) or []
        for c in f.get("classes", []) or []
        if c.get("name")
    }


def _significant_class_names(facts: dict) -> set:
    """Real classes worth showing — those with behaviour (methods) or a place in a
    hierarchy (bases). Pure data carriers with neither are legitimately omitted, so
    using this (not every class) as the coverage denominator stops a big repo from
    being punished for a correctly-selected, readable class view."""
    out = set()
    for f in facts.get("files", []) or []:
        for c in f.get("classes", []) or []:
            name = c.get("name")
            if name and (c.get("methods") or c.get("bases")):
                out.add(name)
    return out


def _component_keys(facts: dict) -> set:
    """Lowercased ids for the deterministic components (ground truth)."""
    keys = set()
    for c in facts.get("components", []) or []:
        cid = c.get("id")
        if cid:
            keys.add(str(cid).lower())
    return keys


def _runtime_file_set(facts: dict) -> set:
    """Lowercased paths of every file fact (the denominator for file coverage).
    Uses raw file paths rather than a separate runtime filter so it stays a pure,
    dependency-free signal — the model's members are a subset of these."""
    out = set()
    for f in facts.get("files", []) or []:
        fp = f.get("file")
        if fp:
            out.add(str(fp).replace("\\", "/").lower())
    return out


def _extract_mermaid_classes(content: str) -> set:
    """Class names declared or referenced in a Mermaid classDiagram."""
    names = set()
    # `class Foo {` / `class Foo`
    names |= set(re.findall(r"\bclass\s+([A-Za-z_]\w*)", content or ""))
    # relationship lines: Foo <|-- Bar, Foo --> Bar, Foo ..> Bar, Foo *-- Bar
    for a, b in re.findall(
        r"([A-Za-z_]\w*)\s*(?:<\|--|--\|>|\*--|o--|-->|\.\.>|--)\s*([A-Za-z_]\w*)",
        content or "",
    ):
        names.add(a)
        names.add(b)
    return {n for n in names if n}


# Class relationship connectors. STRUCTURAL ones (inheritance/composition/
# aggregation/realization) express real design; a loose `..>`/`-->` dependency does
# not. Longer tokens are listed first so the alternation matches greedily.
_CLASS_EDGE_RE = re.compile(
    r"([A-Za-z_]\w*)\s*(<\|--|--\|>|\.\.\|>|\.\.>|\*--|o--|-->|--)\s*([A-Za-z_]\w*)"
)
_STRUCTURAL_CONNECTORS = frozenset({"<|--", "--|>", "..|>", "*--", "o--"})


def _extract_class_edges(content: str) -> list:
    """List of (from, to, is_structural) for each relationship line."""
    out = []
    for a, conn, b in _CLASS_EDGE_RE.findall(content or ""):
        if a and b:
            out.append((a, b, conn in _STRUCTURAL_CONNECTORS))
    return out


def _extract_sequence_participants(content: str) -> set:
    """Participant/actor labels in a Mermaid sequenceDiagram (handles `as` aliases)."""
    names = set()
    for m in re.findall(r"\b(?:participant|actor)\s+(.+)", content or ""):
        # `participant API as Gateway` -> take the displayed/source token
        token = m.strip().split(" as ")[0].strip()
        token = token.strip('"')
        if token:
            names.add(token)
    return names


def _compose(axes: dict, notes: list) -> dict:
    """Combine an ORDERED mapping {axis_name: (ratio0..1, weight)} into the scored
    payload. Insertion order of ``axes`` is the display order in the UI.

    Returns {"score", "grade", "breakdown": {axis: 0-100}, "weights": {axis: w},
    "notes"}. The shape is identical across diagram types — only the axis SET and
    weights differ — so the frontend renders any per-type axis set unchanged.
    """
    breakdown = {}
    weights = {}
    for name, (ratio, weight) in axes.items():
        r = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
        breakdown[name] = round(r * 100)
        weights[name] = weight
    total = sum(weights.values()) or 1
    score = round(sum(breakdown[k] * weights[k] for k in breakdown) / total)
    return {
        "score": score,
        "grade": _grade(score),
        "breakdown": breakdown,
        "weights": weights,
        "notes": notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C4 Combined / HLD
# ─────────────────────────────────────────────────────────────────────────────
def _score_hld(facts, model, validation_report, missing_ids) -> dict:
    """C4 Combined: communicate the system, its major runtime units, and how they
    connect. Scored from HLD's purpose-built validate_model() summary."""
    notes = []
    vr = validation_report or {}
    sc = vr.get("score", {}) or {}
    findings = vr.get("findings", []) or []

    discovered = sc.get("discovered_unit_count", 0) or 0
    containers = sc.get("container_count", 0) or 0
    orphans = sc.get("orphan_count", 0) or 0
    rels = sc.get("relationship_count", 0) or 0
    datastores = sc.get("datastore_count", 0) or 0
    externals = sc.get("external_count", 0) or 0
    total = containers + datastores + externals

    # grounding: containers must trace to discovered units — a container count that
    # exceeds what was discovered means invented nodes.
    if containers and discovered:
        invented = max(0, containers - discovered)
        grounding = 1.0 - invented / containers
        if invented:
            notes.append(f"{invented} container(s) exceed discovered units (possible invention).")
    else:
        grounding = 1.0
    if missing_ids:
        grounding *= max(0.0, 1.0 - len(missing_ids) / max(total, 1))
        notes.append(f"{len(missing_ids)} model entity id(s) absent from rendered output.")

    # coverage: discovered architecture represented, capped (abstracting is good).
    # Units folded into a domain representative are still REPRESENTED, so credit
    # represented_unit_count (= pre-consolidation count) rather than the raw rendered
    # container count — otherwise grouping a hairball into domains, which abstraction
    # rewards, would be punished here.
    represented = sc.get("represented_unit_count", 0) or containers
    if discovered:
        coverage = min(1.0, represented / discovered)
        if represented < discovered:
            notes.append(f"{discovered - represented} discovered unit(s) not represented.")
    else:
        coverage = 1.0 if containers else 0.0

    # connectivity: floating nodes + edge sufficiency for a connected narrative spine.
    connectivity = 1.0 - (orphans / total) if total else 1.0
    if total > 1 and rels == 0:
        connectivity *= 0.5
        notes.append("No relationships between containers — diagram is disconnected.")
    elif total > 1 and rels < total - 1:
        # a connected graph on N nodes needs >= N-1 edges; fewer => fragments.
        connectivity *= max(0.4, rels / (total - 1))
        notes.append("Sparse relationships — some nodes can't reach the spine.")
    if orphans:
        notes.append(f"{orphans} floating node(s) with no connection.")

    # abstraction: container count in a readable window; penalize a raw module dump
    # (≈ one container per discovered unit when many were discovered).
    abstraction = _window(containers, 3, 12, floor=0.3)
    if discovered > 12 and containers and containers / discovered > 0.8:
        abstraction = min(abstraction, 0.5)
        notes.append("Little abstraction — nearly one container per discovered unit.")

    fails = sum(1 for f in findings if f.get("level") == "fail")
    warns = sum(1 for f in findings if f.get("level") == "warn")
    validity = max(0.0, 1.0 - 0.34 * fails - 0.1 * warns)
    for f in findings:
        notes.append(f"[{f.get('level')}] {f.get('message')}")

    return _compose({
        "grounding": (grounding, 25),
        "coverage": (coverage, 20),
        "connectivity": (connectivity, 25),
        "abstraction": (abstraction, 15),
        "validity": (validity, 15),
    }, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Class
# ─────────────────────────────────────────────────────────────────────────────
def _score_class(facts, diagrams, content) -> dict:
    """Class: the core types and how they relate. Grounding dominates (35) because
    an invented class is the gravest failure; relationships reward real design."""
    notes = []
    real_norm = {_norm(n) for n in _real_class_names(facts)}
    significant = {_norm(n) for n in _significant_class_names(facts)} or real_norm

    rendered = set()
    edges = []
    for d in (diagrams or [{"content": content}]):
        c = d.get("content", "")
        rendered |= _extract_mermaid_classes(c)
        edges += _extract_class_edges(c)

    # grounding
    if rendered:
        grounded = {n for n in rendered if _norm(n) in real_norm}
        grounding = len(grounded) / len(rendered)
        invented = rendered - grounded
        if invented:
            notes.append(f"{len(invented)} class node(s) not found in code: "
                         + ", ".join(sorted(invented)[:5]))
    else:
        grounding = 0.0
        notes.append("No class nodes detected in the diagram.")

    # coverage vs the SIGNIFICANT subset, capped so big repos aren't punished for a
    # correctly-bounded view.
    if significant:
        shown_sig = {_norm(n) for n in rendered} & significant
        denom = min(len(significant), 12) or 1
        coverage = min(1.0, len(shown_sig) / denom)
        notes.append(f"{len(shown_sig)} of {len(significant)} significant classes shown (target {denom}).")
    else:
        coverage = 1.0

    # relationships: every class connected + structural-edge richness.
    in_edge = set()
    structural = 0
    for a, b, is_struct in edges:
        in_edge.add(a)
        in_edge.add(b)
        if is_struct:
            structural += 1
    connected = (len({n for n in rendered if n in in_edge}) / len(rendered)) if rendered else 0.0
    struct_ratio = (structural / len(edges)) if edges else 0.0
    relationships = 0.7 * connected + 0.3 * struct_ratio
    isolated = len(rendered) - len({n for n in rendered if n in in_edge})
    if isolated:
        notes.append(f"{isolated} class(es) with no relationship.")
    if edges and struct_ratio == 0:
        notes.append("All relationships are loose 'dependency' edges — no structural design shown.")

    readability = _window(len(rendered), 5, 12, floor=0.3)
    if len(rendered) > 12:
        notes.append(f"{len(rendered)} classes exceed the readable window (12).")

    valid = [d.get("validation", {}).get("valid", True) for d in (diagrams or [])]
    validity = (sum(1 for v in valid if v) / len(valid)) if valid else 1.0

    return _compose({
        "grounding": (grounding, 35),
        "coverage": (coverage, 20),
        "relationships": (relationships, 20),
        "readability": (readability, 10),
        "validity": (validity, 15),
    }, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence
# ─────────────────────────────────────────────────────────────────────────────
# Mermaid sequence message line: `Actor -arrow- Actor : text`. A SOLID arrow
# (single dash: ->>, ->) is a call; a DASHED arrow (double dash: -->>, -->) is a
# return, which rarely maps to a named method — so only calls are scored for
# message grounding, while returns are counted toward flow completeness.
_SEQ_MSG_RE = re.compile(
    r"^\s*[A-Za-z_]\w*\s*(-{1,2})(?:>>?|x|\))\s*[+-]?\s*[A-Za-z_]\w*\s*:\s*(.+?)\s*$",
    re.M,
)


def _compact(s: str) -> str:
    """Lowercase, alphanumerics only — collapses `getKnexConnection` and the
    humanized `get knex connection` to the same string so they compare equal."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _sequence_call_corpus(facts: dict) -> set:
    """Compact names of every real method + call target in the repo.

    The sequence diagram's messages are derived from this same call graph, so a
    faithful (un-drifted) message label still compacts to one of these strings.
    Names shorter than 4 chars are dropped to avoid trivial substring hits.
    """
    verbs = set()

    def _add(name):
        v = _compact(name)
        if len(v) >= 4:
            verbs.add(v)

    def _add_call(call):
        # `self.getMeta(x)` / `agent.refine` -> last segment, sans args
        _add(str(call).split("(")[0].rsplit(".", 1)[-1])

    for f in facts.get("files", []) or []:
        for c in f.get("classes", []) or []:
            for m in c.get("methods", []) or []:
                _add(m.get("name"))
                for call in m.get("calls", []) or []:
                    _add_call(call)
        for fn in f.get("functions", []) or []:
            _add(fn.get("name"))
            for call in fn.get("calls", []) or []:
                _add_call(call)
    return verbs


def _extract_sequence_call_messages(content: str) -> list:
    """Texts of the call (solid-arrow) messages in a Mermaid sequenceDiagram."""
    out = []
    for dashes, text in _SEQ_MSG_RE.findall(content or ""):
        if len(dashes) == 1:  # solid arrow == call; dashed == return (skip)
            t = text.strip()
            if t:
                out.append(t)
    return out


def _message_grounded(text: str, corpus: set) -> bool:
    """True if a message label maps to some real method/call name in the repo."""
    t = _compact(text)
    if len(t) < 4:
        return False
    return any(v in t or t in v for v in corpus)


def _score_sequence(facts, arch_ctx, diagrams, content) -> dict:
    """Sequence: how one scenario flows through the system at runtime. Adds a
    `flow` axis — an entry trigger that closes with a return, sized to a readable
    message window — on top of grounding/coverage."""
    real = {n.lower() for n in _real_class_names(facts)}
    comp_keys = _component_keys(facts)
    ext = set()
    for e in (arch_ctx or {}).get("external_systems", []) or []:
        ext |= {str(e.get("id", "")).lower(), str(e.get("label", "")).lower()}
    resolvable = real | comp_keys | {e for e in ext if e}

    rendered = set()
    contents = []
    for d in (diagrams or [{"content": content}]):
        c = d.get("content", "")
        contents.append(c)
        rendered |= _extract_sequence_participants(c)
    notes = []

    def _is_grounded(p):
        pl = p.lower()
        if pl in resolvable:
            return True
        return any(r and (r in pl or pl in r) for r in resolvable)

    if rendered:
        grounded = {p for p in rendered if _is_grounded(p)}
        grounding = len(grounded) / len(rendered)
        if rendered - grounded:
            notes.append(f"{len(rendered - grounded)} participant(s) not traceable to code.")
    else:
        grounding = 0.0
        notes.append("No participants detected in the diagram.")

    # coverage: how faithfully the diagram's call MESSAGES map back to real methods
    # (label drift). Falls back to the participant-span proxy when there's no call
    # data or too few messages to judge.
    corpus = _sequence_call_corpus(facts)
    call_msgs = []
    for c in contents:
        call_msgs += _extract_sequence_call_messages(c)
    span = (min(1.0, len({p for p in rendered if _is_grounded(p)}) / 3.0)
            if rendered else 0.0)
    if corpus and len(call_msgs) >= 3:
        grounded_msgs = sum(1 for m in call_msgs if _message_grounded(m, corpus))
        coverage = grounded_msgs / len(call_msgs)
        notes.append(f"{grounded_msgs}/{len(call_msgs)} call message(s) map to real methods.")
    else:
        coverage = span

    # flow + readability are PER-VIEW window scores, averaged across the views — the
    # output is N separate sequence diagrams, so each must be judged on its own. (A
    # naive sum over all views would pool 3×16 messages / 3×6 participants into a
    # single-diagram window and wrongly floor every multi-workflow result.)
    flow_scores, read_scores = [], []
    worst_msgs = 0
    most_parts = 0
    views_no_return = 0
    for c in contents:
        msgs = 0
        rets = 0
        for dashes, _t in _SEQ_MSG_RE.findall(c):
            msgs += 1
            if len(dashes) == 2:
                rets += 1
        has_ret = rets > 0
        flow_scores.append(_window(msgs, 6, 15, floor=0.3) * (1.0 if has_ret else 0.6))
        if not has_ret:
            views_no_return += 1
        worst_msgs = max(worst_msgs, msgs)
        pc = len(_extract_sequence_participants(c))
        most_parts = max(most_parts, pc)
        read_scores.append(_window(pc, 3, 6, floor=0.3))
    flow = (sum(flow_scores) / len(flow_scores)) if flow_scores else 0.0
    readability = (sum(read_scores) / len(read_scores)) if read_scores else 0.0
    if views_no_return:
        notes.append(f"{views_no_return} view(s) don't close the loop with a return message.")
    if most_parts > 6:
        notes.append(f"Busiest view has {most_parts} participants, over the readable window (6).")
    if worst_msgs > 15:
        notes.append(f"Busiest view has {worst_msgs} messages, over the readable window (15).")

    valid = [d.get("validation", {}).get("valid", True) for d in (diagrams or [])]
    validity = (sum(1 for v in valid if v) / len(valid)) if valid else 1.0

    return _compose({
        "grounding": (grounding, 30),
        "coverage": (coverage, 20),
        "flow": (flow, 20),
        "readability": (readability, 10),
        "validity": (validity, 20),
    }, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Component
# ─────────────────────────────────────────────────────────────────────────────
def _score_component(facts, model, content, arch_checks, mermaid_validation) -> dict:
    """Component: how the system decomposes into cohesive components and the
    contracts between them. Adds readability (node window + hairball guard) and
    abstraction (every component has an interface; infra consolidated)."""
    notes = []
    comps = (model or {}).get("components", []) or []
    deps = (model or {}).get("dependencies", []) or []

    if not comps:
        # no model handed in — fall back to the old text-reference proxy.
        comp_keys = _component_keys(facts)
        shown = {k for k in comp_keys if k and k in (content or "").lower()}
        cov = (len(shown) / len(comp_keys)) if comp_keys else 1.0
        return _compose({
            "grounding": (max(cov, 0.6) if comp_keys else 1.0, 25),
            "coverage": (cov, 20),
            "readability": (1.0, 20),
            "abstraction": (1.0, 15),
            "validity": (1.0, 20),
        }, [f"{len(shown)}/{len(comp_keys)} discovered components referenced (text proxy)."])

    real_classes_norm = {_norm(n) for n in _real_class_names(facts)}
    real_files = _runtime_file_set(facts)

    # full membership for coverage; members[] is display-capped at 12 and would
    # drastically undercount file coverage on large components.
    def _members_of(c):
        return c.get("member_files") or c.get("members") or []

    def _grounded(c):
        mem = [str(m).replace("\\", "/").lower() for m in _members_of(c)]
        if any(any(m in rf or rf.endswith(m) for rf in real_files) for m in mem if m):
            return True
        return any(_norm(e) in real_classes_norm for e in (c.get("owns_entities") or []))

    grounded = [c for c in comps if _grounded(c)]
    grounding = len(grounded) / len(comps)
    if len(grounded) < len(comps):
        notes.append(f"{len(comps) - len(grounded)} component(s) not traceable to code.")

    owned = {_norm(e) for c in comps for e in (c.get("owns_entities") or [])}
    entity_cov = (len(owned & real_classes_norm) / len(real_classes_norm)
                  if real_classes_norm else 0.0)
    members = set()
    for c in comps:
        for m in _members_of(c):
            members.add(str(m).replace("\\", "/").lower())
    file_cov = (len({m for m in members if any(m in rf or rf.endswith(m)
                                               for rf in real_files)}) / len(real_files)
                if real_files else 0.0)
    coverage = max(entity_cov, file_cov)
    notes.append(f"coverage: {round(entity_cov*100)}% entities, {round(file_cov*100)}% files "
                 f"across {len(comps)} components.")

    # readability: node-count window + edge density (hairball guard).
    n = len(comps)
    node_score = _window(n, 3, _COMPONENT_NODE_MAX, floor=0.3)
    density = len(deps) / n if n else 0.0
    if density > 1.5:
        dens_score = max(0.3, 1.0 - (density - 1.5))
        notes.append(f"Dense edges ({len(deps)} for {n} nodes) — risk of a hairball.")
    else:
        dens_score = 1.0
    readability = min(node_score, dens_score)
    if n > _COMPONENT_NODE_MAX:
        notes.append(f"{n} components exceed the readable window ({_COMPONENT_NODE_MAX}).")

    # abstraction: every component should expose a contract; infra to one sink.
    with_iface = sum(1 for c in comps if (c.get("interfaces") or []))
    abstraction = with_iface / len(comps)
    infra = sum(1 for c in comps if c.get("is_infra"))
    if infra > 1:
        abstraction *= max(0.5, 1.0 - 0.2 * (infra - 1))
        notes.append(f"{infra} infrastructure sinks — consider consolidating to one.")
    if with_iface < len(comps):
        notes.append(f"{len(comps) - with_iface} component(s) expose no interface.")

    if arch_checks is not None:
        ok = arch_checks.get("ok", True)
        warns = len(arch_checks.get("warnings", []) or [])
        validity = 1.0 if ok else 0.6
        validity = max(0.0, validity - 0.05 * warns)
        for w in (arch_checks.get("warnings", []) or []):
            notes.append(f"[warn] {w}")
    elif mermaid_validation is not None:
        validity = 1.0 if mermaid_validation.get("valid", True) else 0.4
        for e in (mermaid_validation.get("errors", []) or []):
            notes.append(f"[invalid] {e}")
    else:
        validity = 1.0

    return _compose({
        "grounding": (grounding, 25),
        "coverage": (coverage, 20),
        "readability": (readability, 20),
        "abstraction": (abstraction, 15),
        "validity": (validity, 20),
    }, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency
# ─────────────────────────────────────────────────────────────────────────────
def _score_dependency(facts, model, content, mermaid_validation) -> dict:
    """Dependency: what the repo depends on, internally and externally. Scored on
    the package graph (packages[]/edges[]) — NOT on component member files — with a
    coverage denominator sized to the SAME significant-import set the builder uses.
    """
    notes = []
    pkgs = (model or {}).get("packages", []) or []
    edges = (model or {}).get("edges", []) or []

    if not pkgs:
        # no model handed in — degrade to a benign validity-only score.
        v = (mermaid_validation or {}).get("valid", True) if mermaid_validation else True
        return _compose({
            "grounding": (0.6, 30), "coverage": (0.0, 30),
            "connectivity": (0.0, 20), "validity": (1.0 if v else 0.4, 20),
        }, ["No dependency model available to score."])

    comp_ids = _component_keys(facts)
    internal_tops = internal_top_segments(facts)
    candidates = candidate_external_libs(facts)
    cand_keys = set(candidates.keys())
    cand_labels = {str(v).lower() for v in candidates.values()}

    def _is_ext(p):
        return str(p.get("id", "")).startswith("ext_") or str(p.get("kind", "")).lower() == "external"

    int_pkgs = [p for p in pkgs if not _is_ext(p)]
    ext_pkgs = [p for p in pkgs if _is_ext(p)]

    # grounding: internal pkgs trace to a real component id / module top; external
    # pkgs trace to a significant import the repo actually uses.
    def _int_grounded(p):
        pid = str(p.get("id", "")).lower()
        top = pid.split("/")[0]
        return pid in comp_ids or pid in internal_tops or top in comp_ids or top in internal_tops

    def _ext_grounded(p):
        key = re.sub(r"^ext_", "", str(p.get("id", "")).lower())
        lab = str(p.get("label", "")).lower()
        if key in cand_keys or lab in cand_labels:
            return True
        return any(key and (key in ck or ck in key) for ck in cand_keys)

    grounded = [p for p in int_pkgs if _int_grounded(p)] + [p for p in ext_pkgs if _ext_grounded(p)]
    grounding = len(grounded) / len(pkgs) if pkgs else 0.0
    if len(grounded) < len(pkgs):
        notes.append(f"{len(pkgs) - len(grounded)} package(s) not traceable to imports/import graph.")

    # coverage: internal + external representation, each sized to min(available, cap)
    # so filling the bounded budget with real packages counts as full coverage.
    target_int = min(MAX_INTERNAL_PACKAGES, max(1, len(internal_tops or comp_ids)))
    int_cov = min(1.0, len(int_pkgs) / target_int) if target_int else (1.0 if int_pkgs else 0.0)
    target_ext = min(MAX_EXTERNAL_PACKAGES, len(cand_keys))
    if target_ext:
        ext_cov = min(1.0, len(ext_pkgs) / target_ext)
    else:
        # repo has no significant external libs — showing none is correct, not a miss.
        ext_cov = 1.0
    coverage = (int_cov + ext_cov) / 2
    notes.append(f"coverage: {len(int_pkgs)} internal (target {target_int}), "
                 f"{len(ext_pkgs)}/{target_ext} significant external libs.")

    # connectivity: every package in >=1 edge; edge endpoints must resolve.
    ids = {str(p.get("id", "")) for p in pkgs}
    endpoint = set()
    dangling = 0
    for e in edges:
        fr, to = e.get("from"), e.get("to")
        if fr in ids:
            endpoint.add(fr)
        else:
            dangling += 1
        if to in ids:
            endpoint.add(to)
        else:
            dangling += 1
    orphans = [pid for pid in ids if pid not in endpoint]
    connectivity = (1.0 - len(orphans) / len(ids)) if ids else 0.0
    if dangling:
        connectivity *= 0.7
        notes.append(f"{dangling} edge endpoint(s) reference an undeclared package.")
    if orphans:
        notes.append(f"{len(orphans)} package(s) with no dependency edge.")

    # validity: mermaid soundness + correct internal/external kind tagging.
    if mermaid_validation is not None:
        validity = 1.0 if mermaid_validation.get("valid", True) else 0.4
        for er in (mermaid_validation.get("errors", []) or []):
            notes.append(f"[invalid] {er}")
    else:
        validity = 1.0
    kind_mismatch = 0
    for p in pkgs:
        is_ext_id = str(p.get("id", "")).startswith("ext_")
        kind = str(p.get("kind", "")).lower()
        if (is_ext_id and kind != "external") or (not is_ext_id and kind not in ("internal", "")):
            kind_mismatch += 1
    if kind_mismatch:
        validity = max(0.0, validity - 0.1 * kind_mismatch)
        notes.append(f"{kind_mismatch} package(s) with wrong internal/external kind.")

    return _compose({
        "grounding": (grounding, 30),
        "coverage": (coverage, 30),
        "connectivity": (connectivity, 20),
        "validity": (validity, 20),
    }, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
def compute_accuracy(
    diagram_type: str,
    *,
    facts: dict | None = None,
    arch_ctx: dict | None = None,
    model: dict | None = None,
    diagrams: list | None = None,
    content: str = "",
    validation_report: dict | None = None,
    missing_ids: list | None = None,
    arch_checks: dict | None = None,
    mermaid_validation: dict | None = None,
) -> dict:
    """Score how faithfully a generated diagram reflects the repo (0-100).

    Returns {"score", "grade", "breakdown": {<per-type axes>}, "weights", "notes"}.
    Each diagram type is routed to its own scorer with its own axis set; inputs are
    whatever the calling pipeline has in scope, and the function degrades gracefully
    on missing inputs (worst case: a benign neutral score, never an exception).
    """
    facts = facts or {}
    try:
        if diagram_type in ("combined", "hld", "hld2"):
            return _score_hld(facts, model or {}, validation_report, missing_ids or [])
        if diagram_type == "class":
            return _score_class(facts, diagrams, content)
        if diagram_type == "sequence":
            return _score_sequence(facts, arch_ctx or {}, diagrams, content)
        if diagram_type == "component":
            return _score_component(facts, model or {}, content, arch_checks, mermaid_validation)
        if diagram_type == "dependency":
            return _score_dependency(facts, model or {}, content, mermaid_validation)
        # unknown type: validity-only fallback
        v = (mermaid_validation or {}).get("valid", True)
        return _compose({"grounding": (1.0, 50), "coverage": (1.0, 30),
                         "validity": (1.0 if v else 0.4, 20)},
                        ["Unscored diagram type; validity only."])
    except Exception as e:  # never let scoring break a generation
        return {
            "score": 0,
            "grade": "Low",
            "breakdown": {"grounding": 0, "coverage": 0, "validity": 0},
            "weights": {"grounding": 50, "coverage": 30, "validity": 20},
            "notes": [f"accuracy scoring error: {e}"],
        }
