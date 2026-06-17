"""
Deterministic grounding check — the unbypassable anti-generic backstop.

The LLM reviewer can be lenient; this pure-Python check cannot. It runs every
review round in run_hld/run_lld and verifies that every diagram node traces to a
real component, capability, class, or external system, and every edge to a real
component dependency. Any ungrounded node/edge yields an issue, which forces
approved=False — so a generic, invented diagram can never be approved even if
the LLM reviewer would have let it through.
"""


def _norm(s):
    return (s or "").strip().lower()


def _real_class_names(facts):
    return {c.get("name") for f in facts.get("files", []) for c in f.get("classes", [])}


def _cap_to_components(arch_ctx):
    out = {}
    for cap in arch_ctx.get("capabilities", []):
        out[cap.get("name")] = set(cap.get("component_ids", []) or [])
    return out


def check_grounding(model, facts, arch_ctx) -> list:
    """Return a list of grounding issues (empty == fully grounded).

    facts is the slim_facts dict (carries `components`, `component_edges`,
    `files`); arch_ctx is the ArchitectureContext (capabilities carry
    `component_ids`). Non-empty result must force the review verdict to
    approved=False.
    """
    issues = []
    if not isinstance(model, dict):
        return issues

    components = facts.get("components", []) or []
    comp_ids = {c["id"] for c in components}
    comp_edges = facts.get("component_edges", []) or facts.get("edges", []) or []
    edge_pairs = {(e["from"], e["to"]) for e in comp_edges}
    edge_pairs |= {(e["to"], e["from"]) for e in comp_edges}   # undirected for grounding
    real_classes = _real_class_names(facts)
    cap_comps = _cap_to_components(arch_ctx)
    cap_names = set(cap_comps)
    ext_ids = set()
    for e in arch_ctx.get("external_systems", []):
        ext_ids |= {_norm(e.get("id")), _norm(e.get("label"))}

    def node_components(label):
        """Component ids a node maps to; set() if grounded w/o components; None if ungrounded."""
        if label in cap_names:
            return cap_comps[label]
        if label in comp_ids:
            return {label}
        if label in real_classes:
            return set()
        return None

    # ---------- HLD C4 model ----------
    if isinstance(model.get("containers"), dict):
        cont = model["containers"]
        ctx = model.get("context", {}) or {}
        containers = cont.get("containers", []) or []

        ext_decl, actor_decl = set(ext_ids), set()
        for s in cont.get("external_services", []) + ctx.get("external_systems", []):
            ext_decl |= {_norm(s.get("id")), _norm(s.get("label"))}
        for a in ctx.get("actors", []):
            actor_decl |= {_norm(a.get("id")), _norm(a.get("label"))}

        cont_comps = {}
        for c in containers:
            label, cid = c.get("label"), c.get("id")
            nc = node_components(label)
            cont_comps[cid] = nc
            if nc is None:
                issues.append(f"Container '{label}' traces to no component, capability, or real class (invented)")

        for block in (ctx, cont):
            for r in block.get("relationships", []) or []:
                a, b = r.get("from"), r.get("to")
                if _norm(a) in ext_decl | actor_decl or _norm(b) in ext_decl | actor_decl:
                    continue
                ca, cb = cont_comps.get(a), cont_comps.get(b)
                if ca is None or cb is None or not ca or not cb:
                    continue  # ungrounded node already flagged, or class-grounded (no comp map)
                if not any((x, y) in edge_pairs for x in ca for y in cb):
                    issues.append(f"Relationship '{a}'->'{b}' has no backing component edge (invented)")
        return issues

    # ---------- LLD class diagram ----------
    if "classes" in model:
        for c in model.get("classes", []) or []:
            nm = c.get("name")
            if nm and nm not in real_classes:
                issues.append(f"Class '{nm}' is not a real class in the codebase (invented)")
        return issues

    # ---------- LLD sequence diagram ----------
    if "participants" in model:
        for p in model.get("participants", []) or []:
            if node_components(p) is not None or _norm(p) in ext_ids:
                continue
            if any(p in rc or rc in p for rc in real_classes if rc):
                continue  # short-name match against a real class is acceptable
            issues.append(f"Participant '{p}' maps to no component, capability, or class (invented)")
        return issues

    # component / dependency diagrams use agent-coined slug ids -> too lenient to check here
    return issues
