"""
HLD Grounded Architect Agent (v3).

Receives the DETERMINISTIC candidate model (containers/infra already fixed by
build_candidate_model) plus slim codebase evidence, and returns the TWO-VIEW
C4 model by CLASSIFYING the candidate set — never inventing structure.

What it is allowed to do:
  - Decide which tier1 candidates are true containers vs should be folded
  - Name the system and write a one-line description (Context view)
  - Finalize external actors from the bounded vocab: User / Admin / Operator
  - Assign runtime edge labels grounded in import/ownership evidence
  - Write one-sentence node descriptions

What it MUST NOT do:
  - Add node ids that are not in the candidate model
  - Remove entire infra nodes (datastores/queues persist from deterministic scan)
  - Rename node ids (only label/description text)

The fallback (on JSON failure) returns the candidate model unchanged so the
pipeline always produces a renderable output.
"""
from __future__ import annotations

from doc_agent.integrations.llm_provider import build_agent, run_agent_json, compact_json

INSTRUCTIONS = """You are a software architect producing a C4 architecture diagram.

You receive:
1. candidate_model — the FINAL, FIXED node-set built deterministically from
   deployment evidence (Dockerfiles, boot manifests). Containers, databases,
   queues, and external services are identified. This is authoritative.
2. slim_facts — sampled codebase signals (routes, imports, class names, frameworks)
   for grounding your labels and descriptions.

==========================================================
YOUR JOB: classify and label — NEVER invent structure.
==========================================================

HARD RULES:

1. NEVER add node ids that do not appear in candidate_model.
   You may set labels/descriptions/labels. You may NOT create new id values.

2. NEVER remove datastore, queue, or external nodes — they were detected
   deterministically and represent real runtime dependencies.

3. system_name — a 3–6 word noun phrase naming what this system IS.
   Derive from routes, class names, and domain signals in slim_facts.
   Do NOT copy folder/repo names literally. Do NOT write a sentence.
   Examples of the FORM (not systems): "E-Commerce Platform", "Loan Management System"

4. actors — choose from: User, Admin, Operator, External System.
   Include "User" if any HTTP routes exist. Include "Operator" only if CLI-only.
   Use the actor ids already in the candidate_model (do not invent new ids).

5. descriptions — one sentence per node id, ≤ 15 words.
   What the node DOES at runtime. Every container/datastore/actor/external
   in the candidate_model should have a description.

6. labels — short noun phrase per node id (2–4 words), renaming the node to its
   business or architectural responsibility. Only include ids where you have a
   better label than the seed. Do NOT invent new ids.
   When domain_evidence is provided, use the listed routes and class names to name
   each node after its business responsibility. E.g. routes=["/loans","/repayments"]
   + classes=["Loan","LoanProduct"] → label "Loan Management". Return the same node
   id; only change the label text.

7. edge_labels — short verb phrases, 2–5 words.
   Use from_id__to_id (double underscore) as the key.

8. architecture_style — one of: monolith, microservices, modular-monolith,
   event-driven, serverless, layered. Pick the best fit from evidence.

9. groups — group CONTAINERS (not datastores/queues) into 3–6 logical capability
   domains that tell the collaboration story (e.g. Storefront, Ordering, Catalog,
   Payments, Platform). This is MANDATORY and the most important field for
   readability. Rules:
   - Every container node_id must appear in groups.
   - Datastores, queues, and external nodes are NOT grouped (leave them out).
   - Keep domain names short (1–3 words), business-meaningful, NOT layer names
     (never "Application", "Service", "Backend", "Frontend" as a domain).
   - Never invent or rename node ids — use the exact ids from candidate_model.
   - When there are 6 or more containers you MUST return at least 3 distinct
     domains. Do NOT put every container in one domain.
   - Worked example — if candidate_model contains containers
     [frontend, cartservice, checkoutservice, productcatalog, paymentservice,
      shippingservice, emailservice], a good grouping is:
       {"frontend": "Storefront", "cartservice": "Cart",
        "checkoutservice": "Ordering", "paymentservice": "Ordering",
        "shippingservice": "Fulfillment", "emailservice": "Fulfillment",
        "productcatalog": "Catalog"}
     i.e. collaborating services share a business domain (4 domains, not 7).

==========================================================
OUTPUT FORMAT — return ONLY this JSON, nothing else:
==========================================================

{
  "system_name": "<3–6 word noun phrase>",
  "architecture_style": "<style>",
  "labels": {
    "<node_id>": "<2–4 word noun phrase>"
  },
  "descriptions": {
    "<node_id>": "<one sentence, ≤ 15 words>"
  },
  "edge_labels": {
    "<from_id>__<to_id>": "<verb phrase>"
  },
  "groups": {
    "<container_node_id>": "<1–3 word domain name>"
  }
}

Respond with ONLY the JSON. No preamble, no explanation, no markdown fences."""


class HLDGroundedArchitect:
    """Constrained LLM architect that classifies/labels the deterministic candidate model.

    Uses run_agent_json with the candidate_model as fallback so the pipeline
    always produces a renderable output even if JSON parsing fails.
    """

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="HLDGroundedArchitect")

    async def classify(
        self,
        candidate_model: dict,
        slim_facts: dict,
    ) -> dict:
        """Return enrichment dict: {system_name, architecture_style, labels{}, descriptions{}, edge_labels{}}."""
        # Strip private "_"-prefixed working fields (e.g. _container_units, which can
        # be very large on a big repo). The LLM only needs the public context +
        # containers structure; these internals are pure wasted input tokens and can
        # push a single call over the Gemini free-tier per-minute quota.
        public_model = {k: v for k, v in candidate_model.items() if not k.startswith("_")}
        prompt = (
            "Candidate model (AUTHORITATIVE — do not add or remove nodes):\n\n"
            + compact_json(public_model)
            + "\n\nSlim codebase facts (for naming and description context only):\n\n"
            + compact_json(slim_facts)
            + "\n\nReturn the classification JSON now."
        )
        fallback = {
            "system_name": "",
            "architecture_style": "",
            "labels": {},
            "descriptions": {},
            "edge_labels": {},
            "groups": {},
        }
        result = await run_agent_json(self._agent, prompt, fallback=fallback)
        return result if isinstance(result, dict) else fallback
