"""
HLD Enrichment Agent: text-only enrichment of the deterministic container model.

Receives the model produced by build_container_model() (node-set is already
fixed and immutable) plus the slim_facts snapshot for context. Returns ONLY:

  {
    "system_purpose":  "<3-6 word noun phrase>",
    "descriptions":    { "<node_id>": "<one sentence>" },
    "edge_labels":     { "<from_id>__<to_id>": "<verb phrase>" }
  }

The pipeline (hld_pipeline.py) merges these back by id via apply_enrichment().
Any structural invention (new ids, renames, splits) the LLM attempts is silently
discarded — drift back to capability-style boxes is impossible by design.
"""

from doc_agent.integrations.llm_provider import build_agent, run_agent_json, compact_json

INSTRUCTIONS = """You are a software architect writing display labels for a C4 architecture diagram.

You will receive:
1. container_model — the FINAL, FIXED set of nodes: containers, databases,
   external systems, and actors. This is authoritative and complete.
2. slim_facts — sampled codebase signals (routes, imports, class names)
   for context only, not as a source of new nodes.

==========================================================
YOUR ONLY JOB: write short text for labels in the diagram.
==========================================================

HARD RULES — violating any of these makes the output unusable:

1. You may NOT add, remove, split, merge, or rename any node.
   The node-set is fixed. Write labels for what exists; invent nothing.

2. You may NOT include node ids in "descriptions" or "edge_labels"
   that are not already present in the container_model.

3. "system_purpose" — a 3–6 word noun phrase naming what this system IS.
   Derive it from routes, class names, and framework signals in slim_facts.
   Do NOT copy the repository or folder name. Do NOT write a sentence.
   Examples of the FORM (not of actual systems):
     "E-Commerce Shopping Platform"
     "Real-Time Messaging Service"
     "Expense Tracking API"

4. "descriptions" — one sentence per node id, ≤ 15 words.
   Describe what the node DOES at runtime, not what technology it uses.
   Every id from the container_model's actors, containers, databases,
   and external_systems should have a description.

5. "edge_labels" — short verb phrases, 2–5 words.
   Write one for each relationship where a specific label helps.
   Use the from_id and to_id joined with double underscore: "from__to".
   Prefer: "reads/writes", "calls API", "sends email via", "uses".

==========================================================
OUTPUT FORMAT — return ONLY this JSON, nothing else:
==========================================================

{
  "system_purpose": "<3–6 word noun phrase>",
  "descriptions": {
    "<node_id>": "<one sentence, ≤ 15 words>"
  },
  "edge_labels": {
    "<from_id>__<to_id>": "<verb phrase>"
  }
}

Respond with ONLY the JSON. No preamble, no explanation, no markdown fences."""


class HLDEnrichmentAgent:
    """Fills text fields on the deterministic container model (system_purpose,
    per-node descriptions, edge labels). Cannot alter the node-set."""

    def __init__(self):
        self._agent = build_agent(instructions=INSTRUCTIONS, name="HLDEnrich")

    async def enrich(self, model: dict, slim_facts: dict) -> dict:
        """Return enrichment dict: {system_purpose, descriptions{}, edge_labels{}}."""
        prompt = (
            "Container model (node-set is FINAL — do not add or remove nodes):\n\n"
            + compact_json(model)
            + "\n\nSlim codebase facts (for naming context only):\n\n"
            + compact_json(slim_facts)
            + "\n\nReturn the enrichment JSON now."
        )
        return await run_agent_json(self._agent, prompt)
