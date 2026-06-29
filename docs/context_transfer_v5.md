# Context Handoff v5 — Folder Restructure DONE (uncommitted)

**Date:** 2026-06-29. Previous brief: `docs/context_transfer_v4.md` (the restructure ask).
This file = what actually happened + what's left.

---

## What was done this session
Reorganized the repo into the generic **production-grade Agentic-AI template**, using
**Option B**: keep the importable package name `doc_agent`, adopt the template's internal
layers (do NOT rename the package to `app/`). User chose **"map only what exists"** — no
empty placeholder directories were created.

### New layout (ground truth)
```
doc_agent/
  main.py            # FastAPI entry  (was api/app.py)
  index.html         # UI, beside main.py  (was api/index.html)
  agents/            # UNCHANGED  (arch_context, hld_*, lld_*, qa, reviewer, writer)
  tools/             # UNCHANGED  (extractor, manifest_parser, container_model, …, extractors/)
  orchestrators/     # pipeline, hld_pipeline, lld_pipeline, qa   (was workflow/)
  services/          # grounding.py                                (was workflow/grounding.py)
  evaluation/        # fidelity_scorer.py   (kept INSIDE pkg — runtime code) (was workflow/)
  integrations/      # llm_provider.py                             (was core/llm.py)
  observability/     # gcp_monitoring.py                           (was core/gcp_monitoring.py)
  retrieval/         # indexer.py                                  (was rag/indexer.py)
docs/                # context_transfer*.md (v1–v5)
pyproject.toml       # NEW (PEP 621 / setuptools)
requirements.txt     # KEPT (user wants both for now)
Dockerfile           # CMD updated → uvicorn doc_agent.main:app
```
Removed (now empty): old `api/ core/ rag/ workflow/` dirs.

### Import-path changes (memorize these)
- `doc_agent.api.app`            → `doc_agent.main`
- `doc_agent.workflow.*`         → `doc_agent.orchestrators.*` (pipeline, hld_pipeline, lld_pipeline, qa)
- `doc_agent.workflow.grounding` → `doc_agent.services.grounding`
- `doc_agent.workflow.fidelity_scorer` → `doc_agent.evaluation.fidelity_scorer`
- `doc_agent.core.llm`           → `doc_agent.integrations.llm_provider`
- `doc_agent.core.gcp_monitoring`→ `doc_agent.observability.gcp_monitoring`
- `doc_agent.rag.indexer`        → `doc_agent.retrieval.indexer`
~35 refs across 22 files rewritten. **0 residual** old paths (one harmless docstring
example `"doc_agent.core.RuntimeManager"` in `agents/lld_agents.py` left intentionally).

### Verified before handoff
- `./venv/Scripts/python.exe -c "import doc_agent.main"` → OK
- `index.html` resolves beside `main.py` → True
- All **8 test suites green** via project venv
- App boots: `GET /` → 200, serves the UI

---

## Git state (IMPORTANT — nothing committed)
- Base was clean (prior HLD/QA work already committed: `770963a`).
- The restructure is **UNCOMMITTED on `main`**:
  - File moves are **staged as renames** (history preserved).
  - `pyproject.toml` and `docs/*.md` are **UNTRACKED** — need `git add` before committing.
- User asked NOT to commit. When ready:
  `git add pyproject.toml docs/ && git commit` (suggest message: "Restructure into enterprise template layers").

---

## Open / deferred (not done, by choice)
- `requirements.txt` kept alongside `pyproject.toml` (user: "keep both for now"). Drop later if desired.
- Empty template layers NOT created: `config.py`, `dependencies.py`, `domain/`, `prompts/`,
  `memory/`, `security/`, `middleware/`, `data/`, `infra/`, `ci-cd/`, `frontend/`,
  `scripts/`, `api/v1/routes/`, request/response model files. Add only when there's real code.
- `index.html` left beside `main.py` (pure move, no logic change). Could relocate to a
  `frontend/` dir later if you split the UI out.
- Inline Pydantic request models still live in `main.py`; template would put them in
  `api/v1/request_models.py` / `response_models.py` — deferred.

---

## How to run / test (unchanged tooling)
- Run app: `./venv/Scripts/python.exe -m uvicorn doc_agent.main:app --reload`
- Tests (custom harness, NOT pytest — project `venv` has networkx, the Downloads venv does not):
  `for f in tests/test_*.py; do ./venv/Scripts/python.exe "$f"; done`
- Sanity: `./venv/Scripts/python.exe -c "import doc_agent.main; print('OK')"`

## Deploy reminders
- Render via Docker runtime; injects `$PORT` (keep `${PORT:-8000}`).
- Dockerfile installs graphviz binary + git (diagrams lib + GitPython clones) — don't drop.
