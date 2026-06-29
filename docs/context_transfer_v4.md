# Context Handoff — Next Task: Generic Production-Grade Folder Structure

**Task:** Restructure this repo into a clean, generic, **production-grade folder layout** —
the kind you'd expect of a shippable Python service (clear package boundaries, proper
project metadata, no loose root junk, deterministic entry points, tests/docs/config in
their conventional homes). "Generic" = not doc-agent-specific conventions; follow standard
Python-packaging norms so a new engineer recognizes it instantly.

> Nothing for this task has been done yet. This file is the cold-start brief.
> Prior session work (HLD accuracy + QA-agent shelving) is **UNCOMMITTED on `main`** — see
> `context_transfer_v3.md`. Decide with the user whether to commit that first (clean base)
> before any move/rename churn, so the restructure diff stays readable.

---

## Why now / what "production-grade" should buy us
- A single, declared **build system** (`pyproject.toml`) instead of a bare `requirements.txt`.
- One obvious **entry point** and run story (currently `uvicorn doc_agent.api.app:app`).
- Root directory free of session artifacts and one-off scripts.
- Conventional homes: `src/` layout (optional), `tests/`, `docs/`, `scripts/`, `config/`.
- Imports and the Dockerfile still work unchanged (or are updated in lockstep).

---

## Current layout (ground truth, `git ls-files`)
Package root is `doc_agent/` (flat-in-root, NOT `src/`):
```
doc_agent/
  __init__.py
  agents/        arch_context, hld_*, lld_agents, lld_reviewer, qa, reviewer, writer
  api/           app.py (FastAPI), index.html (single-file UI), __init__
  core/          llm.py, gcp_monitoring.py (NEW, untracked)
  rag/           indexer.py
  tools/         extractor, manifest_parser, container_model, dependency_graph,
                 component_arch, view_planner, output, c4_views, diagram_*, … (large)
    extractors/  base, registry, python/ts/java/csharp_extractor
  workflow/      pipeline, hld_pipeline, lld_pipeline, grounding, qa, fidelity_scorer (NEW)
tests/           custom-harness scripts test_*.py (NOT pytest — see below)
```
Root-level files:
- Tracked & legit: `Dockerfile`, `.dockerignore`, `.gitignore`, `.env.example`, `requirements.txt`
- **Clutter to relocate/remove:** `context_transfer*.md` (x3), `_audit_probe.py`, `__pycache__/`
- `.env` exists on disk (gitignored — do NOT commit; confirm it's not tracked before moving)

---

## Hard constraints / things that WILL break if moved carelessly
1. **Entry point**: `Dockerfile` CMD is `uvicorn doc_agent.api.app:app`. Any package
   move (e.g. to `src/doc_agent/`) must update the Dockerfile CMD + any docs.
2. **Static UI path**: `api/app.py` serves `FileResponse(Path(__file__).parent / "index.html")`.
   `index.html` must stay beside `app.py` (or update that path).
3. **Imports are absolute** (`from doc_agent.workflow... import ...`) everywhere. A `src/`
   move keeps imports identical; a package *rename* would touch every file — avoid unless asked.
4. **Tests are a custom harness**, not pytest: each `tests/test_*.py` uses `check(label, cond,
   detail)` and exits non-zero on failure (pytest sees no asserts). Run with the **project venv**:
   `./venv/Scripts/python.exe tests/test_X.py`. The `Downloads/doc-agent/.venv` python LACKS
   networkx — don't use it. Run-all:
   `for f in tests/test_*.py; do ./venv/Scripts/python.exe "$f"; done`
5. **Dockerfile system deps** matter: graphviz binary (diagrams lib) + git (GitPython clones).
   Don't drop these when reworking build/deploy.
6. **Deploy target is Render** via the Docker runtime; it injects `$PORT`. Keep `${PORT:-8000}`.

---

## Suggested scope (confirm with user before executing — these are choices, not decisions)
- **`src/` layout?** Move `doc_agent/` → `src/doc_agent/`. Pro: standard, prevents accidental
  import-from-cwd. Con: touches Dockerfile + any tooling paths. (Imports unchanged.)
- **`pyproject.toml`**: introduce (PEP 621) with deps migrated from `requirements.txt`; pick a
  build backend (hatchling/setuptools). Keep `requirements.txt` as a generated lock or drop it.
- **`tests/`**: keep as-is, OR migrate the custom harness to real pytest (bigger task — flag it
  separately; do NOT silently rewrite). At minimum, add a `conftest.py`/`pytest.ini` only if migrating.
- **`docs/`**: move the three `context_transfer*.md` here (or `.handoff/`); they're session notes,
  arguably should be gitignored, not shipped. Ask.
- **`scripts/`**: home for `_audit_probe.py` and any one-off tooling (currently `_audit_*.py` is gitignored).
- **`config/` or `.env` handling**: standardize env loading (already uses python-dotenv).
- **Housekeeping**: ensure `__pycache__/` isn't tracked; `.env` not tracked.

---

## Recommended execution order (low-risk first)
1. (Optional) Commit the uncommitted HLD/QA work so the restructure is a clean isolated diff.
2. Housekeeping: relocate clutter (`context_transfer*`, `_audit_probe.py`), confirm gitignore covers them.
3. Add `pyproject.toml` (no moves yet); verify `pip install -e .` works in venv.
4. (If chosen) `src/` move via `git mv` to preserve history; update Dockerfile CMD + verify import.
5. Re-run ALL 8 test suites green + `python -c "import doc_agent.api.app"` + a local uvicorn smoke.
6. Update Dockerfile / docs to match; re-confirm `index.html` still served.

**Verify-after-every-step**: `./venv/Scripts/python.exe -c "import doc_agent.api.app; print('OK')"`
and the full test loop. Use `git mv` (not delete+add) so history follows the files.

---

## Open questions for the user (answer before big moves)
- `src/` layout yes/no?
- `pyproject.toml` only, or also kill `requirements.txt`? Which build backend?
- Migrate tests to pytest now, or keep the custom harness?
- Should `context_transfer*.md` be shipped (`docs/`) or gitignored session-only?
- Commit the pending HLD/QA changes first, or stack the restructure on top?

---

## Pointers
- Run app locally: `./venv/Scripts/python.exe -m uvicorn doc_agent.api.app:app --reload`
- Test runner: project `venv` only (has networkx). 8 suites, all green as of v3 handoff.
- Untracked-but-real files to fold in: `doc_agent/core/gcp_monitoring.py`,
  `doc_agent/workflow/fidelity_scorer.py`, and their tests.
- Full prior-session detail: `context_transfer_v3.md` (today's HLD work + QA shelving).
