# Context Transfer — Per-Diagram-Type Fidelity Scoring (next session: TEST it)

**Branch:** `main` · **All changes UNCOMMITTED** · **Model under test:** gemini-2.5-flash-lite
(free tier, ~20 req/day) · **Date:** 2026-06-28

---

## 0. What we just did
Rewrote the **fidelity scorer** from a one-size-fits-all 3-axis model into a
**per-diagram-type, architecture-aware framework**. Each diagram type now has its OWN
named axes + weights, quality metrics **actively penalize** (not just annotate), and a
score of **90 means "excellent execution of THIS diagram's purpose"**. Fully
deterministic, repo-agnostic, no LLM in scoring, no name/keyword matching.

**Root bug that started this** (the dependency screenshot scoring 50/100): the
`"dependency"` call site never passed the `model`, so dependency was routed through the
component scorer (`_score_component_like`), `comps=[]` → text-proxy fallback →
grounding 0.6 / coverage 0 / validity 1.0 → `(60·50+0·30+100·20)/100 = 50`. FIXED.

---

## 1. The new per-type axes (all live in `doc_agent/workflow/fidelity_scorer.py`)
| Type | Axes (weights) | Scorer fn |
|---|---|---|
| C4 Combined (`combined`/`hld`/`hld2`) | grounding25 · coverage20 · connectivity25 · abstraction15 · validity15 | `_score_hld` |
| Component (`component`) | grounding25 · coverage20 · readability20 · abstraction15 · validity20 | `_score_component` |
| Dependency (`dependency`) | grounding30 · coverage30 · connectivity20 · validity20 | `_score_dependency` (NEW) |
| Class (`class`) | grounding35 · coverage20 · relationships20 · readability10 · validity15 | `_score_class` |
| Sequence (`sequence`) | grounding30 · coverage20 · flow20 · readability10 · validity20 | `_score_sequence` |

Shared scaffolding in the same file:
- `_compose(axes, notes)` — ordered `{name:(ratio0..1, weight)}` → `{score, grade, breakdown, weights, notes}`. Insertion order = UI display order. (Replaced old `_axis`.)
- `_window(n, lo, hi, floor)` — 1.0 inside the readable window, linear decay outside → the "right number of nodes" penalty (hairball / over-collapse / too-many-participants).
- Grounding helpers reused: `_norm`, `_real_class_names`, `_runtime_file_set`, `_component_keys`, plus NEW `_significant_class_names` (classes with methods/bases = class coverage denominator, so big repos aren't punished for a correctly-bounded view).

---

## 2. Files changed (all UNCOMMITTED on `main`)
1. **`doc_agent/workflow/fidelity_scorer.py`** — full rewrite (see above). `_score_component_like` split into `_score_component` + new `_score_dependency`; dispatch updated.
2. **`doc_agent/tools/dependency_graph.py`** — added two `code_dir`-free helpers so the
   dependency scorer agrees with the builder's selection (single source of truth):
   - `candidate_external_libs(rich_facts) -> {key: label}` — the UNBOUNDED significant-external set (same `_external_lib_key` + stdlib/internal filters the builder uses).
   - `internal_top_segments(rich_facts) -> set` — internal package roots from `import_graph` keys.
3. **`doc_agent/workflow/lld_pipeline.py`** (~line 334) — dependency call site now passes `model=model` (the bug fix).
4. **`doc_agent/api/index.html`** (`renderAccuracy`, ~752/778) — panel now iterates
   `Object.keys(a.breakdown)` with a `prettyAxis` formatter → renders 4–5 bars per type
   automatically (no longer hardcoded to grounding/coverage/validity).
5. **`tests/test_fidelity_scoring.py`** (NEW, 36 checks) — combiner unit tests + an
   "excellent" (≥85/High) and a "bad" (Low) fixture per type + the dependency regression.

---

## 3. Test status (deterministic, all GREEN — run with `PYTHONIOENCODING=utf-8`)
`test_fidelity_scoring 36/0`, `test_component_render_score 16/0`, `test_view_planner 34/34`,
`test_component_arch 39/0`, `test_container_model 293/0`, `test_dependency_graph 37/0`.
Run one: `PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe tests/test_fidelity_scoring.py`

Dispatch smoke (each type emits its own axes; dependency regression fixture = 100/High, was 50):
```
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -c "import doc_agent.workflow.fidelity_scorer as fs; print(fs.compute_accuracy('dependency', facts={}, model={}, content='', mermaid_validation={'valid':True})['breakdown'])"
```

---

## 4. NEXT SESSION — TEST THE FIDELITY SCORER (priority)
The deterministic unit tests pass; what's NOT yet done is **live end-to-end verification
through the real pipeline/UI**. Scoring never calls the LLM, so the scorer itself needs no
Gemini quota — but generating the diagram to feed it does.

1. **Run each diagram type E2E and read the fidelity panel** (UI `/generate/lld` &
   `/generate/hld2`, or `run_lld(repo, diagram_type)` / `run_hld(repo)` directly):
   - **dependency** on `https://github.com/PrefectHQ/prefect` — the screenshot case. Confirm
     the panel now shows **4 bars** (grounding / coverage / connectivity / validity) and a
     **High** score, not 50.
   - one each of **component / class / sequence / combined** — confirm each renders its OWN
     axis set and the score reflects readability/abstraction/flow, not just grounding.
2. **Sanity-check calibration**: a clean, readable diagram should land ~90; a deliberately
   degraded one (hand-edit the model to add orphans / a hairball / drop the return message)
   should visibly drop the RIGHT axis. The unit fixtures already assert this — the goal here
   is to confirm it holds on REAL repo data, where counts/windows aren't synthetic.
3. **Watch for**: real `rich_facts` shape mismatches — confirm the scorer reads the live
   fields (`facts["components"]`, `facts["import_graph"]`, `facts["files"][].imports`,
   `files[].classes[].methods`/`.bases`). If an axis is unexpectedly 0 on a good diagram, it's
   almost always a denominator/ground-truth field named/shaped differently than the fixture.
4. If a real dependency diagram still scores low on **coverage**, check
   `candidate_external_libs` vs what `build_dependency_model` actually selected — they share
   `_external_lib_key`, but the builder maps via members→components (needs `code_dir`) while
   the scorer walks all files directly; a large repo could surface a mismatch worth tightening.

## 5. Watch-outs / gotchas
- `extract_rich_from_directory` is in `doc_agent.tools.extractor` (NOT `extractors`).
- Free-tier Gemini ~20 req/day — a few E2E runs exhaust it; the deterministic fallback path
  still renders so you can validate structure/scoring without quota.
- Component empty-model fallback returns score 100 (text proxy, graceful degradation) — that's
  pre-existing behavior, not a scoring bug.
- Scorers never raise: `compute_accuracy` wraps everything in try/except → benign 0 score on
  error, so a silent low score may be hiding an exception in `notes` ("accuracy scoring error").
- Memory index: `…/memory/MEMORY.md`. Prior component-diagram work is in `project_session_jun25.md`.

## 6. Full uncommitted file list on `main`
Modified: `doc_agent/agents/lld_agents.py`, `doc_agent/api/app.py`, `doc_agent/api/index.html`,
`doc_agent/core/llm.py`, `doc_agent/tools/component_arch.py`, `doc_agent/tools/dependency_graph.py`,
`doc_agent/tools/extractors/python_extractor.py`, `doc_agent/tools/output.py`,
`doc_agent/tools/view_planner.py`, `doc_agent/workflow/fidelity_scorer.py`,
`doc_agent/workflow/hld_pipeline.py`, `doc_agent/workflow/lld_pipeline.py`, `requirements.txt`,
`tests/test_view_planner.py`.
Untracked: `doc_agent/core/gcp_monitoring.py`, `tests/test_component_arch.py`,
`tests/test_component_render_score.py`, `tests/test_fidelity_scoring.py`, `context_transfer.md`.
