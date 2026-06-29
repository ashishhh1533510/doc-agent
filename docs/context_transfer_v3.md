# Context Handoff — Session 2026-06-28

**Next session focus:** something related to a **QA agent** (new area — not started yet).
This file summarizes everything done *today* so we can resume cold.

> Status of all work below: **UNCOMMITTED on `main`**. The user has not asked for a commit yet.
> Test runner is the project venv: `./venv/Scripts/python.exe` (has networkx; the
> `Downloads/doc-agent/.venv` shell python does NOT — don't use it).
> Tests are custom-harness scripts (`check(label, cond, detail)`), run as
> `./venv/Scripts/python.exe tests/test_X.py` (exit code non-zero on failure; pytest sees no asserts).

---

## TL;DR of today

The headline work was making the **HLD (C4 Combined) diagram accurate AND readable on large
monorepos** (validated live on `github.com/microsoft/PowerToys`). Went from **88 (hairball) →
96/100 (High)** with a meaningful capability-level decomposition. Two smaller fixes (C# alias
noise, sequence per-view scoring) were also completed. Everything is deterministic and
**repo-agnostic** (no hardcoded repo/module names — audited via `git diff | grep`).

PowerToys HLD progression across live renders this session:
- **88** — hairball, abstraction 30 (one box per module)
- **95** — readable but WRONG containers (test projects admitted as containers)
- **96** — accurate capability domains, readable ✓ (final)

---

## All changes today (by file)

### 1. `doc_agent/tools/manifest_parser.py` — C# admission accuracy (the key diagram-accuracy fix)
`_csproj_is_deployable` had two deterministic bugs, confirmed against real PowerToys `.csproj`:
- It only matched `<OutputType>Exe` — but **Windows GUI apps are `<OutputType>WinExe`** (WPF/WinForms/WinUI). So every real module was REJECTED. → now also matches `WinExe`.
- **MSTest/xUnit test runners build as `Exe`** → were admitted as fake containers ("FancyZones UI Tests"). → new `_csproj_is_test()` excludes them (signals: `<IsTestProject>`, MSTest/xunit/nunit/`Microsoft.NET.Test.Sdk` package refs, or name suffix `*Tests/*UnitTests/*UITests/*FuzzTests/*Benchmarks`).
- Net effect: real WinExe GUI modules admitted; test projects + libraries excluded.

### 2. `doc_agent/tools/container_model.py` — abstraction / consolidation (the big one)
- **`build_candidate_model`**: now stores `model["_container_paths"] = {node_id: relpath}` (private field for path-based grouping).
- **`_path_domain(rel_path)` + `_GENERIC_ROOT_DIRS`** (new): derive a domain from a unit's path = the **TOPMOST non-generic ancestor folder** (skip leading `src/`, `packages/`, `services/`, `apps/`…). Groups deep-nested domain folders (`src/modules/launcher/Plugins/Calc` → `modules`) while keeping generic-root siblings standalone (`services/cart` → standalone). NOTE: a first attempt used *immediate parent* and it fragmented PowerToys into Plugins/previewpane/ext → fixed to topmost-ancestor.
- **`assign_domains`**: added path-based grouping as a higher-priority deterministic signal (used when it yields ≥2 groups with ≥1 multi-member group), before the old slug-prefix/layer fallback. Existing behavior preserved when `_container_paths` absent.
- **`consolidate_containers_for_abstraction(model)`** (new, wired into pipeline): when `container_count > 12`, fold each multi-member domain `group` into ONE representative container (C4-legal "group of containers as one"). Guards: ≥2 result groups, never collapse below 3, must reduce count. **Protects only the real ingress face** = the single `_pick_entrypoint` node + gateway nodes (NOT the whole presentation layer — important: GUI modules named `*UI` classify as presentation, so protecting the layer defeated folding). Remaps all edges + `_spine_edges` + `_candidate_ids`; drops self-loops; sets `model["_represented_unit_count"]`.
- **`validate_model`**: added `represented_unit_count` to score dict; Rule 2 (under-discovery) now judged on represented units, not raw container count (so consolidation isn't flagged as "dropping containers").

### 3. `doc_agent/workflow/fidelity_scorer.py` — coverage must not punish abstraction
- `_score_hld` `coverage` now uses `represented_unit_count` (= pre-consolidation count) instead of raw rendered container count. The old `containers/discovered` math *punished* the very abstraction the comment said it rewarded.
- (Earlier today) `_score_sequence`: **per-view scoring** fix — was pooling all 3 sequence views into one window and flooring Flow/Readability to 30. Now scores each view independently and averages. This was a SCORER bug, not generation.

### 4. `doc_agent/workflow/hld_pipeline.py` — wiring
- Imports + calls `consolidate_containers_for_abstraction(model)` in Stage 4, **after `assign_domains`, before `reduce_edges_for_readability`**.

### 5. `doc_agent/tools/extractors/csharp_extractor.py` — C# alias `using` noise (earlier today)
- `_using_name` now resolves alias directives to their TARGET: `using Deferral = Windows.Foundation.Deferral;` records `Windows.Foundation.Deferral` (→ filtered), not a bogus `Deferral` package. Strips `global::`. Fixes Deferral/Dispatcher noise in the dependency diagram.

### 6. `doc_agent/tools/dependency_graph.py` (earlier today)
- Added `_typeshed`, `windows`, `abi` to `_STDLIB_NOISE`; added `CommunityToolkit` to `_FRAMEWORK_MAP`.

### Tests added/updated (all green)
- `tests/test_container_model.py` → **319** (+ consolidation, path-domain, presentation-fold, edge-remap, over-collapse guard, represented-coverage end-to-end).
- `tests/test_manifest_parser.py` → **40** (WinExe deployable, test-project exclusion, library not deployable).
- `tests/test_fidelity_scoring.py` → **38** (sequence per-view).
- `tests/test_extractor_hld_signals.py` → **29** (C# alias using).
- `tests/test_dependency_graph.py` → **43** (noise filters/labels).

---

## Current test state (all 8 suites GREEN)
```
component_arch 39 · component_render_score 16 · container_model 319 · dependency_graph 43
extractor_hld_signals 29 · fidelity_scoring 38 · manifest_parser 40 · view_planner 34
```
Run all:
```
for f in tests/test_*.py; do ./venv/Scripts/python.exe "$f"; done
```

---

## Final PowerToys HLD outcome (live, verified)
- **96/100 High** — grounding 94, coverage 100, connectivity 94, abstraction 92, validity 100.
- Diagram shows ~13 real **capability domains** (Input & Productivity, File & Content, Configuration, Layout, System Utilities, Command Palette, Measure Tool, PowerOCR, …) + Workspaces Editor entrypoint + SQLite + OpenAI external.
- Abstraction 92 (not 100) is CORRECT: 13 capability domains is 1 over the soft readable window (12); forcing lower would merge distinct capabilities. We intentionally stopped here.

### How the final result is produced (the stack)
1. `manifest_parser` admits real WinExe modules, excludes tests → real modules only.
2. The **LLM (`HLDGroundedArchitect`) assigned coarse capability domains** (the one non-deterministic step) — it behaved well, grouping into ~13 capabilities not per-module.
3. `consolidate_containers_for_abstraction` folded by those domains.
4. `represented_unit_count` kept coverage at 100 despite folding.

---

## Known caveat / watch-item (only partly in our control)
`apply_enrichment` runs BEFORE `assign_domains` and sets each node's `group` from the **LLM**.
Our deterministic path-grouping + consolidation handle missing/coarse groups. **If a future
live render of some repo shows many unfolded boxes**, the cause is likely the LLM assigning
*distinct per-module groups* (so `assign_domains` won't re-group and consolidation folds by the
per-module groups). Fix-if-needed: have `assign_domains` override per-module LLM groups with the
deterministic path-domain grouping. On PowerToys the LLM grouped well, so we did NOT need this.

Minor cosmetic: a `.github/.../cache-generator` Exe tool can surface as a tiny standalone box
(not caught by `_NON_DEPLOYABLE_SEGMENTS`). Harmless; add `.github` to the filter if it bothers.

---

## Diagram-type status snapshot (for context)
- **HLD / C4 Combined** — DONE today, ship-ready on large monorepos (96 on PowerToys).
- **Dependency** — fixed earlier (noise filters + grounding 100); ship-ready.
- **Sequence** — scorer per-view fix done; diagrams were already accurate.
- **Class** (~91) and **Component** (~94) — already high, ship-ready.

---

## Useful facts for next session
- Extractors: csharp, ts, java (tree-sitter) + python (`ast`). **No Go extractor** — Go repos (e.g. grafana/loki) produce empty output; that's a known limitation, not a bug.
- HLD entry: `POST /generate/hld2`, UI action "C4 Combined", `doc_agent/workflow/hld_pipeline.py::run_hld`. LLM = Gemini 2.5 flash-lite (free tier), `doc_agent/core/llm.py`. Free tier wall ≈ 20 requests/day.
- The fidelity scorer (`doc_agent/workflow/fidelity_scorer.py`) is deterministic, never calls the LLM; per-diagram axes/weights; `_window(n, lo, hi, floor)` for "readable count".
- Repro/sim scripts used today live in the session scratchpad (synthetic monorepo repro + a faithful PowerToys sim that fetches all 196 csproj from GitHub raw and runs the deterministic chain). Re-creatable from the descriptions above if needed.
- Memory updated: `~/.claude/.../memory/project_hld_pipeline.md` has the full blow-by-blow.

## Open items
- Commit the stack if/when the user wants (single coherent change: manifest_parser + container_model + fidelity_scorer + hld_pipeline + tests). Suggested branch off `main`.
- No QA-agent code exists yet — that's net-new for the next session.
