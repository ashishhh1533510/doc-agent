"""
Deterministic tests for component rendering (flat layout) + model-based fidelity scoring.
No pytest, no LLM, no network.

Run:  ./venv/Scripts/python.exe tests/test_component_render_score.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.tools.output import render_component_view_set
from doc_agent.tools.view_planner import plan_single_view
from doc_agent.evaluation.fidelity_scorer import compute_accuracy

_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


def _model():
    return {
        "components": [
            {"id": "comp_00", "label": "Order Context", "layer": "domain",
             "stereotype": "domain", "members": ["order.py"], "member_count": 1,
             "has_routes": False, "has_db": False, "owns_entities": ["Order"],
             "interfaces": ["Data Access"]},
            {"id": "comp_01", "label": "Order API", "layer": "presentation",
             "stereotype": "presentation", "members": ["order_api.py"], "member_count": 1,
             "has_routes": True, "has_db": False, "owns_entities": [],
             "interfaces": ["API"]},
            {"id": "comp_02", "label": "Persistence", "layer": "infrastructure",
             "stereotype": "infrastructure", "members": ["db.py"], "member_count": 1,
             "has_routes": False, "has_db": True, "owns_entities": [], "is_infra": True,
             "interfaces": ["Persistence"]},
        ],
        "dependencies": [
            {"from": "comp_01", "to": "comp_00", "label": "requires", "weight": 2},
            {"from": "comp_00", "to": "comp_02", "label": "requires", "weight": 1},
        ],
        "packages": [],
    }


def _facts():
    return {"files": [
        {"file": "order.py", "classes": [{"name": "Order"}]},
        {"file": "order_api.py", "classes": []},
        {"file": "db.py", "classes": [{"name": "OrderRepo", "is_db_model": True}]},
    ]}


# ── flat render (no per-layer package rectangles) ────────────────────────────

def test_render_is_flat_no_packages():
    out = render_component_view_set(plan_single_view(_model(), []))
    content = out[0]["content"]
    check("render: produced one diagram", len(out) == 1, str(len(out)))
    check("render: no per-layer package rectangle", 'package "' not in content, content[:200])
    check("render: components present", content.count("component ") >= 3, content)
    check("render: lollipop interfaces present", 'interface "' in content, content)
    check("render: left-to-right direction", "left to right direction" in content)


def test_render_single_system_boundary():
    """Internal components live inside ONE system rectangle; the infra/platform
    component sits OUTSIDE it on the sink side (reference shape)."""
    out = render_component_view_set(plan_single_view(_model(), []))
    content = out[0]["content"]
    check("render: exactly one system rectangle", content.count('rectangle "') == 1, content[:300])
    # the infra component (comp_02) must be declared OUTSIDE the rectangle block
    inside = content.split("rectangle ", 1)[1].split("}", 1)[0]
    outside = content.split("}", 1)[1]
    check("render: domain comp inside boundary", "Order Context" in inside, inside)
    check("render: platform/infra comp outside boundary", "Persistence" in outside, outside[:300])


def test_render_external_datastore_wired():
    ext = [{"id": "ext_postgresql", "label": "PostgreSQL", "kind": "datastore",
            "stereotype": "database"}]
    out = render_component_view_set(plan_single_view(_model(), ext))
    content = out[0]["content"]
    check("render: external db node present", "PostgreSQL" in content, content[-400:])
    # external sinks render OUTSIDE the system boundary
    outside = content.split("}", 1)[1]
    check("render: external db outside boundary", "PostgreSQL" in outside, outside[:400])


# ── model-based fidelity scoring ─────────────────────────────────────────────

def test_score_coverage_nonzero():
    """Coverage must be > 0 when components own real entities / real files."""
    out = render_component_view_set(plan_single_view(_model(), []))
    acc = compute_accuracy("component", facts=_facts(), model=_model(),
                           content=out[0]["content"],
                           arch_checks={"ok": True, "warnings": []})
    check("score: coverage > 0", acc["breakdown"]["coverage"] > 0, str(acc["breakdown"]))
    check("score: grounding high (real members/entities)",
          acc["breakdown"]["grounding"] >= 90, str(acc["breakdown"]))
    check("score: overall well above the old 33 plateau", acc["score"] > 60, str(acc["score"]))


def test_score_grounding_penalizes_phantom():
    """A component with no real members and no real entity is not grounded."""
    m = _model()
    m["components"].append({"id": "comp_99", "label": "Ghost", "layer": "application",
                            "members": ["does_not_exist.py"], "member_count": 1,
                            "has_routes": False, "has_db": False, "owns_entities": ["Nope"]})
    acc = compute_accuracy("component", facts=_facts(), model=m, content="",
                           arch_checks={"ok": True, "warnings": []})
    check("score: phantom lowers grounding below 100",
          acc["breakdown"]["grounding"] < 100, str(acc["breakdown"]))


def test_score_uses_full_member_files_for_coverage():
    """members[] is display-capped at 12; coverage must score against full member_files."""
    facts = {"files": [{"file": f"f{i}.py", "classes": []} for i in range(20)]}
    # one component whose display members show only 2 files but really owns all 20
    m = {"components": [{
        "id": "comp_00", "label": "Big", "layer": "application",
        "members": ["f0.py", "f1.py"],
        "member_files": [f"f{i}.py" for i in range(20)],
        "has_routes": False, "has_db": False, "owns_entities": []}],
        "dependencies": [], "packages": []}
    acc = compute_accuracy("component", facts=facts, model=m, content="",
                           arch_checks={"ok": True, "warnings": []})
    check("score: coverage reflects full member_files (not the capped 2)",
          acc["breakdown"]["coverage"] >= 90, str(acc["breakdown"]))


def test_score_empty_model_safe():
    acc = compute_accuracy("component", facts=_facts(), model={"components": []},
                           content="", arch_checks={"ok": True, "warnings": []})
    check("score: empty model does not crash", isinstance(acc.get("score"), int), str(acc))


def main():
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for case in cases:
        try:
            case()
        except Exception as e:
            _FAILURES.append(f"{case.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{_PASS} checks passed, {len(_FAILURES)} failed")
    for fail in _FAILURES:
        print(f"  FAIL  {fail[:200]}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
