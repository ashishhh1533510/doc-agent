"""
Per-diagram-type fidelity scoring tests — proves each diagram type is judged by its
OWN axes, an "excellent" model of each type scores High (>=85), a known-bad one
scores Low, and the dependency regression (the screenshot that scored 50) is fixed.

No pytest, no LLM, no network.
Run:  PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe tests/test_fidelity_scoring.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_agent.workflow.fidelity_scorer import compute_accuracy, _compose, _window

_FAILURES: list[str] = []
_PASS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
    else:
        _FAILURES.append(label + (f"  ({detail})" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Combiner
# ─────────────────────────────────────────────────────────────────────────────
def test_compose_weighted_mean_and_shape():
    out = _compose({"grounding": (1.0, 50), "coverage": (0.0, 50)}, ["n"])
    check("compose: weighted mean", out["score"] == 50, str(out["score"]))
    check("compose: breakdown values", out["breakdown"] == {"grounding": 100, "coverage": 0},
          str(out["breakdown"]))
    check("compose: weights echoed", out["weights"] == {"grounding": 50, "coverage": 50},
          str(out["weights"]))
    check("compose: order preserved (display order)",
          list(out["breakdown"].keys()) == ["grounding", "coverage"], str(out["breakdown"]))
    check("compose: notes carried", out["notes"] == ["n"], str(out["notes"]))


def test_compose_clamps_out_of_range_ratios():
    out = _compose({"a": (1.7, 50), "b": (-0.4, 50)}, [])
    check("compose: clamps high to 100", out["breakdown"]["a"] == 100, str(out["breakdown"]))
    check("compose: clamps low to 0", out["breakdown"]["b"] == 0, str(out["breakdown"]))


def test_window_inside_and_outside():
    check("window: inside is 1.0", _window(6, 5, 9) == 1.0)
    check("window: above hi decays", _window(18, 3, 9, floor=0.0) < 0.6, str(_window(18, 3, 9)))
    check("window: below lo decays", _window(1, 3, 9, floor=0.0) < 0.5, str(_window(1, 3, 9)))
    check("window: respects floor", _window(99, 3, 9, floor=0.3) == 0.3, str(_window(99, 3, 9, 0.3)))


# ─────────────────────────────────────────────────────────────────────────────
# Dependency — the regression + per-axis behaviour
# ─────────────────────────────────────────────────────────────────────────────
def _dep_facts():
    return {
        "components": [{"id": "orchestration_core"}, {"id": "job_orchestration"}],
        "import_graph": {"prefect/core/engine": [], "prefect/api/server": []},
        "files": [
            {"file": "prefect/api/server.py", "imports": ["fastapi", "sqlalchemy", "prefect.core"]},
            {"file": "prefect/core/engine.py", "imports": ["pydantic", "prefect.tasks"]},
        ],
    }


def _dep_model_good():
    return {
        "diagram_type": "dependency",
        "packages": [
            {"id": "orchestration_core", "label": "Orchestration Core", "kind": "internal"},
            {"id": "job_orchestration", "label": "Job Orchestration", "kind": "internal"},
            {"id": "ext_fastapi", "label": "fastapi", "kind": "external"},
            {"id": "ext_sqlalchemy", "label": "sqlalchemy", "kind": "external"},
            {"id": "ext_pydantic", "label": "pydantic", "kind": "external"},
        ],
        "edges": [
            {"from": "orchestration_core", "to": "job_orchestration", "label": "depends on"},
            {"from": "orchestration_core", "to": "ext_fastapi", "label": "uses"},
            {"from": "orchestration_core", "to": "ext_sqlalchemy", "label": "uses"},
            {"from": "job_orchestration", "to": "ext_pydantic", "label": "uses"},
        ],
    }


def test_dependency_regression_now_scores_high():
    """The screenshot case (clean internal->external 'uses' graph) scored 50 because
    dependency was routed through the component scorer with no model. It must now be High."""
    acc = compute_accuracy("dependency", facts=_dep_facts(), model=_dep_model_good(),
                           content="x", mermaid_validation={"valid": True})
    check("dep: coverage no longer 0 (the bug)", acc["breakdown"]["coverage"] > 0, str(acc["breakdown"]))
    check("dep: scored on dependency axes", "connectivity" in acc["breakdown"], str(acc["breakdown"]))
    check("dep: excellent dependency graph is High", acc["score"] >= 85, str(acc))
    check("dep: grade High", acc["grade"] == "High", acc["grade"])


def test_dependency_bad_scores_low():
    """Ungrounded packages + an orphan + wrong kind tag drop the score."""
    model = {
        "packages": [
            {"id": "orchestration_core", "label": "x", "kind": "external"},   # wrong kind
            {"id": "ext_madeuplib", "label": "madeuplib", "kind": "external"},  # ungrounded
            {"id": "ext_fastapi", "label": "fastapi", "kind": "external"},
        ],
        "edges": [{"from": "orchestration_core", "to": "ext_fastapi", "label": "uses"}],  # madeuplib orphan
    }
    acc = compute_accuracy("dependency", facts=_dep_facts(), model=model,
                           content="x", mermaid_validation={"valid": True})
    check("dep-bad: scores Low", acc["score"] < 70, str(acc))
    check("dep-bad: grounding penalized", acc["breakdown"]["grounding"] < 70, str(acc["breakdown"]))


# ─────────────────────────────────────────────────────────────────────────────
# C4 Combined / HLD
# ─────────────────────────────────────────────────────────────────────────────
def test_hld_excellent_high():
    vr = {"score": {"discovered_unit_count": 8, "container_count": 6, "orphan_count": 0,
                    "relationship_count": 7, "datastore_count": 1, "external_count": 1},
          "findings": []}
    acc = compute_accuracy("combined", facts={}, model={}, validation_report=vr, missing_ids=[])
    check("hld: connectivity axis present", "connectivity" in acc["breakdown"], str(acc["breakdown"]))
    check("hld: excellent is High", acc["score"] >= 85, str(acc))


def test_hld_disconnected_dump_low():
    """A raw module dump (no abstraction) with floating nodes and no relationships."""
    vr = {"score": {"discovered_unit_count": 20, "container_count": 18, "orphan_count": 8,
                    "relationship_count": 0, "datastore_count": 0, "external_count": 0},
          "findings": [{"level": "fail", "message": "x"}, {"level": "fail", "message": "y"}]}
    acc = compute_accuracy("combined", facts={}, model={}, validation_report=vr, missing_ids=[])
    check("hld-bad: scores Low", acc["score"] < 70, str(acc))
    check("hld-bad: connectivity tanks", acc["breakdown"]["connectivity"] < 50, str(acc["breakdown"]))
    check("hld-bad: abstraction penalized", acc["breakdown"]["abstraction"] <= 50, str(acc["breakdown"]))


# ─────────────────────────────────────────────────────────────────────────────
# Component
# ─────────────────────────────────────────────────────────────────────────────
def _comp_facts(n):
    return {"files": [{"file": f"f{i}.py", "classes": [{"name": f"C{i}"}]} for i in range(n)]}


def test_component_excellent_high():
    facts = _comp_facts(5)
    comps = [{"id": f"c{i}", "label": f"Comp{i}", "layer": "core",
              "member_files": [f"f{i}.py"], "owns_entities": [f"C{i}"],
              "interfaces": [f"Iface{i}"], "is_infra": i == 4} for i in range(5)]
    model = {"components": comps,
             "dependencies": [{"from": f"c{i}", "to": f"c{i+1}"} for i in range(4)]}
    acc = compute_accuracy("component", facts=facts, model=model, content="",
                           arch_checks={"ok": True, "warnings": []})
    check("comp: readability axis present", "readability" in acc["breakdown"], str(acc["breakdown"]))
    check("comp: excellent is High", acc["score"] >= 85, str(acc))


def test_component_hairball_penalized():
    """15 components, dense edges, no interfaces — still grounded, but readability +
    abstraction must drop hard and the overall score must fall well below excellent."""
    facts = _comp_facts(15)
    comps = [{"id": f"c{i}", "label": f"Comp{i}", "layer": "core",
              "member_files": [f"f{i}.py"], "owns_entities": [f"C{i}"],
              "interfaces": [], "is_infra": i < 3} for i in range(15)]
    deps = [{"from": f"c{i}", "to": f"c{j}"} for i in range(15) for j in range(i + 1, i + 4) if j < 15]
    model = {"components": comps, "dependencies": deps}
    acc = compute_accuracy("component", facts=facts, model=model, content="",
                           arch_checks={"ok": True, "warnings": []})
    check("comp-hairball: readability low", acc["breakdown"]["readability"] < 50, str(acc["breakdown"]))
    check("comp-hairball: abstraction low (no interfaces)",
          acc["breakdown"]["abstraction"] < 50, str(acc["breakdown"]))
    check("comp-hairball: well below excellent", acc["score"] < 80, str(acc))


# ─────────────────────────────────────────────────────────────────────────────
# Class
# ─────────────────────────────────────────────────────────────────────────────
def _class_facts():
    names = ["OrderService", "OrderRepository", "Order", "Payment", "Customer", "Invoice"]
    return {"files": [{"file": f"{n.lower()}.py",
                       "classes": [{"name": n, "methods": [{"name": "do"}], "bases": []}]}
                      for n in names]}


def test_class_excellent_high():
    content = """classDiagram
class OrderService
class OrderRepository
class Order
class Payment
class Customer
class Invoice
OrderService --> OrderRepository
OrderService --> Order
Order *-- Payment
Customer <|-- Invoice
Order --> Customer
"""
    acc = compute_accuracy("class", facts=_class_facts(), content=content)
    check("class: relationships axis present", "relationships" in acc["breakdown"], str(acc["breakdown"]))
    check("class: excellent is High", acc["score"] >= 85, str(acc))


def test_class_invented_and_isolated_low():
    """Two invented classes, one isolated, all loose dependency edges, low coverage."""
    facts = {"files": [{"file": f"{n.lower()}.py",
                        "classes": [{"name": n, "methods": [{"name": "do"}]}]}
                       for n in ["Order", "Payment", "Customer", "Invoice",
                                 "Cart", "Shipment", "Refund", "Ledger", "Tax", "Audit"]]}
    content = """classDiagram
class Order
class Payment
class Ghost
class Phantom
Order ..> Payment
Order ..> Ghost
"""  # Phantom isolated, Ghost+Phantom invented, only dependency edges
    acc = compute_accuracy("class", facts=facts, content=content)
    check("class-bad: grounding penalized", acc["breakdown"]["grounding"] < 70, str(acc["breakdown"]))
    check("class-bad: scores Low", acc["score"] < 70, str(acc))


# ─────────────────────────────────────────────────────────────────────────────
# Sequence
# ─────────────────────────────────────────────────────────────────────────────
def _seq_facts():
    return {"files": [{"file": "x.py", "classes": [
        {"name": "ApiService", "methods": [{"name": "createOrder"}]},
        {"name": "OrderService", "methods": [{"name": "validateOrder"}]},
        {"name": "OrderRepo", "methods": [{"name": "saveOrder"}]},
    ]}]}


def test_sequence_excellent_high():
    content = """sequenceDiagram
participant ApiService
participant OrderService
participant OrderRepo
ApiService->>OrderService: createOrder
OrderService->>OrderRepo: validateOrder
OrderRepo->>OrderRepo: saveOrder
OrderRepo-->>OrderService: return
OrderService-->>ApiService: return
ApiService-->>ApiService: return
"""
    acc = compute_accuracy("sequence", facts=_seq_facts(), content=content)
    check("seq: flow axis present", "flow" in acc["breakdown"], str(acc["breakdown"]))
    check("seq: excellent is High", acc["score"] >= 85, str(acc))


def test_sequence_no_return_and_too_many_participants_low():
    content = """sequenceDiagram
participant Zed
participant Yara
participant Xun
participant Wim
participant Vex
participant Uri
participant Tao
participant Sol
Zed->>Yara: frobnicate
Yara->>Xun: wibble
Xun->>Wim: splork
Wim->>Vex: gribble
"""  # 8 ungrounded participants, no return messages
    acc = compute_accuracy("sequence", facts=_seq_facts(), content=content)
    check("seq-bad: flow penalized (no return)", acc["breakdown"]["flow"] < 70, str(acc["breakdown"]))
    check("seq-bad: readability penalized (>6 participants)",
          acc["breakdown"]["readability"] < 80, str(acc["breakdown"]))
    check("seq-bad: scores Low", acc["score"] < 70, str(acc))


def _seq_view(prefix: str) -> dict:
    """A self-contained, readable sequence view: 6 distinct participants, 6 calls + a
    closing return. Distinct prefixes make each view's participants unique so a
    pooling scorer would (wrongly) sum them into one over-budget window."""
    parts = [f"{prefix}{n}" for n in ("A", "B", "C", "D", "E", "F")]
    lines = ["sequenceDiagram"] + [f"participant {p}" for p in parts]
    for i in range(5):
        lines.append(f"{parts[i]}->>{parts[i+1]}: step{i}")
    lines.append(f"{parts[0]}->>{parts[1]}: step5")
    lines.append(f"{parts[1]}-->>{parts[0]}: return")
    return {"content": "\n".join(lines)}


def test_sequence_per_view_scoring_not_pooled():
    """Three individually-readable views must score readability/flow HIGH. Under the
    old pooled scoring, 18 participants / 21 messages floored both axes to ~30-60."""
    diagrams = [_seq_view("P"), _seq_view("Q"), _seq_view("R")]
    acc = compute_accuracy("sequence", facts=_seq_facts(), diagrams=diagrams)
    check("seq-multi: readability per-view high",
          acc["breakdown"]["readability"] >= 85, str(acc["breakdown"]))
    check("seq-multi: flow per-view high",
          acc["breakdown"]["flow"] >= 85, str(acc["breakdown"]))


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
