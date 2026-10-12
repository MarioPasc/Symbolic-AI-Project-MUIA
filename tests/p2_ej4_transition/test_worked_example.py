"""Tests of the worked example (R1, ``worked_example.py``): the confirmed numbers of §7 R1."""

from __future__ import annotations

import json

import ej4_toys
import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary, load_pharmacokinetics
from symbolic_ai.p2_ej4_transition import EJ4_DATA_VERSION, SolverError, TransitionProblem
from symbolic_ai.p2_ej4_transition.instances import case_study_problem
from symbolic_ai.p2_ej4_transition.solve import ac3_domains
from symbolic_ai.p2_ej4_transition.worked_example import (
    ScheduleTable,
    WorkedExample,
    build_worked_example,
    horizon_domains,
    instance_data,
    value_text,
)


@pytest.fixture(scope="module")
def case_study() -> TransitionProblem:
    """The case study of design §1 from the real data (1.3.0)."""
    version = EJ4_DATA_VERSION
    return case_study_problem(
        load_formulary(version=version),
        load_drug_costs(version=version),
        load_pharmacokinetics(version=version),
    )


@pytest.fixture(scope="module")
def example(case_study: TransitionProblem) -> WorkedExample:
    """R1 on the case study, with the formulary's contraindications."""
    return build_worked_example(
        case_study, load_formulary(version=EJ4_DATA_VERSION).contraindications
    )


# --- the confirmed numbers of design §7 R1 (real data) -------------------------------------------


@pytest.mark.integration
def test_minimum_horizon_is_12(example: WorkedExample) -> None:
    assert example.h_star == 12
    assert [run.status for run in example.horizon_runs] == ["infeasible"] * 12 + ["solved"]
    assert [d.horizon for d in example.domains] == [11, 12]


@pytest.mark.integration
def test_ac3_proves_horizon_11_infeasible_without_search(example: WorkedExample) -> None:
    h11 = example.at(11)
    assert h11.ac3_consistent is False
    assert (h11.run.status, h11.run.assignments, h11.run.checks, h11.run.ac3_removed) == (
        "infeasible",
        0,
        1133,
        116,
    )
    assert h11.removed_by_ac3 == 116


@pytest.mark.integration
def test_ac3_empties_dep_day_2_at_horizon_11(
    example: WorkedExample, case_study: TransitionProblem
) -> None:
    h11 = example.at(11)
    assert h11.emptied == ("DEP", 2)
    assert h11.cell("DEP", 2).after_ac3 == ()
    # the trace's wipe-out is the first empty domain in static order
    _consistent, domains = ac3_domains(case_study, 11)
    assert next(v for v, values in domains.items() if not values) == ("DEP", 2)


@pytest.mark.integration
def test_ac3_at_horizon_12_leaves_one_value_except_pain_day_12(example: WorkedExample) -> None:
    h12 = example.at(12)
    assert h12.ac3_consistent is True and h12.emptied is None
    open_cells = {(c.condition_id, c.day): c.after_ac3 for c in h12.cells if len(c.after_ac3) != 1}
    assert open_cells == {("PAIN", 12): (("ibuprofen", 0), ("tramadol", 0))}
    assert (h12.run.status, h12.run.assignments, h12.run.checks, h12.run.ac3_removed) == (
        "solved",
        26,
        2070,
        133,
    )
    assert h12.removed_by_ac3 == 133


@pytest.mark.integration
def test_search_at_h_star_never_backtracks(example: WorkedExample) -> None:
    counts = dict(example.trace_counts)
    assert counts == {
        "ac3": 133,
        "select": 26,
        "try": 26,
        "prune": 0,
        "wipeout": 0,
        "backtrack": 0,
        "solution": 1,
    }
    assert [step.kind for step in example.trace].count("select") == 26
    assert example.trace[0].step == 133  # the search starts after the AC-3 events
    assert example.trace[-1].kind == "solution"


@pytest.mark.integration
def test_schedule_tapers_sertraline_then_starts_the_analgesic(example: WorkedExample) -> None:
    table = example.schedule
    dep = [value_text(v) for v in table.row("DEP")]
    pain = [value_text(v) for v in table.row("PAIN")]
    assert dep == ["sertraline", *(f"sertraline↓{k}" for k in range(1, 5))] + ["mirtazapine"] * 8
    assert pain == ["∅"] * 12 + ["ibuprofen"]
    assert example.valid and example.violations == ()


@pytest.mark.integration
def test_instance_rows(example: WorkedExample) -> None:
    data = example.instance
    assert data.problem_id == "DEP|LIVER|+cond:PAIN"
    assert (data.event_kind, data.event_target) == ("new_condition", "PAIN")
    assert [(r.drug_id, r.washout_days, r.taper_days) for r in data.drugs] == [
        ("mirtazapine", 4, 3),
        ("sertraline", 7, 4),
        ("ibuprofen", 1, 0),
        ("paracetamol", 1, 0),
        ("tramadol", 2, 2),
    ]
    assert [r.drug_id for r in data.drugs if not r.acceptable] == ["paracetamol"]
    assert [(c.risk_factor_id, c.drug_id) for c in data.contraindications] == [
        ("LIVER", "paracetamol")
    ]
    assert data.interactions == (("ibuprofen", "sertraline"), ("sertraline", "tramadol"))
    assert data.start == (("DEP", "sertraline"), ("PAIN", None))


@pytest.mark.integration
def test_to_dict_is_json_and_deterministic(case_study: TransitionProblem) -> None:
    contraindications = load_formulary(version=EJ4_DATA_VERSION).contraindications
    dumps = [
        json.dumps(build_worked_example(case_study, contraindications).to_dict(), sort_keys=True)
        for _ in range(2)
    ]
    assert dumps[0] == dumps[1]
    content = json.loads(dumps[0])
    assert content["valid"] is True
    assert content["domains"][0]["emptied"] == ["DEP", 2]


# --- toys (no data) ------------------------------------------------------------------------------


def test_toy_with_a_taper_has_domains_at_both_horizons() -> None:
    example = build_worked_example(ej4_toys.T2.problem)
    assert example.h_star == 3
    assert [d.horizon for d in example.domains] == [2, 3]
    assert example.at(2).ac3_consistent is False
    assert example.at(2).emptied is not None
    assert example.valid


def test_trivial_toy_has_only_horizon_0() -> None:
    example = build_worked_example(ej4_toys.T0.problem)
    assert example.h_star == 0
    assert [d.horizon for d in example.domains] == [0]
    assert example.instance.contraindications == ()


@pytest.mark.parametrize("toy", [ej4_toys.T4, ej4_toys.T6], ids=["T4", "T6"])
def test_no_transition_raises(toy: ej4_toys.KnownAnswer) -> None:
    with pytest.raises(SolverError):
        build_worked_example(toy.problem)


def test_removed_values_are_the_node_consistent_ones_ac3_dropped() -> None:
    domains = horizon_domains(ej4_toys.T3.problem, 4)
    for cell in domains.cells:
        assert set(cell.after_ac3) | set(cell.removed_by_ac3) == set(cell.node_consistent)
        assert not set(cell.after_ac3) & set(cell.removed_by_ac3)
    assert domains.removed_by_ac3 == domains.run.ac3_removed


def test_missing_cell_raises_key_error() -> None:
    domains = horizon_domains(ej4_toys.T1.problem, 1)
    with pytest.raises(KeyError):
        domains.cell("X", 5)
    with pytest.raises(KeyError):
        build_worked_example(ej4_toys.T1.problem).at(7)


def test_schedule_table_needs_every_key() -> None:
    with pytest.raises(KeyError):
        ScheduleTable.from_schedule(ej4_toys.T1.problem, {("X", 0): ("x1", 0)}, 1)


def test_instance_data_without_contraindications() -> None:
    data = instance_data(ej4_toys.T5.problem)
    assert data.contraindications == ()
    assert data.interactions == (("x1", "z1"), ("y1", "z1"))


@pytest.mark.parametrize(
    ("value", "text"),
    [(None, "∅"), (("sertraline", 0), "sertraline"), (("sertraline", 3), "sertraline↓3")],
)
def test_value_text(value: tuple[str, int] | None, text: str) -> None:
    assert value_text(value) == text
