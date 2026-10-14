"""Tests of the worked example (R1, ``worked_example.py``): the confirmed numbers of §7 R1."""

from __future__ import annotations

import json

import ej4_toys
import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary, load_pharmacokinetics
from symbolic_ai.p2_ej4_transition import EJ4_DATA_VERSION, SolverError, TransitionProblem
from symbolic_ai.p2_ej4_transition.experiments import R3_CONFIGS
from symbolic_ai.p2_ej4_transition.instances import case_study_problem
from symbolic_ai.p2_ej4_transition.oracle import oracle_min_horizon
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    SolverConfig,
    ac3_domains,
    solve_horizon,
)
from symbolic_ai.p2_ej4_transition.worked_example import (
    ScheduleTable,
    Walkthrough,
    WorkedExample,
    build_walkthrough,
    build_worked_example,
    daily_state_graph,
    horizon_domains,
    instance_data,
    miniature_problem,
    search_tree,
    value_text,
    walkthrough_text,
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
        1234,
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
        2264,
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


# --- the miniature of the slides (slides-and-guide.md §3, frozen numbers) -----------------------

S, S1, M = ("sertraline", 0), ("sertraline", 1), ("mirtazapine", 0)
IBU, TRA = ("ibuprofen", 0), ("tramadol", 0)


@pytest.fixture(scope="module")
def miniature(case_study: TransitionProblem) -> TransitionProblem:
    """The case study with every washout 1 day and every taper 0 days except sertraline's 1."""
    return miniature_problem(case_study)


@pytest.fixture(scope="module")
def walk(miniature: TransitionProblem) -> Walkthrough:
    """The walkthrough of the miniature, with the formulary's contraindications."""
    return build_walkthrough(miniature, load_formulary(version=EJ4_DATA_VERSION).contraindications)


@pytest.mark.integration
def test_miniature_shortens_every_washout_and_taper(
    case_study: TransitionProblem, miniature: TransitionProblem
) -> None:
    assert miniature.problem_id == "DEP|LIVER|+cond:PAIN|miniature"
    assert dict(miniature.washout_days) == dict.fromkeys(case_study.washout_days, 1)
    assert dict(miniature.taper_days) == {
        drug: 1 if drug == "sertraline" else 0 for drug in case_study.taper_days
    }
    for field in ("conditions", "start", "candidates", "acceptable", "interactions", "event"):
        assert getattr(miniature, field) == getattr(case_study, field)


@pytest.mark.integration
def test_miniature_h_star_is_3_for_csp_oracle_and_checker(walk: Walkthrough) -> None:
    assert (walk.h_star, walk.oracle_h_star, walk.state_graph.h_star) == (3, 3, 3)
    assert walk.violations == ()
    assert walk.schedule.row("DEP") == (S, S1, M, M)
    assert walk.schedule.row("PAIN") == (None, None, None, IBU)


@pytest.mark.integration
def test_miniature_horizon_loop(walk: Walkthrough) -> None:
    rows = [
        (s.run.horizon, s.run.status, s.run.assignments, s.run.checks, s.run.ac3_removed)
        for s in walk.loop
    ]
    assert rows == [
        (0, "infeasible", 0, 0, 0),
        (1, "infeasible", 0, 4, 2),
        (2, "infeasible", 0, 22, 6),
        (3, "solved", 8, 124, 9),
    ]
    assert [(s.node_emptied, s.ac3_emptied) for s in walk.loop] == [
        (("PAIN", 0), None),
        (None, ("PAIN", 1)),
        (None, ("DEP", 1)),
        (None, None),
    ]


@pytest.mark.integration
def test_miniature_node_consistent_domains(walk: Walkthrough) -> None:
    dep_middle, pain_middle = (S, S1, M), (IBU, TRA, None)
    expected = {
        2: {
            ("DEP", 0): (S,),
            ("PAIN", 0): (None,),
            ("DEP", 1): dep_middle,
            ("PAIN", 1): pain_middle,
            ("DEP", 2): (S, M),
            ("PAIN", 2): (IBU, TRA),
        },
        3: {
            ("DEP", 0): (S,),
            ("PAIN", 0): (None,),
            ("DEP", 1): dep_middle,
            ("PAIN", 1): pain_middle,
            ("DEP", 2): dep_middle,
            ("PAIN", 2): pain_middle,
            ("DEP", 3): (S, M),
            ("PAIN", 3): (IBU, TRA),
        },
    }
    assert [d.horizon for d in walk.domains] == [2, 3]
    for domains in walk.domains:
        cells = {(c.condition_id, c.day): c.node_consistent for c in domains.cells}
        assert cells == expected[domains.horizon]


@pytest.mark.integration
def test_miniature_ac3_at_h2_ends_in_the_wipe_out_of_dep_day_1(walk: Walkthrough) -> None:
    ac3 = walk.ac3[0]
    assert ac3.horizon == 2
    assert [(r.order, r.variable, r.value, r.neighbour, r.constraint) for r in ac3.removals] == [
        (1, ("PAIN", 1), IBU, ("DEP", 0), "I"),
        (2, ("PAIN", 1), TRA, ("DEP", 0), "I"),
        (3, ("DEP", 1), M, ("DEP", 0), "T"),
        (4, ("DEP", 2), S, ("PAIN", 2), "I"),
        (5, ("DEP", 1), S, ("PAIN", 2), "I"),
        (6, ("DEP", 1), S1, ("PAIN", 2), "I"),
    ]
    assert ac3.emptied == ("DEP", 1)
    assert walk.domains[0].cell("DEP", 1).after_ac3 == ()


@pytest.mark.integration
def test_miniature_ac3_at_h3_leaves_only_the_analgesic_open(walk: Walkthrough) -> None:
    ac3 = walk.ac3[1]
    assert ac3.horizon == 3
    assert [(r.variable, r.value, r.neighbour, r.constraint) for r in ac3.removals] == [
        (("PAIN", 1), IBU, ("DEP", 0), "I"),
        (("PAIN", 1), TRA, ("DEP", 0), "I"),
        (("DEP", 1), M, ("DEP", 0), "T"),
        (("DEP", 3), S, ("PAIN", 3), "I"),
        (("DEP", 2), S, ("PAIN", 3), "I"),
        (("DEP", 2), S1, ("PAIN", 3), "I"),
        (("PAIN", 2), IBU, ("DEP", 1), "I"),
        (("PAIN", 2), TRA, ("DEP", 1), "I"),
        (("DEP", 1), S, ("DEP", 2), "T"),
    ]
    assert ac3.emptied is None
    h3 = walk.domains[1]
    assert [h3.cell("DEP", day).after_ac3 for day in range(4)] == [(S,), (S1,), (M,), (M,)]
    assert [h3.cell("PAIN", day).after_ac3 for day in range(4)] == [
        (None,),
        (None,),
        (None,),
        (IBU, TRA),
    ]


@pytest.mark.integration
def test_miniature_configuration_table(walk: Walkthrough) -> None:
    table = [
        (row.key, [(r.status, r.assignments, r.checks, r.ac3_removed) for r in row.runs])
        for row in walk.configurations
    ]
    assert table == [
        ("bt", [("infeasible", 8, 41, 0), ("solved", 17, 120, 0)]),
        ("bt_fc", [("infeasible", 4, 19, 0), ("solved", 13, 101, 0)]),
        ("bt_fc_mrv", [("infeasible", 5, 27, 0), ("solved", 13, 108, 0)]),
        ("main", [("infeasible", 0, 22, 6), ("solved", 8, 124, 9)]),
    ]
    assert [row.label for row in walk.configurations] == [c.label for _k, c in R3_CONFIGS]


@pytest.mark.integration
def test_miniature_plain_bt_tree_has_17_assignments(walk: Walkthrough) -> None:
    assigned = [(s.number, s.variable, s.value) for s in walk.bt_tree if s.kind == "assign"]
    order = [
        (("DEP", 0), S),
        (("PAIN", 0), None),
        (("DEP", 1), S),
        (("PAIN", 1), None),
        (("DEP", 2), S),
        (("PAIN", 2), None),
        (("DEP", 3), S),
        (("DEP", 2), S1),
        (("PAIN", 2), None),
        (("DEP", 3), M),
        (("DEP", 1), S1),
        (("PAIN", 1), None),
        (("DEP", 2), M),
        (("PAIN", 2), None),
        (("DEP", 3), S),
        (("DEP", 3), M),
        (("PAIN", 3), IBU),
    ]
    assert assigned == [(n, v, x) for n, (v, x) in enumerate(order, start=1)]
    assert walk.bt_tree[-1].kind == "solution"
    # The first dead end (after assignment 7): both analgesics meet sertraline on day 3, and
    # mirtazapine cannot follow full-dose sertraline (T).
    after_7 = walk.bt_tree[[s.number for s in walk.bt_tree].index(7) + 1 :][:5]
    assert [(s.kind, s.variable, s.value) for s in after_7] == [
        ("reject", ("PAIN", 3), IBU),
        ("reject", ("PAIN", 3), TRA),
        ("backtrack", ("PAIN", 3), None),
        ("reject", ("DEP", 3), M),
        ("backtrack", ("DEP", 3), None),
    ]
    assert after_7[0].reason == "I with DEP day 2 = sertraline, I with DEP day 3 = sertraline"
    assert after_7[3].reason == "T with DEP day 2 = sertraline"


@pytest.mark.integration
def test_miniature_main_search_is_a_straight_path(walk: Walkthrough) -> None:
    assert [s.kind for s in walk.main_path] == ["assign"] * 8 + ["solution"]
    assert [s.depth for s in walk.main_path[:8]] == list(range(8))
    assert walk.main_path[7].variable == ("PAIN", 3)
    assert walk.main_path[7].value == IBU


@pytest.mark.integration
def test_miniature_daily_state_graph(walk: Walkthrough, miniature: TransitionProblem) -> None:
    graph = walk.state_graph
    states = {n.label: (n.depth, n.state.values, n.state.days_left) for n in graph.states}
    assert states == {
        "A": (0, (S, None), (("sertraline", 1),)),
        "B": (1, (S1, None), (("sertraline", 1),)),
        "C": (2, (M, None), (("mirtazapine", 1),)),
        "D": (3, (M, IBU), (("ibuprofen", 1), ("mirtazapine", 1))),
        "E": (3, (M, TRA), (("mirtazapine", 1), ("tramadol", 1))),
    }
    assert [(e.source, e.target, e.kind) for e in graph.edges] == [
        ("A", "A", "repeated"),
        ("A", "B", "new"),
        ("B", "C", "new"),
        ("C", "C", "repeated"),
        ("C", "D", "goal"),
        ("C", "E", "goal"),
        ("C", "A", "repeated"),
    ]
    oracle = oracle_min_horizon(miniature)
    assert graph.states_expanded == oracle.states_expanded == 3
    assert graph.path == ("A", "B", "C", "D")
    witness = {
        (condition, day): value
        for day, label in enumerate(graph.path)
        for condition, value in zip(graph.conditions, states[label][1], strict=True)
    }
    assert witness == oracle.schedule
    assert [walk.schedule.rows[i][day] for day in range(4) for i in range(2)] == [
        witness[c, day] for day in range(4) for c in graph.conditions
    ]


@pytest.mark.integration
def test_miniature_text_is_deterministic(walk: Walkthrough, miniature: TransitionProblem) -> None:
    text = walkthrough_text(walk)
    assert text == walkthrough_text(
        build_walkthrough(miniature, load_formulary(version=EJ4_DATA_VERSION).contraindications)
    )
    assert "H* = 3 (CSP), 3 (oracle); schedule (checker: valid):" in text
    assert "   5. DEP day 1 - sertraline: no support in PAIN day 2 (constraint I)" in text


# --- the walkthrough's pieces on the toys -------------------------------------------------------


@pytest.mark.parametrize(
    "toy",
    [k for k in ej4_toys.KNOWN_ANSWERS if k.h_star is not None],
    ids=lambda k: k.problem.problem_id,
)
def test_daily_state_graph_agrees_with_the_oracle(toy: ej4_toys.KnownAnswer) -> None:
    graph = daily_state_graph(toy.problem)
    oracle = oracle_min_horizon(toy.problem)
    assert (graph.h_star, graph.states_expanded) == (oracle.h_star, oracle.states_expanded)
    assert len(graph.path) == (oracle.h_star or 0) + 1


@pytest.mark.parametrize("key_config", R3_CONFIGS, ids=[k for k, _c in R3_CONFIGS])
@pytest.mark.parametrize(
    "toy",
    [k for k in ej4_toys.KNOWN_ANSWERS if k.h_star is not None],
    ids=lambda k: k.problem.problem_id,
)
def test_search_tree_numbers_every_assignment(
    toy: ej4_toys.KnownAnswer, key_config: tuple[str, SolverConfig]
) -> None:
    _key, config = key_config
    assert toy.h_star is not None
    for horizon in range(toy.h_star + 1):
        run = solve_horizon(toy.problem, horizon, config)
        steps = search_tree(toy.problem, horizon, config)
        numbers = [s.number for s in steps if s.kind == "assign"]
        assert numbers == list(range(1, run.stats.assignments + 1))
        assert (steps[-1].kind == "solution") == (run.status == "solved")


def test_main_config_never_lists_rejections() -> None:
    steps = search_tree(ej4_toys.T2.problem, ej4_toys.T2.h_star or 0, MAIN_CONFIG)
    assert all(s.kind != "reject" for s in steps)
