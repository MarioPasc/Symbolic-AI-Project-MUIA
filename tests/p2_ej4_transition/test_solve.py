"""Tests of the solver layer of EJ4 (``solve.py``): known answers, worked example, determinism."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

import ej4_toys
import pytest

import symbolic_ai
from aima import csp as aima_csp
from symbolic_ai.p2_ej4_transition import (
    Event,
    EventKind,
    ModelError,
    SolverError,
    TransitionProblem,
    Value,
    csp_builder,
    solve,
)
from symbolic_ai.p2_ej4_transition.csp_builder import Variable, build_csp, node_consistent_domains
from symbolic_ai.p2_ej4_transition.solve import (
    ALL_CONFIGS,
    MAIN_CONFIG,
    SolverConfig,
    TraceEvent,
    ac3_domains,
    find_min_horizon,
    solve_horizon,
    solve_min_conflicts,
)

TOYS = ej4_toys.KNOWN_ANSWERS
TOY_IDS = [k.problem.problem_id for k in TOYS]
CONFIG_IDS = [c.label for c in ALL_CONFIGS]


def worked_example() -> TransitionProblem:
    """The case study of design §1: depression on sertraline, then chronic pain with LIVER."""
    return TransitionProblem(
        problem_id="DEP+LIVER/new PAIN",
        event=Event(EventKind.NEW_CONDITION, "PAIN"),
        conditions=("DEP", "PAIN"),
        risk_factors=("LIVER",),
        start={"DEP": "sertraline", "PAIN": None},
        candidates={
            "DEP": ("mirtazapine", "sertraline"),
            "PAIN": ("ibuprofen", "paracetamol", "tramadol"),
        },
        acceptable={"DEP": ("mirtazapine", "sertraline"), "PAIN": ("ibuprofen", "tramadol")},
        interactions=frozenset(
            {frozenset({"ibuprofen", "sertraline"}), frozenset({"sertraline", "tramadol"})}
        ),
        washout_days={
            "sertraline": 7,
            "mirtazapine": 4,
            "tramadol": 2,
            "ibuprofen": 1,
            "paracetamol": 1,
        },
        taper_days={
            "sertraline": 4,
            "mirtazapine": 3,
            "tramadol": 2,
            "ibuprofen": 0,
            "paracetamol": 0,
        },
    )


def _assert_valid(
    problem: TransitionProblem, horizon: int, schedule: dict[Variable, Value]
) -> None:
    """The schedule covers days 0..H in static order, stays in the domains and passes goal_test."""
    domains = node_consistent_domains(problem, horizon)
    assert list(schedule) == list(domains)
    assert all(schedule[v] in domains[v] for v in domains)
    assert build_csp(problem, horizon).goal_test(tuple(schedule.items()))


# --- configurations ---------------------------------------------------------------------------


def test_all_configs_are_the_24_complete_configurations() -> None:
    assert len(set(ALL_CONFIGS)) == 24
    assert MAIN_CONFIG in ALL_CONFIGS
    assert MAIN_CONFIG.label == "AC-3 + BT + FC + MRV + keep-first"
    assert SolverConfig("none", "static", "keep_first", False).label == "BT + static + keep-first"


@pytest.mark.parametrize(
    "changes",
    [
        {"inference": "cbj"},
        {"variable_order": "degree"},
        {"value_order": "random"},
        {"timeout_s": 0.0},
        {"timeout_s": -1.0},
    ],
)
def test_invalid_config_raises(changes: dict[str, object]) -> None:
    with pytest.raises(SolverError):
        SolverConfig(**changes)  # type: ignore[arg-type]  # deliberately outside the Literals


def test_invalid_arguments_raise() -> None:
    problem = ej4_toys.T1.problem
    with pytest.raises(SolverError):
        find_min_horizon(problem, h_max=-1)
    with pytest.raises(SolverError):
        solve_min_conflicts(problem, 1, max_steps=-1)
    with pytest.raises(ModelError):
        solve_horizon(problem, -1)


# --- known answers (design §9.3) ---------------------------------------------------------------


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=CONFIG_IDS)
@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
def test_known_answer_under_every_configuration(
    known: ej4_toys.KnownAnswer, config: SolverConfig
) -> None:
    result = find_min_horizon(known.problem, config, h_max=10)
    assert result.h_star == known.h_star
    assert [run.horizon for run in result.runs] == list(range(len(result.runs)))
    assert all(run.status == "infeasible" for run in result.runs[:-1])
    if known.h_star is None:
        assert result.status == "no_transition_within_h_max"
        assert len(result.runs) == 11
        assert result.schedule is None
    else:
        assert result.status == "solved"
        assert result.schedule is not None
        assert result.schedule == result.runs[-1].schedule
        _assert_valid(known.problem, known.h_star, result.schedule)


@pytest.mark.parametrize(
    "known", [k for k in TOYS if k.h_star is not None], ids=lambda k: k.problem.problem_id
)
@pytest.mark.parametrize("config", ALL_CONFIGS, ids=CONFIG_IDS)
def test_monotonicity_l2(known: ej4_toys.KnownAnswer, config: SolverConfig) -> None:
    assert known.h_star is not None
    for horizon in range(known.h_star, known.h_star + 4):
        run = solve_horizon(known.problem, horizon, config)
        assert run.status == "solved", horizon
        assert run.schedule is not None
        _assert_valid(known.problem, horizon, run.schedule)


def test_start_drugs_that_interact_make_every_horizon_infeasible() -> None:
    interactions = frozenset({frozenset({"x1", "y1"})})
    problem = dataclasses.replace(ej4_toys.T5.problem, interactions=interactions)
    result = find_min_horizon(problem, SolverConfig("none", "static", "keep_first", False), 4)
    assert result.status == "no_transition_within_h_max"


# --- the worked example (design §1, §7 R1) -----------------------------------------------------


def _expected_h12_domains() -> dict[Variable, tuple[Value, ...]]:
    """The domains the design predicts after AC-3 at H = 12: all forced but PAIN on day 12."""
    dep: list[Value] = [("sertraline", k) for k in range(5)] + [("mirtazapine", 0)] * 8
    domains: dict[Variable, tuple[Value, ...]] = {}
    for day in range(13):
        domains[("DEP", day)] = (dep[day],)
        domains[("PAIN", day)] = (None,)
    domains[("PAIN", 12)] = (("ibuprofen", 0), ("tramadol", 0))
    return domains


def test_worked_example_h_star_is_12() -> None:
    result = find_min_horizon(worked_example())
    assert result.h_star == 12
    assert result.status == "solved"
    expected = {v: d[0] for v, d in _expected_h12_domains().items()}  # keep-first: ibuprofen
    assert result.schedule == expected


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=CONFIG_IDS)
def test_worked_example_h_star_under_every_configuration(config: SolverConfig) -> None:
    result = find_min_horizon(worked_example(), config, h_max=15)
    assert result.h_star == 12
    assert result.schedule is not None
    _assert_valid(worked_example(), 12, result.schedule)


def test_ac3_alone_proves_horizon_11_infeasible() -> None:
    consistent, domains = ac3_domains(worked_example(), 11)
    assert consistent is False
    # Which domain empties first depends on AC-3's arc order (here: DEP on day 2, where the taper
    # chain from day 0 meets the sertraline-free days 4..11 that the day-11 analgesic requires).
    assert [v for v, d in domains.items() if not d] == [("DEP", 2)]
    run = solve_horizon(worked_example(), 11)
    assert run.status == "infeasible"
    assert run.stats.assignments == 0


def test_ac3_at_horizon_12_leaves_only_the_analgesic_open() -> None:
    assert ac3_domains(worked_example(), 12) == (True, _expected_h12_domains())


def test_main_config_solves_horizon_12_without_backtracking() -> None:
    trace: list[TraceEvent] = []
    run = solve_horizon(worked_example(), 12, MAIN_CONFIG, trace)
    kinds = [e.kind for e in trace]
    assert run.status == "solved"
    assert run.stats.assignments == 26  # every variable assigned exactly once
    assert {"backtrack", "wipeout", "prune"}.isdisjoint(kinds)
    assert kinds.count("ac3") == run.stats.ac3_removed == 133
    assert kinds.count("select") == kinds.count("try") == 26
    assert trace[-1].kind == "solution"
    assert trace[-2] == TraceEvent(
        len(trace) - 2, "try", ("PAIN", 12), ("ibuprofen", 0), "assigned"
    )


@pytest.mark.parametrize(
    ("horizon", "assignments", "checks", "removed"),
    [(11, 0, 1234, 116), (12, 26, 2264, 133)],
)
def test_main_config_effort_on_the_worked_example(
    horizon: int, assignments: int, checks: int, removed: int
) -> None:
    stats = solve_horizon(worked_example(), horizon).stats
    assert (stats.assignments, stats.checks, stats.ac3_removed) == (assignments, checks, removed)


def test_worked_example_constraints_are_symmetric() -> None:
    csp = build_csp(worked_example(), 12)
    for a in csp.variables:
        for b in csp.neighbors[a]:
            for x in csp.domains[a]:
                for y in csp.domains[b]:
                    assert csp.constraints(a, x, b, y) == csp.constraints(b, y, a, x)


# --- counters, trace, timeout, MAC ---------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [MAIN_CONFIG, SolverConfig("none", "static", "lcv", False), SolverConfig("mac", "mrv")],
    ids=lambda c: c.label,
)
@pytest.mark.parametrize("horizon", [11, 12])
def test_checks_count_every_call_of_the_constraint_function(
    monkeypatch: pytest.MonkeyPatch, config: SolverConfig, horizon: int
) -> None:
    calls = 0
    original = csp_builder.constraint_function

    def counting(problem: TransitionProblem) -> csp_builder.Constraint:
        relation = original(problem)

        def counted(a: Variable, x: Value, b: Variable, y: Value) -> bool:
            nonlocal calls
            calls += 1
            return relation(a, x, b, y)

        return counted

    monkeypatch.setattr(csp_builder, "constraint_function", counting)
    run = solve_horizon(worked_example(), horizon, config)
    assert run.stats.checks == calls > 0


def test_tracing_does_not_change_the_run() -> None:
    for config in ALL_CONFIGS:
        traced = solve_horizon(worked_example(), 12, config, trace=[])
        plain = solve_horizon(worked_example(), 12, config)
        assert traced.schedule == plain.schedule
        assert (traced.stats.assignments, traced.stats.checks) == (
            plain.stats.assignments,
            plain.stats.checks,
        )


def test_trace_of_plain_backtracking_shows_assignments_and_backtracks() -> None:
    trace: list[TraceEvent] = []
    config = SolverConfig("none", "static", "keep_first", False)
    run = solve_horizon(worked_example(), 11, config, trace)
    assert run.status == "infeasible"
    assert [e.step for e in trace] == list(range(len(trace)))
    tries = [e for e in trace if e.kind == "try"]
    assert len(tries) == run.stats.assignments  # one "try" per assignment
    assert {e.detail for e in tries} == {"assigned"}
    assert trace[-1].kind == "backtrack"
    assert trace[-1].variable == ("DEP", 0)  # the search gave up at the first variable


def test_keep_first_order_survives_a_restore() -> None:
    # T2 at H = 3: (X, 1) holds x2, then the withdrawn x1 and its two taper days.
    csp = build_csp(ej4_toys.T2.problem, 3)
    variable: Variable = ("X", 1)
    static = csp.domains[variable]
    assert static == [("x2", 0), ("x1", 0), ("x1", 1), ("x1", 2)]
    csp.restore(csp.suppose(variable, ("x1", 1)))  # what backtracking does after a failed value
    assert csp.choices(variable) == [("x1", 1), ("x2", 0), ("x1", 0), ("x1", 2)]  # aima appends
    assert solve._keep_first(variable, {}, csp) == static
    values = solve._value_orderer(MAIN_CONFIG, None)
    assert list(values(variable, {}, csp)) == static


def test_keep_first_tries_follow_the_static_order_on_every_visit() -> None:
    # Plain backtracking revisits variables after backtracks (244 assignments for 26 variables);
    # within one visit (from its "select" to the next one) the values assigned to a variable keep
    # their static order. With aima's own unordered_domain_values this fails on (DEP, 11).
    trace: list[TraceEvent] = []
    run = solve_horizon(worked_example(), 12, PLAIN_BT, trace)
    domains = build_csp(worked_example(), 12).domains
    visits: dict[Variable, list[int]] = {}
    for event in trace:
        if event.kind == "select" and event.variable is not None:
            visits[event.variable] = []
        elif event.kind == "try" and event.variable is not None:
            positions = visits[event.variable]
            positions.append(domains[event.variable].index(event.value))
            assert positions == sorted(positions), event
    assert run.status == "solved"


def test_trace_of_ac3_preprocessing_on_t2() -> None:
    trace: list[TraceEvent] = []
    run = solve_horizon(ej4_toys.T2.problem, 3, MAIN_CONFIG, trace)
    removed = [(e.variable, e.value) for e in trace if e.kind == "ac3"]
    assert run.stats.ac3_removed == len(removed) == 6
    assert sorted(removed) == [
        (("X", 1), ("x1", 0)),
        (("X", 1), ("x1", 2)),
        (("X", 1), ("x2", 0)),
        (("X", 2), ("x1", 0)),
        (("X", 2), ("x1", 1)),
        (("X", 2), ("x2", 0)),
    ]


def test_ac3_domains_of_t2_force_the_taper() -> None:
    assert ac3_domains(ej4_toys.T2.problem, 3) == (
        True,
        {
            ("X", 0): (("x1", 0),),
            ("X", 1): (("x1", 1),),
            ("X", 2): (("x1", 2),),
            ("X", 3): (("x2", 0),),
        },
    )
    assert ac3_domains(ej4_toys.T2.problem, 2)[0] is False


def test_timeout_is_reported_as_unknown() -> None:
    config = SolverConfig("none", "static", "keep_first", False, timeout_s=1e-9)
    run = solve_horizon(ej4_toys.T2.problem, 3, config)
    assert run.status == "timeout"
    assert run.schedule is None
    result = find_min_horizon(ej4_toys.T2.problem, config)
    assert (result.status, result.h_star, len(result.runs)) == ("timeout", None, 1)


def test_aima_mac_returns_a_pair_that_is_always_true() -> None:
    # T4 at H = 1: x1 on day 0 leaves no support for x2 on day 1 (Inter(x1, x2), w_x1 = 1).
    csp = build_csp(ej4_toys.T4.problem, 1)
    variable, value = ("X", 0), ("x1", 0)
    removals = csp.suppose(variable, value)
    result = aima_csp.mac(
        csp, variable, value, {variable: value}, removals, constraint_propagation=aima_csp.AC3
    )
    assert isinstance(result, tuple)
    assert result[0] is False
    assert bool(result) is True  # why solve.py unpacks it before backtracking_search sees it


def test_mac_wipeout_stops_the_branch() -> None:
    trace: list[TraceEvent] = []
    config = SolverConfig("mac", "static", "keep_first", False)
    run = solve_horizon(ej4_toys.T4.problem, 1, config, trace)
    assert run.status == "infeasible"
    assert run.stats.assignments == 1
    assert [e.kind for e in trace].count("wipeout") == 1


# --- determinism -------------------------------------------------------------------------------


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=CONFIG_IDS)
def test_repeated_runs_are_identical(config: SolverConfig) -> None:
    for problem, horizon in ((worked_example(), 12), (ej4_toys.T5.problem, 4)):
        first = solve_horizon(problem, horizon, config)
        second = solve_horizon(problem, horizon, config)
        assert repr(first.schedule) == repr(second.schedule)
        assert first.stats.assignments == second.stats.assignments
        assert first.stats.checks == second.stats.checks
        assert first.stats.ac3_removed == second.stats.ac3_removed


def _worked_example_digest() -> str:
    """A hash of every configuration's status, counts and schedule on the worked example."""
    lines = []
    for horizon in (11, 12):
        lines.append(repr(ac3_domains(worked_example(), horizon)))
        for config in ALL_CONFIGS:
            run = solve_horizon(worked_example(), horizon, config)
            stats = run.stats
            lines.append(
                f"{config.label} {run.status} {stats.assignments} {stats.checks} "
                f"{stats.ac3_removed} {run.schedule!r}"
            )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_results_do_not_depend_on_string_hashing() -> None:
    # aima's own AC-3 arc order breaks ties by set order, which changes with PYTHONHASHSEED;
    # solve.py breaks them by static order, so separate processes must agree.
    paths = [str(Path(symbolic_ai.__file__).resolve().parents[1]), str(Path(__file__).parent)]
    code = "import test_solve; print(test_solve._worked_example_digest())"
    digests = set()
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": os.pathsep.join(paths)}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert digests == {_worked_example_digest()}


# --- min-conflicts (failure case 2) ------------------------------------------------------------


def test_min_conflicts_finds_the_worked_example_at_horizon_12() -> None:
    run = solve_min_conflicts(worked_example(), 12, max_steps=10_000, seed=0)
    assert (run.status, run.seed) == ("solved", 0)
    assert run.schedule is not None
    _assert_valid(worked_example(), 12, run.schedule)


def test_min_conflicts_cannot_prove_horizon_11_infeasible() -> None:
    run = solve_min_conflicts(worked_example(), 11, max_steps=10_000, seed=0)
    assert run.status == "not_found"
    assert run.schedule is None


def test_min_conflicts_is_reproducible_with_its_seed() -> None:
    first = solve_min_conflicts(ej4_toys.T5.problem, 4, max_steps=500, seed=3)
    second = solve_min_conflicts(ej4_toys.T5.problem, 4, max_steps=500, seed=3)
    assert repr(first.schedule) == repr(second.schedule)
    assert (first.stats.assignments, first.stats.checks) == (
        second.stats.assignments,
        second.stats.checks,
    )


def test_min_conflicts_restores_the_global_random_state() -> None:
    random.seed(12345)
    state = random.getstate()
    solve_min_conflicts(worked_example(), 12, max_steps=100, seed=0)
    assert random.getstate() == state


@pytest.mark.parametrize(
    ("problem", "horizon"),
    [(ej4_toys.T1.problem, 0), (ej4_toys.T4.problem, 2)],
    ids=["empty-domain", "infeasible"],
)
def test_min_conflicts_reports_not_found_never_infeasible(
    problem: TransitionProblem, horizon: int
) -> None:
    run = solve_min_conflicts(problem, horizon, max_steps=200, seed=0)
    assert run.status == "not_found"


# --- deterministic assignment budget (wave 2, R3) -------------------------------------------------

PLAIN_BT = SolverConfig("none", "static", "keep_first", False)


def test_budget_stops_the_run_at_the_same_assignment() -> None:
    config = dataclasses.replace(PLAIN_BT, max_assignments=10)
    runs = [solve_horizon(worked_example(), 11, config) for _ in range(2)]
    for run in runs:
        assert (run.status, run.schedule) == ("budget", None)
        # checked when the next variable is chosen, so the run stops one assignment past it
        assert run.stats.assignments == 11
    assert runs[0].stats.checks == runs[1].stats.checks


@pytest.mark.parametrize(
    ("horizon", "status", "assignments"), [(11, "infeasible", 218), (12, "solved", 244)]
)
def test_a_large_enough_budget_changes_nothing(horizon: int, status: str, assignments: int) -> None:
    unbudgeted = solve_horizon(worked_example(), horizon, PLAIN_BT)
    budgeted = solve_horizon(
        worked_example(), horizon, dataclasses.replace(PLAIN_BT, max_assignments=assignments)
    )
    assert (unbudgeted.status, unbudgeted.stats.assignments) == (status, assignments)
    assert (budgeted.status, budgeted.stats.assignments) == (status, assignments)
    assert budgeted.schedule == unbudgeted.schedule


def test_budget_ends_the_horizon_loop_as_unknown() -> None:
    config = dataclasses.replace(PLAIN_BT, max_assignments=5)
    result = find_min_horizon(worked_example(), config)
    assert (result.status, result.h_star, result.schedule) == ("budget", None, None)
    assert result.runs[-1].status == "budget"


def test_default_config_has_no_budget() -> None:
    assert MAIN_CONFIG.max_assignments is None


def test_negative_budget_raises() -> None:
    with pytest.raises(SolverError):
        SolverConfig(max_assignments=-1)


# --- AC-3 names the arc of each removal (the miniature of the slides) ---------------------------


@pytest.mark.parametrize(
    ("horizon", "status", "removed", "checks"),
    [(11, "infeasible", 116, 1234), (12, "solved", 133, 2264)],
)
def test_naming_the_arcs_changes_no_removal_and_no_count(
    horizon: int, status: str, removed: int, checks: int
) -> None:
    # Before: aima's AC-3 with the plain arc queue; after: the traced run, whose queue logs arcs.
    plain: list[tuple[Variable, Value]] = []
    consistent, plain_checks = aima_csp.AC3(
        build_csp(worked_example(), horizon), removals=plain, arc_heuristic=solve._arc_order
    )
    trace: list[TraceEvent] = []
    run = solve_horizon(worked_example(), horizon, MAIN_CONFIG, trace)
    assert [(e.variable, e.value) for e in trace if e.kind == "ac3"] == plain
    assert (run.status, len(plain), run.stats.ac3_removed, run.stats.checks) == (
        status,
        removed,
        removed,
        checks,
    )
    if not consistent:
        assert plain_checks == checks  # the run stopped right after AC-3


@pytest.mark.parametrize("horizon", [11, 12])
def test_each_ac3_removal_has_no_support_in_the_neighbour_it_names(horizon: int) -> None:
    # Replay the removals on the node-consistent domains: when a value is removed, no value left
    # in the named neighbour's domain may satisfy the constraint with it.
    problem = worked_example()
    domains = {v: list(d) for v, d in node_consistent_domains(problem, horizon).items()}
    neighbours = build_csp(problem, horizon).neighbors
    check = csp_builder.constraint_function(problem)
    trace: list[TraceEvent] = []
    solve_horizon(problem, horizon, MAIN_CONFIG, trace)
    for event in trace:
        if event.kind != "ac3":
            assert event.neighbour is None
            continue
        variable, neighbour = event.variable, event.neighbour
        assert variable is not None and neighbour is not None
        assert neighbour in neighbours[variable]
        assert not any(check(variable, event.value, neighbour, y) for y in domains[neighbour])
        assert event.detail.endswith(f"(constraint {solve.arc_constraint(variable, neighbour)})")
        domains[variable].remove(event.value)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (("DEP", 1), ("DEP", 2), "T"),
        (("DEP", 2), ("DEP", 1), "T"),
        (("DEP", 1), ("DEP", 3), "I"),
        (("DEP", 1), ("PAIN", 1), "I"),
        (("DEP", 1), ("PAIN", 2), "I"),
    ],
)
def test_arc_constraint(first: Variable, second: Variable, expected: str) -> None:
    assert solve.arc_constraint(first, second) == expected
