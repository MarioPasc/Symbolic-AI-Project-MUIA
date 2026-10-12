"""Tests of the CSP builder of EJ4 (``csp_builder.py``): values, domains, constraints."""

from __future__ import annotations

import dataclasses
import itertools

import ej4_toys
import pytest

from symbolic_ai.p2_ej4_transition import (
    Event,
    EventKind,
    ModelError,
    TransitionProblem,
    Value,
    synthetic,
)
from symbolic_ai.p2_ej4_transition.csp_builder import (
    Variable,
    build_csp,
    condition_values,
    constraint_function,
    interaction_forbids,
    node_consistent_domains,
    transition_allowed,
)

TOYS = ej4_toys.KNOWN_ANSWERS
TOY_IDS = [k.problem.problem_id for k in TOYS]
#: Horizons on which the structural properties are checked for every toy.
HORIZONS = (0, 1, 2, 3, 5)


@pytest.mark.parametrize(
    ("problem", "condition", "expected"),
    [
        (ej4_toys.T2.problem, "X", (("x1", 0), ("x1", 1), ("x1", 2), ("x2", 0))),
        (ej4_toys.T3.problem, "Y", (None, ("y1", 0))),
        (ej4_toys.T3.problem, "X", (("x1", 0), ("x1", 1), ("x2", 0))),
        (ej4_toys.T6.problem, "X", (("x1", 0), ("x2", 0), ("x3", 0))),
        (
            dataclasses.replace(ej4_toys.T1.problem, start={"X": "x2"}),
            "X",
            (("x2", 0), ("x1", 0)),
        ),
    ],
    ids=["T2-taper", "T3-untreated-first", "T3-X", "T6", "start-drug-first"],
)
def test_condition_values_are_in_keep_first_order(
    problem: TransitionProblem, condition: str, expected: tuple[Value, ...]
) -> None:
    assert condition_values(problem, condition) == expected


def test_domains_of_t2_at_horizon_3() -> None:
    middle = (("x1", 0), ("x1", 1), ("x1", 2), ("x2", 0))
    assert node_consistent_domains(ej4_toys.T2.problem, 3) == {
        ("X", 0): (("x1", 0),),
        ("X", 1): middle,
        ("X", 2): middle,
        ("X", 3): (("x2", 0),),
    }


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (ej4_toys.T0.problem, {("X", 0): (("x1", 0),)}),
        (ej4_toys.T1.problem, {("X", 0): ()}),
        (ej4_toys.T3.problem, {("X", 0): (("x1", 0),), ("Y", 0): ()}),
    ],
    ids=["T0-start-acceptable", "T1-start-contraindicated", "T3-untreated"],
)
def test_horizon_0_intersects_the_start_with_the_acceptable_drugs(
    problem: TransitionProblem, expected: dict[Variable, tuple[Value, ...]]
) -> None:
    assert node_consistent_domains(problem, 0) == expected


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_untreated_value_only_for_the_new_condition_before_the_horizon(
    known: ej4_toys.KnownAnswer, horizon: int
) -> None:
    problem = known.problem
    for (condition, day), domain in node_consistent_domains(problem, horizon).items():
        expected = problem.start[condition] is None and day < horizon
        assert (None in domain) is expected, (condition, day)


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_new_contraindicated_drug_is_ever_given(
    known: ej4_toys.KnownAnswer, horizon: int
) -> None:
    problem = known.problem
    for (condition, _day), domain in node_consistent_domains(problem, horizon).items():
        allowed = {*problem.acceptable[condition], problem.start[condition], None}
        assert {v if v is None else v[0] for v in domain} <= allowed


def test_contraindicated_bridge_drug_never_appears() -> None:
    domains = node_consistent_domains(ej4_toys.T6.problem, 4)
    assert all(value is None or value[0] != "x3" for d in domains.values() for value in d)


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_last_day_holds_the_acceptable_drugs_at_full_dose(
    known: ej4_toys.KnownAnswer, horizon: int
) -> None:
    problem = known.problem
    domains = node_consistent_domains(problem, horizon)
    for condition in problem.conditions:
        expected = tuple(
            v
            for v in condition_values(problem, condition)
            if v is not None and v[1] == 0 and v[0] in problem.acceptable[condition]
        )
        assert domains[(condition, horizon)] == expected


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
def test_variables_are_in_static_order(known: ej4_toys.KnownAnswer) -> None:
    problem = known.problem
    expected = [(c, day) for day in range(4) for c in problem.conditions]
    assert list(node_consistent_domains(problem, 3)) == expected
    assert build_csp(problem, 3).variables == expected


def test_build_csp_uses_the_node_consistent_domains() -> None:
    problem = ej4_toys.T5.problem
    csp = build_csp(problem, 4)
    domains = node_consistent_domains(problem, 4)
    assert csp.domains == {v: list(d) for v, d in domains.items()}


def test_negative_horizon_raises() -> None:
    with pytest.raises(ModelError):
        node_consistent_domains(ej4_toys.T0.problem, -1)
    with pytest.raises(ModelError):
        build_csp(ej4_toys.T0.problem, -1)


#: X starts on x1 (2-day taper), which is no longer acceptable; x2 and x3 (no taper) are.
SWITCH = TransitionProblem(
    problem_id="switch",
    event=Event(EventKind.NEW_RISK_FACTOR, "R"),
    conditions=("X",),
    risk_factors=("R",),
    start={"X": "x1"},
    candidates={"X": ("x1", "x2", "x3")},
    acceptable={"X": ("x2", "x3")},
    interactions=frozenset(),
    washout_days=dict.fromkeys(ej4_toys.DRUGS, 0),
    taper_days={**dict.fromkeys(ej4_toys.DRUGS, 0), "x1": 2},
)


@pytest.mark.parametrize(
    ("before", "after", "allowed"),
    [
        (("x1", 0), ("x1", 0), True),  # keep (the old drug may stay while it is withdrawn)
        (("x1", 0), ("x1", 1), True),  # start the taper
        (("x1", 1), ("x1", 2), True),  # continue it
        (("x1", 2), ("x2", 0), True),  # switch after the whole taper
        (("x2", 0), ("x3", 0), True),  # x2 needs no taper: direct switch to an acceptable drug
        (None, None, True),  # a new condition waits
        (None, ("x2", 0), True),  # and starts an acceptable drug at full dose
        (("x2", 0), ("x1", 0), False),  # x1 is contraindicated: never started again (v2.1)
        (("x3", 0), ("x1", 0), False),
        (None, ("x1", 0), False),  # nor started as a first drug
        (("x1", 0), ("x2", 0), False),  # x1 must be tapered first
        (("x1", 1), ("x2", 0), False),  # no aborted taper
        (("x1", 1), ("x1", 0), False),  # no return to full dose
        (("x1", 2), ("x1", 0), False),
        (("x1", 1), ("x1", 1), False),  # a taper day is not repeated
        (("x1", 2), ("x1", 3), False),  # beyond the taper
        (("x2", 0), ("x2", 1), False),  # x2 has no taper
        (("x1", 0), None, False),  # a treated condition is never untreated
        (("x1", 2), None, False),
        (None, ("x2", 1), False),  # no start at a taper day
    ],
)
def test_transition_constraint(before: Value, after: Value, allowed: bool) -> None:
    assert transition_allowed(SWITCH, "X", before, after) is allowed


def _compact_transition_allowed(
    problem: TransitionProblem, condition: str, before: Value, after: Value
) -> bool:
    """The compact implementation of T merged in wave 1 (design v2.1), kept as the reference."""
    if after is None:
        return before is None
    if before is not None:
        drug, phase = before
        if after == before and phase == 0:
            return True
        if after == (drug, phase + 1) and phase < problem.taper_days[drug]:
            return True
        if after[0] == drug or phase != problem.taper_days[drug]:
            return False
    return after[1] == 0 and after[0] in problem.acceptable[condition]


def test_transition_rows_match_the_compact_implementation() -> None:
    # Every ordered pair of values of every condition: the toys, the worked example and 200
    # synthetic problems (same-condition interactions, contraindicated tapers, both events).
    from test_solve import worked_example

    problems = [k.problem for k in TOYS]
    problems += [SWITCH, worked_example(), *synthetic.random_problems(seed=0, count=200)]
    pairs = 0
    for problem in problems:
        for condition in problem.conditions:
            values = condition_values(problem, condition)
            for before, after in itertools.product(values, repeat=2):
                expected = _compact_transition_allowed(problem, condition, before, after)
                assert transition_allowed(problem, condition, before, after) is expected, (
                    problem.problem_id,
                    condition,
                    before,
                    after,
                )
                pairs += 1
    assert pairs > 30_000


@pytest.mark.parametrize("day", [1, 2, 3])
def test_a_withdrawn_contraindicated_drug_is_never_restarted(day: int) -> None:
    constraints = constraint_function(SWITCH)
    before, after = ("X", day), ("X", day + 1)
    for other in (("x2", 0), ("x3", 0)):
        assert not constraints(before, other, after, ("x1", 0))
        assert not constraints(after, ("x1", 0), before, other)


def test_no_solution_switches_back_to_the_contraindicated_drug() -> None:
    # Every complete assignment at H = 7 (time enough to taper x1, switch, restart x1 and taper it
    # again), checked with aima's goal test.
    csp = build_csp(SWITCH, 7)
    solutions = [
        values
        for values in itertools.product(*(csp.domains[v] for v in csp.variables))
        if csp.goal_test(tuple(zip(csp.variables, values, strict=True)))
    ]
    assert solutions  # e.g. x1, x1 taper 1, x1 taper 2, x2, x2, x2
    for values in solutions:
        drugs = [value[0] for value in values if value is not None]
        first_other = next(i for i, drug in enumerate(drugs) if drug != "x1")
        assert "x1" not in drugs[first_other:], values


def test_interactions_are_checked_on_day_0() -> None:
    # Two start drugs that interact: the start regimen itself breaks I.
    problem = dataclasses.replace(
        ej4_toys.T5.problem,
        interactions=frozenset({frozenset({"x1", "y1"})}),
    )
    csp = build_csp(problem, 2)
    assert ("Y", 0) in csp.neighbors[("X", 0)]
    assert not csp.constraints(("X", 0), ("x1", 0), ("Y", 0), ("y1", 0))


# T3: Inter(x1, y1), w_x1 = 2, w_y1 = 0.
@pytest.mark.parametrize(
    ("drug_a", "day_a", "drug_b", "day_b", "forbidden"),
    [
        ("x1", 1, "y1", 1, True),  # same day
        ("x1", 1, "y1", 3, True),  # x1 still present on day 1 + 2
        ("x1", 1, "y1", 4, False),  # x1 gone
        ("x1", 1, "y1", 0, False),  # y1 (washout 0) gone by day 1
        ("y1", 3, "x1", 3, True),
        ("y1", 3, "x1", 4, False),
        ("y1", 3, "x1", 1, True),  # the same pair as (x1, 1, y1, 3), arguments swapped
        ("x1", 1, "x2", 1, False),  # no interaction
        ("x1", 1, None, 1, False),  # nothing given
    ],
)
def test_interaction_window(
    drug_a: str | None, day_a: int, drug_b: str | None, day_b: int, forbidden: bool
) -> None:
    problem = ej4_toys.T3.problem
    assert interaction_forbids(problem, drug_a, day_a, drug_b, day_b) is forbidden
    assert interaction_forbids(problem, drug_b, day_b, drug_a, day_a) is forbidden


def test_interaction_applies_within_one_condition() -> None:
    constraints = constraint_function(ej4_toys.T4.problem)  # Inter(x1, x2), w_x1 = 1
    assert not constraints(("X", 1), ("x1", 0), ("X", 2), ("x2", 0))
    assert constraints(("X", 1), ("x1", 0), ("X", 3), ("x2", 0))  # gap 2 > w_x1: x1 is gone


def _all_value_pairs(
    problem: TransitionProblem, horizon: int
) -> list[tuple[Variable, Value, Variable, Value]]:
    domains = node_consistent_domains(problem, horizon)
    return [
        (a, x, b, y)
        for a, b in itertools.permutations(domains, 2)
        for x in domains[a]
        for y in domains[b]
    ]


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_constraint_function_is_symmetric(known: ej4_toys.KnownAnswer, horizon: int) -> None:
    constraints = constraint_function(known.problem)
    for a, x, b, y in _all_value_pairs(known.problem, horizon):
        assert constraints(a, x, b, y) == constraints(b, y, a, x), (a, x, b, y)


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_neighbours_are_symmetric_and_in_static_order(
    known: ej4_toys.KnownAnswer, horizon: int
) -> None:
    csp = build_csp(known.problem, horizon)
    position = {v: i for i, v in enumerate(csp.variables)}
    for variable, neighbours in csp.neighbors.items():
        assert variable not in neighbours
        assert neighbours == sorted(neighbours, key=position.__getitem__)
        assert all(variable in csp.neighbors[other] for other in neighbours)


@pytest.mark.parametrize("known", TOYS, ids=TOY_IDS)
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_non_neighbours_never_constrain_each_other(
    known: ej4_toys.KnownAnswer, horizon: int
) -> None:
    csp = build_csp(known.problem, horizon)
    constraints = constraint_function(known.problem)
    for a, x, b, y in _all_value_pairs(known.problem, horizon):
        if b not in csp.neighbors[a]:
            assert constraints(a, x, b, y), (a, x, b, y)


def test_neighbours_of_a_single_condition_without_interactions_are_the_adjacent_days() -> None:
    csp = build_csp(ej4_toys.T2.problem, 3)
    assert csp.neighbors[("X", 0)] == [("X", 1)]
    assert csp.neighbors[("X", 1)] == [("X", 0), ("X", 2)]


def test_neighbours_follow_the_washout_window() -> None:
    # T3: Inter(x1, y1), w_x1 = 2, w_y1 = 0. (X, 1) may hold x1, present through day 3, so it is a
    # neighbour of Y on days 1..3; Y on day 0 holds only None, Y on days 4 and 5 is after x1 left.
    csp = build_csp(ej4_toys.T3.problem, 5)
    assert csp.neighbors[("X", 1)] == [("X", 0), ("Y", 1), ("X", 2), ("Y", 2), ("Y", 3)]
