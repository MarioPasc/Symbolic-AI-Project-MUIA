"""Tests of the EJ4 checker: hand-written schedules pass, one targeted mutation per rule fails."""

from __future__ import annotations

import ej4_toys
import pytest

from symbolic_ai.p2_ej4_transition.checker import RULES, Violation, check_schedule
from symbolic_ai.p2_ej4_transition.errors import VerificationError
from symbolic_ai.p2_ej4_transition.problem import (
    Event,
    EventKind,
    TransitionProblem,
    Value,
)

ScheduleDict = dict[tuple[str, int], Value]


def full(drug: str) -> Value:
    return (drug, 0)


def taper(drug: str, phase: int) -> Value:
    return (drug, phase)


def schedule(rows: dict[str, list[Value]]) -> ScheduleDict:
    """Turn one list of daily values per condition into a schedule keyed (condition, day)."""
    return {(c, day): value for c, values in rows.items() for day, value in enumerate(values)}


def rules_of(violations: tuple[Violation, ...]) -> set[str]:
    return {violation.rule for violation in violations}


def worked_example() -> TransitionProblem:
    """The case study of design §1 (DEP + severe liver disease, new PAIN), values of §8."""
    return TransitionProblem(
        problem_id="worked-example",
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


def worked_example_schedule(analgesic: str = "tramadol", start_day: int = 12) -> ScheduleDict:
    """Taper sertraline on days 1-4, mirtazapine from day 5, the analgesic from ``start_day``."""
    dep = [full("sertraline")] + [taper("sertraline", k) for k in range(1, 5)]
    dep += [full("mirtazapine")] * (start_day + 1 - len(dep))
    pain: list[Value] = [None] * start_day + [full(analgesic)]
    return schedule({"DEP": dep, "PAIN": pain})


# Hand-written expected solutions (from the docstrings of ej4_toys), each at its known H*.
T0_SCHEDULE = schedule({"X": [full("x1")]})
T1_SCHEDULE = schedule({"X": [full("x1"), full("x2")]})
T2_SCHEDULE = schedule({"X": [full("x1"), taper("x1", 1), taper("x1", 2), full("x2")]})
T3_SCHEDULE = schedule(
    {
        "X": [full("x1"), taper("x1", 1), full("x2"), full("x2"), full("x2")],
        "Y": [None, None, None, None, full("y1")],
    }
)
T5_SCHEDULE = schedule(
    {
        "X": [full("x1")] + [full("x2")] * 4,
        "Y": [full("y1")] + [full("y2")] * 4,
        "Z": [None] * 4 + [full("z1")],
    }
)


@pytest.mark.parametrize(
    ("toy", "rows", "horizon"),
    [
        (ej4_toys.T0, T0_SCHEDULE, 0),
        (ej4_toys.T1, T1_SCHEDULE, 1),
        (ej4_toys.T2, T2_SCHEDULE, 3),
        (ej4_toys.T3, T3_SCHEDULE, 4),
        (ej4_toys.T5, T5_SCHEDULE, 4),
    ],
    ids=["T0", "T1", "T2", "T3", "T5"],
)
def test_hand_written_solution_is_accepted(
    toy: ej4_toys.KnownAnswer, rows: ScheduleDict, horizon: int
) -> None:
    assert toy.h_star == horizon
    assert check_schedule(toy.problem, rows, horizon) == ()


def test_longer_than_minimum_schedule_is_accepted() -> None:
    # L2: repeating the last day keeps a schedule valid.
    rows = schedule({"X": [full("x1"), full("x2"), full("x2"), full("x2")]})
    assert check_schedule(ej4_toys.T1.problem, rows, 3) == ()


def test_new_condition_may_wait_longer_than_needed() -> None:
    rows = dict(T3_SCHEDULE)
    rows[("Y", 4)] = None
    rows[("X", 5)], rows[("Y", 5)] = full("x2"), full("y1")
    assert check_schedule(ej4_toys.T3.problem, rows, 5) == ()


def _mutated(base: ScheduleDict, **changes: Value) -> ScheduleDict:
    """Return ``base`` with cells replaced; a keyword ``X3`` names the cell (condition X, day 3)."""
    rows = dict(base)
    for cell, value in changes.items():
        rows[(cell[0], int(cell[1:]))] = value
    return rows


# (problem, schedule, horizon, exact set of rules the checker must report)
MUTATIONS: dict[str, tuple[TransitionProblem, ScheduleDict, int, set[str]]] = {
    "drop a taper day (T2)": (
        ej4_toys.T2.problem,
        _mutated(T2_SCHEDULE, X2=full("x2")),
        3,
        {"transition"},
    ),
    "skip the whole taper (T3)": (
        ej4_toys.T3.problem,
        _mutated(T3_SCHEDULE, X1=full("x2")),
        4,
        {"transition"},
    ),
    "start the new drug one day early (T3)": (
        ej4_toys.T3.problem,
        _mutated(T3_SCHEDULE, Y3=full("y1")),
        4,
        {"interaction"},
    ),
    "start the new drug one day early (T5)": (
        ej4_toys.T5.problem,
        _mutated(T5_SCHEDULE, Z3=full("z1")),
        4,
        {"interaction"},
    ),
    "switch to a drug that interacts with the old one (T4)": (
        ej4_toys.T4.problem,
        schedule({"X": [full("x1"), full("x2")]}),
        1,
        {"interaction"},
    ),
    "make a treated condition untreated (T5)": (
        ej4_toys.T5.problem,
        _mutated(T5_SCHEDULE, Y2=None),
        4,
        {"transition", "domain"},
    ),
    "contraindicated bridge on an intermediate day (T6)": (
        ej4_toys.T6.problem,
        schedule({"X": [full("x1"), full("x3"), full("x2")]}),
        2,
        {"domain", "transition"},  # v2.1: a switch target must be acceptable
    ),
    "drug of another condition on an intermediate day (T5)": (
        ej4_toys.T5.problem,
        _mutated(T5_SCHEDULE, X1=full("y2")),
        4,
        {"domain", "transition"},  # y2 is not an acceptable drug of X
    ),
    "taper phase beyond the taper days (T2)": (
        ej4_toys.T2.problem,
        schedule({"X": [full("x1"), taper("x1", 1), taper("x1", 2), taper("x1", 3), full("x2")]}),
        4,
        {"domain", "transition"},
    ),
    "change day 0 (T1)": (
        ej4_toys.T1.problem,
        schedule({"X": [full("x2"), full("x2")]}),
        1,
        {"day0"},
    ),
    "end on a tapered value (T2)": (
        ej4_toys.T2.problem,
        schedule({"X": [full("x1"), taper("x1", 1), taper("x1", 2)]}),
        2,
        {"final"},
    ),
    "end on an unacceptable drug (T1)": (
        ej4_toys.T1.problem,
        schedule({"X": [full("x1"), full("x1")]}),
        1,
        {"final"},
    ),
    "end with the new condition untreated (T3)": (
        ej4_toys.T3.problem,
        _mutated(T3_SCHEDULE, Y4=None),
        4,
        {"final"},
    ),
    "delete a key (T3)": (
        ej4_toys.T3.problem,
        {key: value for key, value in T3_SCHEDULE.items() if key != ("Y", 2)},
        4,
        {"shape"},
    ),
    "start the analgesic on day 11 (worked example)": (
        worked_example(),
        worked_example_schedule(start_day=11),
        11,
        {"interaction"},
    ),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_targeted_mutation_is_rejected_with_its_rule(name: str) -> None:
    problem, rows, horizon, expected = MUTATIONS[name]
    assert rules_of(check_schedule(problem, rows, horizon)) == expected


def test_presence_lasts_exactly_washout_days_after_the_last_dose() -> None:
    # T3: x1 last given on day 1 with w = 2 is present on days 2 and 3, not on day 4.
    violations = check_schedule(ej4_toys.T3.problem, _mutated(T3_SCHEDULE, Y3=full("y1")), 4)
    assert [(v.rule, v.condition, v.day) for v in violations] == [("interaction", "Y", 3)]
    assert "x1" in violations[0].detail


def test_pair_given_on_the_same_day_is_reported_once() -> None:
    rows = _mutated(T5_SCHEDULE, X1=full("x1"), X2=full("x1"), X3=full("x1"), X4=full("x1"))
    violations = check_schedule(ej4_toys.T5.problem, rows, 4)
    assert [(v.rule, v.day) for v in violations] == [("interaction", 4)]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (("X", 2), full("x2")),  # a day beyond the horizon
        (("W", 0), full("x2")),  # an unknown condition
        ("X0", full("x2")),  # not a (condition, day) pair
    ],
)
def test_extra_key_is_a_shape_violation(key: object, value: Value) -> None:
    rows: dict[object, Value] = dict(T1_SCHEDULE)
    rows[key] = value
    assert rules_of(check_schedule(ej4_toys.T1.problem, rows, 1)) == {"shape"}


@pytest.mark.parametrize(
    "value",
    [("x2",), ("x2", "0"), ("x2", True), 3, ["x2", 0], ("q9", 0)],
    ids=["one-tuple", "str-phase", "bool-phase", "int", "list", "unknown-drug"],
)
def test_malformed_value_or_unknown_drug_is_a_shape_violation(value: object) -> None:
    rows: dict[tuple[str, int], object] = dict(T1_SCHEDULE)
    rows[("X", 1)] = value
    violations = check_schedule(ej4_toys.T1.problem, rows, 1)
    assert [(v.rule, v.condition, v.day) for v in violations] == [("shape", "X", 1)]


def test_worked_example_schedules_are_accepted() -> None:
    problem = worked_example()
    for analgesic in ("ibuprofen", "tramadol"):
        assert check_schedule(problem, worked_example_schedule(analgesic), 12) == ()


def test_worked_example_paracetamol_is_contraindicated() -> None:
    problem = worked_example()
    rows = worked_example_schedule("paracetamol")
    # v2.1: starting a non-acceptable drug also breaks the transition rule.
    assert rules_of(check_schedule(problem, rows, 12)) == {"final", "transition"}
    rows[("PAIN", 13)], rows[("DEP", 13)] = full("tramadol"), full("mirtazapine")
    assert rules_of(check_schedule(problem, rows, 13)) == {"domain", "transition"}


def contraindicated_start_problem() -> TransitionProblem:
    """X treated with x1, now contraindicated; x2 and x3 acceptable; no taper, no washout."""
    return ej4_toys._x_only("contraindicated-start", ("x2", "x3"), candidates=("x1", "x2", "x3"))


def test_returning_to_the_contraindicated_start_drug_is_a_transition_violation() -> None:
    # Design v2.1: once the condition has left x1, x1 is never given again unless acceptable.
    rows = schedule({"X": [full("x1"), full("x2"), full("x1"), full("x2")]})
    violations = check_schedule(contraindicated_start_problem(), rows, 3)
    assert [(v.rule, v.condition, v.day) for v in violations] == [("transition", "X", 2)]


def test_keeping_the_contraindicated_start_drug_is_still_allowed() -> None:
    rows = schedule({"X": [full("x1"), full("x1"), full("x1"), full("x3")]})
    assert check_schedule(contraindicated_start_problem(), rows, 3) == ()


def test_unsafe_day_zero_is_an_interaction_violation() -> None:
    problem = _two_condition_problem(interactions={frozenset({"x1", "y1"})})
    rows = schedule({"X": [full("x1")], "Y": [full("y1")]})
    violations = check_schedule(problem, rows, 0)
    assert [(v.rule, v.day) for v in violations] == [("interaction", 0)]


def test_violations_are_sorted_by_day_condition_rule() -> None:
    rows = _mutated(T5_SCHEDULE, Y2=None, Z3=full("z1"), X4=full("x1"))
    del rows[("X", 1)]
    violations = check_schedule(ej4_toys.T5.problem, rows, 4)
    keys = [(v.day, v.condition, RULES.index(v.rule)) for v in violations]
    assert keys == sorted(keys)
    assert {v.rule for v in violations} == {"shape", "domain", "transition", "interaction"}
    assert check_schedule(ej4_toys.T5.problem, rows, 4) == violations


@pytest.mark.parametrize("horizon", [-1, True, 1.0])
def test_invalid_horizon_raises(horizon: object) -> None:
    with pytest.raises(VerificationError):
        check_schedule(ej4_toys.T1.problem, T1_SCHEDULE, horizon)


def _two_condition_problem(interactions: frozenset[frozenset[str]]) -> TransitionProblem:
    """X (x1, x2) and Y (y1, y2) treated with x1 and y1, all acceptable, nothing to taper."""
    return TransitionProblem(
        problem_id="two",
        event=Event(EventKind.NEW_RISK_FACTOR, "R"),
        conditions=("X", "Y"),
        risk_factors=("R",),
        start={"X": "x1", "Y": "y1"},
        candidates={"X": ("x1", "x2"), "Y": ("y1", "y2")},
        acceptable={"X": ("x1", "x2"), "Y": ("y1", "y2")},
        interactions=interactions,
        washout_days=dict.fromkeys(("x1", "x2", "y1", "y2"), 0),
        taper_days=dict.fromkeys(("x1", "x2", "y1", "y2"), 0),
    )
