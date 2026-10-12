"""Tests of the EJ4 oracle: known answers, witnesses the checker accepts, budgets, aima."""

from __future__ import annotations

import itertools
import random

import ej4_toys
import pytest

from aima.search import breadth_first_graph_search
from symbolic_ai.p2_ej4_transition.checker import check_schedule
from symbolic_ai.p2_ej4_transition.errors import VerificationError
from symbolic_ai.p2_ej4_transition.oracle import DailyStateProblem, oracle_min_horizon
from symbolic_ai.p2_ej4_transition.problem import (
    Event,
    EventKind,
    TransitionProblem,
    Value,
)

TOY_IDS = [answer.problem.problem_id for answer in ej4_toys.KNOWN_ANSWERS]


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


def knock_on_problem(taper_x1: int, washout_x1: int) -> TransitionProblem:
    """T3 with free taper and washout of x1: H* = taper + 1 + washout (y1 waits for x1 to leave)."""
    pk = dict.fromkeys(ej4_toys.DRUGS, 0)
    return TransitionProblem(
        problem_id=f"knock-on-{taper_x1}-{washout_x1}",
        event=Event(EventKind.NEW_CONDITION, "Y"),
        conditions=("X", "Y"),
        risk_factors=(),
        start={"X": "x1", "Y": None},
        candidates={"X": ("x1", "x2"), "Y": ("y1",)},
        acceptable={"X": ("x1", "x2"), "Y": ("y1",)},
        interactions=frozenset({frozenset({"x1", "y1"})}),
        washout_days={**pk, "x1": washout_x1},
        taper_days={**pk, "x1": taper_x1},
    )


def random_problem(
    rng: random.Random,
    n_conditions: int,
    max_pk: int,
    *,
    sizes: tuple[int, ...] = (2, 3),
    density: float = 0.25,
) -> TransitionProblem:
    """A random problem with a safe starting regimen; the last condition is the new one.

    ``sizes``: possible numbers of candidates per condition; ``density``: probability that two
    drugs of different conditions interact; ``max_pk``: largest washout and taper days.
    """
    while True:
        conditions = tuple(f"C{i}" for i in range(n_conditions))
        candidates = {
            c: tuple(f"{c.lower()}{j}" for j in range(rng.choice(sizes))) for c in conditions
        }
        drugs = sorted(d for c in conditions for d in candidates[c])
        start = {c: candidates[c][0] for c in conditions[:-1]} | {conditions[-1]: None}
        interactions = frozenset(
            frozenset((a, b))
            for a, b in itertools.combinations(drugs, 2)
            if a[:2] != b[:2] and rng.random() < density
        )
        regimen = [d for d in start.values() if d is not None]
        if any(frozenset(pair) in interactions for pair in itertools.combinations(regimen, 2)):
            continue
        return TransitionProblem(
            problem_id="random",
            event=Event(EventKind.NEW_CONDITION, conditions[-1]),
            conditions=conditions,
            risk_factors=(),
            start=start,
            candidates=candidates,
            acceptable={
                c: tuple(d for d in candidates[c] if rng.random() < 0.7) or (candidates[c][-1],)
                for c in conditions
            },
            interactions=interactions,
            washout_days={d: rng.randint(0, max_pk) for d in drugs},
            taper_days={d: rng.randint(0, max_pk) for d in drugs},
        )


@pytest.mark.parametrize("answer", ej4_toys.KNOWN_ANSWERS, ids=TOY_IDS)
def test_known_answer_is_reproduced(answer: ej4_toys.KnownAnswer) -> None:
    result = oracle_min_horizon(answer.problem)
    assert result.h_star == answer.h_star
    assert result.status == ("impossible" if answer.h_star is None else "found")


@pytest.mark.parametrize("answer", ej4_toys.KNOWN_ANSWERS, ids=TOY_IDS)
def test_witness_has_every_day_and_passes_the_checker(answer: ej4_toys.KnownAnswer) -> None:
    result = oracle_min_horizon(answer.problem)
    if result.h_star is None:
        assert result.schedule is None
        return
    assert result.schedule is not None
    expected_keys = {(c, t) for c in answer.problem.conditions for t in range(result.h_star + 1)}
    assert set(result.schedule) == expected_keys
    assert check_schedule(answer.problem, result.schedule, result.h_star) == ()


def test_worked_example_needs_twelve_days() -> None:
    result = oracle_min_horizon(worked_example())
    assert (result.h_star, result.status) == (12, "found")
    assert result.schedule is not None
    dep = [result.schedule[("DEP", t)] for t in range(13)]
    pain = [result.schedule[("PAIN", t)] for t in range(13)]
    assert dep[:5] == [("sertraline", k) for k in range(5)]
    assert dep[5:] == [("mirtazapine", 0)] * 8
    assert pain[:12] == [None] * 12
    assert pain[12] in {("ibuprofen", 0), ("tramadol", 0)}
    assert check_schedule(worked_example(), result.schedule, 12) == ()


@pytest.mark.parametrize(("taper_x1", "washout_x1"), list(itertools.product((0, 1, 2), (0, 1, 3))))
def test_presence_off_by_one(taper_x1: int, washout_x1: int) -> None:
    # x1 is last given on day taper_x1 and present through taper_x1 + washout_x1 (inclusive).
    result = oracle_min_horizon(knock_on_problem(taper_x1, washout_x1))
    assert result.h_star == taper_x1 + 1 + washout_x1


def test_unsafe_starting_regimen_is_impossible() -> None:
    problem = TransitionProblem(
        problem_id="unsafe-start",
        event=Event(EventKind.NEW_RISK_FACTOR, "R"),
        conditions=("X", "Y"),
        risk_factors=("R",),
        start={"X": "x1", "Y": "y1"},
        candidates={"X": ("x1",), "Y": ("y1",)},
        acceptable={"X": ("x1",), "Y": ("y1",)},
        interactions=frozenset({frozenset({"x1", "y1"})}),
        washout_days={"x1": 0, "y1": 0},
        taper_days={"x1": 0, "y1": 0},
    )
    assert oracle_min_horizon(problem) == oracle_min_horizon(problem)
    result = oracle_min_horizon(problem)
    assert (result.h_star, result.status, result.states_expanded) == (None, "impossible", 0)


def test_oracle_never_switches_to_a_contraindicated_drug() -> None:
    # Design v2.1: switch and start targets are acceptable; the bridge x3 of T6 stays forbidden.
    problem = ej4_toys._x_only("contraindicated-start", ("x2", "x3"), candidates=("x1", "x2", "x3"))
    result = oracle_min_horizon(problem)
    assert (result.h_star, result.status) == (1, "found")
    search = DailyStateProblem(problem)
    for state in [search.initial, search.result(search.initial, (("x2", 0),))]:
        targets = {values[0] for values in search.actions(state)}
        assert ("x1", 0) not in targets - {state.values[0]}


def test_no_acceptable_drug_is_impossible() -> None:
    problem = ej4_toys._x_only("none-acceptable", (), taper={"x1": 2}, washout={"x1": 3})
    assert oracle_min_horizon(problem).status == "impossible"


@pytest.mark.parametrize(
    ("h_max", "max_states", "status", "h_star"),
    [
        (12, 200_000, "found", 12),
        (11, 200_000, "budget_exhausted", None),
        (60, 5, "budget_exhausted", None),
        (60, 0, "budget_exhausted", None),
    ],
)
def test_budget_on_the_worked_example(
    h_max: int, max_states: int, status: str, h_star: int | None
) -> None:
    result = oracle_min_horizon(worked_example(), h_max=h_max, max_states=max_states)
    assert (result.status, result.h_star) == (status, h_star)
    assert result.states_expanded <= max_states


def test_depth_zero_budget_still_finds_a_goal_at_day_zero() -> None:
    assert oracle_min_horizon(ej4_toys.T0.problem, h_max=0).status == "found"
    assert oracle_min_horizon(ej4_toys.T1.problem, h_max=0).status == "budget_exhausted"


@pytest.mark.parametrize(("h_max", "max_states"), [(-1, 10), (10, -1)])
def test_negative_budget_raises(h_max: int, max_states: int) -> None:
    with pytest.raises(VerificationError):
        oracle_min_horizon(ej4_toys.T1.problem, h_max=h_max, max_states=max_states)


def test_result_is_deterministic() -> None:
    problems = [answer.problem for answer in ej4_toys.KNOWN_ANSWERS] + [worked_example()]
    assert [oracle_min_horizon(p) for p in problems] == [oracle_min_horizon(p) for p in problems]


def test_aima_breadth_first_graph_search_agrees() -> None:
    # The same search problem solved by aima's own BFS (3rd ed. Fig. 3.11) has the same depth.
    rng = random.Random(7)
    problems = [answer.problem for answer in ej4_toys.KNOWN_ANSWERS] + [worked_example()]
    problems += [random_problem(rng, n, max_pk=3) for n in (2, 3, 4) for _ in range(10)]
    for problem in problems:
        node = breadth_first_graph_search(DailyStateProblem(problem))
        assert (None if node is None else node.depth) == oracle_min_horizon(problem).h_star


def test_random_witnesses_pass_the_checker() -> None:
    rng = random.Random(11)
    for n in (2, 3, 4, 5):
        for _ in range(15):
            problem = random_problem(rng, n, max_pk=4)
            result = oracle_min_horizon(problem)
            assert result.status in {"found", "impossible"}
            if result.schedule is not None:
                assert check_schedule(problem, result.schedule, result.h_star or 0) == ()


def _all_values(problem: TransitionProblem, condition: str) -> list[Value]:
    """Every value of a condition: each candidate at each phase, and untreated for the new one."""
    values: list[Value] = [None] if problem.start[condition] is None else []
    for drug in problem.candidates[condition]:
        values += [(drug, phase) for phase in range(problem.taper_days[drug] + 1)]
    return values


def _brute_force_min_horizon(problem: TransitionProblem, h_limit: int) -> int | None:
    """The smallest H <= h_limit with a schedule the checker accepts, by enumerating them all."""
    conditions = problem.conditions
    day0 = tuple(None if problem.start[c] is None else (problem.start[c], 0) for c in conditions)
    day_choices = list(itertools.product(*(_all_values(problem, c) for c in conditions)))
    for horizon in range(h_limit + 1):
        for days in itertools.product(day_choices, repeat=horizon):
            rows = {
                (c, t): values[i]
                for t, values in enumerate((day0, *days))
                for i, c in enumerate(conditions)
            }
            if check_schedule(problem, rows, horizon) == ():
                return horizon
    return None


@pytest.mark.parametrize("answer", ej4_toys.KNOWN_ANSWERS, ids=TOY_IDS)
def test_brute_force_over_the_checker_agrees_on_toys(answer: ej4_toys.KnownAnswer) -> None:
    # Every schedule up to H = 4, judged by the checker alone: the first accepted H is H*.
    assert _brute_force_min_horizon(answer.problem, 4) == answer.h_star


def test_brute_force_over_the_checker_agrees_on_random_problems() -> None:
    rng = random.Random(3)
    for _ in range(40):
        problem = random_problem(rng, 2, max_pk=2, sizes=(2,), density=0.5)
        h_star = oracle_min_horizon(problem).h_star
        expected = h_star if h_star is not None and h_star <= 3 else None
        assert _brute_force_min_horizon(problem, 3) == expected
