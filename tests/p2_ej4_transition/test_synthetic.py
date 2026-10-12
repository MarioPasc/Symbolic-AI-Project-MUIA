"""Tests of the synthetic problem generator: validity, determinism, coverage, argument checks."""

from __future__ import annotations

import itertools
import random

import ej4_toys
import pytest

from symbolic_ai.p2_ej4_transition.errors import InstanceError
from symbolic_ai.p2_ej4_transition.problem import EventKind, TransitionProblem
from symbolic_ai.p2_ej4_transition.synthetic import (
    Coverage,
    coverage,
    has_contraindicated_taper_start,
    has_same_condition_interaction,
    random_problem,
    random_problems,
)


def _start_is_safe(problem: TransitionProblem) -> bool:
    """Whether no two drugs of the regimen before the event interact."""
    regimen = [d for d in problem.start.values() if d is not None]
    return not any(
        frozenset(pair) in problem.interactions for pair in itertools.combinations(regimen, 2)
    )


def _has_planted_bridge(problem: TransitionProblem) -> bool:
    """Whether some treated condition has the pattern of ``synthetic._plant_bridge``."""
    new = problem.event.target_id
    for condition, old in problem.start.items():
        if old is None:
            continue
        others = [d for d in problem.acceptable[condition] if d != old]
        for target, bridge in itertools.permutations(others, 2):
            blocked = all(
                frozenset((drug, d)) in problem.interactions
                for drug in (old, bridge)
                for d in problem.acceptable[new]
            )
            if (
                blocked
                and frozenset((old, target)) in problem.interactions
                and frozenset((old, bridge)) not in problem.interactions
                and frozenset((target, bridge)) not in problem.interactions
                and problem.washout_days[old] >= 1
            ):
                return True
    return False


def test_batch_is_valid_and_starts_safe() -> None:
    # The contract's __post_init__ already validated every problem on construction.
    problems = random_problems(seed=1, count=300)
    assert all(_start_is_safe(p) for p in problems)
    assert all(2 <= len(p.conditions) <= 5 for p in problems)


def test_batch_ids_are_numbered_from_one() -> None:
    problems = random_problems(seed=1, count=3)
    assert [p.problem_id for p in problems] == ["S0001", "S0002", "S0003"]


def test_same_seed_gives_the_same_batch() -> None:
    assert random_problems(seed=42, count=50) == random_problems(seed=42, count=50)


def test_different_seeds_give_different_batches() -> None:
    assert random_problems(seed=42, count=20) != random_problems(seed=43, count=20)


def test_empty_batch() -> None:
    assert random_problems(seed=0, count=0) == ()
    assert coverage(()) == Coverage(0, 0, 0, 0, 0)


def test_range_of_conditions_is_respected() -> None:
    problems = random_problems(seed=3, count=100, min_conditions=1, max_conditions=1)
    assert {len(p.conditions) for p in problems} == {1}


def test_new_condition_events_keep_the_start_drugs_acceptable() -> None:
    for problem in random_problems(seed=5, count=200):
        if problem.event.kind is EventKind.NEW_CONDITION:
            assert problem.risk_factors == ()
            assert all(d is None or d in problem.acceptable[c] for c, d in problem.start.items())


def test_zero_pharmacokinetics_and_no_same_condition_interaction() -> None:
    rng = random.Random(9)
    problems = [
        random_problem(
            rng, 3, max_washout=0, max_taper=0, p_same_condition_interaction=0.0, p_bridge=0.0
        )
        for _ in range(50)
    ]
    assert all(set(p.washout_days.values()) == {0} == set(p.taper_days.values()) for p in problems)
    assert not any(map(has_same_condition_interaction, problems))
    assert not any(map(has_contraindicated_taper_start, problems))


def test_planted_bridge_has_its_pattern() -> None:
    rng = random.Random(13)
    problems = [random_problem(rng, 2, p_bridge=1.0) for _ in range(200)]
    planted = [
        p
        for p in problems
        if p.event.kind is EventKind.NEW_CONDITION
        and any(len(p.candidates[c]) >= 3 for c, d in p.start.items() if d is not None)
    ]
    assert planted
    assert all(_has_planted_bridge(p) for p in planted)
    assert all(_start_is_safe(p) for p in planted)


def test_batch_coverage_exercises_every_special_situation() -> None:
    cov = coverage(random_problems(seed=2026, count=200))
    assert cov.problems == 200
    assert cov.new_condition_events + cov.new_risk_factor_events == 200
    assert cov.same_condition_interactions >= 0.2 * cov.problems
    assert cov.contraindicated_taper_starts >= 0.1 * cov.problems
    assert min(cov.new_condition_events, cov.new_risk_factor_events) >= 0.3 * cov.problems


def test_coverage_helpers_on_the_toys() -> None:
    # T2: x1 is no longer acceptable and has a 2-day taper; T4: x1 and x2 (both of X) interact.
    assert has_contraindicated_taper_start(ej4_toys.T2.problem)
    assert not has_contraindicated_taper_start(ej4_toys.T1.problem)
    assert has_same_condition_interaction(ej4_toys.T4.problem)
    assert not has_same_condition_interaction(ej4_toys.T3.problem)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_conditions": 0},
        {"n_conditions": 10},
        {"p_interaction": -0.1},
        {"p_same_condition_interaction": 1.5},
        {"p_bridge": 2.0},
        {"max_washout": -1},
        {"max_taper": -1},
        {"max_candidates": 1},
    ],
)
def test_bad_arguments_raise(kwargs: dict[str, float]) -> None:
    arguments: dict[str, float] = {"n_conditions": 3} | kwargs
    n_conditions = int(arguments.pop("n_conditions"))
    with pytest.raises(InstanceError):
        random_problem(random.Random(0), n_conditions, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(("count", "low", "high"), [(-1, 2, 5), (5, 0, 3), (5, 4, 3), (5, 2, 10)])
def test_bad_batch_arguments_raise(count: int, low: int, high: int) -> None:
    with pytest.raises(InstanceError):
        random_problems(seed=0, count=count, min_conditions=low, max_conditions=high)
