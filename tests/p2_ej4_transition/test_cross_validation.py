"""Cross-check of the CSP against the independent oracle and checker on synthetic problems (R2).

On every problem of a seeded batch (``synthetic.random_problems``), the horizon loop of the CSP
with the main configuration and with plain backtracking + forward checking returns the oracle's
H* (or "no transition" when the oracle proves none exists), and every CSP schedule passes the
checker. The batch is also required to contain enough of each situation the real formulary never
produces, so that the agreement cannot be vacuous.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pytest

from symbolic_ai.p2_ej4_transition.checker import check_schedule
from symbolic_ai.p2_ej4_transition.oracle import OracleResult, oracle_min_horizon
from symbolic_ai.p2_ej4_transition.problem import TransitionProblem, Value, drug_of
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    SolverConfig,
    TransitionResult,
    find_min_horizon,
)
from symbolic_ai.p2_ej4_transition.synthetic import coverage, random_problems

#: The last horizon the main configuration tries (the oracle searches up to its default of 60
#: days); the largest H* in both batches is 16.
H_MAX = 20
#: The last horizon plain backtracking + FC tries. It is exponential in H on infeasible horizons
#: (failure case 1, design §7 R3): on the slowest problem of the slow batch, H = 6 takes 6 s and
#: H = 7 takes 44 s. Problems with a larger H* still check that FC finds nothing up to here.
FC_H_MAX = 6
#: Chronological backtracking with forward checking, static order, no AC-3 preprocessing.
FC_STATIC = SolverConfig(inference="fc", variable_order="static", ac3_preprocess=False)

FAST_BATCH = (2026, 200)
SLOW_BATCH = (2027, 1500)


@dataclass(frozen=True, slots=True)
class Outcome:
    """The answers of the oracle and of both CSP configurations on one problem."""

    problem: TransitionProblem
    oracle: OracleResult
    main: TransitionResult
    fc: TransitionResult


@functools.cache
def _outcomes(seed: int, count: int) -> tuple[Outcome, ...]:
    """Solve every problem of the batch with the oracle and both CSP configurations (cached)."""
    outcomes = []
    for problem in random_problems(seed, count):
        outcomes.append(
            Outcome(
                problem,
                oracle_min_horizon(problem),
                find_min_horizon(problem, MAIN_CONFIG, h_max=H_MAX),
                find_min_horizon(problem, FC_STATIC, h_max=FC_H_MAX),
            )
        )
    return tuple(outcomes)


def _disagreements(outcome: Outcome) -> list[str]:
    """Every way in which the CSP's answers differ from the oracle's or fail the checker."""
    oracle, problem_id = outcome.oracle, outcome.problem.problem_id
    if oracle.status == "budget_exhausted":
        return [f"{problem_id}: the oracle ran out of budget"]
    problems = []
    for name, result, h_max in (("main", outcome.main, H_MAX), ("fc", outcome.fc, FC_H_MAX)):
        found = oracle.h_star is not None and oracle.h_star <= h_max
        expected = oracle.h_star if found else None
        status = "solved" if found else "no_transition_within_h_max"
        if (result.h_star, result.status) != (expected, status):
            problems.append(
                f"{problem_id} {name}: CSP {result.h_star} ({result.status}), "
                f"oracle {oracle.h_star} ({oracle.status})"
            )
        if result.schedule is not None and result.h_star is not None:
            violations = check_schedule(outcome.problem, result.schedule, result.h_star)
            if violations:
                problems.append(f"{problem_id} {name}: checker rejects the schedule: {violations}")
    return problems


def _uses_bridge(outcome: Outcome) -> bool:
    """Whether the oracle's witness gives a condition a drug other than its first and last."""
    problem, schedule, horizon = outcome.problem, outcome.oracle.schedule, outcome.oracle.h_star
    if schedule is None or horizon is None:
        return False
    for condition in problem.conditions:
        row: list[Value] = [schedule[(condition, day)] for day in range(horizon + 1)]
        drugs = {drug_of(value) for value in row} - {None}
        if drugs - {problem.start[condition], drug_of(row[-1])}:
            return True
    return False


def _assert_agreement(seed: int, count: int) -> None:
    """Assert that the CSP and the oracle agree on every problem of the batch."""
    problems = [line for outcome in _outcomes(seed, count) for line in _disagreements(outcome)]
    assert problems == []


def _assert_not_vacuous(seed: int, count: int) -> None:
    """Assert that the batch exercises every situation the real formulary never produces."""
    outcomes = _outcomes(seed, count)
    cov = coverage([outcome.problem for outcome in outcomes])
    impossible = [o for o in outcomes if o.oracle.status == "impossible"]
    h_stars = [o.oracle.h_star for o in outcomes if o.oracle.h_star is not None]
    assert cov.same_condition_interactions >= 0.20 * count
    assert cov.contraindicated_taper_starts >= 0.10 * count
    assert min(cov.new_condition_events, cov.new_risk_factor_events) >= 0.30 * count
    assert len(impossible) >= 0.05 * count
    # Impossible because of interactions, not merely because a condition lost every drug.
    blocked = [o for o in impossible if all(o.problem.acceptable.values())]
    assert len(blocked) >= 0.05 * count
    assert sum(h >= 2 for h in h_stars) >= 0.10 * count
    assert sum(h >= 5 for h in h_stars) >= 0.02 * count
    assert sum(map(_uses_bridge, outcomes)) >= 0.02 * count


def test_fast_batch_csp_agrees_with_oracle_and_checker() -> None:
    _assert_agreement(*FAST_BATCH)


def test_fast_batch_is_not_vacuous() -> None:
    _assert_not_vacuous(*FAST_BATCH)


@pytest.mark.slow
def test_slow_batch_csp_agrees_with_oracle_and_checker() -> None:
    _assert_agreement(*SLOW_BATCH)


@pytest.mark.slow
def test_slow_batch_is_not_vacuous() -> None:
    _assert_not_vacuous(*SLOW_BATCH)
