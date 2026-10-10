"""Tests for symbolic_ai.p1_ej3_search.search: the four algorithms against the reference oracle."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Formulary
from symbolic_ai.p1_ej1_logic import FullRecord, RegimenSpace
from symbolic_ai.p1_ej3_search import Algorithm, DomainPruning, RegimenProblem, search

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ej3_fixture_data import toy_formulary  # noqa: E402

OPTIMAL = [algorithm for algorithm in Algorithm if algorithm.optimal]


def _record(conditions: str, risk_factors: str = "") -> FullRecord:
    return FullRecord(frozenset(conditions.split()), frozenset(risk_factors.split()))


def _subsets(items: tuple[str, ...], smallest: int = 0) -> list[frozenset[str]]:
    return [
        frozenset(combo)
        for size in range(smallest, len(items) + 1)
        for combo in combinations(items, size)
    ]


def test_only_greedy_is_not_optimal() -> None:
    assert [algorithm for algorithm in Algorithm if not algorithm.optimal] == [Algorithm.GREEDY]


# --- reference regimens on the full formulary ---------------------------------------------------


@pytest.mark.parametrize("algorithm", OPTIMAL)
@pytest.mark.parametrize(
    ("conditions", "risk_factors", "regimen", "cost"),
    [
        # E001 under Γ⁺: CKD leaves linagliptin; warfarin forces mirtazapine.
        (
            "AF DEP HTN PAIN T2D",
            "AGE65 CKD",
            {"enalapril", "linagliptin", "mirtazapine", "paracetamol", "warfarin"},
            5010,
        ),
        ("AF HTN", "", {"hydrochlorothiazide", "warfarin"}, 400),
        ("DEP PAIN", "AGE65 EPI LIVER", {"ibuprofen", "mirtazapine", "omeprazole"}, 1110),
        # The cheapest drug of each condition, warfarin and sertraline, interact.
        ("AF DEP", "", {"mirtazapine", "warfarin"}, 870),
    ],
)
def test_optimal_algorithms_return_the_cheapest_regimen(
    formulary: Formulary,
    costs: Mapping[str, int],
    algorithm: Algorithm,
    conditions: str,
    risk_factors: str,
    regimen: set[str],
    cost: int,
) -> None:
    result = search(RegimenProblem(formulary, costs, _record(conditions, risk_factors)), algorithm)
    assert result.algorithm is algorithm
    assert result.regimen == regimen
    assert result.cost == cost


def test_the_companion_decides_between_two_analgesics(
    formulary: Formulary, costs: Mapping[str, int]
) -> None:
    """At 65 with liver disease ibuprofen (2.30) needs omeprazole (2.70): tramadol (4.80) wins.

    With reflux the omeprazole is prescribed anyway, and ibuprofen is the cheaper analgesic.
    """
    alone = search(RegimenProblem(formulary, costs, _record("PAIN", "AGE65 LIVER")))
    with_reflux = search(RegimenProblem(formulary, costs, _record("GERD PAIN", "AGE65 LIVER")))
    assert (alone.regimen, alone.cost) == ({"tramadol"}, 480)
    assert (with_reflux.regimen, with_reflux.cost) == ({"ibuprofen", "omeprazole"}, 500)


def test_chosen_lists_one_drug_per_condition_and_no_companion(
    formulary: Formulary, costs: Mapping[str, int]
) -> None:
    result = search(RegimenProblem(formulary, costs, _record("DEP PAIN", "AGE65 EPI LIVER")))
    assert result.chosen == ("mirtazapine", "ibuprofen")
    assert result.regimen == {"ibuprofen", "mirtazapine", "omeprazole"}


def test_greedy_ignores_the_cost_already_paid(
    formulary: Formulary, costs: Mapping[str, int]
) -> None:
    """After apixaban (54.00) h is 4.20, after warfarin (2.60) it is 6.10: greedy takes apixaban.

    Both antidepressants then reach a goal with h = 0, and the first generated is returned.
    """
    result = search(RegimenProblem(formulary, costs, _record("AF DEP")), Algorithm.GREEDY)
    assert result.regimen == {"apixaban", "mirtazapine"}
    assert result.cost == 6010


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_a_record_without_conditions_needs_no_drug(
    formulary: Formulary, costs: Mapping[str, int], algorithm: Algorithm
) -> None:
    result = search(RegimenProblem(formulary, costs, _record("")), algorithm)
    assert (result.regimen, result.cost, result.chosen) == (frozenset(), 0, ())
    assert result.expanded == 0


@pytest.mark.parametrize("algorithm", list(Algorithm))
@pytest.mark.parametrize(
    ("conditions", "risk_factors"),
    [("AF", "PREG"), ("PAIN", "CKD EPI LIVER"), ("AF PAIN", "EPI LIVER"), ("AF DEP HTN", "PREG")],
)
def test_an_untreatable_record_has_no_regimen(
    formulary: Formulary,
    costs: Mapping[str, int],
    algorithm: Algorithm,
    conditions: str,
    risk_factors: str,
) -> None:
    result = search(RegimenProblem(formulary, costs, _record(conditions, risk_factors)), algorithm)
    assert (result.regimen, result.cost, result.chosen) == (None, None, ())


# --- effort -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("algorithm", "expanded", "generated", "goal_tests"),
    [
        (Algorithm.ASTAR, 5, 12, 6),
        (Algorithm.IDASTAR, 6, 14, 7),
        (Algorithm.GREEDY, 5, 13, 6),
        (Algorithm.UNIFORM_COST, 21, 33, 22),
    ],
)
def test_effort_on_e001(
    formulary: Formulary,
    costs: Mapping[str, int],
    algorithm: Algorithm,
    expanded: int,
    generated: int,
    goal_tests: int,
) -> None:
    problem = RegimenProblem(formulary, costs, _record("AF DEP HTN PAIN T2D", "AGE65 CKD"))
    result = search(problem, algorithm)
    assert (result.expanded, result.generated, result.goal_tests) == (
        expanded,
        generated,
        goal_tests,
    )


def test_search_is_repeatable(formulary: Formulary, costs: Mapping[str, int]) -> None:
    problem = RegimenProblem(formulary, costs, _record("AF DEP GERD HTN PAIN T2D"))
    assert search(problem) == search(problem)


# --- a formulary where the heuristic shares costs -----------------------------------------------


@pytest.mark.parametrize("algorithm", OPTIMAL)
def test_a_drug_that_treats_two_conditions_is_found_when_it_is_cheaper(
    algorithm: Algorithm,
) -> None:
    """``m`` (10) covers X and Y; one drug each costs 8 + 9 = 17."""
    formulary = toy_formulary({"X": "mx", "Y": "my"})
    problem = RegimenProblem(formulary, {"m": 10, "x": 8, "y": 9}, _record("X Y"))
    result = search(problem, algorithm)
    assert (result.regimen, result.cost) == ({"m"}, 10)


@pytest.mark.parametrize("algorithm", OPTIMAL)
def test_two_single_drugs_are_found_when_the_shared_one_is_dearer(algorithm: Algorithm) -> None:
    formulary = toy_formulary({"X": "mx", "Y": "my"})
    problem = RegimenProblem(formulary, {"m": 10, "x": 4, "y": 5}, _record("X Y"))
    result = search(problem, algorithm)
    assert (result.regimen, result.cost) == ({"x", "y"}, 9)


# --- every fully observed record against the oracle ---------------------------------------------


def _check_against_the_oracle(
    formulary: Formulary,
    costs: Mapping[str, int],
    space: RegimenSpace,
    risk_factor_sets: list[frozenset[str]],
) -> int:
    """Check every algorithm on every condition set with each of ``risk_factor_sets``.

    The verdict, the minimum cost, and A* expanding no more than uniform cost. The pruning is
    taken from the oracle here (drugs in every model, drugs in none); Agent 1's own
    classification is exercised in ``test_ej3_agent.py`` and by ``main --experiments``. Returns
    the number of satisfiable records.
    """
    satisfiable = 0
    for conditions in _subsets(formulary.condition_ids, smallest=1):
        for present in risk_factor_sets:
            record = FullRecord(conditions, present)
            regimens = space.regimens(record)
            plain = RegimenProblem(formulary, costs, record)
            if not regimens:
                assert all(search(plain, algorithm).regimen is None for algorithm in Algorithm)
                continue
            satisfiable += 1
            minimum = min(plain.cost(regimen) for regimen in regimens)
            pruning = DomainPruning(
                essential=frozenset.intersection(*regimens),
                excluded=frozenset(formulary.drug_ids) - frozenset.union(*regimens),
            )
            results = {algorithm: search(plain, algorithm) for algorithm in Algorithm}
            assert all(result.regimen in regimens for result in results.values())
            assert all(results[algorithm].cost == minimum for algorithm in OPTIMAL)
            assert results[Algorithm.ASTAR].expanded <= results[Algorithm.UNIFORM_COST].expanded
            pruned = search(RegimenProblem(formulary, costs, record, pruning))
            assert (pruned.regimen, pruned.cost) == (results[Algorithm.ASTAR].regimen, minimum)
            assert pruned.generated <= results[Algorithm.ASTAR].generated
    return satisfiable


def test_records_with_at_most_one_risk_factor_match_the_oracle(
    formulary: Formulary, costs: Mapping[str, int], space: RegimenSpace
) -> None:
    few = [present for present in _subsets(formulary.risk_factor_ids) if len(present) <= 1]
    assert _check_against_the_oracle(formulary, costs, space, few) == 409


@pytest.mark.slow
def test_every_record_matches_the_oracle(
    formulary: Formulary, costs: Mapping[str, int], space: RegimenSpace
) -> None:
    every = _subsets(formulary.risk_factor_ids)
    assert _check_against_the_oracle(formulary, costs, space, every) == 2752
