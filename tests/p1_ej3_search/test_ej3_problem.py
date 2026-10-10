"""Tests for symbolic_ai.p1_ej3_search.problem: states, actions, costs and the heuristic."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Formulary
from symbolic_ai.p1_ej1_logic import DrugClassification, FullRecord
from symbolic_ai.p1_ej3_search import CostError, DomainPruning, RecordError, RegimenProblem
from symbolic_ai.p1_ej3_search.problem import State

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ej3_fixture_data import toy_formulary  # noqa: E402

EMPTY: State = frozenset()


def _record(conditions: str, risk_factors: str = "") -> FullRecord:
    """Build a record from space-separated identifiers."""
    return FullRecord(frozenset(conditions.split()), frozenset(risk_factors.split()))


def _problem(
    formulary: Formulary,
    costs: Mapping[str, int],
    conditions: str,
    risk_factors: str = "",
    pruning: DomainPruning | None = None,
) -> RegimenProblem:
    return RegimenProblem(formulary, costs, _record(conditions, risk_factors), pruning)


# --- states, actions and the transition model ---------------------------------------------------


def test_initial_state_is_the_empty_regimen() -> None:
    problem = _problem(toy_formulary({"X": "a"}), {"a": 1}, "X")
    assert problem.initial == EMPTY
    assert not problem.goal_test(EMPTY)


def test_a_record_without_conditions_is_already_a_goal() -> None:
    problem = _problem(toy_formulary({"X": "a"}), {"a": 1}, "")
    assert problem.goal_test(EMPTY)
    assert problem.actions(EMPTY) == ()
    assert problem.heuristic(EMPTY) == 0


def test_actions_are_the_sorted_candidates_of_the_first_uncovered_condition() -> None:
    problem = _problem(toy_formulary({"X": "ba", "Y": "c"}), {"a": 1, "b": 1, "c": 1}, "X Y")
    assert problem.uncovered(EMPTY) == ("X", "Y")
    assert problem.actions(EMPTY) == ("a", "b")
    assert problem.actions(frozenset({"a"})) == ("c",)


def test_a_goal_state_has_no_actions() -> None:
    problem = _problem(toy_formulary({"X": "a", "Y": "c"}), {"a": 1, "c": 1}, "X Y")
    goal = frozenset({"a", "c"})
    assert problem.goal_test(goal)
    assert problem.actions(goal) == ()


def test_conditions_absent_from_the_record_are_not_covered() -> None:
    problem = _problem(toy_formulary({"X": "a", "Y": "c"}), {"a": 1, "c": 1}, "Y")
    assert problem.actions(EMPTY) == ("c",)
    assert problem.goal_test(frozenset({"c"}))


@pytest.mark.parametrize(("risk_factors", "offered"), [("", ("a", "b")), ("R", ("b",))])
def test_a4_a_contraindicated_drug_is_offered_only_without_its_risk_factor(
    risk_factors: str, offered: tuple[str, ...]
) -> None:
    formulary = toy_formulary({"X": "ab"}, risk_factors="R", contraindications=[("R", "a")])
    problem = _problem(formulary, {"a": 1, "b": 1}, "X", risk_factors)
    assert problem.actions(EMPTY) == offered


def test_a3_a_drug_interacting_with_a_prescribed_one_is_not_offered() -> None:
    formulary = toy_formulary({"X": "a", "Y": "bc"}, interactions=[("a", "b")])
    problem = _problem(formulary, {"a": 1, "b": 1, "c": 1}, "X Y")
    assert problem.actions(frozenset({"a"})) == ("c",)


def test_a6_a_second_drug_of_a_family_is_not_offered() -> None:
    formulary = toy_formulary({"X": "a", "Y": "bc"}, families=["ab"])
    problem = _problem(formulary, {"a": 1, "b": 1, "c": 1}, "X Y")
    assert problem.actions(frozenset({"a"})) == ("c",)


@pytest.mark.parametrize(
    ("risk_factors", "reached"), [("", {"a"}), ("R", {"a", "g", "s"}), ("Q", {"a"})]
)
def test_a5_the_result_adds_the_companions_required_under_the_risk_factor(
    risk_factors: str, reached: set[str]
) -> None:
    """``a`` requires ``g`` and ``g`` requires ``s``, both under R: the bundle is transitive."""
    formulary = toy_formulary(
        {"X": "a"},
        extra_drugs="gs",
        risk_factors="RQ",
        coprescriptions=[("a", "R", "g"), ("g", "R", "s")],
    )
    problem = _problem(formulary, {"a": 1, "g": 1, "s": 1}, "X", risk_factors)
    assert problem.result(EMPTY, "a") == reached


def test_a_companion_covers_the_condition_it_treats() -> None:
    formulary = toy_formulary(
        {"X": "a", "Y": "g"}, risk_factors="R", coprescriptions=[("a", "R", "g")]
    )
    problem = _problem(formulary, {"a": 1, "g": 1}, "X Y", "R")
    assert problem.goal_test(problem.result(EMPTY, "a"))


def test_a_drug_whose_companion_is_contraindicated_is_not_offered() -> None:
    formulary = toy_formulary(
        {"X": "ab"},
        extra_drugs="g",
        risk_factors="R",
        contraindications=[("R", "g")],
        coprescriptions=[("a", "R", "g")],
    )
    problem = _problem(formulary, {"a": 1, "b": 1, "g": 1}, "X", "R")
    assert problem.actions(EMPTY) == ("b",)


def test_a_drug_whose_companion_conflicts_with_the_state_is_not_offered() -> None:
    formulary = toy_formulary(
        {"X": "c", "Y": "ab"},
        extra_drugs="g",
        risk_factors="R",
        interactions=[("c", "g")],
        coprescriptions=[("a", "R", "g")],
    )
    problem = _problem(formulary, {"a": 1, "b": 1, "c": 1, "g": 1}, "X Y", "R")
    assert problem.actions(frozenset({"c"})) == ("b",)


def test_a_drug_that_interacts_with_its_own_companion_is_never_offered() -> None:
    formulary = toy_formulary(
        {"X": "ab"},
        extra_drugs="g",
        risk_factors="R",
        interactions=[("a", "g")],
        coprescriptions=[("a", "R", "g")],
    )
    problem = _problem(formulary, {"a": 1, "b": 1, "g": 1}, "X", "R")
    assert problem.actions(EMPTY) == ("b",)


# --- costs --------------------------------------------------------------------------------------


def test_path_cost_adds_the_drug_and_its_new_companions() -> None:
    formulary = toy_formulary(
        {"X": "a"}, extra_drugs="g", risk_factors="R", coprescriptions=[("a", "R", "g")]
    )
    problem = _problem(formulary, {"a": 4, "g": 9}, "X", "R")
    state = problem.result(EMPTY, "a")
    assert problem.path_cost(0, EMPTY, "a", state) == 13


def test_a_companion_already_prescribed_is_not_paid_twice() -> None:
    formulary = toy_formulary(
        {"X": "g", "Y": "a"}, risk_factors="R", coprescriptions=[("a", "R", "g")]
    )
    problem = _problem(formulary, {"a": 4, "g": 9}, "X Y", "R")
    first = problem.result(EMPTY, "g")
    second = problem.result(first, "a")
    cost = problem.path_cost(problem.path_cost(0, EMPTY, "g", first), first, "a", second)
    assert cost == 13 == problem.cost(second)


def test_cost_of_a_drug_outside_the_formulary_raises() -> None:
    problem = _problem(toy_formulary({"X": "a"}), {"a": 1}, "X")
    with pytest.raises(RecordError, match="aspirin"):
        problem.cost(frozenset({"aspirin"}))


# --- the heuristic ------------------------------------------------------------------------------


def test_heuristic_adds_the_cheapest_candidate_of_each_uncovered_condition() -> None:
    formulary = toy_formulary({"X": "ab", "Y": "cd"})
    problem = _problem(formulary, {"a": 5, "b": 3, "c": 7, "d": 11}, "X Y")
    assert problem.heuristic(EMPTY) == 3 + 7
    assert problem.heuristic(frozenset({"a"})) == 7
    assert problem.heuristic(frozenset({"a", "d"})) == 0


def test_heuristic_ignores_candidates_that_conflict_with_the_state() -> None:
    formulary = toy_formulary({"X": "a", "Y": "cd"}, interactions=[("a", "c")])
    problem = _problem(formulary, {"a": 5, "c": 7, "d": 11}, "X Y")
    assert problem.heuristic(EMPTY) == 5 + 7
    assert problem.heuristic(frozenset({"a"})) == 11


def test_heuristic_does_not_charge_the_companion() -> None:
    formulary = toy_formulary(
        {"X": "a"}, extra_drugs="g", risk_factors="R", coprescriptions=[("a", "R", "g")]
    )
    problem = _problem(formulary, {"a": 4, "g": 9}, "X", "R")
    assert problem.heuristic(EMPTY) == 4


def test_heuristic_is_infinite_when_a_condition_has_no_compatible_candidate() -> None:
    formulary = toy_formulary({"X": "ab", "Y": "c"}, interactions=[("a", "c")])
    problem = _problem(formulary, {"a": 1, "b": 2, "c": 1}, "X Y")
    assert problem.heuristic(frozenset({"a"})) == math.inf
    assert problem.actions(frozenset({"a"})) == ()


def test_heuristic_shares_the_cost_of_a_drug_among_the_conditions_it_treats() -> None:
    """``m`` treats X and Y for 10: charging 10 to each would overestimate the optimum, 10."""
    formulary = toy_formulary({"X": "mx", "Y": "my"})
    problem = _problem(formulary, {"m": 10, "x": 8, "y": 9}, "X Y")
    assert problem.heuristic(EMPTY) == Fraction(10, 2) + Fraction(10, 2)
    assert problem.heuristic(frozenset({"x"})) == 9


def test_h_reads_the_state_of_a_search_node() -> None:
    class Node:
        state: State = frozenset({"a"})

    problem = _problem(toy_formulary({"X": "ab", "Y": "c"}), {"a": 1, "b": 1, "c": 6}, "X Y")
    assert problem.h(Node()) == 6


# --- the pruning of Agent 1 ---------------------------------------------------------------------


def test_an_essential_candidate_is_the_only_action() -> None:
    formulary = toy_formulary({"X": "abc"})
    pruning = DomainPruning(essential=frozenset({"b"}))
    problem = _problem(formulary, {"a": 1, "b": 2, "c": 3}, "X", pruning=pruning)
    assert problem.actions(EMPTY) == ("b",)


def test_an_excluded_drug_is_not_offered_alone_or_as_a_companion() -> None:
    formulary = toy_formulary(
        {"X": "abc"}, extra_drugs="g", risk_factors="R", coprescriptions=[("a", "R", "g")]
    )
    pruning = DomainPruning(excluded=frozenset({"c", "g"}))
    problem = _problem(formulary, {"a": 1, "b": 2, "c": 3, "g": 1}, "X", "R", pruning)
    assert problem.actions(EMPTY) == ("b",)
    assert problem.heuristic(EMPTY) == 2


def test_pruning_from_a_classification_keeps_optional_drugs_free() -> None:
    pruning = DomainPruning.from_classification(
        {
            "a": DrugClassification.ESSENTIAL,
            "b": DrugClassification.EXCLUDED,
            "c": DrugClassification.OPTIONAL,
        }
    )
    assert pruning == DomainPruning(essential=frozenset({"a"}), excluded=frozenset({"b"}))


# --- failure paths ------------------------------------------------------------------------------


def test_a_drug_without_a_cost_raises() -> None:
    with pytest.raises(CostError, match=r"without a cost: \['b'\]"):
        _problem(toy_formulary({"X": "ab"}), {"a": 1}, "X")


def test_a_negative_cost_raises() -> None:
    with pytest.raises(CostError, match=r"negative cost: \[\('b', -3\)\]"):
        _problem(toy_formulary({"X": "ab"}), {"a": 1, "b": -3}, "X")


@pytest.mark.parametrize(
    ("conditions", "risk_factors", "pruning", "named"),
    [
        ("X NOPE", "", None, "conditions ['NOPE']"),
        ("X", "NOPE", None, "risk factors ['NOPE']"),
        ("X", "", DomainPruning(essential=frozenset({"nope"})), "drugs ['nope']"),
        ("X", "", DomainPruning(excluded=frozenset({"nope"})), "drugs ['nope']"),
    ],
)
def test_an_identifier_outside_the_formulary_raises(
    conditions: str, risk_factors: str, pruning: DomainPruning | None, named: str
) -> None:
    with pytest.raises(RecordError) as excinfo:
        _problem(toy_formulary({"X": "a"}), {"a": 1}, conditions, risk_factors, pruning)
    assert named in str(excinfo.value)


# --- admissibility and consistency, checked on every reachable state ----------------------------


def _reachable(problem: RegimenProblem) -> set[State]:
    seen = {problem.initial}
    pending = [problem.initial]
    while pending:
        state = pending.pop()
        for action in problem.actions(state):
            child = problem.result(state, action)
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return seen


def _cost_to_go(problem: RegimenProblem, state: State, memo: dict[State, float]) -> float:
    """The true cost of the cheapest completion of ``state`` (``inf`` if there is none)."""
    if state in memo:
        return memo[state]
    best = 0.0 if problem.goal_test(state) else math.inf
    for action in problem.actions(state):
        child = problem.result(state, action)
        step = problem.path_cost(0, state, action, child)
        best = min(best, step + _cost_to_go(problem, child, memo))
    memo[state] = best
    return best


_TOY_COUPLED = toy_formulary(
    {"X": "mxa", "Y": "myb", "Z": "g"},
    extra_drugs="s",
    risk_factors="R",
    interactions=[("x", "y"), ("a", "g")],
    contraindications=[("R", "b")],
    coprescriptions=[("a", "R", "g"), ("x", "R", "s")],
    families=["my"],
)
_TOY_COSTS = {"m": 10, "x": 3, "a": 1, "y": 2, "b": 1, "g": 6, "s": 5}


def _cases() -> list[tuple[str, str, str]]:
    """(formulary name, conditions, risk factors) of the problems whose heuristic is checked."""
    return [
        ("toy", "X Y Z", ""),
        ("toy", "X Y Z", "R"),
        ("toy", "X Y", "R"),
        ("full", "AF DEP GERD HTN PAIN T2D", ""),
        ("full", "AF DEP HTN PAIN T2D", "AGE65 CKD"),
        ("full", "AF DEP PAIN", "AGE65 LIVER"),
        ("full", "DEP GERD PAIN", "AGE65 EPI LIVER"),
        ("full", "AF PAIN", "EPI LIVER"),
    ]


@pytest.fixture(params=_cases(), ids=lambda case: f"{case[0]}:{case[1]}|{case[2]}")
def checked_problem(
    request: pytest.FixtureRequest, formulary: Formulary, costs: Mapping[str, int]
) -> RegimenProblem:
    name, conditions, risk_factors = request.param
    if name == "toy":
        return _problem(_TOY_COUPLED, _TOY_COSTS, conditions, risk_factors)
    return _problem(formulary, costs, conditions, risk_factors)


def test_heuristic_never_overestimates_the_cheapest_completion(
    checked_problem: RegimenProblem,
) -> None:
    memo: dict[State, float] = {}
    for state in _reachable(checked_problem):
        assert checked_problem.heuristic(state) <= _cost_to_go(checked_problem, state, memo)


def test_heuristic_is_consistent_along_every_action(checked_problem: RegimenProblem) -> None:
    for state in _reachable(checked_problem):
        for action in checked_problem.actions(state):
            child = checked_problem.result(state, action)
            step = checked_problem.path_cost(0, state, action, child)
            assert checked_problem.heuristic(state) <= step + checked_problem.heuristic(child)


def test_heuristic_is_zero_exactly_on_goal_states(checked_problem: RegimenProblem) -> None:
    for state in _reachable(checked_problem):
        assert (checked_problem.heuristic(state) == 0) == checked_problem.goal_test(state)
