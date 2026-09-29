"""Tests for symbolic_ai.p1_ej1_logic.explain: deletion-based minimal unsatisfiable subsets."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import (
    AxiomTag,
    Clause,
    DPLLSolver,
    ExplanationError,
    Literal,
    gamma,
    gamma_minus,
    gamma_plus,
    minimal_unsatisfiable_subset,
    unknown_risk_factors,
)


def _unit(symbol: str, positive: bool) -> Clause:
    return Clause((Literal(symbol, positive),), AxiomTag.PERCEPT, symbol)


def _tags(clauses: tuple[Clause, ...]) -> set[tuple[str, str]]:
    return {(clause.axiom.value, clause.source) for clause in clauses}


def _assert_minimal(subset: tuple[Clause, ...], solver: DPLLSolver) -> None:
    assert not solver.solve(subset).satisfiable
    for index in range(len(subset)):
        rest = subset[:index] + subset[index + 1 :]
        assert solver.solve(rest).satisfiable, f"clause {subset[index]} is not necessary"


def test_contradictory_units_are_the_whole_conflict() -> None:
    solver = DPLLSolver()
    clauses = (_unit("A", True), _unit("B", True), _unit("A", False))
    assert minimal_unsatisfiable_subset(clauses, solver) == (clauses[0], clauses[2])


def test_e002_worst_case_conflict_is_af_under_pregnancy(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    solver = DPLLSolver()
    encounter = encounters["E002"]
    base = gamma(formulary, encounter)
    subset = minimal_unsatisfiable_subset(
        gamma_plus(base, unknown_risk_factors(encounter, formulary)), solver
    )
    assert _tags(subset) == {
        ("A1", "AF"),
        ("A4", "PREG,apixaban"),
        ("A4", "PREG,warfarin"),
        ("PERCEPT", "AF"),
        ("ASSUMPTION", "PREG"),
    }
    _assert_minimal(subset, solver)


def test_e005_conflict_leaves_pain_without_analgesic(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    solver = DPLLSolver()
    encounter = encounters["E005"]
    base = gamma(formulary, encounter)
    subset = minimal_unsatisfiable_subset(
        gamma_minus(base, unknown_risk_factors(encounter, formulary)), solver
    )
    assert _tags(subset) == {
        ("A1", "PAIN"),
        ("A4", "CKD,ibuprofen"),
        ("A4", "EPI,tramadol"),
        ("A4", "LIVER,paracetamol"),
        ("PERCEPT", "PAIN"),
        ("PERCEPT", "CKD"),
        ("PERCEPT", "EPI"),
        ("PERCEPT", "LIVER"),
    }
    _assert_minimal(subset, solver)


def test_satisfiable_input_raises(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    with pytest.raises(ExplanationError, match="satisfiable"):
        minimal_unsatisfiable_subset(gamma(formulary, encounters["E004"]), DPLLSolver())


def test_empty_input_raises() -> None:
    with pytest.raises(ExplanationError):
        minimal_unsatisfiable_subset((), DPLLSolver())
