"""Tests for symbolic_ai.p1_ej1_logic.semantics: the set-semantics oracle of axioms A1-A6."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import date
from itertools import combinations
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Drug, Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic import (
    FullRecord,
    OracleError,
    RegimenSpace,
    gamma,
    unknown_risk_factors,
)

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from oracle import count_models_of_clauses  # noqa: E402


def _bounds(encounter: Encounter, formulary: Formulary) -> tuple[FullRecord, FullRecord]:
    """Γ⁺ and Γ⁻ of ``encounter`` as full records."""
    present = encounter.risk_factors_with(RiskStatus.PRESENT)
    unknowns = unknown_risk_factors(encounter, formulary)
    return (
        FullRecord(encounter.conditions, present | unknowns),
        FullRecord(encounter.conditions, present),
    )


def test_a3_and_a6_leave_18432_regimens(formulary: Formulary) -> None:
    assert RegimenSpace(formulary).n_candidate_regimens == 18_432


@pytest.mark.parametrize(
    ("encounter_id", "plus_models", "minus_models"),
    [("E001", 115, 1645), ("E002", 0, 94), ("E003", 94, 94), ("E004", 1, 1), ("E005", 0, 0)],
)
def test_reference_model_counts(
    formulary: Formulary,
    encounters: Mapping[str, Encounter],
    encounter_id: str,
    plus_models: int,
    minus_models: int,
) -> None:
    space = RegimenSpace(formulary)
    plus, minus = _bounds(encounters[encounter_id], formulary)
    assert space.count(plus) == plus_models
    assert space.count(minus) == minus_models


def test_counts_agree_with_the_truth_table_over_every_mini_record(
    elena_formulary: Formulary,
) -> None:
    """Set semantics and the CNF truth table are independent; they must agree on every record."""
    space = RegimenSpace(elena_formulary)
    risk_ids = elena_formulary.risk_factor_ids
    for size in range(1, len(elena_formulary.condition_ids) + 1):
        for conditions in combinations(elena_formulary.condition_ids, size):
            for n_present in range(len(risk_ids) + 1):
                for present in combinations(risk_ids, n_present):
                    statuses = {
                        r: RiskStatus.PRESENT if r in present else RiskStatus.ABSENT
                        for r in risk_ids
                    }
                    encounter = Encounter(
                        "X", "P", 1, date(2026, 1, 1), 50, frozenset(conditions), statuses
                    )
                    expected = count_models_of_clauses(gamma(elena_formulary, encounter))
                    record = FullRecord(frozenset(conditions), frozenset(present))
                    assert space.count(record) == expected, (conditions, present)


def test_minimum_size(formulary: Formulary, encounters: Mapping[str, Encounter]) -> None:
    space = RegimenSpace(formulary)
    assert space.minimum_size(_bounds(encounters["E001"], formulary)[0]) == 5
    assert space.minimum_size(_bounds(encounters["E004"], formulary)[0]) == 3
    assert space.minimum_size(_bounds(encounters["E005"], formulary)[0]) is None


@pytest.mark.parametrize(
    ("conditions", "present", "regimen", "violated"),
    [
        (("AF", "PAIN"), ("AGE65",), ("paracetamol", "warfarin"), ()),
        (("PAIN",), ("AGE65",), (), ("A1",)),
        (("PAIN",), ("AGE65",), ("omeprazole", "paracetamol"), ("A2",)),
        (("AF", "PAIN"), ("AGE65",), ("ibuprofen", "omeprazole", "warfarin"), ("A3",)),
        (("AF", "PAIN"), ("LIVER",), ("paracetamol", "warfarin"), ("A4",)),
        (("PAIN",), ("AGE65",), ("ibuprofen",), ("A5",)),
        (("PAIN",), ("AGE65",), ("ibuprofen", "omeprazole", "paracetamol"), ("A6",)),
    ],
)
def test_violations_name_each_axiom(
    elena_formulary: Formulary,
    conditions: tuple[str, ...],
    present: tuple[str, ...],
    regimen: tuple[str, ...],
    violated: tuple[str, ...],
) -> None:
    space = RegimenSpace(elena_formulary)
    record = FullRecord(frozenset(conditions), frozenset(present))
    assert space.violations(record, regimen) == violated
    assert space.is_valid(record, regimen) is (not violated)


class TestFailurePaths:
    def test_unknown_condition_raises(self, formulary: Formulary) -> None:
        with pytest.raises(OracleError, match="outside formulary"):
            RegimenSpace(formulary).count(FullRecord(frozenset({"ASTHMA_ATTACK"}), frozenset()))

    def test_unknown_risk_factor_raises(self, formulary: Formulary) -> None:
        with pytest.raises(OracleError, match="outside formulary"):
            RegimenSpace(formulary).minimum_size(FullRecord(frozenset({"HTN"}), frozenset({"X"})))

    def test_unknown_drug_in_regimen_raises(self, formulary: Formulary) -> None:
        with pytest.raises(OracleError, match="outside the formulary"):
            RegimenSpace(formulary).is_valid(
                FullRecord(frozenset({"HTN"}), frozenset()), ["aspirin"]
            )

    def test_too_many_drugs_raises(self) -> None:
        drugs = tuple(Drug(f"d{i:02d}", "x", "x") for i in range(23))
        formulary = Formulary("big", (), drugs, (), {}, (), (), ())
        with pytest.raises(OracleError, match="at most 22"):
            RegimenSpace(formulary)
