"""Tests for ``RegimenSpace.regimens``: the oracle's list of the models it counts (used by EJ3)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic import FullRecord, OracleError, RegimenSpace, unknown_risk_factors


def _gamma_plus(encounter: Encounter, formulary: Formulary) -> FullRecord:
    present = encounter.risk_factors_with(RiskStatus.PRESENT)
    return FullRecord(encounter.conditions, present | unknown_risk_factors(encounter, formulary))


@pytest.mark.parametrize("encounter_id", ["E001", "E002", "E003", "E004", "E005"])
def test_regimens_are_the_models_counted(
    formulary: Formulary, encounters: Mapping[str, Encounter], encounter_id: str
) -> None:
    space = RegimenSpace(formulary)
    record = _gamma_plus(encounters[encounter_id], formulary)
    regimens = space.regimens(record)
    assert len(regimens) == len(set(regimens)) == space.count(record)
    assert all(space.is_valid(record, regimen) for regimen in regimens)


def test_regimens_are_ordered_by_size_then_by_drug(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    space = RegimenSpace(formulary)
    regimens = space.regimens(_gamma_plus(encounters["E003"], formulary))
    keys = [(len(regimen), sorted(regimen)) for regimen in regimens]
    assert keys == sorted(keys)
    assert regimens[0] == {"amlodipine", "apixaban"}
    assert len(regimens[0]) == space.minimum_size(_gamma_plus(encounters["E003"], formulary))


def test_a_forced_record_has_one_regimen_and_an_untreatable_one_none(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    space = RegimenSpace(formulary)
    forced = space.regimens(_gamma_plus(encounters["E004"], formulary))
    assert forced == (frozenset({"ibuprofen", "mirtazapine", "omeprazole"}),)
    assert space.regimens(_gamma_plus(encounters["E005"], formulary)) == ()


def test_a_record_without_conditions_has_the_empty_regimen_only(formulary: Formulary) -> None:
    assert RegimenSpace(formulary).regimens(FullRecord(frozenset(), frozenset())) == (frozenset(),)


def test_regimens_of_an_unknown_condition_raises(formulary: Formulary) -> None:
    with pytest.raises(OracleError, match="GOUT"):
        RegimenSpace(formulary).regimens(FullRecord(frozenset({"GOUT"}), frozenset()))
