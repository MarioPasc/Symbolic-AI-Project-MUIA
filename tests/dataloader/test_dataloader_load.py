"""Loader tests against the real, committed database (docs 01-database.md, section 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import (
    RiskStatus,
    UnknownIdError,
    default_data_dir,
    load_encounter,
    load_encounters,
    load_formulary,
    load_patients,
)

pytestmark = pytest.mark.integration


def test_formulary_table_counts(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir)
    assert len(formulary.conditions) == 6
    assert len(formulary.drugs) == 18
    assert len(formulary.risk_factors) == 6
    assert sum(len(drugs) for drugs in formulary.candidates.values()) == 18
    assert len(formulary.interactions) == 7
    assert len(formulary.contraindications) == 11
    assert len(formulary.coprescriptions) == 1
    assert len(formulary.drug_classes) == 5
    assert sum(len(members) for members in formulary.class_members.values()) == 11


def test_exclusive_family_pair_count_is_seven(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir)
    families = formulary.exclusive_families()
    assert len(families) == 5
    pair_count = sum(len(family) * (len(family) - 1) // 2 for family in families)
    assert pair_count == 7


def test_indications_of_omeprazole_is_gerd_only(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir)
    assert formulary.indications("omeprazole") == frozenset({"GERD"})


def test_formulary_tuples_are_sorted_by_id(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir)
    assert [c.condition_id for c in formulary.conditions] == sorted(formulary.condition_ids)
    assert [d.drug_id for d in formulary.drugs] == sorted(formulary.drug_ids)
    assert [r.risk_factor_id for r in formulary.risk_factors] == sorted(formulary.risk_factor_ids)


@pytest.mark.parametrize(
    ("encounter_id", "patient_id", "age_years", "conditions"),
    [
        ("E001", "P001", 78, frozenset({"AF", "DEP", "HTN", "PAIN", "T2D"})),
        ("E002", "P002", 34, frozenset({"AF", "HTN"})),
        ("E003", "P002", 34, frozenset({"AF", "HTN"})),
        ("E004", "P003", 70, frozenset({"DEP", "PAIN"})),
        ("E005", "P003", 70, frozenset({"DEP", "PAIN"})),
    ],
)
def test_encounter_conditions_and_demographics(
    real_data_dir: Path,
    encounter_id: str,
    patient_id: str,
    age_years: int,
    conditions: frozenset[str],
) -> None:
    encounter = load_encounter(encounter_id, data_dir=real_data_dir)
    assert encounter.patient_id == patient_id
    assert encounter.age_years == age_years
    assert encounter.conditions == conditions


@pytest.mark.parametrize(
    ("encounter_id", "present", "unknown"),
    [
        ("E001", frozenset({"AGE65"}), frozenset({"CKD"})),
        ("E002", frozenset(), frozenset({"PREG"})),
        ("E003", frozenset(), frozenset()),
        ("E004", frozenset({"AGE65", "LIVER", "EPI"}), frozenset()),
        ("E005", frozenset({"AGE65", "LIVER", "EPI", "CKD"}), frozenset()),
    ],
)
def test_encounter_risk_factor_statuses(
    real_data_dir: Path,
    encounter_id: str,
    present: frozenset[str],
    unknown: frozenset[str],
) -> None:
    encounter = load_encounter(encounter_id, data_dir=real_data_dir)
    assert encounter.risk_factors_with(RiskStatus.PRESENT) == present
    assert encounter.risk_factors_with(RiskStatus.UNKNOWN) == unknown
    all_risk_factors = frozenset(encounter.risk_factors)
    assert encounter.risk_factors_with(RiskStatus.ABSENT) == all_risk_factors - present - unknown


def test_patients_sex_and_alias(real_data_dir: Path) -> None:
    patients = load_patients(data_dir=real_data_dir)
    assert {patient_id: patient.sex for patient_id, patient in patients.items()} == {
        "P001": "F",
        "P002": "F",
        "P003": "M",
    }
    assert patients["P001"].alias == "Rosa"
    assert patients["P002"].alias == "Lucía"
    assert patients["P003"].alias == "Tomás"


def test_load_encounters_and_load_patients_return_sorted_mappings(real_data_dir: Path) -> None:
    encounters = load_encounters(data_dir=real_data_dir)
    assert list(encounters) == ["E001", "E002", "E003", "E004", "E005"]
    patients = load_patients(data_dir=real_data_dir)
    assert list(patients) == ["P001", "P002", "P003"]


def test_load_encounter_unknown_id_raises(real_data_dir: Path) -> None:
    with pytest.raises(UnknownIdError):
        load_encounter("E999", data_dir=real_data_dir)


def test_default_data_dir_is_repository_root_data(
    monkeypatch: pytest.MonkeyPatch, real_data_dir: Path
) -> None:
    monkeypatch.delenv("SYMAI_DATA_DIR", raising=False)
    assert default_data_dir() == real_data_dir


def test_symai_data_dir_env_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch, sandbox_data_dir: Path
) -> None:
    monkeypatch.setenv("SYMAI_DATA_DIR", str(sandbox_data_dir))
    assert default_data_dir() == sandbox_data_dir
    patients = load_patients()
    assert set(patients) == {"P001", "P002", "P003"}
