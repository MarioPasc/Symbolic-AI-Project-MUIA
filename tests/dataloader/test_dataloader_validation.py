"""One failing-input test per validation rule of ``01-database.md``, section 5.

Each test copies the real database to ``tmp_path`` (``sandbox_data_dir``), corrupts exactly one
thing, and asserts that ``validate_database`` reports the specific problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import DatabaseValidationError, validate_database

pytestmark = pytest.mark.integration


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture text not found in {path}: {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _problems(data_dir: Path) -> tuple[str, ...]:
    with pytest.raises(DatabaseValidationError) as excinfo:
        validate_database(data_dir=data_dir)
    return excinfo.value.problems


# Rule 1: primary keys unique; foreign keys resolve; enums and booleans parse; since <= version.


def test_duplicate_primary_key_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "conditions.csv"
    path.write_text(
        path.read_text(encoding="utf-8") + "AF,Duplicado,Duplicate,I48,1.0.0\n", encoding="utf-8"
    )
    problems = _problems(sandbox_data_dir)
    assert any("duplicate primary key" in p and "AF" in p for p in problems)


def test_unresolved_foreign_key_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "candidates.csv"
    _replace(path, "HTN,amlodipine,1.0.0", "HTN,nonexistent_drug,1.0.0")
    problems = _problems(sandbox_data_dir)
    assert any("nonexistent_drug" in p and "does not match" in p for p in problems)


def test_invalid_enum_value_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "adverse_interactions.csv"
    _replace(
        path,
        "apixaban,ibuprofen,bleeding,major,,1.0.0",
        "apixaban,ibuprofen,bleeding,extreme,,1.0.0",
    )
    problems = _problems(sandbox_data_dir)
    assert any("extreme" in p and "not one of" in p for p in problems)


def test_invalid_boolean_value_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "drug_classes.csv"
    _replace(
        path,
        "analgesics,Analgésicos,Analgesics,,,true,1.0.0",
        "analgesics,Analgésicos,Analgesics,,,yes,1.0.0",
    )
    problems = _problems(sandbox_data_dir)
    assert any("not a valid boolean" in p for p in problems)


def test_since_newer_than_package_version_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "drugs.csv"
    _replace(
        path,
        "amlodipine,Amlodipino,Amlodipine,C08CA01,1.0.0",
        "amlodipine,Amlodipino,Amlodipine,C08CA01,9.9.9",
    )
    problems = _problems(sandbox_data_dir)
    assert any("newer than the package" in p for p in problems)


# Rule 2: every condition has at least one candidate drug.


def test_condition_without_candidate_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "candidates.csv"
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith("GERD,")
    ]
    path.write_text("".join(lines), encoding="utf-8")
    problems = _problems(sandbox_data_dir)
    assert any("GERD" in p and "no candidate drug" in p for p in problems)


# Rule 3: adverse_interactions drug_a < drug_b, no self-pairs; coprescriptions companion != trigger.


def test_adverse_interaction_wrong_order_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "adverse_interactions.csv"
    _replace(
        path, "apixaban,ibuprofen,bleeding,major,,1.0.0", "ibuprofen,apixaban,bleeding,major,,1.0.0"
    )
    problems = _problems(sandbox_data_dir)
    assert any("lexicographically" in p for p in problems)


def test_adverse_interaction_self_pair_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "adverse_interactions.csv"
    _replace(
        path, "apixaban,ibuprofen,bleeding,major,,1.0.0", "apixaban,apixaban,bleeding,major,,1.0.0"
    )
    problems = _problems(sandbox_data_dir)
    assert any("self-pair is not allowed" in p for p in problems)


def test_coprescription_companion_equals_trigger_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "coprescriptions.csv"
    _replace(
        path,
        "ibuprofen,AGE65,omeprazole,gastroprotection,,1.0.0",
        "ibuprofen,AGE65,ibuprofen,gastroprotection,,1.0.0",
    )
    problems = _problems(sandbox_data_dir)
    assert any("companion_drug_id must differ" in p for p in problems)


# Rule 4: drug_classes parent_class_id chains are acyclic.


def test_drug_class_cyclic_parent_chain_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "formulary" / "drug_classes.csv"
    _replace(
        path,
        "beta_blockers,Betabloqueantes,Beta blockers,,C07,true,1.0.0",
        "beta_blockers,Betabloqueantes,Beta blockers,raas_blockers,C07,true,1.0.0",
    )
    _replace(
        path,
        "RAAS blockers (ACEI/ARB),,C09,true,1.0.0",
        "RAAS blockers (ACEI/ARB),beta_blockers,C09,true,1.0.0",
    )
    problems = _problems(sandbox_data_dir)
    assert any("cyclic" in p for p in problems)


# Rule 5: encounters seq unique per patient and increasing with visit_date.


def test_duplicate_seq_per_patient_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounters.csv"
    _replace(path, "E003,P002,2,2026-09-09,34,", "E003,P002,1,2026-09-09,34,")
    problems = _problems(sandbox_data_dir)
    assert any("has 2 encounters with seq=1" in p for p in problems)


def test_seq_not_increasing_with_visit_date_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounters.csv"
    _replace(path, "E003,P002,2,2026-09-09,34,", "E003,P002,2,2026-08-01,34,")
    problems = _problems(sandbox_data_dir)
    assert any("does not have an increasing visit_date" in p for p in problems)


# Rule 6: every encounter has exactly one status row for every risk factor.


def test_missing_risk_factor_status_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounter_risk_factors.csv"
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.strip() != "E001,PREG,absent"
    ]
    path.write_text("".join(lines), encoding="utf-8")
    problems = _problems(sandbox_data_dir)
    assert any("E001" in p and "PREG" in p and "missing a status" in p for p in problems)


# Rule 7: AGE65 present iff age_years >= 65, never unknown; PREG absent when sex = M.


def test_age65_inconsistent_with_age_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounter_risk_factors.csv"
    _replace(path, "E002,AGE65,absent", "E002,AGE65,present")
    problems = _problems(sandbox_data_dir)
    assert any("AGE65" in p and "inconsistent" in p for p in problems)


def test_age65_unknown_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounter_risk_factors.csv"
    _replace(path, "E001,AGE65,present", "E001,AGE65,unknown")
    problems = _problems(sandbox_data_dir)
    assert any("AGE65 must never be" in p for p in problems)


def test_preg_present_for_male_is_reported(sandbox_data_dir: Path) -> None:
    path = sandbox_data_dir / "patients" / "encounter_risk_factors.csv"
    _replace(path, "E004,PREG,absent", "E004,PREG,present")
    problems = _problems(sandbox_data_dir)
    assert any("PREG must be 'absent' for a male" in p for p in problems)


def test_valid_database_has_no_problems(sandbox_data_dir: Path) -> None:
    validate_database(data_dir=sandbox_data_dir)
