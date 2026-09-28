"""The only reader of the database under ``data/``: loads, validates and returns immutable models.

The signatures below are a frozen contract (docs SPECIFICATIONS/02-code-architecture.md, section 4).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from symbolic_ai.dataloader import validation
from symbolic_ai.dataloader.errors import DatabaseValidationError, DataLoaderError, UnknownIdError
from symbolic_ai.dataloader.io import (
    DataPackage,
    FieldValue,
    parse_semver,
    read_descriptor,
    read_rows,
)
from symbolic_ai.dataloader.models import (
    AdverseInteraction,
    Condition,
    Contraindication,
    Coprescription,
    Drug,
    DrugClass,
    Encounter,
    Formulary,
    Patient,
    RiskFactor,
    RiskStatus,
)

__all__ = [
    "AdverseInteraction",
    "Condition",
    "Contraindication",
    "Coprescription",
    "DataLoaderError",
    "DatabaseValidationError",
    "Drug",
    "DrugClass",
    "Encounter",
    "Formulary",
    "Patient",
    "RiskFactor",
    "RiskStatus",
    "UnknownIdError",
    "default_data_dir",
    "load_encounter",
    "load_encounters",
    "load_formulary",
    "load_patients",
    "validate_database",
]

_DATA_DIR_ENV_VAR = "SYMAI_DATA_DIR"
_SemVer = tuple[int, int, int]


def default_data_dir() -> Path:
    """Return ``$SYMAI_DATA_DIR`` if set, else ``<repository root>/data``."""
    env_value = os.environ.get(_DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[3] / "data"


def _resolve_data_dir(data_dir: Path | None) -> Path:
    return data_dir if data_dir is not None else default_data_dir()


def validate_database(data_dir: Path | None = None) -> None:
    """Run every check of the database specification; raises ``DatabaseValidationError``."""
    directory = _resolve_data_dir(data_dir)
    problems = validation.collect_problems(directory)
    if problems:
        raise DatabaseValidationError(problems)


def _as_str(value: FieldValue) -> str:
    assert isinstance(value, str)
    return value


def _as_optional_str(value: FieldValue) -> str | None:
    assert value is None or isinstance(value, str)
    return value


def _as_int(value: FieldValue) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _as_bool(value: FieldValue) -> bool:
    assert isinstance(value, bool)
    return value


def _as_date(value: FieldValue) -> date:
    assert isinstance(value, date)
    return value


def _since_le(row: dict[str, FieldValue], target_version: _SemVer) -> bool:
    since_value = row.get("since")
    assert isinstance(since_value, str)
    return parse_semver(since_value) <= target_version


def _versioned_rows(
    directory: Path, package: DataPackage, resource_name: str, target_version: _SemVer
) -> tuple[dict[str, FieldValue], ...]:
    resource = package.resource(resource_name)
    rows = read_rows(directory, resource)
    if "since" in resource.field_names:
        rows = tuple(row for row in rows if _since_le(row, target_version))
    return rows


def _conditions(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[Condition, ...]:
    rows = _versioned_rows(directory, package, "conditions", target_version)
    items = [
        Condition(
            condition_id=_as_str(row["condition_id"]),
            name_es=_as_str(row["name_es"]),
            name_en=_as_str(row["name_en"]),
            icd10=_as_optional_str(row["icd10"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: c.condition_id))


def _drugs(directory: Path, package: DataPackage, target_version: _SemVer) -> tuple[Drug, ...]:
    rows = _versioned_rows(directory, package, "drugs", target_version)
    items = [
        Drug(
            drug_id=_as_str(row["drug_id"]),
            name_es=_as_str(row["name_es"]),
            name_en=_as_str(row["name_en"]),
            atc_code=_as_optional_str(row["atc_code"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda d: d.drug_id))


def _risk_factors(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[RiskFactor, ...]:
    rows = _versioned_rows(directory, package, "risk_factors", target_version)
    items = [
        RiskFactor(
            risk_factor_id=_as_str(row["risk_factor_id"]),
            name_es=_as_str(row["name_es"]),
            name_en=_as_str(row["name_en"]),
            definition=_as_optional_str(row["definition"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda r: r.risk_factor_id))


def _grouped_frozensets(
    rows: tuple[dict[str, FieldValue], ...], key_field: str, value_field: str
) -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[_as_str(row[key_field])].add(_as_str(row[value_field]))
    return {key: frozenset(grouped[key]) for key in sorted(grouped)}


def _candidates(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> dict[str, frozenset[str]]:
    rows = _versioned_rows(directory, package, "candidates", target_version)
    return _grouped_frozensets(rows, "condition_id", "drug_id")


def _class_members(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> dict[str, frozenset[str]]:
    rows = _versioned_rows(directory, package, "drug_class_members", target_version)
    return _grouped_frozensets(rows, "class_id", "drug_id")


def _interactions(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[AdverseInteraction, ...]:
    rows = _versioned_rows(directory, package, "adverse_interactions", target_version)
    items = [
        AdverseInteraction(
            drug_a=_as_str(row["drug_a"]),
            drug_b=_as_str(row["drug_b"]),
            effect=_as_str(row["effect"]),
            severity=_as_str(row["severity"]),
            source=_as_optional_str(row["source"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda i: (i.drug_a, i.drug_b)))


def _contraindications(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[Contraindication, ...]:
    rows = _versioned_rows(directory, package, "contraindications", target_version)
    items = [
        Contraindication(
            risk_factor_id=_as_str(row["risk_factor_id"]),
            drug_id=_as_str(row["drug_id"]),
            reason=_as_str(row["reason"]),
            source=_as_optional_str(row["source"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: (c.risk_factor_id, c.drug_id)))


def _coprescriptions(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[Coprescription, ...]:
    rows = _versioned_rows(directory, package, "coprescriptions", target_version)
    items = [
        Coprescription(
            drug_id=_as_str(row["drug_id"]),
            risk_factor_id=_as_str(row["risk_factor_id"]),
            companion_drug_id=_as_str(row["companion_drug_id"]),
            reason=_as_str(row["reason"]),
            source=_as_optional_str(row["source"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: (c.drug_id, c.risk_factor_id, c.companion_drug_id)))


def _drug_classes(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[DrugClass, ...]:
    rows = _versioned_rows(directory, package, "drug_classes", target_version)
    items = [
        DrugClass(
            class_id=_as_str(row["class_id"]),
            name_es=_as_str(row["name_es"]),
            name_en=_as_str(row["name_en"]),
            parent_class_id=_as_optional_str(row["parent_class_id"]),
            atc_code=_as_optional_str(row["atc_code"]),
            exclusive=_as_bool(row["exclusive"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: c.class_id))


def load_formulary(data_dir: Path | None = None, version: str | None = None) -> Formulary:
    """Load the formulary as of ``version`` (default: the database version); validates first."""
    directory = _resolve_data_dir(data_dir)
    validate_database(directory)
    package = read_descriptor(directory)
    version_text = version if version is not None else package.version
    target_version = parse_semver(version_text)
    return Formulary(
        version=version_text,
        conditions=_conditions(directory, package, target_version),
        drugs=_drugs(directory, package, target_version),
        risk_factors=_risk_factors(directory, package, target_version),
        candidates=_candidates(directory, package, target_version),
        interactions=_interactions(directory, package, target_version),
        contraindications=_contraindications(directory, package, target_version),
        coprescriptions=_coprescriptions(directory, package, target_version),
        drug_classes=_drug_classes(directory, package, target_version),
        class_members=_class_members(directory, package, target_version),
    )


def load_patients(data_dir: Path | None = None) -> Mapping[str, Patient]:
    """Load every patient, keyed and sorted by ``patient_id``."""
    directory = _resolve_data_dir(data_dir)
    validate_database(directory)
    package = read_descriptor(directory)
    rows = read_rows(directory, package.resource("patients"))
    patients = {
        _as_str(row["patient_id"]): Patient(
            patient_id=_as_str(row["patient_id"]),
            alias=_as_str(row["alias"]),
            sex=_as_str(row["sex"]),
        )
        for row in rows
    }
    return {patient_id: patients[patient_id] for patient_id in sorted(patients)}


def _encounter_conditions_by_id(directory: Path, package: DataPackage) -> dict[str, set[str]]:
    rows = read_rows(directory, package.resource("encounter_conditions"))
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[_as_str(row["encounter_id"])].add(_as_str(row["condition_id"]))
    return grouped


def _encounter_risk_factors_by_id(
    directory: Path, package: DataPackage
) -> dict[str, dict[str, RiskStatus]]:
    rows = read_rows(directory, package.resource("encounter_risk_factors"))
    grouped: dict[str, dict[str, RiskStatus]] = defaultdict(dict)
    for row in rows:
        encounter_id = _as_str(row["encounter_id"])
        risk_factor_id = _as_str(row["risk_factor_id"])
        grouped[encounter_id][risk_factor_id] = RiskStatus(_as_str(row["status"]))
    return grouped


def _encounter_from_row(
    row: dict[str, FieldValue],
    conditions_by_encounter: dict[str, set[str]],
    risk_factors_by_encounter: dict[str, dict[str, RiskStatus]],
) -> Encounter:
    encounter_id = _as_str(row["encounter_id"])
    risk_factors = risk_factors_by_encounter.get(encounter_id, {})
    return Encounter(
        encounter_id=encounter_id,
        patient_id=_as_str(row["patient_id"]),
        seq=_as_int(row["seq"]),
        visit_date=_as_date(row["visit_date"]),
        age_years=_as_int(row["age_years"]),
        conditions=frozenset(conditions_by_encounter.get(encounter_id, set())),
        risk_factors=dict(sorted(risk_factors.items())),
        note=_as_optional_str(row["note"]) or "",
    )


def load_encounters(data_dir: Path | None = None) -> Mapping[str, Encounter]:
    """Load every encounter, keyed and sorted by ``encounter_id``."""
    directory = _resolve_data_dir(data_dir)
    validate_database(directory)
    package = read_descriptor(directory)
    conditions_by_encounter = _encounter_conditions_by_id(directory, package)
    risk_factors_by_encounter = _encounter_risk_factors_by_id(directory, package)
    rows = read_rows(directory, package.resource("encounters"))
    encounters = {
        _as_str(row["encounter_id"]): _encounter_from_row(
            row, conditions_by_encounter, risk_factors_by_encounter
        )
        for row in rows
    }
    return {encounter_id: encounters[encounter_id] for encounter_id in sorted(encounters)}


def load_encounter(encounter_id: str, data_dir: Path | None = None) -> Encounter:
    """Load one encounter; raises ``UnknownIdError`` if it does not exist."""
    encounters = load_encounters(data_dir)
    if encounter_id not in encounters:
        raise UnknownIdError(f"Unknown encounter_id {encounter_id!r}")
    return encounters[encounter_id]
