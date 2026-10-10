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
    Category,
    CategoryDefinition,
    Condition,
    Contraindication,
    ContraindicationLink,
    Coprescription,
    CoprescriptionLink,
    DisjointSet,
    Drug,
    DrugClass,
    DrugCost,
    Encounter,
    Formulary,
    IndicationLink,
    InteractionLink,
    Membership,
    OntologyData,
    Patient,
    RiskFactor,
    RiskStatus,
    SubcategoryEdge,
)

__all__ = [
    "AdverseInteraction",
    "Category",
    "CategoryDefinition",
    "Condition",
    "Contraindication",
    "ContraindicationLink",
    "Coprescription",
    "CoprescriptionLink",
    "DataLoaderError",
    "DatabaseValidationError",
    "DisjointSet",
    "Drug",
    "DrugClass",
    "DrugCost",
    "Encounter",
    "Formulary",
    "IndicationLink",
    "InteractionLink",
    "Membership",
    "OntologyData",
    "Patient",
    "RiskFactor",
    "RiskStatus",
    "SubcategoryEdge",
    "UnknownIdError",
    "default_data_dir",
    "load_drug_costs",
    "load_encounter",
    "load_encounters",
    "load_formulary",
    "load_ontology",
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


def load_drug_costs(
    data_dir: Path | None = None, version: str | None = None
) -> Mapping[str, DrugCost]:
    """Load the cost of every drug as of ``version`` (default: the database version).

    The costs are a table of their own (data 1.2.0), not a field of :class:`Formulary`, so that
    ``load_formulary`` returns exactly what it returned before for every version.

    Parameters
    ----------
    data_dir : Path | None
        Database directory (default: :func:`default_data_dir`).
    version : str | None
        Data version; rows with ``since`` newer than it are left out.

    Returns
    -------
    Mapping[str, DrugCost]
        One cost per drug of that version, keyed and sorted by ``drug_id``.

    Raises
    ------
    DatabaseValidationError
        If the database is invalid, or a drug of that version has no cost (as for every version
        before 1.2.0).
    """
    directory = _resolve_data_dir(data_dir)
    validate_database(directory)
    package = read_descriptor(directory)
    version_text = version if version is not None else package.version
    target_version = parse_semver(version_text)
    rows = _versioned_rows(directory, package, "drug_costs", target_version)
    costs = {
        _as_str(row["drug_id"]): DrugCost(
            drug_id=_as_str(row["drug_id"]),
            monthly_cost_cents=_as_int(row["monthly_cost_cents"]),
            source=_as_optional_str(row["source"]),
        )
        for row in rows
    }
    drugs = _drugs(directory, package, target_version)
    problems = validation.missing_drug_costs((d.drug_id for d in drugs), costs)
    if problems:
        raise DatabaseValidationError(problems)
    return {drug_id: costs[drug_id] for drug_id in sorted(costs)}


def _categories(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[Category, ...]:
    rows = _versioned_rows(directory, package, "ontology_categories", target_version)
    items = [
        Category(
            category_id=_as_str(row["category_id"]),
            name_es=_as_str(row["name_es"]),
            name_en=_as_str(row["name_en"]),
            atc_code=_as_optional_str(row["atc_code"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: c.category_id))


def _subcategories(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[SubcategoryEdge, ...]:
    rows = _versioned_rows(directory, package, "ontology_subcategories", target_version)
    items = [
        SubcategoryEdge(_as_str(row["category_id"]), _as_str(row["parent_id"])) for row in rows
    ]
    return tuple(sorted(items, key=lambda e: (e.category_id, e.parent_id)))


def _definitions(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[CategoryDefinition, ...]:
    rows = _versioned_rows(directory, package, "ontology_definitions", target_version)
    conjuncts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        conjuncts[_as_str(row["category_id"])].add(_as_str(row["conjunct_id"]))
    return tuple(
        CategoryDefinition(category_id, tuple(sorted(conjuncts[category_id])))
        for category_id in sorted(conjuncts)
    )


def _memberships(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[Membership, ...]:
    rows = _versioned_rows(directory, package, "ontology_memberships", target_version)
    items = [Membership(_as_str(row["object_id"]), _as_str(row["category_id"])) for row in rows]
    return tuple(sorted(items, key=lambda m: (m.object_id, m.category_id)))


def _indication_links(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[IndicationLink, ...]:
    rows = _versioned_rows(directory, package, "ontology_indications", target_version)
    items = [
        IndicationLink(_as_str(row["subject_id"]), _as_str(row["condition_id"])) for row in rows
    ]
    return tuple(sorted(items, key=lambda i: (i.subject_id, i.condition_id)))


def _contraindication_links(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[ContraindicationLink, ...]:
    rows = _versioned_rows(directory, package, "ontology_contraindications", target_version)
    items = [
        ContraindicationLink(
            subject_id=_as_str(row["subject_id"]),
            risk_factor_id=_as_str(row["risk_factor_id"]),
            reason=_as_str(row["reason"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: (c.subject_id, c.risk_factor_id)))


def _interaction_links(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[InteractionLink, ...]:
    rows = _versioned_rows(directory, package, "ontology_interactions", target_version)
    items = [
        InteractionLink(
            subject_a=_as_str(row["subject_a"]),
            subject_b=_as_str(row["subject_b"]),
            effect=_as_str(row["effect"]),
            severity=_as_str(row["severity"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda i: (i.subject_a, i.subject_b)))


def _coprescription_links(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[CoprescriptionLink, ...]:
    rows = _versioned_rows(directory, package, "ontology_coprescriptions", target_version)
    items = [
        CoprescriptionLink(
            subject_id=_as_str(row["subject_id"]),
            risk_factor_id=_as_str(row["risk_factor_id"]),
            companion_drug_id=_as_str(row["companion_drug_id"]),
            reason=_as_str(row["reason"]),
        )
        for row in rows
    ]
    return tuple(sorted(items, key=lambda c: (c.subject_id, c.risk_factor_id, c.companion_drug_id)))


def _families(directory: Path, package: DataPackage, target_version: _SemVer) -> tuple[str, ...]:
    rows = _versioned_rows(directory, package, "ontology_families", target_version)
    return tuple(sorted(_as_str(row["category_id"]) for row in rows))


def _disjoint_sets(
    directory: Path, package: DataPackage, target_version: _SemVer
) -> tuple[DisjointSet, ...]:
    rows = _versioned_rows(directory, package, "ontology_disjoint_sets", target_version)
    members: dict[str, set[str]] = defaultdict(set)
    partition_of: dict[str, str | None] = {}
    for row in rows:
        set_id = _as_str(row["set_id"])
        members[set_id].add(_as_str(row["category_id"]))
        # Validation guarantees one partition_of value per set.
        partition_of[set_id] = _as_optional_str(row["partition_of"])
    return tuple(
        DisjointSet(set_id, tuple(sorted(members[set_id])), partition_of[set_id])
        for set_id in sorted(members)
    )


def load_ontology(data_dir: Path | None = None, version: str | None = None) -> OntologyData:
    """Load the ontology tables as of ``version`` (default: the database version); validates first.

    The drug-level formulary tables (candidates, contraindications, adverse interactions,
    coprescriptions, drug classes and members) are never read into the result, so that what is
    derived from the ontology can be compared with them (EJ2, decision K6).

    Parameters
    ----------
    data_dir : Path | None
        Database directory (default: :func:`default_data_dir`).
    version : str | None
        Data version; rows with ``since`` newer than it are left out.

    Returns
    -------
    OntologyData
        The ontology tables, with the drugs, conditions and risk factors of that version.

    Raises
    ------
    DatabaseValidationError
        If the database is invalid, or a drug of that version has no told membership.
    """
    directory = _resolve_data_dir(data_dir)
    validate_database(directory)
    package = read_descriptor(directory)
    version_text = version if version is not None else package.version
    target_version = parse_semver(version_text)
    drugs = _drugs(directory, package, target_version)
    memberships = _memberships(directory, package, target_version)
    problems = validation.missing_memberships(
        (d.drug_id for d in drugs), (m.object_id for m in memberships)
    )
    if problems:
        raise DatabaseValidationError(problems)
    return OntologyData(
        version=version_text,
        drugs=drugs,
        conditions=_conditions(directory, package, target_version),
        risk_factors=_risk_factors(directory, package, target_version),
        categories=_categories(directory, package, target_version),
        subcategories=_subcategories(directory, package, target_version),
        memberships=memberships,
        indications=_indication_links(directory, package, target_version),
        contraindications=_contraindication_links(directory, package, target_version),
        interactions=_interaction_links(directory, package, target_version),
        coprescriptions=_coprescription_links(directory, package, target_version),
        families=_families(directory, package, target_version),
        disjoint_sets=_disjoint_sets(directory, package, target_version),
        definitions=_definitions(directory, package, target_version),
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
