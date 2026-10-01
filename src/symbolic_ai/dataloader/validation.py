"""Validation of the database against every rule of ``01-database.md`` section 5.

Since data 1.1.0 it also checks the ontology tables (EJ2 results design §6.1, amended by §11.1):
link subjects resolve to a category or a drug, the two namespaces are disjoint,
``subject_a <= subject_b`` (a self-link is allowed), every defined category has at least two
conjuncts, no told parent and no told member, the graph of subcategory and definition edges is
acyclic, and every disjoint set has at least two members and one consistent ``partition_of``.

Every check reads the raw (untyped) rows of every resource and returns the problems it finds
instead of raising, so that :func:`collect_problems` reports every violation of the database in one
pass, as the specification requires.
"""

from __future__ import annotations

import graphlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import TypeAlias

from symbolic_ai.dataloader.io import (
    DataPackage,
    FieldSchema,
    ResourceSchema,
    parse_semver,
    parse_value,
    read_descriptor,
    read_raw_rows,
)

RawRow: TypeAlias = dict[str, str | None]
RawTable: TypeAlias = dict[str, tuple[RawRow, ...]]
_StatusIndex: TypeAlias = dict[tuple[str | None, str | None], str | None]

_AGE65 = "AGE65"
_PREG = "PREG"
_MALE = "M"


def _rows(raw: RawTable, resource_name: str) -> tuple[RawRow, ...]:
    return raw.get(resource_name, ())


def _row_label(resource: ResourceSchema, row: RawRow) -> str:
    values = ", ".join(f"{name}={row.get(name)!r}" for name in resource.primary_key)
    return f"{resource.name}[{values}]"


def _format_key(names: tuple[str, ...], values: tuple[str | None, ...]) -> str:
    return ", ".join(f"{n}={v!r}" for n, v in zip(names, values, strict=True))


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _check_field_value(label: str, field: FieldSchema, value: str) -> list[str]:
    problems: list[str] = []
    try:
        parse_value(value, field.type)
    except ValueError as exc:
        problems.append(
            f"{label}: field {field.name!r} = {value!r} is not a valid {field.type} ({exc})"
        )
        return problems
    if field.enum is not None and value not in field.enum:
        problems.append(f"{label}: field {field.name!r} = {value!r} is not one of {field.enum}")
    return problems


def _check_fields(resource: ResourceSchema, rows: tuple[RawRow, ...]) -> list[str]:
    """Check every field of every row: required, type parses, enum membership (rule 1, part)."""
    problems: list[str] = []
    for row in rows:
        label = _row_label(resource, row)
        for field in resource.fields:
            value = row.get(field.name)
            if value is None:
                if field.required:
                    problems.append(f"{label}: field {field.name!r} is required but empty")
                continue
            problems.extend(_check_field_value(label, field, value))
    return problems


def _check_primary_key(resource: ResourceSchema, rows: tuple[RawRow, ...]) -> list[str]:
    """Check that the primary key is unique within the resource (rule 1, part)."""
    counts: Counter[tuple[str | None, ...]] = Counter(
        tuple(row.get(name) for name in resource.primary_key) for row in rows
    )
    duplicates = sorted(
        (_format_key(resource.primary_key, key), count)
        for key, count in counts.items()
        if count > 1
    )
    return [
        f"{resource.name}: duplicate primary key ({key}), {count} rows" for key, count in duplicates
    ]


def _field_value_set(rows: tuple[RawRow, ...], field_name: str) -> frozenset[str]:
    return frozenset(value for row in rows if (value := row.get(field_name)) is not None)


def _check_foreign_keys(
    package: DataPackage, resource: ResourceSchema, rows: tuple[RawRow, ...], raw: RawTable
) -> list[str]:
    """Check that every foreign key resolves to an existing row (rule 1, part)."""
    problems: list[str] = []
    for fk in resource.foreign_keys:
        target = package.resource_for_self_reference(resource, fk.reference_resource)
        target_values = _field_value_set(_rows(raw, target.name), fk.reference_field)
        for row in rows:
            value = row.get(fk.field)
            if value is not None and value not in target_values:
                problems.append(
                    f"{_row_label(resource, row)}: field {fk.field!r} = {value!r} does not match "
                    f"any {target.name}.{fk.reference_field}"
                )
    return problems


def _check_since(
    resource: ResourceSchema, rows: tuple[RawRow, ...], package_version: tuple[int, int, int]
) -> list[str]:
    """Check that ``since`` is a valid version, not newer than the package (rule 1, part)."""
    if "since" not in resource.field_names:
        return []
    problems: list[str] = []
    package_version_text = ".".join(str(part) for part in package_version)
    for row in rows:
        raw_since = row.get("since")
        if raw_since is None:
            continue  # already reported by _check_fields as a missing required field
        try:
            since = parse_semver(raw_since)
        except ValueError as exc:
            problems.append(f"{_row_label(resource, row)}: invalid 'since' {raw_since!r} ({exc})")
            continue
        if since > package_version:
            problems.append(
                f"{_row_label(resource, row)}: 'since' {raw_since!r} is newer than the package "
                f"version {package_version_text!r}"
            )
    return problems


def _check_every_condition_has_candidate(raw: RawTable) -> list[str]:
    """Every condition has at least one candidate drug (rule 2)."""
    condition_ids = {
        cid for row in _rows(raw, "conditions") if (cid := row.get("condition_id")) is not None
    }
    covered = {
        cid for row in _rows(raw, "candidates") if (cid := row.get("condition_id")) is not None
    }
    missing = sorted(condition_ids - covered)
    return [f"conditions[condition_id={c!r}]: no candidate drug in 'candidates'" for c in missing]


def _check_adverse_interaction_order(raw: RawTable) -> list[str]:
    """``adverse_interactions``: ``drug_a < drug_b``, no self-pairs (rule 3, part)."""
    problems: list[str] = []
    for row in _rows(raw, "adverse_interactions"):
        a, b = row.get("drug_a"), row.get("drug_b")
        if a is None or b is None:
            continue
        if a == b:
            problems.append(
                f"adverse_interactions[drug_a={a!r}, drug_b={b!r}]: self-pair is not allowed"
            )
        elif a > b:
            problems.append(
                f"adverse_interactions[drug_a={a!r}, drug_b={b!r}]: drug_a must be "
                "lexicographically before drug_b"
            )
    return problems


def _check_coprescription_companion(raw: RawTable) -> list[str]:
    """``coprescriptions``: companion drug differs from the trigger drug (rule 3, part)."""
    problems: list[str] = []
    for row in _rows(raw, "coprescriptions"):
        drug_id, companion = row.get("drug_id"), row.get("companion_drug_id")
        if drug_id is not None and companion is not None and drug_id == companion:
            problems.append(
                f"coprescriptions[drug_id={drug_id!r}, companion_drug_id={companion!r}]: "
                "companion_drug_id must differ from drug_id"
            )
    return problems


def _has_cycle(start: str, parent_of: dict[str, str | None]) -> bool:
    seen: set[str] = set()
    current: str | None = start
    while current is not None:
        if current in seen:
            return True
        seen.add(current)
        current = parent_of.get(current)
    return False


def _check_drug_class_acyclic(raw: RawTable) -> list[str]:
    """``drug_classes``: ``parent_class_id`` chains are acyclic (rule 4)."""
    parent_of: dict[str, str | None] = {
        class_id: row.get("parent_class_id")
        for row in _rows(raw, "drug_classes")
        if (class_id := row.get("class_id")) is not None
    }
    return [
        f"drug_classes[class_id={class_id!r}]: parent_class_id chain is cyclic"
        for class_id in sorted(parent_of)
        if _has_cycle(class_id, parent_of)
    ]


def _check_patient_sequence(patient_id: str, rows: list[RawRow]) -> list[str]:
    problems: list[str] = []
    seq_counts: Counter[int] = Counter()
    ordered_pairs: list[tuple[int, date]] = []
    for row in rows:
        seq = _safe_int(row.get("seq"))
        visit_date = _safe_date(row.get("visit_date"))
        if seq is not None:
            seq_counts[seq] += 1
            if visit_date is not None:
                ordered_pairs.append((seq, visit_date))
    for seq in sorted(seq for seq, count in seq_counts.items() if count > 1):
        problems.append(
            f"encounters: patient {patient_id!r} has {seq_counts[seq]} encounters with seq={seq}"
        )
    ordered_pairs.sort(key=lambda pair: pair[0])
    for (prev_seq, prev_date), (cur_seq, cur_date) in itertools.pairwise(ordered_pairs):
        if cur_date <= prev_date:
            problems.append(
                f"encounters: patient {patient_id!r} seq {prev_seq} -> {cur_seq} does not have an "
                "increasing visit_date"
            )
    return problems


def _check_encounter_sequence(raw: RawTable) -> list[str]:
    """Encounters: ``seq`` unique per patient and increasing with ``visit_date`` (rule 5)."""
    by_patient: dict[str, list[RawRow]] = defaultdict(list)
    for row in _rows(raw, "encounters"):
        patient_id = row.get("patient_id")
        if patient_id is not None:
            by_patient[patient_id].append(row)
    problems: list[str] = []
    for patient_id in sorted(by_patient):
        problems.extend(_check_patient_sequence(patient_id, by_patient[patient_id]))
    return problems


def _check_encounter_risk_factor_completeness(raw: RawTable) -> list[str]:
    """Every encounter has exactly one status row for every risk factor (rule 6).

    Duplicate status rows for the same pair are reported by the primary-key check; this check
    reports the missing pairs.
    """
    encounter_ids = sorted(
        eid for row in _rows(raw, "encounters") if (eid := row.get("encounter_id")) is not None
    )
    risk_factor_ids = sorted(
        rid for row in _rows(raw, "risk_factors") if (rid := row.get("risk_factor_id")) is not None
    )
    existing = {
        (row.get("encounter_id"), row.get("risk_factor_id"))
        for row in _rows(raw, "encounter_risk_factors")
    }
    return [
        f"encounter_risk_factors: encounter {encounter_id!r} is missing a status for risk factor "
        f"{risk_factor_id!r}"
        for encounter_id in encounter_ids
        for risk_factor_id in risk_factor_ids
        if (encounter_id, risk_factor_id) not in existing
    ]


def _check_age65(
    encounter_id: str, age_years: int | None, status_by_pair: _StatusIndex
) -> list[str]:
    status = status_by_pair.get((encounter_id, _AGE65))
    if status is None or age_years is None:
        return []
    if status == "unknown":
        return [
            f"encounter_risk_factors[encounter_id={encounter_id!r}]: AGE65 must never be 'unknown'"
        ]
    expected_present = age_years >= 65
    if (status == "present") != expected_present:
        return [
            f"encounter_risk_factors[encounter_id={encounter_id!r}]: AGE65={status!r} is "
            f"inconsistent with age_years={age_years}"
        ]
    return []


def _check_preg(encounter_id: str, sex: str | None, status_by_pair: _StatusIndex) -> list[str]:
    status = status_by_pair.get((encounter_id, _PREG))
    if status is None or sex != _MALE or status == "absent":
        return []
    return [
        f"encounter_risk_factors[encounter_id={encounter_id!r}]: PREG must be 'absent' for a male "
        f"patient, got {status!r}"
    ]


def _check_age65_and_preg_consistency(raw: RawTable) -> list[str]:
    """AGE65 matches age_years and is never unknown; PREG is absent for male patients (rule 7)."""
    patients_sex = {
        row["patient_id"]: row.get("sex") for row in _rows(raw, "patients") if row.get("patient_id")
    }
    status_by_pair: _StatusIndex = {
        (row.get("encounter_id"), row.get("risk_factor_id")): row.get("status")
        for row in _rows(raw, "encounter_risk_factors")
    }
    problems: list[str] = []
    for row in _rows(raw, "encounters"):
        encounter_id = row.get("encounter_id")
        if encounter_id is None:
            continue
        age_years = _safe_int(row.get("age_years"))
        sex = patients_sex.get(row.get("patient_id"))
        problems.extend(_check_age65(encounter_id, age_years, status_by_pair))
        problems.extend(_check_preg(encounter_id, sex, status_by_pair))
    return problems


# --- ontology tables (data 1.1.0; results design §6.1) ----------------------------------------

_ONTOLOGY_CATEGORIES = "ontology_categories"
_ONTOLOGY_SUBCATEGORIES = "ontology_subcategories"
_ONTOLOGY_DEFINITIONS = "ontology_definitions"
_ONTOLOGY_MEMBERSHIPS = "ontology_memberships"
_ONTOLOGY_DISJOINT_SETS = "ontology_disjoint_sets"
_ONTOLOGY_INTERACTIONS = "ontology_interactions"
#: Link tables whose subject is a category or a drug: (resource, subject fields).
_ONTOLOGY_SUBJECT_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ontology_indications", ("subject_id",)),
    ("ontology_contraindications", ("subject_id",)),
    (_ONTOLOGY_INTERACTIONS, ("subject_a", "subject_b")),
    ("ontology_coprescriptions", ("subject_id",)),
)


def _ids(raw: RawTable, resource_name: str, field_name: str) -> frozenset[str]:
    return _field_value_set(_rows(raw, resource_name), field_name)


def _check_subject_namespaces_disjoint(raw: RawTable) -> list[str]:
    """Category and drug identifiers never coincide, so a link subject is unambiguous."""
    shared = sorted(_ids(raw, _ONTOLOGY_CATEGORIES, "category_id") & _ids(raw, "drugs", "drug_id"))
    return [f"ontology_categories[category_id={c!r}]: also a drug_id" for c in shared]


def _check_link_subjects_resolve(raw: RawTable) -> list[str]:
    """Every link subject is a category or a drug (a foreign key into either table)."""
    known = _ids(raw, _ONTOLOGY_CATEGORIES, "category_id") | _ids(raw, "drugs", "drug_id")
    problems: list[str] = []
    for resource_name, subject_fields in _ONTOLOGY_SUBJECT_FIELDS:
        for row in _rows(raw, resource_name):
            problems.extend(
                f"{resource_name}: field {name!r} = {value!r} is neither a category nor a drug"
                for name in subject_fields
                if (value := row.get(name)) is not None and value not in known
            )
    return problems


def _check_interaction_subject_order(raw: RawTable) -> list[str]:
    """``ontology_interactions``: ``subject_a <= subject_b`` (a self-link a = b is allowed)."""
    problems: list[str] = []
    for row in _rows(raw, _ONTOLOGY_INTERACTIONS):
        a, b = row.get("subject_a"), row.get("subject_b")
        if a is not None and b is not None and not a <= b:
            problems.append(
                f"ontology_interactions[subject_a={a!r}, subject_b={b!r}]: subject_a must not be "
                "lexicographically after subject_b"
            )
    return problems


def _edges(raw: RawTable, resource_name: str, child: str, parent: str) -> list[tuple[str, str]]:
    """Return the (child, parent) pairs of a two-column link table, skipping incomplete rows."""
    return [
        (c, p)
        for row in _rows(raw, resource_name)
        if (c := row.get(child)) is not None and (p := row.get(parent)) is not None
    ]


def _check_taxonomy_acyclic(raw: RawTable) -> list[str]:
    """Check that subcategory and definition edges together (child -> parents) have no cycle.

    A defined category sits below each of its conjuncts, so a definition edge counts as a parent
    edge: a category defined through one of its own subcategories would be circular.
    """
    parents: dict[str, set[str]] = defaultdict(set)
    edges = _edges(raw, _ONTOLOGY_SUBCATEGORIES, "category_id", "parent_id")
    edges += _edges(raw, _ONTOLOGY_DEFINITIONS, "category_id", "conjunct_id")
    for child, parent in edges:
        parents[child].add(parent)
    try:
        tuple(graphlib.TopologicalSorter(parents).static_order())
    except graphlib.CycleError as exc:
        cycle = " -> ".join(str(node) for node in exc.args[1])
        return [
            "ontology_subcategories + ontology_definitions: the graph of subcategory and "
            f"definition edges has a cycle ({cycle})"
        ]
    return []


def _check_definitions(raw: RawTable) -> list[str]:
    """Check that a defined category has at least two conjuncts, no told parent and no told member.

    Its place in the taxonomy and its members follow from the definition alone; a told parent or
    member would duplicate (or contradict) what the definition entails.
    """
    conjuncts: dict[str, set[str]] = defaultdict(set)
    for category_id, conjunct in _edges(raw, _ONTOLOGY_DEFINITIONS, "category_id", "conjunct_id"):
        conjuncts[category_id].add(conjunct)
    with_parent = _ids(raw, _ONTOLOGY_SUBCATEGORIES, "category_id")
    with_member = _ids(raw, _ONTOLOGY_MEMBERSHIPS, "category_id")
    problems: list[str] = []
    for category_id in sorted(conjuncts):
        label = f"ontology_definitions[category_id={category_id!r}]"
        if len(conjuncts[category_id]) < 2:
            problems.append(f"{label}: fewer than two conjuncts")
        if category_id in with_parent:
            problems.append(f"{label}: a defined category has no told parent (subcategories row)")
        if category_id in with_member:
            problems.append(f"{label}: a defined category has no told member (memberships row)")
    return problems


def _check_disjoint_sets(raw: RawTable) -> list[str]:
    """Every disjoint set has at least two members and one consistent ``partition_of``."""
    members: dict[str, list[str | None]] = defaultdict(list)
    partition_of: dict[str, set[str | None]] = defaultdict(set)
    for row in _rows(raw, _ONTOLOGY_DISJOINT_SETS):
        set_id = row.get("set_id")
        if set_id is not None:
            members[set_id].append(row.get("category_id"))
            partition_of[set_id].add(row.get("partition_of"))
    problems: list[str] = []
    for set_id in sorted(members):
        if len(members[set_id]) < 2:
            problems.append(f"ontology_disjoint_sets[set_id={set_id!r}]: fewer than two members")
        if len(partition_of[set_id]) > 1:
            values = sorted(str(value) for value in partition_of[set_id])
            problems.append(
                f"ontology_disjoint_sets[set_id={set_id!r}]: inconsistent partition_of {values}"
            )
    return problems


def _check_ontology(raw: RawTable) -> list[str]:
    """Run the structural checks of the ontology tables (results design §6.1)."""
    problems = _check_subject_namespaces_disjoint(raw)
    problems.extend(_check_link_subjects_resolve(raw))
    problems.extend(_check_interaction_subject_order(raw))
    problems.extend(_check_taxonomy_acyclic(raw))
    problems.extend(_check_definitions(raw))
    problems.extend(_check_disjoint_sets(raw))
    return problems


def missing_memberships(drug_ids: Iterable[str], member_ids: Iterable[str]) -> tuple[str, ...]:
    """Report every drug without a told membership, for :func:`load_ontology`.

    "Every drug has at least one told membership" is checked when the ontology is loaded, not in
    :func:`collect_problems`: a drug may be added to the formulary in a later data version before
    it is placed in the ontology, and that must not invalidate the database for EJ1. A drug may
    have several told memberships (data 1.1.0, amended: tramadol is told in two categories).

    Parameters
    ----------
    drug_ids : Iterable[str]
        The drugs the ontology must classify.
    member_ids : Iterable[str]
        The objects of the told memberships.

    Returns
    -------
    tuple[str, ...]
        One problem message per drug with no told membership, sorted by drug.
    """
    missing = sorted(set(drug_ids) - set(member_ids))
    return tuple(
        f"ontology_memberships: drug {drug_id!r} has no told membership, expected at least one"
        for drug_id in missing
    )


def _read_all_raw(data_dir: Path, package: DataPackage) -> RawTable:
    return {resource.name: read_raw_rows(data_dir, resource) for resource in package.resources}


def _check_schema_rules(package: DataPackage, raw: RawTable) -> list[str]:
    package_version = parse_semver(package.version)
    problems: list[str] = []
    for resource in package.resources:
        rows = raw[resource.name]
        problems.extend(_check_fields(resource, rows))
        problems.extend(_check_primary_key(resource, rows))
        problems.extend(_check_foreign_keys(package, resource, rows, raw))
        problems.extend(_check_since(resource, rows, package_version))
    return problems


def collect_problems(data_dir: Path) -> tuple[str, ...]:
    """Validate the whole database and return every problem found.

    Parameters
    ----------
    data_dir : Path
        Root directory of the database (holding ``datapackage.json``).

    Returns
    -------
    tuple[str, ...]
        One message per violation of ``01-database.md`` section 5; empty if the database is valid.
    """
    package = read_descriptor(data_dir)
    raw = _read_all_raw(data_dir, package)
    problems = _check_schema_rules(package, raw)
    problems.extend(_check_every_condition_has_candidate(raw))
    problems.extend(_check_adverse_interaction_order(raw))
    problems.extend(_check_coprescription_companion(raw))
    problems.extend(_check_drug_class_acyclic(raw))
    problems.extend(_check_encounter_sequence(raw))
    problems.extend(_check_encounter_risk_factor_completeness(raw))
    problems.extend(_check_age65_and_preg_consistency(raw))
    problems.extend(_check_ontology(raw))
    return tuple(problems)
