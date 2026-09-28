"""Immutable domain objects returned by the dataloader: the contract every exercise codes against.

Fields are never renamed or removed. A field added later must have a default, so that code
written against an earlier version keeps working (docs: SPECIFICATIONS/02-code-architecture.md §4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class RiskStatus(StrEnum):
    """Status of a risk factor at an encounter."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Condition:
    """A condition the formulary can treat (symbol ``H_<condition_id>`` in EJ1)."""

    condition_id: str
    name_es: str
    name_en: str
    icd10: str | None = None


@dataclass(frozen=True, slots=True)
class Drug:
    """A drug of the formulary (symbol ``T_<drug_id>`` in EJ1)."""

    drug_id: str
    name_es: str
    name_en: str
    atc_code: str | None = None


@dataclass(frozen=True, slots=True)
class RiskFactor:
    """A patient risk factor (symbol ``R_<risk_factor_id>`` in EJ1)."""

    risk_factor_id: str
    name_es: str
    name_en: str
    definition: str | None = None


@dataclass(frozen=True, slots=True)
class AdverseInteraction:
    """A pair of drugs that must not be prescribed together (axiom A3); ``drug_a < drug_b``."""

    drug_a: str
    drug_b: str
    effect: str
    severity: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Contraindication:
    """A risk factor that excludes a drug (axiom A4)."""

    risk_factor_id: str
    drug_id: str
    reason: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Coprescription:
    """A drug that, under a risk factor, requires a companion drug (axiom A5)."""

    drug_id: str
    risk_factor_id: str
    companion_drug_id: str
    reason: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class DrugClass:
    """A class of drugs; ``exclusive`` classes are the families of axiom A6."""

    class_id: str
    name_es: str
    name_en: str
    parent_class_id: str | None = None
    atc_code: str | None = None
    exclusive: bool = False


@dataclass(frozen=True, slots=True)
class Formulary:
    """The stable knowledge of the environment, as of one data version.

    Tuples are sorted by their identifiers. ``candidates`` maps a condition to its candidate drugs
    (the sets D_c); ``class_members`` maps a class to its direct member drugs.
    """

    version: str
    conditions: tuple[Condition, ...]
    drugs: tuple[Drug, ...]
    risk_factors: tuple[RiskFactor, ...]
    candidates: Mapping[str, frozenset[str]]
    interactions: tuple[AdverseInteraction, ...]
    contraindications: tuple[Contraindication, ...]
    coprescriptions: tuple[Coprescription, ...]
    drug_classes: tuple[DrugClass, ...] = ()
    class_members: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def condition_ids(self) -> tuple[str, ...]:
        """Identifiers of all conditions, sorted."""
        return tuple(sorted(c.condition_id for c in self.conditions))

    @property
    def drug_ids(self) -> tuple[str, ...]:
        """Identifiers of all drugs, sorted."""
        return tuple(sorted(d.drug_id for d in self.drugs))

    @property
    def risk_factor_ids(self) -> tuple[str, ...]:
        """Identifiers of all risk factors, sorted."""
        return tuple(sorted(r.risk_factor_id for r in self.risk_factors))

    def indications(self, drug_id: str) -> frozenset[str]:
        """Return Ind(d), the conditions whose candidate set contains ``drug_id``.

        Parameters
        ----------
        drug_id : str
            Identifier of a drug.

        Returns
        -------
        frozenset[str]
            Condition identifiers; empty if the drug treats no condition.
        """
        return frozenset(c for c, drugs in self.candidates.items() if drug_id in drugs)

    def exclusive_families(self) -> tuple[frozenset[str], ...]:
        """Return the direct member sets of the exclusive classes (axiom A6), sorted by class id.

        Returns
        -------
        tuple[frozenset[str], ...]
            One set of drug identifiers per class with ``exclusive = True``.
        """
        exclusive = sorted(c.class_id for c in self.drug_classes if c.exclusive)
        return tuple(self.class_members.get(class_id, frozenset()) for class_id in exclusive)


@dataclass(frozen=True, slots=True)
class Patient:
    """A fictitious patient."""

    patient_id: str
    alias: str
    sex: str


@dataclass(frozen=True, slots=True)
class Encounter:
    """The clinical record of one appointment: the input of one agent decision.

    ``conditions`` holds the conditions present (closed world: any other condition is absent);
    ``risk_factors`` maps every risk factor to its status at this encounter.
    """

    encounter_id: str
    patient_id: str
    seq: int
    visit_date: date
    age_years: int
    conditions: frozenset[str]
    risk_factors: Mapping[str, RiskStatus]
    note: str = ""

    def risk_factors_with(self, status: RiskStatus) -> frozenset[str]:
        """Return the risk factors that have ``status`` at this encounter.

        Parameters
        ----------
        status : RiskStatus
            The status to select.

        Returns
        -------
        frozenset[str]
            Risk factor identifiers.
        """
        return frozenset(r for r, s in self.risk_factors.items() if s is status)
