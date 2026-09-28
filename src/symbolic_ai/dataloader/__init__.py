"""The only reader of the database under ``data/``: loads, validates and returns immutable models.

The signatures below are a frozen contract (docs SPECIFICATIONS/02-code-architecture.md, section 4).
The bodies are stubs until the dataloader implementation lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from symbolic_ai.dataloader.errors import DatabaseValidationError, DataLoaderError, UnknownIdError
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


def default_data_dir() -> Path:
    """Return ``$SYMAI_DATA_DIR`` if set, else ``<repository root>/data``."""
    raise NotImplementedError


def load_formulary(data_dir: Path | None = None, version: str | None = None) -> Formulary:
    """Load the formulary as of ``version`` (default: the database version); validates first."""
    raise NotImplementedError


def load_patients(data_dir: Path | None = None) -> Mapping[str, Patient]:
    """Load every patient, keyed and sorted by ``patient_id``."""
    raise NotImplementedError


def load_encounters(data_dir: Path | None = None) -> Mapping[str, Encounter]:
    """Load every encounter, keyed and sorted by ``encounter_id``."""
    raise NotImplementedError


def load_encounter(encounter_id: str, data_dir: Path | None = None) -> Encounter:
    """Load one encounter; raises ``UnknownIdError`` if it does not exist."""
    raise NotImplementedError


def validate_database(data_dir: Path | None = None) -> None:
    """Run every check of the database specification; raises ``DatabaseValidationError``."""
    raise NotImplementedError
