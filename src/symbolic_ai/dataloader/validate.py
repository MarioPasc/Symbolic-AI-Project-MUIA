"""CLI entry point: ``python -m symbolic_ai.dataloader.validate [--data-dir PATH]``.

Runs every validation rule of ``01-database.md`` section 5 against the database, prints a summary of
table counts and exits 0 when the database is valid, or prints every problem found and exits 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from symbolic_ai.dataloader import (
    DatabaseValidationError,
    default_data_dir,
    load_encounters,
    load_formulary,
    load_patients,
    validate_database,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the symbolic_ai course database.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory of the database (default: $SYMAI_DATA_DIR or <repo root>/data).",
    )
    return parser.parse_args(argv)


def _print_summary(data_dir: Path) -> None:
    formulary = load_formulary(data_dir)
    patients = load_patients(data_dir)
    encounters = load_encounters(data_dir)
    total_class_members = sum(len(members) for members in formulary.class_members.values())
    candidate_pairs = sum(len(drugs) for drugs in formulary.candidates.values())
    print("OK: the database is valid.")
    print(f"data version: {formulary.version}")
    print(f"conditions: {len(formulary.conditions)}")
    print(f"drugs: {len(formulary.drugs)}")
    print(f"risk_factors: {len(formulary.risk_factors)}")
    print(f"candidates: {candidate_pairs}")
    print(f"adverse_interactions: {len(formulary.interactions)}")
    print(f"contraindications: {len(formulary.contraindications)}")
    print(f"coprescriptions: {len(formulary.coprescriptions)}")
    print(f"drug_classes: {len(formulary.drug_classes)} ({total_class_members} members)")
    print(f"patients: {len(patients)}")
    print(f"encounters: {len(encounters)}")


def main(argv: list[str] | None = None) -> int:
    """Run the validation CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` if the database is valid, ``1`` otherwise.
    """
    args = _parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    try:
        validate_database(data_dir)
    except DatabaseValidationError as exc:
        print(f"FAILED: {len(exc.problems)} database problem(s):")
        for problem in exc.problems:
            print(f"- {problem}")
        return 1
    _print_summary(data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
