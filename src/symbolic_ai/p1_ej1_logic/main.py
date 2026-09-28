"""Command-line entry point for the EJ1 agent (``python -m symbolic_ai.p1_ej1_logic.main``).

The only module of this exercise allowed to print or write files: it reads the formulary and the
requested encounters through :mod:`symbolic_ai.dataloader`, runs :class:`PrescribingAgent`, prints
one readable block per encounter, and optionally writes the results as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from symbolic_ai.dataloader import load_encounter, load_encounters, load_formulary
from symbolic_ai.dataloader.models import Encounter
from symbolic_ai.p1_ej1_logic import (
    EJ1_FORMULARY_VERSION,
    Action,
    Decision,
    DPLLSolver,
    PrescribingAgent,
)

__all__ = ["build_arg_parser", "main"]

_DEFAULT_JSON_PATH = Path("outputs") / "ej1_decisions.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``[--encounter ID ... | --all] [--first-value {false,true}] [--no-classify]
        [--data-dir PATH] [--json OUT]``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_ai.p1_ej1_logic.main",
        description="EJ1: decide PRESCRIBE / REQUEST_TEST / REFER for one or more encounters.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--encounter",
        nargs="+",
        metavar="ID",
        help="Encounter identifiers to decide, e.g. E001 E002.",
    )
    selection.add_argument(
        "--all", action="store_true", help="Decide every encounter of the database."
    )
    parser.add_argument(
        "--first-value",
        choices=("false", "true"),
        default="false",
        help="Value tried first for decision symbols T_d (default: false; see solver-choice.md).",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip the essential / excluded / optional classification after a PRESCRIBE.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Database directory (default: $SYMAI_DATA_DIR or <repository root>/data).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        nargs="?",
        const=_DEFAULT_JSON_PATH,
        default=None,
        metavar="OUT",
        help=f"Write results as JSON to OUT (default: {_DEFAULT_JSON_PATH} with no path given).",
    )
    return parser


def _select_encounters(args: argparse.Namespace) -> list[Encounter]:
    """Load the encounters requested by ``--encounter`` or ``--all``, in a deterministic order."""
    if args.all:
        encounters = load_encounters(data_dir=args.data_dir)
        return [encounters[encounter_id] for encounter_id in sorted(encounters)]
    return [load_encounter(encounter_id, data_dir=args.data_dir) for encounter_id in args.encounter]


def _decision_to_dict(decision: Decision) -> dict[str, object]:
    """Serialise one :class:`Decision` into a JSON-ready ``dict``."""
    classification = (
        {drug_id: status.value for drug_id, status in sorted(decision.classification.items())}
        if decision.classification is not None
        else None
    )
    return {
        "action": decision.action.value,
        "regimen": sorted(decision.regimen),
        "tests": list(decision.tests),
        "classification": classification,
        "trace": [
            {"name": call.name, "satisfiable": call.satisfiable, "calls": call.calls}
            for call in decision.trace
        ],
    }


def _print_decision(encounter: Encounter, decision: Decision) -> None:
    """Print one readable block describing ``decision`` for ``encounter``."""
    print(f"=== {encounter.encounter_id} ({encounter.patient_id}, visit {encounter.seq}) ===")
    print(f"action: {decision.action.value}")
    if decision.action is Action.PRESCRIBE:
        print(f"regimen ({len(decision.regimen)} drugs): {', '.join(sorted(decision.regimen))}")
        if decision.classification is not None:
            for status in ("essential", "excluded", "optional"):
                drugs = sorted(
                    drug_id
                    for drug_id, classification in decision.classification.items()
                    if classification.value == status
                )
                print(f"  {status} ({len(drugs)}): {', '.join(drugs)}")
    elif decision.action is Action.REQUEST_TEST:
        print(f"tests: {', '.join(decision.tests)}")
    total_calls = sum(call.calls for call in decision.trace)
    print(f"trace: {len(decision.trace)} SAT call(s), {total_calls} solver call(s) total")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI: decide one or more encounters, print, and optionally save as JSON.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, excluding the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = build_arg_parser().parse_args(argv)
    formulary = load_formulary(data_dir=args.data_dir, version=EJ1_FORMULARY_VERSION)
    encounters = _select_encounters(args)

    solver = DPLLSolver(decision_first_value=args.first_value == "true")
    agent = PrescribingAgent(formulary, solver, classify=not args.no_classify)

    results: dict[str, dict[str, object]] = {}
    for encounter in encounters:
        decision = agent.decide(encounter)
        _print_decision(encounter, decision)
        results[encounter.encounter_id] = _decision_to_dict(decision)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
