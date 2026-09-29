"""Command-line entry point for the EJ1 agent (``python -m symbolic_ai.p1_ej1_logic.main``).

The only module of this exercise allowed to print or write files. It reads the formulary and the
encounters through :mod:`symbolic_ai.dataloader` and has three modes: decide encounters with
:class:`PrescribingAgent` (``--encounter`` / ``--all``), run the experiments of the report's
*Resultados* and save them (``--experiments``), or redraw the figures from a saved results file
(``--plot``).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import symbolic_ai
from symbolic_ai.dataloader import load_encounter, load_encounters, load_formulary
from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import (
    EJ1_FORMULARY_VERSION,
    Action,
    Decision,
    DPLLSolver,
    PrescribingAgent,
)
from symbolic_ai.p1_ej1_logic.errors import ResultsFormatError
from symbolic_ai.p1_ej1_logic.experiments import (
    PatternsResult,
    ScenarioResult,
    ValueOrderingResult,
    VerificationResult,
    as_jsonable,
    run_scenarios,
    run_value_ordering,
    run_verification,
    untreatable_patterns,
)
from symbolic_ai.p1_ej1_logic.plot import VALUE_ORDERING_FIGURE, plot_value_ordering
from symbolic_ai.p1_ej1_logic.semantics import RegimenSpace
from symbolic_ai.p1_ej1_logic.solver import AIMA_PYTHON_COMMIT

__all__ = ["RESULTS_SCHEMA", "build_arg_parser", "main"]

#: Identifier and version of the layout of ``results.json``; bump it when a field changes meaning.
RESULTS_SCHEMA = "symai.ej1.results/1"
#: The command recorded in the results' provenance (worker count and output directory do not
#: change the results, so they are not part of it).
_EXPERIMENTS_COMMAND = "python -m symbolic_ai.p1_ej1_logic.main --experiments"

_DEFAULT_JSON_PATH = Path("outputs") / "ej1_decisions.json"
_DEFAULT_OUT_DIR = Path("outputs") / "ej1"
_RESULTS_FILE = "results.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``(--encounter ID ... | --all | --experiments | --plot RESULTS_JSON)
        [--first-value {false,true}] [--no-classify] [--data-dir PATH] [--json OUT]
        [--out-dir DIR] [--workers N]``.
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
    selection.add_argument(
        "--experiments",
        action="store_true",
        help="Run the report's experiments (scenarios, exhaustive verification, value ordering) "
        "and write results.json and the figure to --out-dir.",
    )
    selection.add_argument(
        "--plot",
        type=Path,
        metavar="RESULTS_JSON",
        help="Only redraw the figure from a saved results.json, into --out-dir.",
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory of --experiments and --plot (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for --experiments (default: 1; the results do not depend on it).",
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
            {
                "name": call.name,
                "satisfiable": call.satisfiable,
                "calls": call.calls,
                "decisions": call.decisions,
                "failures": call.failures,
            }
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


def _decide(args: argparse.Namespace, formulary: Formulary) -> int:
    """Mode ``--encounter`` / ``--all``: decide, print, and optionally save as JSON."""
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


def _git(*arguments: str) -> str | None:
    """Run ``git`` in the repository root; ``None`` if git or the repository is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    """Installed version of distribution ``name``, or ``None``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _provenance(formulary: Formulary) -> dict[str, object]:
    """Record what produced the results: command, code and data versions, libraries."""
    status = _git("status", "--porcelain")
    return {
        "command": _EXPERIMENTS_COMMAND,
        "package_version": symbolic_ai.__version__,
        "formulary_version": formulary.version,
        "aima_python_commit": AIMA_PYTHON_COMMIT,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "matplotlib": _package_version("matplotlib"),
    }


def _experiment_results(args: argparse.Namespace, formulary: Formulary) -> dict[str, object]:
    """Run P1-P3 and the untreatable patterns; return the content of ``results.json``."""
    encounters = load_encounters(data_dir=args.data_dir)
    space = RegimenSpace(formulary)
    scenarios = run_scenarios(formulary, [encounters[k] for k in sorted(encounters)], space)
    verification = run_verification(formulary, space, workers=args.workers)
    value_ordering = run_value_ordering(formulary, space, workers=args.workers)
    patterns = untreatable_patterns(formulary, space)
    _print_summary(scenarios, verification, value_ordering, patterns)
    return {
        "schema": RESULTS_SCHEMA,
        "provenance": _provenance(formulary),
        "oracle": {
            "drugs": len(space.drug_ids),
            "regimens": 2 ** len(space.drug_ids),
            "regimens_satisfying_a3_a6": space.n_candidate_regimens,
        },
        "scenarios": as_jsonable(scenarios),
        "verification": as_jsonable(verification),
        "value_ordering": as_jsonable(value_ordering),
        "untreatable_patterns": as_jsonable(patterns),
    }


def _print_summary(
    scenarios: Sequence[ScenarioResult],
    verification: VerificationResult,
    value_ordering: ValueOrderingResult,
    patterns: PatternsResult,
) -> None:
    """Print the headline numbers of each experiment."""
    print("P1 scenarios")
    for s in scenarios:
        output = ", ".join(s.regimen or s.tests) or "-"
        print(
            f"  {s.encounter_id}: {s.action} [{output}] "
            f"Gamma+ {s.gamma_plus_models} models, Gamma- {s.gamma_minus_models} models, "
            f"conflict of {len(s.conflict)} clauses"
        )
    print(
        f"P2 verification: {verification.n_records} records, {verification.actions}, "
        f"{len(verification.disagreements)} disagreement(s)"
    )
    for summary in value_ordering.orderings:
        print(
            f"P3 {summary.ordering}: minimal in {summary.n_minimal}/{summary.n_satisfiable}, "
            f"mean excess {summary.mean_excess}, max {summary.max_excess}, "
            f"mean DPLL calls {summary.mean_calls}"
        )
    print(
        f"untreatable: {patterns.n_untreatable}/{patterns.n_records} records, "
        f"{len(patterns.patterns)} minimal pattern(s), upward closed: {patterns.upward_closed}"
    )


def _load_results(path: Path) -> Mapping[str, object]:
    """Read a saved ``results.json`` and check its schema."""
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict) or loaded.get("schema") != RESULTS_SCHEMA:
        raise ResultsFormatError(f"{path} is not a {RESULTS_SCHEMA} results file")
    return loaded


def _draw_figures(results_path: Path, out_dir: Path) -> None:
    """Draw every figure from the saved results file and print the paths written."""
    results = _load_results(results_path)
    section = results.get("value_ordering")
    if not isinstance(section, Mapping):
        raise ResultsFormatError(f"{results_path} has no value_ordering section")
    for path in plot_value_ordering(section, out_dir / VALUE_ORDERING_FIGURE):
        print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI in the mode selected by the arguments.

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
    if args.plot is not None:
        _draw_figures(args.plot, args.out_dir)
        return 0

    formulary = load_formulary(data_dir=args.data_dir, version=EJ1_FORMULARY_VERSION)
    if not args.experiments:
        return _decide(args, formulary)

    results = _experiment_results(args, formulary)
    results_path = args.out_dir / _RESULTS_FILE
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {results_path}")
    # The figure is drawn from the file just written, so it can only show saved results.
    _draw_figures(results_path, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
