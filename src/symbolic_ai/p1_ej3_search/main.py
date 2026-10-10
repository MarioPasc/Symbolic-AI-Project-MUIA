"""Command-line entry point for the EJ3 agent (``python -m symbolic_ai.p1_ej3_search.main``).

The only module of this exercise allowed to print or write files. It reads the formulary, the
drug costs and the encounters through :mod:`symbolic_ai.dataloader` and has two modes: decide
encounters with :class:`CostAwareAgent` (``--encounter`` / ``--all``), or run the experiments P6
and P7 and save them (``--experiments``).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import symbolic_ai
from symbolic_ai.dataloader import (
    UnknownIdError,
    load_drug_costs,
    load_encounter,
    load_encounters,
    load_formulary,
)
from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import Action, DPLLSolver, RegimenSpace
from symbolic_ai.p1_ej1_logic.experiments import as_jsonable
from symbolic_ai.p1_ej1_logic.solver import AIMA_PYTHON_COMMIT
from symbolic_ai.p1_ej3_search import EJ3_DATA_VERSION, Algorithm, CostAwareAgent, CostedDecision
from symbolic_ai.p1_ej3_search.experiments import (
    ComparisonResult,
    ScenarioResult,
    run_scenarios,
    run_search_comparison,
)

__all__ = ["RESULTS_SCHEMA", "build_arg_parser", "main"]

#: Identifier and version of the layout of ``results.json``; bump it when a field changes meaning.
RESULTS_SCHEMA = "symai.ej3.results/1"
#: The command recorded in the results' provenance (worker count and output directory do not
#: change the results, so they are not part of it).
_EXPERIMENTS_COMMAND = "python -m symbolic_ai.p1_ej3_search.main --experiments"

_DEFAULT_JSON_PATH = Path("outputs") / "ej3_decisions.json"
_DEFAULT_OUT_DIR = Path("outputs") / "ej3"
_RESULTS_FILE = "results.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``(--encounter ID ... | --all | --experiments)
        [--algorithm {astar,idastar,greedy,uniform_cost}] [--no-prune] [--data-dir PATH]
        [--json OUT] [--out-dir DIR] [--workers N]``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_ai.p1_ej3_search.main",
        description="EJ3: decide as the EJ1 agent and, when it prescribes, search the cheapest "
        "safe regimen.",
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
        help="Run the experiments (P6 scenarios; P7 every fully observed record) and write "
        "results.json to --out-dir.",
    )
    parser.add_argument(
        "--algorithm",
        choices=tuple(algorithm.value for algorithm in Algorithm),
        default=Algorithm.ASTAR.value,
        help="Search algorithm of --encounter / --all (default: astar; greedy is not optimal).",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip Agent 1's essential / excluded classification and search the plain problem.",
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
        help=f"Write decisions as JSON to OUT (default: {_DEFAULT_JSON_PATH} with no path given).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory of --experiments (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for --experiments (default: 1; the results do not depend on it).",
    )
    return parser


def _euros(cents: int | None) -> str:
    """Format a monthly cost given in euro cents."""
    return "-" if cents is None else f"{cents / 100:.2f} EUR/month"


def _select_encounters(args: argparse.Namespace) -> list[Encounter]:
    """Load the encounters requested by ``--encounter`` or ``--all``, in a deterministic order."""
    if args.all:
        encounters = load_encounters(data_dir=args.data_dir)
        return [encounters[encounter_id] for encounter_id in sorted(encounters)]
    return [load_encounter(encounter_id, data_dir=args.data_dir) for encounter_id in args.encounter]


def _decision_to_dict(costed: CostedDecision) -> dict[str, object]:
    """Serialise one :class:`CostedDecision` into a JSON-ready ``dict``."""
    found = costed.search
    return {
        "action": costed.action.value,
        "tests": list(costed.decision.tests),
        "agent1_regimen": sorted(costed.decision.regimen),
        "agent1_cost_cents": costed.baseline_cost,
        "regimen": sorted(costed.regimen),
        "cost_cents": costed.cost,
        "saving_cents": costed.saving,
        "search": None
        if found is None
        else {
            "algorithm": found.algorithm.value,
            "chosen": list(found.chosen),
            "expanded": found.expanded,
            "generated": found.generated,
            "goal_tests": found.goal_tests,
        },
    }


def _print_decision(encounter: Encounter, costed: CostedDecision) -> None:
    """Print one readable block describing ``costed`` for ``encounter``."""
    print(f"=== {encounter.encounter_id} ({encounter.patient_id}, visit {encounter.seq}) ===")
    print(f"action: {costed.action.value}")
    found = costed.search
    if costed.action is Action.PRESCRIBE and found is not None:
        baseline = costed.decision.regimen
        print(
            f"agent 1 regimen ({len(baseline)} drugs, {_euros(costed.baseline_cost)}): "
            f"{', '.join(sorted(baseline))}"
        )
        print(
            f"{found.algorithm.value} regimen ({len(costed.regimen)} drugs, "
            f"{_euros(costed.cost)}): {', '.join(sorted(costed.regimen))}"
        )
        print(f"saving: {_euros(costed.saving)}")
        print(
            f"search: {found.expanded} node(s) expanded, {found.generated} generated, "
            f"{found.goal_tests} goal test(s)"
        )
    elif costed.action is Action.REQUEST_TEST:
        print(f"tests: {', '.join(costed.decision.tests)}")
    print()


def _decide(args: argparse.Namespace, formulary: Formulary, costs: Mapping[str, int]) -> int:
    """Mode ``--encounter`` / ``--all``: decide, print, and optionally save as JSON."""
    try:
        encounters = _select_encounters(args)
    except UnknownIdError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    agent = CostAwareAgent(
        formulary, costs, DPLLSolver(), Algorithm(args.algorithm), prune=not args.no_prune
    )
    results: dict[str, dict[str, object]] = {}
    for encounter in encounters:
        costed = agent.decide(encounter)
        _print_decision(encounter, costed)
        results[encounter.encounter_id] = _decision_to_dict(costed)

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
        "data_version": formulary.version,
        "aima_python_commit": AIMA_PYTHON_COMMIT,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
    }


def _experiment_results(
    args: argparse.Namespace, formulary: Formulary, costs: Mapping[str, int]
) -> dict[str, object]:
    """Run P6 and P7; return the content of ``results.json``."""
    encounters = load_encounters(data_dir=args.data_dir)
    space = RegimenSpace(formulary)
    scenarios = run_scenarios(formulary, costs, [encounters[k] for k in sorted(encounters)], space)
    comparison = run_search_comparison(formulary, costs, space, workers=args.workers)
    _print_summary(scenarios, comparison)
    return {
        "schema": RESULTS_SCHEMA,
        "provenance": _provenance(formulary),
        "costs_cents_per_month": dict(sorted(costs.items())),
        "oracle": {
            "drugs": len(space.drug_ids),
            "regimens": 2 ** len(space.drug_ids),
            "regimens_satisfying_a3_a6": space.n_candidate_regimens,
        },
        "scenarios": as_jsonable(scenarios),
        "search_comparison": as_jsonable(comparison),
    }


def _print_summary(scenarios: Sequence[ScenarioResult], comparison: ComparisonResult) -> None:
    """Print the headline numbers of each experiment."""
    print("P6 scenarios")
    for s in scenarios:
        if s.cost is None:
            print(f"  {s.encounter_id}: {s.action} [{', '.join(s.tests) or '-'}]")
            continue
        print(
            f"  {s.encounter_id}: {s.action} [{', '.join(s.regimen)}] {_euros(s.cost)}; "
            f"agent 1 {_euros(s.baseline_cost)}; oracle minimum {_euros(s.minimum_cost)}; "
            f"{s.expanded} expanded, {s.generated} generated"
        )
    n = comparison.n_satisfiable
    print(
        f"P7 records: {comparison.n_instances}, satisfiable {n}; search verdicts agree with the "
        f"oracle: {comparison.verdicts_agree_with_oracle}; unique optimum in "
        f"{comparison.n_unique_optimum}"
    )
    print(
        f"P7 heuristic: admissible at the root: {comparison.heuristic_admissible_at_root}; "
        f"equal to the minimum cost in {comparison.n_root_heuristic_exact}/{n}"
    )
    baseline = comparison.baseline
    print(
        f"P7 agent 1: cheapest in {baseline.n_optimal}/{n}; mean cost "
        f"{_euros(round(baseline.mean_cost))} against {_euros(round(baseline.mean_minimum_cost))}; "
        f"mean excess {_euros(round(baseline.mean_excess_cost))}, max "
        f"{_euros(baseline.max_excess_cost)}"
    )
    for c in comparison.configurations:
        print(
            f"P7 {c.configuration}: cheapest in {c.n_optimal}/{n}, mean excess "
            f"{_euros(round(c.mean_excess_cost))}; expanded mean {c.mean_expanded} max "
            f"{c.max_expanded}; generated mean {c.mean_generated} max {c.max_generated}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI in the mode selected by the arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, excluding the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``2`` for an unknown encounter.
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    formulary = load_formulary(data_dir=args.data_dir, version=EJ3_DATA_VERSION)
    priced = load_drug_costs(data_dir=args.data_dir, version=EJ3_DATA_VERSION)
    costs = {drug_id: cost.monthly_cost_cents for drug_id, cost in priced.items()}
    if not args.experiments:
        return _decide(args, formulary, costs)

    results = _experiment_results(args, formulary, costs)
    results_path = args.out_dir / _RESULTS_FILE
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
