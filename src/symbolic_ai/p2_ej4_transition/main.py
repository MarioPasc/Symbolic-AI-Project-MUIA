"""Command-line entry point for EJ4 (``python -m symbolic_ai.p2_ej4_transition.main``).

The only module of EJ4 that prints or writes files. It reads the formulary, the costs and the
pharmacokinetics through :mod:`symbolic_ai.dataloader` (data 1.3.0) and has four modes:

- ``--experiments``: R1 (worked example), R2 (validation), R3 (algorithm table), the Gantt figures
  and the LaTeX tables, written to ``--out-dir`` with a deterministic ``results.json`` (no times;
  they go to ``timings.json``);
- ``--worked-example``: R1 as text (domains at ``H* - 1`` and ``H*``, the emptied domain, the
  schedule); with ``--miniature``, the step-by-step miniature of the slides instead;
- ``--problem ID``: solve one problem with the main configuration and print its schedule;
- ``--list``: the problem identifiers, optionally of one category.

Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §7, §9 (private docs).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import symbolic_ai
from symbolic_ai.dataloader import (
    DataLoaderError,
    DrugCost,
    Formulary,
    Pharmacokinetics,
    load_drug_costs,
    load_formulary,
    load_pharmacokinetics,
)
from symbolic_ai.p1_ej1_logic.solver import AIMA_PYTHON_COMMIT
from symbolic_ai.p1_ej3_search import EJ3_DATA_VERSION
from symbolic_ai.p2_ej4_transition import EJ4_DATA_VERSION, EJ4Error, TransitionProblem
from symbolic_ai.p2_ej4_transition.experiments import (
    EXTENDED_CASE_ID,
    R3_BUDGET,
    R3_CONFIGS,
    R3_TIMEOUT_S,
    AlgorithmRun,
    CategoryAgreement,
    PopulationRow,
    SolvedEntry,
    SyntheticResult,
    ValidationResult,
    census_matches_design,
    check_categories,
    entries_in_category,
    find_entry,
    run_algorithm_table,
    run_knock_on_population,
    run_min_conflicts_rows,
    run_synthetic,
    run_validation,
    solve_entries,
)
from symbolic_ai.p2_ej4_transition.instances import (
    Category,
    CensusRow,
    ContinuityReport,
    ProblemEntry,
    case_study_problem,
    census,
    continuity_with_ej3,
    enumerate_problems,
)
from symbolic_ai.p2_ej4_transition.plot import GANTT_EXTENDED_FIGURE, GANTT_FIGURE, save_gantt
from symbolic_ai.p2_ej4_transition.solve import MAIN_CONFIG, SolverConfig, find_min_horizon
from symbolic_ai.p2_ej4_transition.tables import (
    LABELS_ES,
    algorithm_legend,
    algorithm_table,
    domain_legend,
    domain_table,
    drug_codes,
    validation_table,
)
from symbolic_ai.p2_ej4_transition.worked_example import (
    HorizonDomains,
    ScheduleTable,
    WorkedExample,
    build_walkthrough,
    build_worked_example,
    instance_data,
    miniature_problem,
    value_text,
    walkthrough_text,
)

__all__ = ["RESULTS_SCHEMA", "build_arg_parser", "main"]

#: Identifier and version of the layout of ``results.json``; bump it when a field changes meaning.
RESULTS_SCHEMA = "symai.ej4.results/1"
#: The command recorded in the results' provenance (the output directory does not change them).
_EXPERIMENTS_COMMAND = "python -m symbolic_ai.p2_ej4_transition.main --experiments"
#: Min-conflicts in R3 (design §7 R3, failure case 2).
MIN_CONFLICTS_STEPS = 10_000
MIN_CONFLICTS_SEED = 0

_DEFAULT_OUT_DIR = Path("outputs") / "ej4"
_RESULTS_FILE = "results.json"
_TIMINGS_FILE = "timings.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
#: Label keys of the titles of the two instances of R3 (column groups of its table).
_R3_TITLES = ("r3.worked", "r3.extended")
_LISTABLE = tuple(c.value for c in Category if c is not Category.REFER)
#: The fields of ``SyntheticResult`` that make R3's synthetic row; the others belong to R2.
_SYNTHETIC_R3_FIELDS = (
    "with_h_star_minus_one",
    "ac3_decides_before",
    "solved",
    "backtracking",
    "median_assignments",
    "max_assignments",
    "max_assignments_id",
)


@dataclass(frozen=True, slots=True)
class _Extensions:
    """The R2 and R3 rows added by design v2.3, computed once and written in several places."""

    categories: tuple[CategoryAgreement, ...]
    synthetic: SyntheticResult | None
    continuity: ContinuityReport
    population: tuple[PopulationRow, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``(--experiments | --worked-example | --problem ID | --list)
        [--miniature] [--category CATEGORY] [--out-dir DIR] [--data-dir PATH] [--no-oracle]``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_ai.p2_ej4_transition.main",
        description="EJ4: the safe transition between two regimens, as a CSP indexed by day.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--experiments",
        action="store_true",
        help="Run R1, R2 and R3; write results.json, timings.json, the figures and the LaTeX "
        "tables to --out-dir.",
    )
    mode.add_argument(
        "--worked-example",
        action="store_true",
        help="Print the worked example (R1): domains at H*-1 and H*, the emptied domain, the "
        "schedule.",
    )
    mode.add_argument(
        "--problem", metavar="ID", help="Solve one problem with the main configuration."
    )
    mode.add_argument("--list", action="store_true", help="List the problem identifiers.")
    parser.add_argument(
        "--miniature",
        action="store_true",
        help="With --worked-example: print the miniature of the slides instead (the case study "
        "with every washout 1 d and every taper 0 d except sertraline's 1 d, H* = 3) step by "
        "step: horizon loop, domains, AC-3 removals with their arcs, configurations, search "
        "tree, the oracle's daily states, schedule.",
    )
    parser.add_argument(
        "--category", choices=_LISTABLE, help="With --list: only the problems of this category."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"With --experiments: the output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Database directory (default: $SYMAI_DATA_DIR or <repository root>/data).",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="With --experiments: skip the oracle in R2 and the synthetic batch of R2/R3 "
        "(faster; their fields are null).",
    )
    return parser


def _load(
    data_dir: Path | None,
) -> tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]]:
    """Load the formulary, the costs and the pharmacokinetics of data 1.3.0."""
    return (
        load_formulary(data_dir=data_dir, version=EJ4_DATA_VERSION),
        load_drug_costs(data_dir=data_dir, version=EJ4_DATA_VERSION),
        load_pharmacokinetics(data_dir=data_dir, version=EJ4_DATA_VERSION),
    )


def _drug_names(formulary: Formulary) -> dict[str, str]:
    """Return the Spanish name of every drug (the labels of the talk)."""
    return {drug.drug_id: drug.name_es for drug in formulary.drugs}


# --- text output --------------------------------------------------------------------------------


def _schedule_lines(table: ScheduleTable) -> list[str]:
    """Return a schedule as a day table: one line per day, one column per condition."""
    width = max(14, *(len(value_text(v)) + 2 for row in table.rows for v in row))
    lines = ["day " + "".join(f"{c:<{width}}" for c in table.conditions)]
    for day in range(table.horizon + 1):
        values = "".join(f"{value_text(row[day]):<{width}}" for row in table.rows)
        lines.append(f"{day:>3} {values}")
    return lines


def _domain_lines(domains: HorizonDomains, conditions: Sequence[str]) -> list[str]:
    """Return the domains after AC-3 at one horizon, one line per condition and day."""
    run = domains.run
    verdict = "consistent" if domains.ac3_consistent else f"emptied {domains.emptied}"
    lines = [
        f"H = {domains.horizon}: AC-3 {verdict}; removed {domains.removed_by_ac3}; main "
        f"configuration: {run.status}, {run.assignments} assignments, {run.checks} checks"
    ]
    for condition in conditions:
        for day in range(domains.horizon + 1):
            cell = domains.cell(condition, day)
            kept = "{" + ", ".join(value_text(v) for v in cell.after_ac3) + "}"
            mark = "   <- emptied by AC-3" if domains.emptied == (condition, day) else ""
            removed = len(cell.removed_by_ac3)
            lines.append(f"  {condition} day {day:>2}: {kept} (removed {removed}){mark}")
    return lines


def _print_worked_example(example: WorkedExample) -> None:
    """Print R1 (see ``--worked-example``)."""
    data = example.instance
    print(f"=== worked example {data.problem_id} ({example.config_label}) ===")
    for row in data.drugs:
        state = "acceptable" if row.acceptable else "contraindicated"
        start = ", start drug" if row.start else ""
        print(
            f"  {row.condition_id} {row.drug_id}: washout {row.washout_days} d, taper "
            f"{row.taper_days} d ({state}{start})"
        )
    for c in data.contraindications:
        print(f"  contraindication: {c.risk_factor_id} -> {c.drug_id} ({c.reason})")
    for a, b in data.interactions:
        print(f"  interaction: {a} - {b}")
    for domains in example.domains:
        print()
        print("\n".join(_domain_lines(domains, data.conditions)))
    print()
    print(f"H* = {example.h_star}; schedule (checker: {'valid' if example.valid else 'INVALID'}):")
    print("\n".join(_schedule_lines(example.schedule)))
    print("trace at H*: " + ", ".join(f"{kind} {n}" for kind, n in example.trace_counts))


def _print_validation(validation: ValidationResult, census_ok: bool) -> None:
    """Print the headline numbers of R2."""
    oracle = validation.oracle
    print(
        f"R2: {validation.problems} problems ({validation.events} events); census as in the "
        f"design: {census_ok}"
    )
    if oracle is None:
        print("R2 oracle: skipped (--no-oracle)")
    else:
        print(
            f"R2 oracle: agree {oracle.agree}/{oracle.compared}, budget exhausted "
            f"{oracle.budget_exhausted}, disagreements {len(oracle.disagreements)}"
        )
    checker = validation.checker
    print(f"R2 checker: valid {checker.valid}/{checker.schedules}")
    for m in validation.mutations:
        print(
            f"R2 mutation {m.name} ({m.expected_rule}): applied {m.applied} on {m.schedules} "
            f"schedules, rejected with the expected rule {m.rejected_with_expected_rule}"
        )
    histogram = ", ".join(f"{h}: {n}" for h, n in validation.context.histogram)
    print(f"context: H* -> events {{{histogram}}}")


def _print_extensions(extensions: _Extensions) -> None:
    """Print the headline numbers of the R2 and R3 rows of design v2.3."""
    for c in extensions.categories:
        print(f"R2 category {c.category}: agree {c.agree}/{c.entries}, H* {dict(c.h_stars)}")
    s = extensions.synthetic
    if s is None:
        print("R2 synthetic: skipped (--no-oracle)")
    else:
        print(
            f"R2 synthetic (seed {s.seed}, {s.count}): CSP = oracle = checker on "
            f"{s.three_way_agree}/{s.count}; oracle {dict(s.oracle_statuses)}"
        )
        print(
            f"R3 synthetic: AC-3 decides H*-1 in {s.ac3_decides_before}/{s.with_h_star_minus_one}; "
            f"backtracks at H* in {len(s.backtracking)}/{s.solved}; assignments at H* median "
            f"{s.median_assignments}, max {s.max_assignments} ({s.max_assignments_id})"
        )
    report = extensions.continuity
    print(
        f"R2 continuity with EJ3 ({report.histories} histories): equal {report.equal}, "
        f"different {len(report.different)}, companion {len(report.companion)}, refer "
        f"{report.refer}, unexpected {len(report.unexpected)}"
    )
    for row in extensions.population:
        print(
            f"R3 knock-on population {row.configuration} ({row.entries} at H*-1, budget "
            f"{row.budget}): median {row.median_assignments}, budget hits {row.budget_hits}"
        )


def _print_algorithm_table(runs: Sequence[AlgorithmRun]) -> None:
    """Print the rows of R3."""
    for r in runs:
        print(
            f"R3 {r.instance_id} {r.configuration} H={r.horizon}: {r.status}, {r.assignments} "
            f"assignments, {r.checks} checks, AC-3 removed {r.ac3_removed}"
        )


# --- modes --------------------------------------------------------------------------------------


def _list(entries: Sequence[ProblemEntry], category: str | None) -> int:
    """Mode ``--list``: print ``problem_id, category, events`` of every listable problem."""
    selected = entries_in_category(entries, Category(category)) if category else entries
    for entry in selected:
        if entry.problem is not None:
            print(f"{entry.problem.problem_id}\t{entry.category.value}\t{len(entry.occurrences)}")
    return 0


def _solve_one(entries: Sequence[ProblemEntry], problem_id: str) -> int:
    """Mode ``--problem``: solve one problem and print H* and its schedule."""
    entry = find_entry(entries, problem_id)
    if entry is None or entry.problem is None:
        print(f"error: no problem with id {problem_id!r} (see --list)", file=sys.stderr)
        return 1
    problem = entry.problem
    result = find_min_horizon(problem, MAIN_CONFIG)
    print(f"=== {problem.problem_id} ({entry.category.value}, {len(entry.occurrences)} events) ===")
    if result.h_star is None or result.schedule is None:
        print(f"no transition: {result.status}")
        return 0
    print(f"H* = {result.h_star} ({MAIN_CONFIG.label})")
    table = ScheduleTable.from_schedule(problem, result.schedule, result.h_star)
    print("\n".join(_schedule_lines(table)))
    return 0


def _git(*arguments: str) -> str | None:
    """Run ``git`` in the repository root; ``None`` if git or the repository is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _provenance(formulary: Formulary, no_oracle: bool) -> dict[str, object]:
    """Record what produced the results: command, code and data versions (no timestamps)."""
    status = _git("status", "--porcelain")
    return {
        "command": _EXPERIMENTS_COMMAND + (" --no-oracle" if no_oracle else ""),
        "package_version": symbolic_ai.__version__,
        "data_version": formulary.version,
        "aima_python_commit": AIMA_PYTHON_COMMIT,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
    }


def _census_rows(rows: Sequence[CensusRow]) -> list[dict[str, object]]:
    """Return the census as JSON-ready rows."""
    return [{**dataclasses.asdict(row), "event_kind": row.event_kind.value} for row in rows]


def _config_dict(key: str, config: SolverConfig) -> dict[str, object]:
    """Return a solver configuration as a JSON-ready ``dict`` with its key and label."""
    return {"key": key, "label": config.label, **dataclasses.asdict(config)}


def _write_tables(
    out_dir: Path,
    example: WorkedExample,
    validation: ValidationResult,
    census_rows: Sequence[CensusRow],
    extensions: _Extensions,
    r3_runs: Sequence[AlgorithmRun],
    r3_instances: Sequence[tuple[str, str, int]],
    names: Mapping[str, str],
) -> list[Path]:
    """Write the four LaTeX fragments (and their legends as comments) to ``out_dir``."""
    drugs = [row.drug_id for row in example.instance.drugs]
    codes = drug_codes(drugs, names)
    legend = domain_legend(codes, names)
    written = []
    for domains in example.domains:
        path = out_dir / f"tab_ej4_domains_h{domains.horizon}.tex"
        table = domain_table(domains, example.instance.conditions, codes)
        path.write_text(f"% Leyenda: {legend}\n{table}", encoding="utf-8")
        written.append(path)
    path = out_dir / "tab_ej4_r2.tex"
    r2_table = validation_table(
        validation,
        census_matches_design(census_rows),
        synthetic=extensions.synthetic,
        continuity=extensions.continuity,
        categories=extensions.categories,
        census_events=sum(row.events for row in census_rows),
    )
    path.write_text(r2_table, encoding="utf-8")
    written.append(path)
    path = out_dir / "tab_ej4_r3.tex"
    population_budget = extensions.population[0].budget if extensions.population else None
    legend = algorithm_legend(example.h_star, R3_BUDGET, population_budget=population_budget)
    r3_table = algorithm_table(
        r3_runs, r3_instances, population=extensions.population, synthetic=extensions.synthetic
    )
    path.write_text(f"% Leyenda: {legend}\n{r3_table}", "utf-8")
    written.append(path)
    return written


def _load_ej3(data_dir: Path | None) -> tuple[Formulary, dict[str, int]]:
    """Load the formulary and the costs exactly as EJ3's CLI does (data 1.2.0)."""
    formulary = load_formulary(data_dir=data_dir, version=EJ3_DATA_VERSION)
    priced = load_drug_costs(data_dir=data_dir, version=EJ3_DATA_VERSION)
    return formulary, {drug_id: cost.monthly_cost_cents for drug_id, cost in priced.items()}


def _run_extensions(
    args: argparse.Namespace,
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    solved: Sequence[SolvedEntry],
    timings: dict[str, object],
) -> _Extensions:
    """Run the R2 and R3 rows of design v2.3, recording each phase's time in ``timings``."""
    clock = time.perf_counter
    phase = clock()
    categories = check_categories(solved)
    synthetic = None if args.no_oracle else run_synthetic()
    timings["synthetic_s"] = clock() - phase
    phase = clock()
    continuity = continuity_with_ej3(formulary, costs, *_load_ej3(args.data_dir))
    timings["continuity_s"] = clock() - phase
    phase = clock()
    population = run_knock_on_population(solved)
    timings["population_s"] = clock() - phase
    timings["population_configs"] = [
        {"configuration": row.configuration, "seconds": row.seconds} for row in population
    ]
    return _Extensions(categories, synthetic, continuity, population)


def _extensions_json(extensions: _Extensions) -> tuple[dict[str, object], dict[str, object]]:
    """Return the JSON of the v2.3 rows: the part that goes to ``r2`` and the part to ``r3``."""
    synthetic: dict[str, object] | None = None
    if extensions.synthetic is not None:
        synthetic = dataclasses.asdict(extensions.synthetic)
    synthetic_r3: dict[str, object] | None = None
    if synthetic is not None:
        synthetic_r3 = {key: synthetic.pop(key) for key in _SYNTHETIC_R3_FIELDS}
        synthetic_r3.update(seed=synthetic["seed"], count=synthetic["count"])
    r2: dict[str, object] = {
        "categories": [dataclasses.asdict(c) for c in extensions.categories],
        "synthetic": synthetic,
        "continuity_with_ej3": dataclasses.asdict(extensions.continuity),
    }
    r3: dict[str, object] = {
        "knock_on_population": [row.to_dict() for row in extensions.population],
        "synthetic_row": synthetic_r3,
    }
    return r2, r3


def _experiments(args: argparse.Namespace) -> int:
    """Mode ``--experiments``: R1, R2, R3, figures and tables; write the results."""
    clock = time.perf_counter
    started = clock()
    # Created first, so that an unusable directory fails now and not after minutes of solving.
    out_dir: Path = args.out_dir or _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    formulary, costs, pharmacokinetics = _load(args.data_dir)
    entries = enumerate_problems(formulary, costs, pharmacokinetics)
    names = _drug_names(formulary)
    timings: dict[str, object] = {"load_and_enumerate_s": clock() - started}

    phase = clock()
    worked = case_study_problem(formulary, costs, pharmacokinetics)
    example = build_worked_example(worked, formulary.contraindications)
    timings["r1_s"] = clock() - phase
    _print_worked_example(example)

    phase = clock()
    solved = solve_entries(entries)
    validation = run_validation(entries, use_oracle=not args.no_oracle, solved_entries=solved)
    rows = census(entries)
    census_ok = census_matches_design(rows)
    timings["r2_s"] = clock() - phase
    _print_validation(validation, census_ok)
    extensions = _run_extensions(args, formulary, costs, solved, timings)

    phase = clock()
    extended = _extended_case(entries)
    extended_result = find_min_horizon(extended, MAIN_CONFIG)
    if extended_result.h_star is None or extended_result.schedule is None:
        raise EJ4Error(f"the extended case {EXTENDED_CASE_ID} has no transition")
    instances = [(worked, example.h_star), (extended, extended_result.h_star)]
    r3_runs = run_algorithm_table(instances) + run_min_conflicts_rows(
        worked,
        (example.h_star - 1, example.h_star),
        MIN_CONFLICTS_STEPS,
        MIN_CONFLICTS_SEED,
    )
    timings["r3_s"] = clock() - phase
    timings["r3_runs"] = [
        {
            "instance_id": r.instance_id,
            "configuration": r.configuration,
            "horizon": r.horizon,
            "seconds": r.seconds,
        }
        for r in r3_runs
    ]
    _print_algorithm_table(r3_runs)
    _print_extensions(extensions)

    phase = clock()
    extended_table = ScheduleTable.from_schedule(
        extended, extended_result.schedule, extended_result.h_star
    )
    written = [
        *save_gantt(example.schedule, names, out_dir / GANTT_FIGURE),
        *save_gantt(extended_table, names, out_dir / GANTT_EXTENDED_FIGURE),
    ]
    r3_instances = [
        (problem.problem_id, LABELS_ES[title], h_star)
        for (problem, h_star), title in zip(instances, _R3_TITLES, strict=True)
    ]
    written += _write_tables(
        out_dir, example, validation, rows, extensions, r3_runs, r3_instances, names
    )
    timings["figures_and_tables_s"] = clock() - phase

    validation_dict = dataclasses.asdict(validation)
    context = validation_dict.pop("context")
    r2_extra, r3_extra = _extensions_json(extensions)
    validation_dict.update(r2_extra)
    results = {
        "schema": RESULTS_SCHEMA,
        "provenance": _provenance(formulary, args.no_oracle),
        "main_config": _config_dict("main", MAIN_CONFIG),
        "r1": example.to_dict(),
        "r2": validation_dict,
        "census": _census_rows(rows),
        "census_matches_design": census_ok,
        "context": context,
        "r3": {
            "budget_assignments": R3_BUDGET,
            "timeout_s": R3_TIMEOUT_S,
            "configurations": [_config_dict(key, config) for key, config in R3_CONFIGS],
            "instances": [{"problem_id": p, "h_star": h} for p, _title, h in r3_instances],
            "min_conflicts": {"max_steps": MIN_CONFLICTS_STEPS, "seed": MIN_CONFLICTS_SEED},
            "runs": [r.to_dict() for r in r3_runs],
            **r3_extra,
        },
        "extended_case": {
            "instance": dataclasses.asdict(instance_data(extended, formulary.contraindications)),
            "h_star": extended_result.h_star,
            "schedule": dataclasses.asdict(extended_table),
        },
        "outputs": sorted(p.name for p in written),
    }
    _write_json(out_dir / _RESULTS_FILE, results)
    timings["total_s"] = clock() - started
    _write_json(out_dir / _TIMINGS_FILE, timings)
    for path in [*written, out_dir / _RESULTS_FILE, out_dir / _TIMINGS_FILE]:
        print(f"wrote {path}")
    print(f"total {timings['total_s']:.1f} s")
    return 0


def _extended_case(entries: Sequence[ProblemEntry]) -> TransitionProblem:
    """Return the extended case of R3 (:data:`EXTENDED_CASE_ID`)."""
    entry = find_entry(entries, EXTENDED_CASE_ID)
    if entry is None or entry.problem is None:
        raise EJ4Error(f"the extended case {EXTENDED_CASE_ID} is not among the problems")
    return entry.problem


def _write_json(path: Path, content: Mapping[str, object]) -> None:
    """Write ``content`` as sorted, indented JSON (byte-identical for equal content)."""
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI in the mode selected by the arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, excluding the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: ``0`` on success; ``1`` for an unknown problem, an EJ4 error, a
        database that cannot be read or an output directory that cannot be written (one line
        ``error: ...`` on stderr, no traceback); ``2`` for invalid arguments (argparse).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _check_option_combinations(parser, args)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.experiments:
            return _experiments(args)
        formulary, costs, pharmacokinetics = _load(args.data_dir)
        if args.worked_example:
            problem = case_study_problem(formulary, costs, pharmacokinetics)
            if args.miniature:
                miniature = miniature_problem(problem)
                print(walkthrough_text(build_walkthrough(miniature, formulary.contraindications)))
                return 0
            _print_worked_example(build_worked_example(problem, formulary.contraindications))
            return 0
        entries = enumerate_problems(formulary, costs, pharmacokinetics)
        if args.list:
            return _list(entries, args.category)
        return _solve_one(entries, args.problem)
    except (EJ4Error, DataLoaderError, OSError) as exc:
        # The CLI boundary: a bad --data-dir or --out-dir is the user's error, not a crash.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _check_option_combinations(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject an option that its mode would silently ignore (``parser.error`` exits with 2)."""
    if args.category is not None and not args.list:
        parser.error("--category needs --list")
    if args.miniature and not args.worked_example:
        parser.error("--miniature needs --worked-example")
    if args.no_oracle and not args.experiments:
        parser.error("--no-oracle needs --experiments")
    if args.out_dir is not None and not args.experiments:
        parser.error("--out-dir needs --experiments")


if __name__ == "__main__":
    sys.exit(main())
