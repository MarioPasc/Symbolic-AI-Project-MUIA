"""Run every experiment of the report, EJ1 then EJ2 then EJ3, with one command.

``python -m symbolic_ai.reproduce`` (console script ``symai-reproduce``) calls the ``main`` of each
exercise in this process with ``--experiments --out-dir <out-root>/ej<k>``, exactly as the
per-exercise commands do, so every results file and figure lands where those commands write it
(``outputs/ej1``, ``outputs/ej2``, ``outputs/ej3`` by default). It keeps going after a failure,
prints one summary line per exercise and exits non-zero if any exercise failed. With
``--report-figures DIR`` it also copies the figure PDFs the report includes to the names the
report's LaTeX source uses.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias, cast

__all__ = [
    "EXERCISES",
    "REPORT_FIGURES",
    "Exercise",
    "ExerciseOutcome",
    "ExerciseSpec",
    "FigureCopy",
    "ReportFigure",
    "build_arg_parser",
    "copy_report_figures",
    "exercise_argv",
    "main",
    "out_dir",
    "run_exercise",
]

logger = logging.getLogger(__name__)

#: The signature of every exercise's ``main``: arguments without the program name, exit code.
ExerciseMain: TypeAlias = Callable[[Sequence[str] | None], int]

_DEFAULT_OUT_ROOT = Path("outputs")
_RESULTS_FILE = "results.json"


class Exercise(StrEnum):
    """The exercises of Práctica 1, in the order they are run."""

    EJ1 = "ej1"
    EJ2 = "ej2"
    EJ3 = "ej3"


@dataclass(frozen=True, slots=True)
class ExerciseSpec:
    """How to run one exercise's experiments.

    Parameters
    ----------
    exercise : Exercise
        The exercise; its value names the output sub-directory (``<out-root>/ej<k>``).
    module : str
        The module whose ``main(argv)`` runs the exercise's CLI.
    accepts_workers : bool
        Whether that CLI has ``--workers``.
    accepts_no_oracle : bool
        Whether that CLI has ``--no-oracle``.
    """

    exercise: Exercise
    module: str
    accepts_workers: bool
    accepts_no_oracle: bool


#: The exercises, in the order they run (EJ2's CLI has no ``--workers``; only EJ2 has an oracle
#: that needs Java).
EXERCISES: tuple[ExerciseSpec, ...] = (
    ExerciseSpec(Exercise.EJ1, "symbolic_ai.p1_ej1_logic.main", True, False),
    ExerciseSpec(Exercise.EJ2, "symbolic_ai.p1_ej2_ontology.main", False, True),
    ExerciseSpec(Exercise.EJ3, "symbolic_ai.p1_ej3_search.main", True, False),
)


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """A figure the report includes: the file an exercise writes and its name in the report.

    Parameters
    ----------
    exercise : Exercise
        The exercise whose ``--experiments`` run writes the figure.
    source : str
        The file name in ``<out-root>/ej<k>``.
    report_name : str
        The file name under ``figures/`` in the report's LaTeX source.
    """

    exercise: Exercise
    source: str
    report_name: str


#: The figure PDFs of the report. EJ1 writes ``fig_ej1_value_ordering`` (``p1_ej1_logic.plot``)
#: and EJ2 writes ``fig_ej2_ontology`` (``p1_ej2_ontology.plot``); the two panels of the deduced
#: taxonomy are the files ``fig_ej2_taxonomy_{a,b}`` of the EJ2 taxonomy figure.
REPORT_FIGURES: tuple[ReportFigure, ...] = (
    ReportFigure(Exercise.EJ1, "fig_ej1_value_ordering.pdf", "ej1_value_ordering.pdf"),
    ReportFigure(Exercise.EJ2, "fig_ej2_ontology.pdf", "ej2_ontology.pdf"),
    ReportFigure(Exercise.EJ2, "fig_ej2_taxonomy_a.pdf", "ej2_taxonomy_a.pdf"),
    ReportFigure(Exercise.EJ2, "fig_ej2_taxonomy_b.pdf", "ej2_taxonomy_b.pdf"),
)


@dataclass(frozen=True, slots=True)
class ExerciseOutcome:
    """What one exercise's run produced.

    Parameters
    ----------
    exercise : Exercise
        The exercise.
    argv : tuple[str, ...]
        The arguments its ``main`` was called with.
    exit_code : int
        Its exit code; ``1`` when it raised an exception.
    seconds : float
        Wall-clock time of the run.
    results_path : Path | None
        The ``results.json`` it wrote, or ``None`` when it failed.
    error : str | None
        The exception it raised, as text, or ``None``.
    """

    exercise: Exercise
    argv: tuple[str, ...]
    exit_code: int
    seconds: float
    results_path: Path | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the exercise exited with code 0."""
        return self.exit_code == 0


@dataclass(frozen=True, slots=True)
class FigureCopy:
    """One report figure copied, or missing.

    Parameters
    ----------
    figure : ReportFigure
        The figure.
    source : Path
        Where the exercise should have written it.
    destination : Path | None
        Where it was copied, or ``None`` when ``source`` does not exist.
    """

    figure: ReportFigure
    source: Path
    destination: Path | None


def out_dir(out_root: Path, exercise: Exercise) -> Path:
    """Return the output directory of ``exercise`` under ``out_root``.

    Parameters
    ----------
    out_root : Path
        The root of every exercise's outputs.
    exercise : Exercise
        The exercise.

    Returns
    -------
    Path
        ``out_root / "ej<k>"``.
    """
    return out_root / exercise.value


def exercise_argv(
    spec: ExerciseSpec,
    out_root: Path,
    *,
    workers: int | None = None,
    data_dir: Path | None = None,
    no_oracle: bool = False,
) -> tuple[str, ...]:
    """Build the arguments of one exercise's ``main``, forwarding only the options it accepts.

    Parameters
    ----------
    spec : ExerciseSpec
        The exercise.
    out_root : Path
        The root of every exercise's outputs.
    workers : int | None
        Processes for the exercises with ``--workers``; ``None`` keeps their default.
    data_dir : Path | None
        Database directory for every exercise; ``None`` keeps their default.
    no_oracle : bool
        Whether to pass ``--no-oracle`` to the exercises that have it.

    Returns
    -------
    tuple[str, ...]
        ``--experiments --out-dir <out_root>/ej<k>`` and the forwarded options.
    """
    argv = ["--experiments", "--out-dir", str(out_dir(out_root, spec.exercise))]
    if workers is not None and spec.accepts_workers:
        argv += ["--workers", str(workers)]
    if data_dir is not None:
        argv += ["--data-dir", str(data_dir)]
    if no_oracle and spec.accepts_no_oracle:
        argv.append("--no-oracle")
    return tuple(argv)


def _exercise_main(spec: ExerciseSpec) -> ExerciseMain:
    """Import the exercise's CLI module only when it runs (its imports cannot stop another)."""
    module = importlib.import_module(spec.module)
    return cast("ExerciseMain", module.main)


def run_exercise(spec: ExerciseSpec, argv: Sequence[str], out_root: Path) -> ExerciseOutcome:
    """Run one exercise's ``main`` in this process and record how it ended.

    An exception, or a ``SystemExit`` from the exercise's argument parser, is logged and turned
    into a non-zero exit code, so the remaining exercises still run.

    Parameters
    ----------
    spec : ExerciseSpec
        The exercise.
    argv : Sequence[str]
        The arguments of its ``main`` (see :func:`exercise_argv`).
    out_root : Path
        The root of every exercise's outputs.

    Returns
    -------
    ExerciseOutcome
        The exit code, the wall time and the results file written.
    """
    start = time.perf_counter()
    error: str | None = None
    try:
        exit_code = _exercise_main(spec)(list(argv))
    except SystemExit as exc:
        # argparse inside the exercise exits with 2; ``sys.exit()`` without a code means success.
        exit_code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        error = f"SystemExit({exc.code!r})" if exit_code else None
    except Exception as exc:
        # Any failure of one exercise is reported and the next one still runs.
        logger.exception("%s failed", spec.exercise.value)
        exit_code, error = 1, f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    results_path = out_dir(out_root, spec.exercise) / _RESULTS_FILE
    written = results_path if exit_code == 0 and results_path.is_file() else None
    return ExerciseOutcome(spec.exercise, tuple(argv), exit_code, seconds, written, error)


def copy_report_figures(
    exercises: Sequence[Exercise], out_root: Path, destination: Path
) -> tuple[FigureCopy, ...]:
    """Copy the report's figures written by ``exercises`` to ``destination`` under report names.

    Parameters
    ----------
    exercises : Sequence[Exercise]
        The exercises whose figures to copy (those that ran successfully).
    out_root : Path
        The root of every exercise's outputs.
    destination : Path
        The directory to copy into (created if needed), e.g. the report's ``figures/``.

    Returns
    -------
    tuple[FigureCopy, ...]
        One entry per figure of :data:`REPORT_FIGURES` of those exercises, in that order; a
        figure the exercise did not write has ``destination=None``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copies = []
    for figure in REPORT_FIGURES:
        if figure.exercise not in exercises:
            continue
        source = out_dir(out_root, figure.exercise) / figure.source
        if not source.is_file():
            logger.error("report figure %s missing: %s was not written", figure.report_name, source)
            copies.append(FigureCopy(figure, source, None))
            continue
        target = destination / figure.report_name
        shutil.copyfile(source, target)
        copies.append(FigureCopy(figure, source, target))
    return tuple(copies)


def _positive_int(text: str) -> int:
    """Parse a positive integer for ``--workers``."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``[--only {ej1,ej2,ej3} ...] [--workers N] [--out-root DIR]
        [--data-dir PATH] [--no-oracle] [--report-figures DIR]``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_ai.reproduce",
        description="Run the experiments of every exercise (EJ1, EJ2, EJ3) and write their "
        "results and figures, as each exercise's --experiments does.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(exercise.value for exercise in Exercise),
        metavar="{ej1,ej2,ej3}",
        help="Run only these exercises (always in the order EJ1, EJ2, EJ3). Default: all.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="Processes for EJ1 and EJ3 (EJ2 has no --workers). Default: each exercise's "
        "default, 1. The results do not depend on it.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_DEFAULT_OUT_ROOT,
        help=f"Root of the outputs; exercise k writes to <out-root>/ej<k> "
        f"(default: {_DEFAULT_OUT_ROOT}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Database directory for every exercise (default: $SYMAI_DATA_DIR or "
        "<repository root>/data).",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="EJ2 only: skip the HermiT oracle (needs Java >= 11); its oracle fields are null.",
    )
    parser.add_argument(
        "--report-figures",
        type=Path,
        default=None,
        metavar="DIR",
        help="Also copy the report's figure PDFs into DIR under the names the report uses.",
    )
    return parser


def _print_summary(
    outcomes: Sequence[ExerciseOutcome], figures: Sequence[FigureCopy] | None
) -> None:
    """Print one line per exercise and, if copied, one line per report figure."""
    print()
    print("=== summary ===")
    print(f"{'exercise':<9} {'exit':>4} {'time (s)':>9}  results")
    for outcome in outcomes:
        where = "-" if outcome.results_path is None else str(outcome.results_path)
        print(
            f"{outcome.exercise.value:<9} {outcome.exit_code:>4} {outcome.seconds:>9.1f}  {where}"
        )
        if outcome.error is not None:
            print(f"{'':<9} error: {outcome.error}")
    print(f"{'total':<9} {'':>4} {sum(o.seconds for o in outcomes):>9.1f}")
    for copy in figures or ():
        state = f"-> {copy.destination}" if copy.destination is not None else "MISSING"
        print(f"figure {copy.figure.report_name}: {copy.source} {state}")
    failed = [o.exercise.value for o in outcomes if not o.ok]
    missing = [c.figure.report_name for c in figures or () if c.destination is None]
    if failed:
        print(f"FAILED: {', '.join(failed)} (the other exercises ran to the end)")
    if missing:
        print(f"MISSING figures: {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected exercises' experiments, copy the figures if asked, print a summary.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, excluding the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` if every exercise exited with 0 (and every requested figure was copied), ``1``
        otherwise.
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    selected = set(args.only) if args.only else {exercise.value for exercise in Exercise}
    outcomes = []
    for spec in EXERCISES:
        if spec.exercise.value not in selected:
            continue
        exercise_args = exercise_argv(
            spec,
            args.out_root,
            workers=args.workers,
            data_dir=args.data_dir,
            no_oracle=args.no_oracle,
        )
        print(f"=== {spec.exercise.value}: python -m {spec.module} {' '.join(exercise_args)} ===")
        outcomes.append(run_exercise(spec, exercise_args, args.out_root))

    figures = None
    if args.report_figures is not None:
        succeeded = [o.exercise for o in outcomes if o.ok]
        figures = copy_report_figures(succeeded, args.out_root, args.report_figures)
    _print_summary(outcomes, figures)
    all_ok = all(o.ok for o in outcomes) and all(c.destination is not None for c in figures or ())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
