"""Tests of the EJ4 command line (``main.py``): modes, exit codes, deterministic results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary, load_pharmacokinetics
from symbolic_ai.p2_ej4_transition import experiments, instances
from symbolic_ai.p2_ej4_transition import main as cli
from symbolic_ai.p2_ej4_transition.experiments import (
    EXTENDED_CASE_ID,
    AlgorithmRun,
    entries_in_category,
)
from symbolic_ai.p2_ej4_transition.instances import Category, ProblemEntry
from symbolic_ai.p2_ej4_transition.problem import TransitionProblem
from symbolic_ai.p2_ej4_transition.solve import SolverConfig

CASE_STUDY_ID = "DEP|LIVER|+cond:PAIN"


# --- arguments ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [[], ["--list", "--worked-example"], ["--worked-example", "--category", "trivial"]],
    ids=["no mode", "two modes", "category without list"],
)
def test_invalid_arguments_exit_with_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2


def test_refer_is_not_a_listable_category() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--list", "--category", "refer"])


# --- modes on the real data ---------------------------------------------------------------------


def _real_entries() -> tuple[ProblemEntry, ...]:
    """Every problem of the real data (the counts are not hard-coded: they change with D23)."""
    version = "1.3.0"
    return instances.enumerate_problems(
        load_formulary(version=version),
        load_drug_costs(version=version),
        load_pharmacokinetics(version=version),
    )


@pytest.mark.integration
def test_list_prints_every_problem(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    solvable = [e for e in _real_entries() if e.problem is not None]
    assert len(lines) == len(solvable)
    assert sum(int(line.split("\t")[2]) for line in lines) == sum(
        len(e.occurrences) for e in solvable
    )


@pytest.mark.integration
def test_list_one_category(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--list", "--category", "knock_on"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(entries_in_category(_real_entries(), Category.KNOCK_ON))
    assert all(line.split("\t")[1] == "knock_on" for line in lines)
    assert any(line.startswith(f"{CASE_STUDY_ID}\t") for line in lines)


@pytest.mark.integration
def test_problem_prints_h_star_and_schedule(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--problem", CASE_STUDY_ID]) == 0
    out = capsys.readouterr().out
    assert "H* = 12" in out
    last_day = out.splitlines()[-1].split()
    assert last_day == ["12", "mirtazapine", "ibuprofen"]


@pytest.mark.integration
def test_unknown_problem_exits_with_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--problem", "no such problem"]) == 1
    assert "no problem with id" in capsys.readouterr().err


@pytest.mark.integration
def test_worked_example_prints_the_domains_and_the_schedule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--worked-example"]) == 0
    out = capsys.readouterr().out
    assert "H = 11: AC-3 emptied ('DEP', 2); removed 116" in out
    assert "DEP day  2: {} (removed 9)   <- emptied by AC-3" in out
    assert "H = 12: AC-3 consistent; removed 133" in out
    assert "  PAIN day 12: {ibuprofen, tramadol} (removed 0)" in out
    assert "H* = 12; schedule (checker: valid):" in out
    assert "contraindication: LIVER -> paracetamol" in out


# --- --experiments on a small population (fast) -------------------------------------------------


@pytest.fixture
def small_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink ``--experiments``: a few problems, a tiny R3 budget, a short min-conflicts run."""
    real_enumerate = instances.enumerate_problems
    keep = {CASE_STUDY_ID, EXTENDED_CASE_ID}

    def few_problems(*args: object) -> tuple[ProblemEntry, ...]:
        entries = real_enumerate(*args)  # type: ignore[arg-type]
        return tuple(e for e in entries if e.problem is not None and e.problem.problem_id in keep)

    tiny = tuple(
        (key, SolverConfig(c.inference, c.variable_order, c.value_order, c.ac3_preprocess, 60, 50))
        for key, c in experiments.R3_CONFIGS
    )

    def small_table(
        problems: Sequence[tuple[TransitionProblem, int]],
    ) -> tuple[AlgorithmRun, ...]:
        return experiments.run_algorithm_table(problems, tiny)

    monkeypatch.setattr(cli, "enumerate_problems", few_problems)
    monkeypatch.setattr(cli, "run_algorithm_table", small_table)
    monkeypatch.setattr(cli, "MIN_CONFLICTS_STEPS", 100)


def _keys(content: object) -> set[str]:
    """Every key of a JSON document, at any depth."""
    if isinstance(content, dict):
        return set(content) | {k for value in content.values() for k in _keys(value)}
    if isinstance(content, list):
        return {k for item in content for k in _keys(item)}
    return set()


@pytest.mark.integration
@pytest.mark.usefixtures("small_run")
def test_experiments_write_identical_results_twice(tmp_path: Path) -> None:
    for name in ("a", "b"):
        assert cli.main(["--experiments", "--out-dir", str(tmp_path / name)]) == 0
    first = (tmp_path / "a" / "results.json").read_bytes()
    assert first == (tmp_path / "b" / "results.json").read_bytes()
    results = json.loads(first)
    assert results["schema"] == cli.RESULTS_SCHEMA
    assert results["provenance"]["data_version"] == "1.3.0"
    measured = {k for k in _keys(results) if k == "seconds" or k.endswith("_s")}
    assert measured == {"timeout_s"}  # a configuration value, not a measured time
    assert sorted(results["outputs"]) == sorted(
        [
            "fig_ej4_gantt.pdf",
            "fig_ej4_gantt.png",
            "fig_ej4_gantt_extended.pdf",
            "fig_ej4_gantt_extended.png",
            "tab_ej4_domains_h11.tex",
            "tab_ej4_domains_h12.tex",
            "tab_ej4_r2.tex",
            "tab_ej4_r3.tex",
        ]
    )
    for name in results["outputs"]:
        assert (tmp_path / "a" / name).is_file()
    timings = json.loads((tmp_path / "a" / "timings.json").read_text())
    assert {"r1_s", "r2_s", "r3_s", "total_s"} <= set(timings)


@pytest.mark.integration
@pytest.mark.usefixtures("small_run")
def test_experiments_record_the_confirmed_r1_numbers(tmp_path: Path) -> None:
    assert cli.main(["--experiments", "--out-dir", str(tmp_path)]) == 0
    r1 = json.loads((tmp_path / "results.json").read_text())["r1"]
    h11, h12 = r1["domains"]
    assert (r1["h_star"], h11["emptied"], h11["removed_by_ac3"]) == (12, ["DEP", 2], 116)
    assert (h12["run"]["assignments"], h12["run"]["checks"], h12["removed_by_ac3"]) == (
        26,
        2264,
        133,
    )
    assert r1["valid"] is True


@pytest.mark.integration
@pytest.mark.usefixtures("small_run")
def test_no_oracle_leaves_the_comparison_null(tmp_path: Path) -> None:
    assert cli.main(["--experiments", "--no-oracle", "--out-dir", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["r2"]["oracle"] is None
    assert results["provenance"]["command"].endswith("--no-oracle")
    assert results["r2"]["problems"] == 2


# --- the real run (slow) ------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_real_experiments(tmp_path: Path) -> None:
    assert cli.main(["--experiments", "--out-dir", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text())
    oracle = results["r2"]["oracle"]
    assert oracle["agree"] == oracle["compared"] > 0 and oracle["disagreements"] == []
    assert results["r2"]["checker"]["invalid"] == []
    rows = {(r["instance_id"], r["configuration"], r["horizon"]): r for r in results["r3"]["runs"]}
    assert rows[EXTENDED_CASE_ID, "bt", 11]["status"] == "budget"
    assert rows[EXTENDED_CASE_ID, "main", 11]["assignments"] == 0
