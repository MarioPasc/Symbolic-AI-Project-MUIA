"""The one-command reproduction: dispatch, forwarding of options, failures, report figures."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from symbolic_ai import reproduce
from symbolic_ai.reproduce import EXERCISES, REPORT_FIGURES, Exercise, main

_MODULES = {spec.exercise.value: spec.module for spec in EXERCISES}

Calls = list[tuple[str, list[str]]]


def _option(argv: Sequence[str], name: str) -> str | None:
    """Return the value following ``name`` in ``argv``, or ``None`` if absent."""
    return argv[list(argv).index(name) + 1] if name in argv else None


def _fake_main(name: str, calls: Calls, exit_code: int) -> Callable[[Sequence[str] | None], int]:
    """A stand-in exercise ``main``: record the call, write results and the exercise's figures."""

    def fake(argv: Sequence[str] | None) -> int:
        assert argv is not None
        calls.append((name, list(argv)))
        if exit_code != 0:
            return exit_code
        out = Path(str(_option(argv, "--out-dir")))
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps({"schema": f"fake.{name}"}))
        for figure in REPORT_FIGURES:
            if figure.exercise.value == name:
                (out / figure.source).write_bytes(f"%PDF {figure.source}".encode())
        return 0

    return fake


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """Replace the three exercise mains by recorders that succeed."""
    recorded: Calls = []
    for name, module in _MODULES.items():
        monkeypatch.setattr(f"{module}.main", _fake_main(name, recorded, 0))
    return recorded


# --- dispatch -----------------------------------------------------------------------------------


def test_runs_every_exercise_in_order_with_experiments(calls: Calls, tmp_path: Path) -> None:
    assert main(["--out-root", str(tmp_path)]) == 0
    assert [name for name, _ in calls] == ["ej1", "ej2", "ej3"]
    for name, argv in calls:
        assert argv[:3] == ["--experiments", "--out-dir", str(tmp_path / name)]
        assert (tmp_path / name / "results.json").is_file()


def test_default_out_root_is_outputs(
    calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
    assert [_option(argv, "--out-dir") for _, argv in calls] == [
        str(Path("outputs") / name) for name in ("ej1", "ej2", "ej3")
    ]


@pytest.mark.parametrize(
    ("only", "expected"),
    [
        (["ej3"], ["ej3"]),
        (["ej3", "ej1"], ["ej1", "ej3"]),
        (["ej2", "ej2"], ["ej2"]),
    ],
)
def test_only_selects_and_keeps_the_report_order(
    calls: Calls, tmp_path: Path, only: list[str], expected: list[str]
) -> None:
    assert main(["--only", *only, "--out-root", str(tmp_path)]) == 0
    assert [name for name, _ in calls] == expected


def test_unknown_only_value_is_an_argparse_error(calls: Calls) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--only", "ej4"])
    assert excinfo.value.code == 2
    assert calls == []


@pytest.mark.parametrize("workers", ["0", "-2", "two"])
def test_invalid_workers_is_an_argparse_error(calls: Calls, workers: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--workers", workers])
    assert excinfo.value.code == 2
    assert calls == []


# --- forwarding ---------------------------------------------------------------------------------


def test_workers_goes_to_ej1_and_ej3_only(calls: Calls, tmp_path: Path) -> None:
    assert main(["--workers", "7", "--out-root", str(tmp_path)]) == 0
    assert {name: _option(argv, "--workers") for name, argv in calls} == {
        "ej1": "7",
        "ej2": None,
        "ej3": "7",
    }


def test_no_workers_forwarded_by_default(calls: Calls, tmp_path: Path) -> None:
    assert main(["--out-root", str(tmp_path)]) == 0
    assert all("--workers" not in argv for _, argv in calls)


def test_data_dir_goes_to_every_exercise(calls: Calls, tmp_path: Path) -> None:
    data = tmp_path / "db"
    assert main(["--data-dir", str(data), "--out-root", str(tmp_path)]) == 0
    assert [_option(argv, "--data-dir") for _, argv in calls] == [str(data)] * 3


def test_no_oracle_goes_to_ej2_only(calls: Calls, tmp_path: Path) -> None:
    assert main(["--no-oracle", "--out-root", str(tmp_path)]) == 0
    assert {name: "--no-oracle" in argv for name, argv in calls} == {
        "ej1": False,
        "ej2": True,
        "ej3": False,
    }


def test_oracle_on_by_default(calls: Calls, tmp_path: Path) -> None:
    assert main(["--out-root", str(tmp_path)]) == 0
    assert all("--no-oracle" not in argv for _, argv in calls)


# --- failures -----------------------------------------------------------------------------------


def test_a_failing_exercise_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded: Calls = []
    for name, module in _MODULES.items():
        monkeypatch.setattr(f"{module}.main", _fake_main(name, recorded, 3 if name == "ej2" else 0))
    assert main(["--out-root", str(tmp_path)]) == 1
    assert [name for name, _ in recorded] == ["ej1", "ej2", "ej3"]
    out = capsys.readouterr().out
    assert "FAILED: ej2" in out
    assert (tmp_path / "ej3" / "results.json").is_file()


def test_an_exception_counts_as_a_failure(
    calls: Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(argv: Sequence[str] | None) -> int:
        raise RuntimeError("no Java")

    monkeypatch.setattr(f"{_MODULES['ej2']}.main", boom)
    spec = EXERCISES[1]
    outcome = reproduce.run_exercise(spec, ["--experiments"], tmp_path)
    assert (outcome.exit_code, outcome.results_path) == (1, None)
    assert outcome.error == "RuntimeError: no Java"
    assert main(["--out-root", str(tmp_path)]) == 1
    assert [name for name, _ in calls] == ["ej1", "ej3"]


@pytest.mark.parametrize(("code", "expected"), [(2, 2), (None, 0), ("bad", 1)])
def test_system_exit_inside_an_exercise_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: object, expected: int
) -> None:
    def exits(argv: Sequence[str] | None) -> int:
        raise SystemExit(code)

    monkeypatch.setattr(f"{_MODULES['ej1']}.main", exits)
    outcome = reproduce.run_exercise(EXERCISES[0], ["--experiments"], tmp_path)
    assert outcome.exit_code == expected
    assert outcome.ok is (expected == 0)


def test_summary_lists_each_exercise_and_its_results(
    calls: Calls, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--only", "ej1", "ej3", "--out-root", str(tmp_path)]) == 0
    summary = capsys.readouterr().out.split("=== summary ===")[1]
    assert str(tmp_path / "ej1" / "results.json") in summary
    assert str(tmp_path / "ej3" / "results.json") in summary
    assert "ej2" not in summary
    assert "FAILED" not in summary


# --- report figures -----------------------------------------------------------------------------


def test_report_figures_are_copied_under_report_names(calls: Calls, tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    assert main(["--out-root", str(tmp_path / "out"), "--report-figures", str(figures)]) == 0
    assert sorted(p.name for p in figures.iterdir()) == [
        "ej1_value_ordering.pdf",
        "ej2_ontology.pdf",
        "ej2_taxonomy_a.pdf",
        "ej2_taxonomy_b.pdf",
    ]
    assert (figures / "ej1_value_ordering.pdf").read_bytes() == b"%PDF fig_ej1_value_ordering.pdf"


def test_report_figures_only_for_the_exercises_run(calls: Calls, tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    assert (
        main(["--only", "ej1", "--out-root", str(tmp_path), "--report-figures", str(figures)]) == 0
    )
    assert [p.name for p in figures.iterdir()] == ["ej1_value_ordering.pdf"]


def test_a_missing_report_figure_is_reported(calls: Calls, tmp_path: Path) -> None:
    out = tmp_path / "out"
    copies = reproduce.copy_report_figures([Exercise.EJ2], out, tmp_path / "figures")
    assert [c.destination for c in copies] == [None, None, None]

    assert main(["--only", "ej2", "--out-root", str(out)]) == 0
    (out / "ej2" / "fig_ej2_taxonomy_b.pdf").unlink()
    copies = reproduce.copy_report_figures([Exercise.EJ2], out, tmp_path / "again")
    assert [c.figure.report_name for c in copies if c.destination is None] == ["ej2_taxonomy_b.pdf"]


def test_missing_figure_makes_main_exit_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_figures(argv: Sequence[str] | None) -> int:
        assert argv is not None
        out = Path(str(_option(argv, "--out-dir")))
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text("{}")
        return 0

    monkeypatch.setattr(f"{_MODULES['ej1']}.main", no_figures)
    code = main(["--only", "ej1", "--out-root", str(tmp_path), "--report-figures", str(tmp_path)])
    assert code == 1
    assert "MISSING figures: ej1_value_ordering.pdf" in capsys.readouterr().out


def test_report_figure_names_match_the_exercise_plots() -> None:
    """The EJ1 and EJ2 sources are the stems their plot modules write."""
    from symbolic_ai.p1_ej1_logic.plot import VALUE_ORDERING_FIGURE
    from symbolic_ai.p1_ej2_ontology.plot import ONTOLOGY_FIGURE

    sources = {figure.source for figure in REPORT_FIGURES}
    assert f"{VALUE_ORDERING_FIGURE}.pdf" in sources
    assert f"{ONTOLOGY_FIGURE}.pdf" in sources


# --- the real run -------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_real_ej3_run_writes_its_results(tmp_path: Path) -> None:
    from symbolic_ai.p1_ej3_search.main import RESULTS_SCHEMA

    assert main(["--only", "ej3", "--workers", "4", "--out-root", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "ej3" / "results.json").read_text())
    assert results["schema"] == RESULTS_SCHEMA
