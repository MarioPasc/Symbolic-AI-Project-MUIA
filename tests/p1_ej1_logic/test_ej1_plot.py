"""Tests for symbolic_ai.p1_ej1_logic.plot and the CLI modes --plot and --experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from symbolic_ai.p1_ej1_logic.errors import ResultsFormatError
from symbolic_ai.p1_ej1_logic.main import RESULTS_SCHEMA, main
from symbolic_ai.p1_ej1_logic.plot import VALUE_ORDERING_FIGURE, plot_value_ordering

_SECTION: dict[str, object] = {
    "orderings": [
        {"ordering": "false_first", "excess_histogram": {"0": 10}},
        {"ordering": "true_first", "excess_histogram": {"0": 6, "1": 3, "2": 1}},
    ],
    "by_n_conditions": [
        {
            "n_conditions": k,
            "mean_minimum": float(k),
            "mean_false_first": float(k),
            "mean_true_first": k + 0.2 * k,
        }
        for k in (1, 2)
    ],
}


def test_figure_is_written_and_reproducible(tmp_path: Path) -> None:
    first = plot_value_ordering(_SECTION, tmp_path / "a")
    second = plot_value_ordering(_SECTION, tmp_path / "b")
    assert [p.suffix for p in first] == [".pdf", ".png"]
    for path_a, path_b in zip(first, second, strict=True):
        assert path_a.read_bytes() == path_b.read_bytes()


@pytest.mark.parametrize("missing", ["orderings", "by_n_conditions"])
def test_missing_section_field_raises(tmp_path: Path, missing: str) -> None:
    section = {k: v for k, v in _SECTION.items() if k != missing}
    with pytest.raises(ResultsFormatError, match=missing):
        plot_value_ordering(section, tmp_path / "fig")


def test_cli_plot_mode_reads_a_saved_file(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"schema": RESULTS_SCHEMA, "value_ordering": _SECTION}))
    assert main(["--plot", str(results), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / f"{VALUE_ORDERING_FIGURE}.pdf").is_file()


def test_cli_plot_mode_rejects_another_schema(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"schema": "other/1", "value_ordering": _SECTION}))
    with pytest.raises(ResultsFormatError, match="results file"):
        main(["--plot", str(results), "--out-dir", str(tmp_path)])


def test_cli_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--experiments"])


@pytest.mark.slow
@pytest.mark.integration
def test_cli_experiments_reproduce_the_report(tmp_path: Path) -> None:
    """The numbers the report quotes, regenerated end to end from the real database."""
    workers = str(max(1, min(os.cpu_count() or 1, 16)))
    assert main(["--experiments", "--workers", workers, "--out-dir", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["schema"] == RESULTS_SCHEMA
    verification = results["verification"]
    assert verification["n_records"] == 45_927
    assert verification["disagreements"] == []
    assert [s["action"] for s in results["scenarios"]] == [
        "prescribe",
        "request_test",
        "prescribe",
        "prescribe",
        "refer",
    ]
    assert len(results["untreatable_patterns"]["patterns"]) == 3
    assert (tmp_path / f"{VALUE_ORDERING_FIGURE}.pdf").is_file()
