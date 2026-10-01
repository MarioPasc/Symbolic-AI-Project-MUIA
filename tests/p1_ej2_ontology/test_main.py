"""The EJ2 command line: its modes, the results file and its schema check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError
from symbolic_ai.p1_ej2_ontology.main import RESULTS_SCHEMA, main

pytestmark = pytest.mark.integration


def test_classify_prints_the_categories(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--classify", "tramadol", "--no-oracle"]) == 0
    out = capsys.readouterr().out
    assert "told categories opioids, serotonergic_drugs" in out
    assert "defined categories: candidate_PAIN, contraindicated_EPI, serotonergic_opioids" in out
    assert "a new member of opioids would receive 4 formulary rows" in out


def test_classify_unknown_drug_exits_with_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--classify", "aspirin"]) == 2
    assert "aspirin" in capsys.readouterr().err


def test_taxonomy_prints_124_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--taxonomy"]) == 0
    assert "124 proper subsumption pairs among 44 categories" in capsys.readouterr().out


def test_formulary_reports_the_extra_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--formulary"]) == 0
    out = capsys.readouterr().out
    for pair in ("apixaban / sertraline", "apixaban / warfarin", "bisoprolol / propranolol"):
        assert f"only derived: {pair}" in out


def test_plot_rejects_a_file_of_another_schema(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"schema": "symai.ej1.results/3"}), encoding="utf-8")
    with pytest.raises(ResultsFormatError):
        main(["--plot", str(path), "--out-dir", str(tmp_path)])


@pytest.mark.slow
def test_experiments_without_oracle_write_a_reproducible_results_file(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert main(["--experiments", "--no-oracle", "--out-dir", str(first)]) == 0
    assert main(["--experiments", "--no-oracle", "--out-dir", str(second)]) == 0
    text = (first / "results.json").read_text(encoding="utf-8")
    assert text == (second / "results.json").read_text(encoding="utf-8")
    results = json.loads(text)
    assert results["schema"] == RESULTS_SCHEMA
    assert set(results) == {
        "schema",
        "provenance",
        "ontology",
        "fixed_point",
        "formulary_derivation",
        "taxonomy",
        "inheritance",
        "consistency",
    }
    assert results["provenance"]["oracle"] is None
    assert results["taxonomy"]["oracle_classification"] is None
    for stem in ("fig_ej2_ontology", "fig_ej2_ontology_inferred"):
        for suffix in (".dot", ".pdf", ".png"):
            assert (first / f"{stem}{suffix}").exists()
