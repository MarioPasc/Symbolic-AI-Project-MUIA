"""The EJ3 command line: its modes on the real database, the results file and determinism."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary
from symbolic_ai.dataloader.models import DrugCost, Encounter, Formulary
from symbolic_ai.p1_ej1_logic import EJ1_FORMULARY_VERSION
from symbolic_ai.p1_ej3_search import EJ3_DATA_VERSION
from symbolic_ai.p1_ej3_search import main as main_module
from symbolic_ai.p1_ej3_search.main import RESULTS_SCHEMA, main

_SCRIPT = Path(__file__).resolve().parent / "_determinism_subprocess.py"


# --- the real database --------------------------------------------------------------------------


@pytest.mark.integration
def test_database_costs_are_the_transcribed_ones(costs: Mapping[str, int]) -> None:
    priced = load_drug_costs(version=EJ3_DATA_VERSION)
    assert {d: c.monthly_cost_cents for d, c in priced.items()} == costs


@pytest.mark.integration
def test_data_1_2_0_has_the_rules_of_the_formulary_of_ej1(formulary: Formulary) -> None:
    """EJ3 loads 1.2.0 and EJ1 is pinned to 1.0.0: only the version label may differ."""
    current = load_formulary(version=EJ3_DATA_VERSION)
    pinned = load_formulary(version=EJ1_FORMULARY_VERSION)
    assert current.version == "1.2.0"
    for name in ("conditions", "drugs", "risk_factors", "candidates", "interactions"):
        assert getattr(current, name) == getattr(pinned, name) == getattr(formulary, name)
    for name in ("contraindications", "coprescriptions", "drug_classes", "class_members"):
        assert getattr(current, name) == getattr(pinned, name)


@pytest.mark.integration
def test_encounter_prints_both_regimens_and_the_saving(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--encounter", "E001"]) == 0
    out = capsys.readouterr().out
    assert (
        "agent 1 regimen (5 drugs, 53.30 EUR/month): "
        "amlodipine, linagliptin, mirtazapine, tramadol, warfarin"
    ) in out
    assert (
        "astar regimen (5 drugs, 50.10 EUR/month): "
        "enalapril, linagliptin, mirtazapine, paracetamol, warfarin"
    ) in out
    assert "saving: 3.20 EUR/month" in out
    assert "search: 5 node(s) expanded, 12 generated, 6 goal test(s)" in out


@pytest.mark.integration
def test_all_writes_one_decision_per_encounter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "decisions.json"
    assert main(["--all", "--json", str(path)]) == 0
    out = capsys.readouterr().out
    assert "=== E002 (P002, visit 1) ===\naction: request_test\ntests: PREG\n" in out
    assert "=== E005 (P003, visit 2) ===\naction: refer\n" in out
    decisions = json.loads(path.read_text(encoding="utf-8"))
    assert list(decisions) == ["E001", "E002", "E003", "E004", "E005"]
    assert decisions["E001"]["cost_cents"] == 5010
    assert decisions["E001"]["agent1_cost_cents"] == 5330
    assert decisions["E001"]["saving_cents"] == 320
    assert decisions["E001"]["search"] == {
        "algorithm": "astar",
        "chosen": ["warfarin", "mirtazapine", "enalapril", "paracetamol", "linagliptin"],
        "expanded": 5,
        "generated": 12,
        "goal_tests": 6,
    }
    assert decisions["E002"] == {
        "action": "request_test",
        "agent1_cost_cents": None,
        "agent1_regimen": [],
        "cost_cents": None,
        "regimen": [],
        "saving_cents": None,
        "search": None,
        "tests": ["PREG"],
    }


@pytest.mark.integration
def test_greedy_is_reported_under_its_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--encounter", "E003", "--algorithm", "greedy"]) == 0
    out = capsys.readouterr().out
    assert "greedy regimen (2 drugs, 55.80 EUR/month): amlodipine, apixaban" in out
    assert "saving: -51.40 EUR/month" in out


@pytest.mark.integration
def test_no_prune_prescribes_the_same(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--encounter", "E004", "--no-prune"]) == 0
    out = capsys.readouterr().out
    assert "astar regimen (3 drugs, 11.10 EUR/month): ibuprofen, mirtazapine, omeprazole" in out
    assert "search: 2 node(s) expanded, 3 generated" in out


@pytest.mark.integration
def test_unknown_encounter_exits_with_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--encounter", "E999"]) == 2
    assert "E999" in capsys.readouterr().err


def test_an_unknown_algorithm_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main(["--all", "--algorithm", "minimax"])


# --- the experiments mode, on the mini formulary ------------------------------------------------


def test_experiments_write_a_reproducible_results_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    elena_formulary: Formulary,
    elena_costs: Mapping[str, int],
    elena_encounter: Encounter,
) -> None:
    priced = {d: DrugCost(d, cost) for d, cost in elena_costs.items()}
    monkeypatch.setattr(main_module, "load_formulary", lambda data_dir, version: elena_formulary)
    monkeypatch.setattr(main_module, "load_drug_costs", lambda data_dir, version: priced)
    monkeypatch.setattr(main_module, "load_encounters", lambda data_dir: {"ELENA": elena_encounter})

    first, second = tmp_path / "a", tmp_path / "b"
    assert main(["--experiments", "--out-dir", str(first)]) == 0
    assert main(["--experiments", "--out-dir", str(second), "--workers", "2"]) == 0
    text = (first / "results.json").read_text(encoding="utf-8")
    assert text == (second / "results.json").read_text(encoding="utf-8")

    results = json.loads(text)
    assert results["schema"] == RESULTS_SCHEMA
    assert set(results) == {
        "schema",
        "provenance",
        "costs_cents_per_month",
        "oracle",
        "scenarios",
        "search_comparison",
    }
    assert results["costs_cents_per_month"] == dict(elena_costs)
    assert results["oracle"]["regimens"] == 2**5
    assert results["scenarios"][0]["regimen"] == ["paracetamol", "warfarin"]
    assert results["search_comparison"]["n_satisfiable"] == 24
    assert len(results["search_comparison"]["instances"]) == 28
    out = capsys.readouterr().out
    assert "P7 astar: cheapest in 24/24" in out
    assert "P7 greedy: cheapest in 8/24" in out


# --- determinism --------------------------------------------------------------------------------


def _run_with_seed(seed: str, algorithm: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), algorithm],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout


@pytest.mark.parametrize("algorithm", ["astar", "greedy"])
def test_same_output_across_hash_seeds(algorithm: str) -> None:
    """States are sets of strings, whose iteration order depends on the hash seed."""
    output_seed_0 = _run_with_seed("0", algorithm)
    output_seed_1 = _run_with_seed("1", algorithm)

    assert output_seed_0 == output_seed_1
    assert "=== E001" in output_seed_0
    assert f"{algorithm} regimen (5 drugs" in output_seed_0
