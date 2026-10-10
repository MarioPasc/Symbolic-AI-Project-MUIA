"""``load_drug_costs`` on the real data 1.2.0 and one broken fixture per rule of the cost table."""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import (
    DatabaseValidationError,
    DrugCost,
    load_drug_costs,
    load_formulary,
    validate_database,
)
from symbolic_ai.dataloader.validate import main as validate_main

pytestmark = pytest.mark.integration


def _costs_path(data_dir: Path) -> Path:
    return data_dir / "formulary" / "drug_costs.csv"


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture text not found in {path}: {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _problems(data_dir: Path) -> tuple[str, ...]:
    with pytest.raises(DatabaseValidationError) as excinfo:
        validate_database(data_dir=data_dir)
    return excinfo.value.problems


# --- the real data ------------------------------------------------------------------------------


def test_every_drug_has_one_cost(real_data_dir: Path) -> None:
    costs = load_drug_costs(data_dir=real_data_dir)
    formulary = load_formulary(data_dir=real_data_dir)
    assert tuple(costs) == formulary.drug_ids
    assert len(costs) == 18
    assert all(cost.drug_id == drug_id for drug_id, cost in costs.items())


def test_reference_costs(real_data_dir: Path) -> None:
    costs = load_drug_costs(data_dir=real_data_dir)
    assert costs["warfarin"] == DrugCost("warfarin", 260)
    assert costs["apixaban"].monthly_cost_cents == 5400
    assert min(c.monthly_cost_cents for c in costs.values()) == 140
    assert sum(c.monthly_cost_cents for c in costs.values()) == 14_030


def test_costs_load_at_the_version_that_added_them(real_data_dir: Path) -> None:
    assert load_drug_costs(data_dir=real_data_dir, version="1.2.0") == load_drug_costs(
        data_dir=real_data_dir
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
def test_no_cost_exists_before_version_1_2_0(real_data_dir: Path, version: str) -> None:
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_drug_costs(data_dir=real_data_dir, version=version)
    assert len(excinfo.value.problems) == 18  # every drug lacks a cost
    assert excinfo.value.problems[0] == "drug_costs: drug 'amlodipine' has no cost"


def test_formulary_1_0_0_is_unchanged_by_data_1_2_0(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir, version="1.0.0")
    assert formulary.version == "1.0.0"
    assert len(formulary.drugs) == 18
    assert sum(len(drugs) for drugs in formulary.candidates.values()) == 18
    assert len(formulary.interactions) == 7
    assert len(formulary.contraindications) == 11


def test_validate_cli_counts_the_costs(
    real_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert validate_main(["--data-dir", str(real_data_dir)]) == 0
    assert "drug_costs: 18" in capsys.readouterr().out


# --- one broken fixture per rule ----------------------------------------------------------------


def test_negative_cost_is_reported(sandbox_data_dir: Path) -> None:
    _replace(_costs_path(sandbox_data_dir), "warfarin,260,,1.2.0", "warfarin,-260,,1.2.0")
    problems = _problems(sandbox_data_dir)
    assert problems == ("drug_costs[drug_id='warfarin']: monthly_cost_cents = -260 is negative",)


def test_non_integer_cost_is_reported(sandbox_data_dir: Path) -> None:
    _replace(_costs_path(sandbox_data_dir), "warfarin,260,,1.2.0", "warfarin,2.60,,1.2.0")
    problems = _problems(sandbox_data_dir)
    assert any("monthly_cost_cents" in p and "not a valid integer" in p for p in problems)


def test_empty_cost_is_reported(sandbox_data_dir: Path) -> None:
    _replace(_costs_path(sandbox_data_dir), "warfarin,260,,1.2.0", "warfarin,,,1.2.0")
    problems = _problems(sandbox_data_dir)
    assert any("'monthly_cost_cents' is required but empty" in p for p in problems)


def test_cost_of_an_unknown_drug_is_reported(sandbox_data_dir: Path) -> None:
    _replace(_costs_path(sandbox_data_dir), "warfarin,260,,1.2.0", "aspirin,260,,1.2.0")
    problems = _problems(sandbox_data_dir)
    assert any("aspirin" in p and "does not match any drugs.drug_id" in p for p in problems)


def test_two_costs_for_one_drug_are_reported(sandbox_data_dir: Path) -> None:
    path = _costs_path(sandbox_data_dir)
    path.write_text(path.read_text(encoding="utf-8") + "warfarin,300,,1.2.0\n", encoding="utf-8")
    problems = _problems(sandbox_data_dir)
    assert any("drug_costs: duplicate primary key" in p and "warfarin" in p for p in problems)


def test_a_drug_without_a_cost_fails_the_load_not_the_validation(
    sandbox_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drug not priced yet must not invalidate the database for EJ1, which reads no cost."""
    _replace(_costs_path(sandbox_data_dir), "warfarin,260,,1.2.0\n", "")
    validate_database(data_dir=sandbox_data_dir)
    assert len(load_formulary(data_dir=sandbox_data_dir, version="1.0.0").drugs) == 18
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_drug_costs(data_dir=sandbox_data_dir)
    assert excinfo.value.problems == ("drug_costs: drug 'warfarin' has no cost",)
    assert validate_main(["--data-dir", str(sandbox_data_dir)]) == 1
    assert "drug 'warfarin' has no cost" in capsys.readouterr().out
