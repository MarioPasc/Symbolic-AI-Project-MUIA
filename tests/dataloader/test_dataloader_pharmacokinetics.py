"""``load_pharmacokinetics`` on the real data 1.3.0 and one broken fixture per rule of the table."""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import (
    DatabaseValidationError,
    Pharmacokinetics,
    load_drug_costs,
    load_formulary,
    load_ontology,
    load_pharmacokinetics,
    validate_database,
)
from symbolic_ai.dataloader.validate import main as validate_main

pytestmark = pytest.mark.integration

_SOURCE = "illustrative (EJ4 design D24)"
_WARFARIN_ROW = f"warfarin,5,0,{_SOURCE},1.3.0"

#: The values of EJ4 design §8 (D24): drug -> (washout days, taper days).
_EXPECTED_DAYS: dict[str, tuple[int, int]] = {
    "amlodipine": (3, 0),
    "apixaban": (2, 0),
    "bisoprolol": (2, 3),
    "enalapril": (1, 0),
    "gliclazide": (1, 0),
    "hydrochlorothiazide": (1, 0),
    "ibuprofen": (1, 0),
    "linagliptin": (2, 0),
    "losartan": (1, 0),
    "metformin": (1, 0),
    "mirtazapine": (4, 3),
    "omeprazole": (1, 0),
    "paracetamol": (1, 0),
    "propranolol": (1, 3),
    "sertraline": (7, 4),
    "tramadol": (2, 2),
    "verapamil": (1, 0),
    "warfarin": (5, 0),
}


def _pk_path(data_dir: Path) -> Path:
    return data_dir / "formulary" / "pharmacokinetics.csv"


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture text not found in {path}: {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _problems(data_dir: Path) -> tuple[str, ...]:
    with pytest.raises(DatabaseValidationError) as excinfo:
        validate_database(data_dir=data_dir)
    return excinfo.value.problems


# --- the real data ------------------------------------------------------------------------------


def test_every_drug_has_one_row(real_data_dir: Path) -> None:
    table = load_pharmacokinetics(data_dir=real_data_dir)
    assert tuple(table) == load_formulary(data_dir=real_data_dir).drug_ids
    assert len(table) == 18
    assert all(row.drug_id == drug_id for drug_id, row in table.items())


def test_values_are_those_of_the_design(real_data_dir: Path) -> None:
    table = load_pharmacokinetics(data_dir=real_data_dir)
    assert {d: (r.washout_days, r.taper_days) for d, r in table.items()} == _EXPECTED_DAYS
    assert table["sertraline"] == Pharmacokinetics("sertraline", 7, 4, _SOURCE)
    assert {row.source for row in table.values()} == {_SOURCE}


def test_rows_load_at_the_version_that_added_them(real_data_dir: Path) -> None:
    assert load_pharmacokinetics(data_dir=real_data_dir, version="1.3.0") == load_pharmacokinetics(
        data_dir=real_data_dir
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0"])
def test_no_row_exists_before_version_1_3_0(real_data_dir: Path, version: str) -> None:
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_pharmacokinetics(data_dir=real_data_dir, version=version)
    assert len(excinfo.value.problems) == 18  # every drug lacks a row
    assert excinfo.value.problems[0] == (
        "pharmacokinetics: drug 'amlodipine' has no washout and taper days"
    )


def test_older_pins_are_unchanged_by_data_1_3_0(real_data_dir: Path) -> None:
    pinned = load_formulary(data_dir=real_data_dir, version="1.0.0")
    current = load_formulary(data_dir=real_data_dir)
    assert current.version == "1.3.0"
    for name in ("conditions", "drugs", "risk_factors", "candidates", "interactions"):
        assert getattr(current, name) == getattr(pinned, name)
    assert load_ontology(data_dir=real_data_dir, version="1.1.0").version == "1.1.0"
    assert load_drug_costs(data_dir=real_data_dir, version="1.2.0") == load_drug_costs(
        data_dir=real_data_dir
    )


def test_validate_cli_counts_the_rows(
    real_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert validate_main(["--data-dir", str(real_data_dir)]) == 0
    out = capsys.readouterr().out
    assert "data version: 1.3.0" in out
    assert "pharmacokinetics: 18" in out


# --- one broken fixture per rule ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("broken_row", "message"),
    [
        (
            f"warfarin,-5,0,{_SOURCE},1.3.0",
            "pharmacokinetics[drug_id='warfarin']: washout_days = -5 is negative",
        ),
        (
            f"warfarin,5,-1,{_SOURCE},1.3.0",
            "pharmacokinetics[drug_id='warfarin']: taper_days = -1 is negative",
        ),
    ],
)
def test_negative_days_are_reported(sandbox_data_dir: Path, broken_row: str, message: str) -> None:
    _replace(_pk_path(sandbox_data_dir), _WARFARIN_ROW, broken_row)
    assert _problems(sandbox_data_dir) == (message,)


@pytest.mark.parametrize("field_name", ["washout_days", "taper_days"])
def test_non_integer_days_are_reported(sandbox_data_dir: Path, field_name: str) -> None:
    broken = (
        f"warfarin,5.5,0,{_SOURCE},1.3.0"
        if field_name == "washout_days"
        else (f"warfarin,5,0.5,{_SOURCE},1.3.0")
    )
    _replace(_pk_path(sandbox_data_dir), _WARFARIN_ROW, broken)
    problems = _problems(sandbox_data_dir)
    assert any(field_name in p and "not a valid integer" in p for p in problems)


def test_empty_days_are_reported(sandbox_data_dir: Path) -> None:
    _replace(_pk_path(sandbox_data_dir), _WARFARIN_ROW, f"warfarin,,0,{_SOURCE},1.3.0")
    problems = _problems(sandbox_data_dir)
    assert any("'washout_days' is required but empty" in p for p in problems)


def test_row_of_an_unknown_drug_is_reported(sandbox_data_dir: Path) -> None:
    _replace(_pk_path(sandbox_data_dir), _WARFARIN_ROW, f"aspirin,5,0,{_SOURCE},1.3.0")
    problems = _problems(sandbox_data_dir)
    assert any("aspirin" in p and "does not match any drugs.drug_id" in p for p in problems)


def test_two_rows_for_one_drug_are_reported(sandbox_data_dir: Path) -> None:
    path = _pk_path(sandbox_data_dir)
    path.write_text(
        path.read_text(encoding="utf-8") + f"warfarin,6,0,{_SOURCE},1.3.0\n", encoding="utf-8"
    )
    problems = _problems(sandbox_data_dir)
    assert any("pharmacokinetics: duplicate primary key" in p and "warfarin" in p for p in problems)


def test_a_drug_without_a_row_fails_the_load_not_the_validation(
    sandbox_data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drug whose washout is not known yet must not invalidate the database for EJ1-EJ3."""
    _replace(_pk_path(sandbox_data_dir), _WARFARIN_ROW + "\n", "")
    validate_database(data_dir=sandbox_data_dir)
    assert len(load_formulary(data_dir=sandbox_data_dir, version="1.0.0").drugs) == 18
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_pharmacokinetics(data_dir=sandbox_data_dir)
    assert excinfo.value.problems == (
        "pharmacokinetics: drug 'warfarin' has no washout and taper days",
    )
    assert validate_main(["--data-dir", str(sandbox_data_dir)]) == 1
    assert "drug 'warfarin' has no washout and taper days" in capsys.readouterr().out
