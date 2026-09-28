"""Version filtering of ``load_formulary`` (docs 01-database.md, section 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from symbolic_ai.dataloader import load_formulary

pytestmark = pytest.mark.integration

_NEW_DRUG_ID = "acetazolamide"
_NEW_DRUG_ROW = f"{_NEW_DRUG_ID},Acetazolamida,Acetazolamide,S01EC01,1.1.0\n"


def _add_minor_version_drug(sandbox_data_dir: Path) -> None:
    """Bump the package to 1.1.0 and add one drug row with ``since = 1.1.0``."""
    descriptor_path = sandbox_data_dir / "datapackage.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["version"] = "1.1.0"
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    drugs_path = sandbox_data_dir / "formulary" / "drugs.csv"
    drugs_path.write_text(drugs_path.read_text(encoding="utf-8") + _NEW_DRUG_ROW, encoding="utf-8")


def test_load_formulary_excludes_rows_newer_than_the_requested_version(
    sandbox_data_dir: Path,
) -> None:
    _add_minor_version_drug(sandbox_data_dir)
    formulary = load_formulary(data_dir=sandbox_data_dir, version="1.0.0")
    assert formulary.version == "1.0.0"
    assert _NEW_DRUG_ID not in formulary.drug_ids
    assert len(formulary.drugs) == 18


def test_load_formulary_includes_rows_up_to_the_default_version(sandbox_data_dir: Path) -> None:
    _add_minor_version_drug(sandbox_data_dir)
    formulary = load_formulary(data_dir=sandbox_data_dir)
    assert formulary.version == "1.1.0"
    assert _NEW_DRUG_ID in formulary.drug_ids
    assert len(formulary.drugs) == 19
