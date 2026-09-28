"""Shared fixtures for the dataloader test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = _REPO_ROOT / "data"


@pytest.fixture
def real_data_dir() -> Path:
    """Return the path to the real, committed database."""
    return REAL_DATA_DIR


@pytest.fixture
def sandbox_data_dir(tmp_path: Path) -> Path:
    """Copy the real database into a temporary directory that a test may freely corrupt."""
    destination = tmp_path / "data"
    shutil.copytree(REAL_DATA_DIR, destination)
    return destination
