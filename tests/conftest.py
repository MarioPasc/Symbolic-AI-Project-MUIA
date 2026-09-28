"""Shared fixtures for the whole test suite: repository paths only.

Data access for any exercise goes through ``symbolic_ai.dataloader``; nothing here opens a file
under ``data/`` directly (``docs/HARNESSES/PYTHON-CODING.md`` Part B.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root: the directory containing ``pyproject.toml``."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    """Return the real database directory, ``<repo_root>/data``."""
    return repo_root / "data"
