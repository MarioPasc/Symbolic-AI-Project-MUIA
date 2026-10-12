"""Fixtures for EJ4 tests: the seven toy problems with hand-computed minimum horizons."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``--import-mode=importlib`` does not add test directories to ``sys.path``; this one holds
# ``ej4_toys`` (the frozen known answers), which test modules import to parametrize over them.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import ej4_toys  # noqa: E402


@pytest.fixture(scope="session")
def known_answers() -> tuple[ej4_toys.KnownAnswer, ...]:
    """The toy problems T0-T6 and their minimum horizons (``None``: no transition exists)."""
    return ej4_toys.KNOWN_ANSWERS
