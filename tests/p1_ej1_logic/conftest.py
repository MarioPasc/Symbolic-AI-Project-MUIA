"""Fixtures for EJ1 tests: the full 1.0.0 formulary, E001-E005, and Elena's/Pablo's mini world.

Transcribed from ``01-database.md`` §6 and ``EJ1-sat/README.md`` §5 by :mod:`fixture_data`, which
stays free of pytest so it can also be loaded by the standalone subprocess script of the
determinism test.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import DPLLSolver, PrescribingAgent

# ``tests/`` is not an installed package (only ``src/`` is, per pyproject.toml), and
# ``--import-mode=importlib`` does not add test directories to ``sys.path``. Adding this
# directory explicitly lets both this file and the subprocess script of
# ``test_ej1_determinism.py`` import the plain module ``fixture_data`` the same way.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fixture_data  # noqa: E402


@pytest.fixture
def formulary() -> Formulary:
    """The full 1.0.0 formulary."""
    return fixture_data.full_formulary()


@pytest.fixture
def encounters() -> Mapping[str, Encounter]:
    """Encounters E001-E005."""
    return fixture_data.encounters()


@pytest.fixture
def elena_formulary() -> Formulary:
    """Elena's and Pablo's 5-drug mini formulary."""
    return fixture_data.elena_formulary()


@pytest.fixture
def elena_encounter() -> Encounter:
    """Elena's clinical record."""
    return fixture_data.elena_encounter()


@pytest.fixture
def pablo_encounter() -> Encounter:
    """Pablo's clinical record."""
    return fixture_data.pablo_encounter()


@pytest.fixture
def false_first_solver() -> DPLLSolver:
    """A solver with the default value ordering (``T_d`` tried false first)."""
    return DPLLSolver(decision_first_value=False)


@pytest.fixture
def true_first_solver() -> DPLLSolver:
    """A solver reproducing the textbook value ordering (``T_d`` tried true first)."""
    return DPLLSolver(decision_first_value=True)


@pytest.fixture
def agent(formulary: Formulary, false_first_solver: DPLLSolver) -> PrescribingAgent:
    """A classifying agent over the full formulary, with the default (false-first) solver."""
    return PrescribingAgent(formulary, false_first_solver, classify=True)
