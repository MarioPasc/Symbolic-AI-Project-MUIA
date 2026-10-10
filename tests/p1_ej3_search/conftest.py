"""Fixtures for EJ3 tests: EJ1's formulary and encounters, the drug costs and the oracle."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import DPLLSolver, RegimenSpace
from symbolic_ai.p1_ej3_search import CostAwareAgent, CostedDecision

# ``--import-mode=importlib`` does not add test directories to ``sys.path``. This directory holds
# ``ej3_fixture_data``; EJ1's holds ``fixture_data`` (the 1.0.0 formulary and E001-E005), reused
# here instead of transcribing the formulary a second time.
_THIS_DIR = Path(__file__).resolve().parent
for directory in (_THIS_DIR, _THIS_DIR.parent / "p1_ej1_logic"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ej3_fixture_data  # noqa: E402
import fixture_data  # noqa: E402


@pytest.fixture(scope="session")
def formulary() -> Formulary:
    """The full 1.0.0 formulary (the rules of data 1.2.0 are the same)."""
    return fixture_data.full_formulary()


@pytest.fixture(scope="session")
def costs() -> Mapping[str, int]:
    """The monthly cost of every drug of the full formulary, in euro cents."""
    return dict(ej3_fixture_data.COSTS)


@pytest.fixture(scope="session")
def space(formulary: Formulary) -> RegimenSpace:
    """The reference oracle over the full formulary."""
    return RegimenSpace(formulary)


@pytest.fixture(scope="session")
def costed(formulary: Formulary, costs: Mapping[str, int]) -> Mapping[str, CostedDecision]:
    """The default agent (A*, pruned with Agent 1's classification) on E001-E005, decided once."""
    agent = CostAwareAgent(formulary, costs, DPLLSolver())
    return {
        encounter_id: agent.decide(encounter)
        for encounter_id, encounter in fixture_data.encounters().items()
    }


@pytest.fixture
def encounters() -> Mapping[str, Encounter]:
    """Encounters E001-E005."""
    return fixture_data.encounters()


@pytest.fixture
def elena_formulary() -> Formulary:
    """Elena's and Pablo's 5-drug mini formulary."""
    return fixture_data.elena_formulary()


@pytest.fixture
def elena_costs(elena_formulary: Formulary) -> Mapping[str, int]:
    """The costs of the five drugs of the mini formulary."""
    return {drug_id: ej3_fixture_data.COSTS[drug_id] for drug_id in elena_formulary.drug_ids}


@pytest.fixture
def elena_encounter() -> Encounter:
    """Elena's clinical record."""
    return fixture_data.elena_encounter()
