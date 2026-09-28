"""Integration test: the real dataloader against the real database (01-database.md §6).

Marked ``integration`` and excluded from the fast suite (``pytest -m "not integration"``). In this
worktree ``symbolic_ai.dataloader``'s load functions are stubs that raise ``NotImplementedError``
(peer ``db-dataloader``'s work lands separately): this test is expected to fail here. It exercises
the exact reference outcomes of ``01-database.md`` §6 and is the regression guard once the
dataloader implementation merges.
"""

from __future__ import annotations

import pytest

from symbolic_ai.dataloader import load_encounter, load_formulary
from symbolic_ai.p1_ej1_logic import EJ1_FORMULARY_VERSION, Action, DPLLSolver, PrescribingAgent

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("encounter_id", "expected_action", "regimen_size"),
    [
        ("E001", Action.PRESCRIBE, 5),
        ("E002", Action.REQUEST_TEST, None),
        ("E003", Action.PRESCRIBE, 2),
        ("E004", Action.PRESCRIBE, 3),
        ("E005", Action.REFER, None),
    ],
)
def test_reference_outcome(
    encounter_id: str, expected_action: Action, regimen_size: int | None
) -> None:
    formulary = load_formulary(version=EJ1_FORMULARY_VERSION)
    encounter = load_encounter(encounter_id)
    agent = PrescribingAgent(formulary, DPLLSolver(), classify=True)

    decision = agent.decide(encounter)

    assert decision.action is expected_action
    if regimen_size is not None:
        assert len(decision.regimen) == regimen_size


def test_e002_requests_preg() -> None:
    formulary = load_formulary(version=EJ1_FORMULARY_VERSION)
    encounter = load_encounter("E002")
    agent = PrescribingAgent(formulary, DPLLSolver(), classify=False)

    decision = agent.decide(encounter)

    assert decision.tests == ("PREG",)


def test_e004_classification() -> None:
    formulary = load_formulary(version=EJ1_FORMULARY_VERSION)
    encounter = load_encounter("E004")
    agent = PrescribingAgent(formulary, DPLLSolver(), classify=True)

    decision = agent.decide(encounter)

    assert decision.regimen == frozenset({"ibuprofen", "mirtazapine", "omeprazole"})
    assert decision.classification is not None
