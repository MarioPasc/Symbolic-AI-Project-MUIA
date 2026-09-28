"""Tests for symbolic_ai.p1_ej1_logic.agent: Algorithm 1 (PRESCRIBE / REQUEST_TEST / REFER).

Uses the worked mini world of ``EJ1-sat/README.md`` §5 (Elena, Pablo) and the reference outcomes
of ``01-database.md`` §6 (encounters E001-E005), both transcribed into fixtures by
``tests/p1_ej1_logic/fixture_data.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic.agent import Action, DrugClassification, PrescribingAgent
from symbolic_ai.p1_ej1_logic.encoding import gamma, gamma_plus, unknown_risk_factors
from symbolic_ai.p1_ej1_logic.errors import LemmaPreconditionError
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import oracle  # noqa: E402

ELENA_MODELS = frozenset(
    {frozenset({"warfarin", "paracetamol"}), frozenset({"apixaban", "paracetamol"})}
)
PABLO_MODELS = frozenset({frozenset({"paracetamol"}), frozenset({"ibuprofen", "omeprazole"})})


class TestElenaAndPablo:
    def test_elena_two_models(
        self, elena_formulary: Formulary, elena_encounter: Encounter, false_first_solver: DPLLSolver
    ) -> None:
        base = gamma(elena_formulary, elena_encounter)
        unknowns = unknown_risk_factors(elena_encounter, elena_formulary)
        clauses = gamma_plus(base, unknowns)
        assert oracle.models_of_clauses(clauses, "T_") == ELENA_MODELS

        agent = PrescribingAgent(elena_formulary, false_first_solver, classify=False)
        decision = agent.decide(elena_encounter)
        assert decision.action is Action.PRESCRIBE
        assert decision.regimen in ELENA_MODELS

    def test_pablo_two_models(
        self, elena_formulary: Formulary, pablo_encounter: Encounter, false_first_solver: DPLLSolver
    ) -> None:
        base = gamma(elena_formulary, pablo_encounter)
        unknowns = unknown_risk_factors(pablo_encounter, elena_formulary)
        clauses = gamma_plus(base, unknowns)
        assert oracle.models_of_clauses(clauses, "T_") == PABLO_MODELS

        agent = PrescribingAgent(elena_formulary, false_first_solver, classify=False)
        decision = agent.decide(pablo_encounter)
        assert decision.action is Action.PRESCRIBE
        assert decision.regimen in PABLO_MODELS

    def test_first_value_true_also_yields_a_valid_regimen(
        self, elena_formulary: Formulary, elena_encounter: Encounter
    ) -> None:
        solver = DPLLSolver(decision_first_value=True)
        agent = PrescribingAgent(elena_formulary, solver, classify=False)
        decision = agent.decide(elena_encounter)
        assert decision.action is Action.PRESCRIBE
        assert decision.regimen in ELENA_MODELS


class TestReferenceOutcomes:
    """The reference outcomes of ``01-database.md`` §6."""

    def test_e001_prescribe_five_drugs(
        self, agent: PrescribingAgent, encounters: Mapping[str, Encounter]
    ) -> None:
        decision = agent.decide(encounters["E001"])
        assert decision.action is Action.PRESCRIBE
        assert len(decision.regimen) == 5

    def test_e002_request_test_preg(
        self, agent: PrescribingAgent, encounters: Mapping[str, Encounter]
    ) -> None:
        decision = agent.decide(encounters["E002"])
        assert decision.action is Action.REQUEST_TEST
        assert decision.tests == ("PREG",)

    def test_e003_prescribe_two_drugs(
        self, agent: PrescribingAgent, encounters: Mapping[str, Encounter]
    ) -> None:
        decision = agent.decide(encounters["E003"])
        assert decision.action is Action.PRESCRIBE
        assert len(decision.regimen) == 2

    def test_e004_prescribe_exact_regimen_and_classification(
        self, agent: PrescribingAgent, encounters: Mapping[str, Encounter]
    ) -> None:
        decision = agent.decide(encounters["E004"])
        expected = frozenset({"ibuprofen", "mirtazapine", "omeprazole"})
        assert decision.action is Action.PRESCRIBE
        assert decision.regimen == expected
        assert decision.classification is not None

        essential = {
            d for d, c in decision.classification.items() if c is DrugClassification.ESSENTIAL
        }
        excluded = {
            d for d, c in decision.classification.items() if c is DrugClassification.EXCLUDED
        }
        optional = {
            d for d, c in decision.classification.items() if c is DrugClassification.OPTIONAL
        }
        assert essential == expected
        assert len(excluded) == 15
        assert optional == set()

    def test_e005_refer(self, agent: PrescribingAgent, encounters: Mapping[str, Encounter]) -> None:
        decision = agent.decide(encounters["E005"])
        assert decision.action is Action.REFER
        assert decision.regimen == frozenset()
        assert decision.classification is None


class TestLemmaPreconditionThroughAgent:
    """Confirms the agent checks the lemma's precondition before solving (wiring, not the logic).

    The check's own logic (a positive unknown ``R_u`` raises) is unit-tested directly against
    ``encoding.check_lemma_precondition`` in ``test_ej1_encoding.py``; no real formulary of this
    design can violate it (every A4/A5 clause negates its risk factor by construction), so here we
    monkeypatch the check to confirm ``PrescribingAgent.decide`` calls it and propagates its error.
    """

    def test_decide_propagates_lemma_precondition_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        elena_formulary: Formulary,
        elena_encounter: Encounter,
        false_first_solver: DPLLSolver,
    ) -> None:
        def _raise(*_args: object, **_kwargs: object) -> None:
            raise LemmaPreconditionError(("forced for test",))

        monkeypatch.setattr("symbolic_ai.p1_ej1_logic.agent.check_lemma_precondition", _raise)
        agent = PrescribingAgent(elena_formulary, false_first_solver, classify=False)
        with pytest.raises(LemmaPreconditionError):
            agent.decide(elena_encounter)
