"""Tests for symbolic_ai.p1_ej3_search.agent: Agent 1 decides, the search picks the regimen."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic import Action, DPLLSolver, FullRecord, PrescribingAgent
from symbolic_ai.p1_ej3_search import (
    Algorithm,
    CostAwareAgent,
    CostedDecision,
    CostError,
    RegimenProblem,
    SearchInvariantError,
    SearchResult,
    worst_case_record,
)

ALL_ENCOUNTERS = ["E001", "E002", "E003", "E004", "E005"]


@pytest.fixture
def agent(formulary: Formulary, costs: Mapping[str, int]) -> CostAwareAgent:
    """A* on the plain problem: Agent 1 does not classify, which keeps these tests fast.

    The default agent (pruned with the classification) is the ``costed`` fixture of conftest.
    """
    return CostAwareAgent(formulary, costs, DPLLSolver(), prune=False)


# --- reference outcomes on E001-E005 ------------------------------------------------------------


@pytest.mark.parametrize(
    ("encounter_id", "regimen", "cost", "baseline_cost"),
    [
        (
            "E001",
            {"enalapril", "linagliptin", "mirtazapine", "paracetamol", "warfarin"},
            5010,
            5330,
        ),
        ("E003", {"hydrochlorothiazide", "warfarin"}, 400, 440),
        ("E004", {"ibuprofen", "mirtazapine", "omeprazole"}, 1110, 1110),
    ],
)
def test_prescribes_the_cheapest_regimen(
    costed: Mapping[str, CostedDecision],
    encounter_id: str,
    regimen: set[str],
    cost: int,
    baseline_cost: int,
) -> None:
    decided = costed[encounter_id]
    assert decided.action is Action.PRESCRIBE
    assert decided.regimen == regimen
    assert (decided.cost, decided.baseline_cost) == (cost, baseline_cost)
    assert decided.saving == baseline_cost - cost
    assert decided.search is not None
    assert decided.search.algorithm is Algorithm.ASTAR


def test_request_test_is_agent_1s_decision_untouched(costed: Mapping[str, CostedDecision]) -> None:
    decided = costed["E002"]
    assert decided.action is Action.REQUEST_TEST
    assert decided.decision.tests == ("PREG",)
    assert (decided.regimen, decided.cost, decided.baseline_cost) == (frozenset(), None, None)
    assert decided.saving is None
    assert decided.search is None


def test_refer_is_agent_1s_decision_untouched(costed: Mapping[str, CostedDecision]) -> None:
    decided = costed["E005"]
    assert decided.action is Action.REFER
    assert (decided.regimen, decided.cost, decided.search) == (frozenset(), None, None)


@pytest.mark.parametrize("encounter_id", ALL_ENCOUNTERS)
def test_agent_1_returns_what_it_returns_on_its_own(
    formulary: Formulary,
    costed: Mapping[str, CostedDecision],
    encounters: Mapping[str, Encounter],
    encounter_id: str,
) -> None:
    """The embedded decision equals Agent 1's: action, regimen, tests, classification and trace."""
    alone = PrescribingAgent(formulary, DPLLSolver(), classify=True)
    assert costed[encounter_id].decision == alone.decide(encounters[encounter_id])


@pytest.mark.parametrize("encounter_id", ALL_ENCOUNTERS)
def test_without_pruning_the_regimen_is_the_same(
    agent: CostAwareAgent,
    costed: Mapping[str, CostedDecision],
    encounters: Mapping[str, Encounter],
    encounter_id: str,
) -> None:
    unpruned = agent.decide(encounters[encounter_id])
    pruned = costed[encounter_id]
    assert unpruned.decision.classification is None
    assert (unpruned.action, unpruned.regimen, unpruned.cost) == (
        pruned.action,
        pruned.regimen,
        pruned.cost,
    )


@pytest.mark.parametrize("algorithm", [Algorithm.IDASTAR, Algorithm.UNIFORM_COST])
def test_the_other_optimal_algorithms_agree_with_astar(
    formulary: Formulary,
    costs: Mapping[str, int],
    encounters: Mapping[str, Encounter],
    algorithm: Algorithm,
) -> None:
    agent = CostAwareAgent(formulary, costs, DPLLSolver(), algorithm, prune=False)
    assert agent.decide(encounters["E001"]).cost == 5010


def test_greedy_may_cost_more_than_agent_1(
    formulary: Formulary, costs: Mapping[str, int], encounters: Mapping[str, Encounter]
) -> None:
    """Greedy is not optimal, so a regimen dearer than Agent 1's is reported, not an error."""
    greedy = CostAwareAgent(formulary, costs, DPLLSolver(), Algorithm.GREEDY, prune=False)
    decided = greedy.decide(encounters["E003"])
    assert decided.regimen == {"amlodipine", "apixaban"}
    assert (decided.cost, decided.baseline_cost, decided.saving) == (5580, 440, -5140)


def test_elena_gets_warfarin_and_paracetamol(
    elena_formulary: Formulary, elena_costs: Mapping[str, int], elena_encounter: Encounter
) -> None:
    """Her two regimens differ in the anticoagulant: warfarin (2.60) against apixaban (54.00)."""
    decided = CostAwareAgent(elena_formulary, elena_costs, DPLLSolver()).decide(elena_encounter)
    assert decided.regimen == {"paracetamol", "warfarin"}
    assert decided.cost == 450


# --- the worst-case record ----------------------------------------------------------------------


def test_worst_case_record_takes_unknown_risk_factors_as_present(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    record = worst_case_record(encounters["E001"], formulary)
    assert record == FullRecord(
        frozenset({"AF", "DEP", "HTN", "PAIN", "T2D"}), frozenset({"AGE65", "CKD"})
    )


def test_worst_case_record_ignores_identifiers_outside_the_formulary(
    elena_formulary: Formulary,
) -> None:
    encounter = Encounter(
        encounter_id="X",
        patient_id="P",
        seq=1,
        visit_date=date(2026, 1, 1),
        age_years=70,
        conditions=frozenset({"PAIN", "GOUT"}),
        risk_factors={"AGE65": RiskStatus.PRESENT, "CKD": RiskStatus.UNKNOWN},
    )
    assert worst_case_record(encounter, elena_formulary) == FullRecord(
        frozenset({"PAIN"}), frozenset({"AGE65"})
    )


# --- failure paths ------------------------------------------------------------------------------


def test_a_missing_cost_raises_when_the_agent_prescribes(
    formulary: Formulary, costs: Mapping[str, int], encounters: Mapping[str, Encounter]
) -> None:
    incomplete = {drug_id: cost for drug_id, cost in costs.items() if drug_id != "warfarin"}
    agent = CostAwareAgent(formulary, incomplete, DPLLSolver(), prune=False)
    with pytest.raises(CostError, match="warfarin"):
        agent.decide(encounters["E003"])


def _stub_search(
    monkeypatch: pytest.MonkeyPatch, regimen: frozenset[str] | None, cost: int | None
) -> None:
    """Make the agent's search return ``regimen`` whatever the problem."""

    def fake(problem: RegimenProblem, algorithm: Algorithm = Algorithm.ASTAR) -> SearchResult:
        return SearchResult(algorithm, regimen, cost, (), 0, 0, 0)

    monkeypatch.setattr("symbolic_ai.p1_ej3_search.agent.search", fake)


def test_a_search_that_finds_nothing_after_prescribe_raises(
    monkeypatch: pytest.MonkeyPatch, agent: CostAwareAgent, encounters: Mapping[str, Encounter]
) -> None:
    _stub_search(monkeypatch, None, None)
    with pytest.raises(SearchInvariantError, match=r"E003.*found nothing"):
        agent.decide(encounters["E003"])


@pytest.mark.parametrize(
    ("regimen", "axiom"),
    [
        ({"warfarin"}, "A1:HTN"),  # hypertension left uncovered
        ({"amlodipine", "warfarin", "omeprazole"}, "A2:omeprazole"),  # nothing justifies it
        ({"amlodipine", "apixaban", "warfarin"}, "A6:anticoagulants"),
    ],
)
def test_a_regimen_that_breaks_a_clause_of_gamma_plus_raises(
    monkeypatch: pytest.MonkeyPatch,
    agent: CostAwareAgent,
    encounters: Mapping[str, Encounter],
    regimen: set[str],
    axiom: str,
) -> None:
    _stub_search(monkeypatch, frozenset(regimen), 0)
    with pytest.raises(SearchInvariantError, match="breaks") as excinfo:
        agent.decide(encounters["E003"])
    assert axiom in str(excinfo.value)


def test_an_optimal_search_dearer_than_agent_1_raises(
    monkeypatch: pytest.MonkeyPatch, agent: CostAwareAgent, encounters: Mapping[str, Encounter]
) -> None:
    """Verapamil with warfarin is a valid regimen for E003, but dearer than Agent 1's (4.40)."""
    _stub_search(monkeypatch, frozenset({"verapamil", "warfarin"}), 780)
    with pytest.raises(SearchInvariantError, match="more than Agent 1's"):
        agent.decide(encounters["E003"])


def test_the_certificate_uses_the_worst_case_of_the_unknown_risk_factors(
    monkeypatch: pytest.MonkeyPatch, agent: CostAwareAgent, encounters: Mapping[str, Encounter]
) -> None:
    """E001 does not know CKD: metformin is fine without it, and still breaks Γ⁺."""
    regimen = {"enalapril", "metformin", "mirtazapine", "paracetamol", "warfarin"}
    _stub_search(monkeypatch, frozenset(regimen), 1380)
    with pytest.raises(SearchInvariantError, match="A4:CKD,metformin"):
        agent.decide(encounters["E001"])


def test_costed_decision_is_immutable(costed: Mapping[str, CostedDecision]) -> None:
    decided = costed["E004"]
    assert replace(decided, cost=1) != decided
    with pytest.raises(AttributeError):
        decided.cost = 1  # type: ignore[misc]
