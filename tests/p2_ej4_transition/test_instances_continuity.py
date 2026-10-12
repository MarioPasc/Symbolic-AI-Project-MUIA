"""Continuity with P1: EJ4's starting regimen ``R_0`` against what P1's EJ3 agent prescribes.

For each of the 4032 fully observed histories (every non-empty set of conditions times every set
of risk factors) the EJ3 agent (``CostAwareAgent``, data 1.2.0, the costs EJ3 uses) decides, and
its decision is compared with ``cheapest_safe_regimen``. Three cases:

(a) EJ3 prescribes one drug per condition: ``R_0`` must be that regimen exactly;
(b) EJ3 prescribes a companion drug that no condition of the history needs (today omeprazole
    with ibuprofen under AGE65 and no GERD): one drug per condition cannot express it, and EJ4
    has no safe regimen for the history;
(c) EJ3 refers: EJ4 has no safe regimen either.

The full run uses the agent without Agent 1's domain pruning (``prune=False``, ~55 s instead of
~14 min); pruning only removes drugs Agent 1 proved essential or excluded, and the regimens of the
two configurations were identical on all 4032 histories (2026-10-12). A deterministic subset
re-checks that with the default, pruned agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import pytest

from symbolic_ai.dataloader import Formulary, load_drug_costs, load_formulary
from symbolic_ai.dataloader.models import Encounter, RiskStatus
from symbolic_ai.p1_ej1_logic import Action, DPLLSolver
from symbolic_ai.p1_ej3_search import EJ3_DATA_VERSION, CostAwareAgent, CostedDecision
from symbolic_ai.p2_ej4_transition.instances import History, cheapest_safe_regimen

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: One history in 64 is re-decided by the default (pruned) agent: 63 histories, about 16 s.
_PRUNED_STRIDE = 64


@dataclass(frozen=True, slots=True)
class _Comparison:
    """EJ3's decision and EJ4's ``R_0`` for one fully observed history."""

    history: History
    decision: CostedDecision
    start: dict[str, str] | None


def _subsets(items: tuple[str, ...], *, smallest: int) -> Iterator[tuple[str, ...]]:
    ordered = sorted(items)
    for size in range(smallest, len(ordered) + 1):
        yield from combinations(ordered, size)


def _histories(formulary: Formulary) -> list[History]:
    return [
        History(conditions, risk_factors)
        for conditions in _subsets(formulary.condition_ids, smallest=1)
        for risk_factors in _subsets(formulary.risk_factor_ids, smallest=0)
    ]


def _encounter(formulary: Formulary, history: History) -> Encounter:
    """A fully observed record: each risk factor of the formulary is present or absent."""
    return Encounter(
        encounter_id="synthetic",
        patient_id="synthetic",
        seq=1,
        visit_date=date(2026, 10, 12),
        age_years=0,
        conditions=frozenset(history.conditions),
        risk_factors={
            r: RiskStatus.PRESENT if r in history.risk_factors else RiskStatus.ABSENT
            for r in formulary.risk_factor_ids
        },
    )


def _condition_of(formulary: Formulary) -> dict[str, str]:
    """The condition each drug treats (each drug treats exactly one, design §4.1)."""
    return {d: c for c, drugs in formulary.candidates.items() for d in drugs}


def _one_per_condition(
    formulary: Formulary, history: History, regimen: frozenset[str]
) -> dict[str, str] | None:
    """The regimen as condition -> drug if it gives exactly one drug per condition, else None."""
    condition_of = _condition_of(formulary)
    by_condition: dict[str, list[str]] = {}
    for drug in sorted(regimen):
        by_condition.setdefault(condition_of[drug], []).append(drug)
    if set(by_condition) != set(history.conditions):
        return None
    if any(len(drugs) != 1 for drugs in by_condition.values()):
        return None
    return {c: drugs[0] for c, drugs in by_condition.items()}


def _uses_an_unneeded_companion(
    formulary: Formulary, history: History, regimen: frozenset[str]
) -> bool:
    """Whether a co-prescription row forces a companion whose condition the history lacks."""
    condition_of = _condition_of(formulary)
    return any(
        row.drug_id in regimen
        and row.risk_factor_id in history.risk_factors
        and row.companion_drug_id in regimen
        and condition_of[row.companion_drug_id] not in history.conditions
        for row in formulary.coprescriptions
    )


@pytest.fixture(scope="module")
def ej3_data() -> tuple[Formulary, Mapping[str, int]]:
    """The formulary and costs exactly as EJ3's CLI loads them (data 1.2.0)."""
    formulary = load_formulary(version=EJ3_DATA_VERSION)
    priced = load_drug_costs(version=EJ3_DATA_VERSION)
    return formulary, {drug_id: cost.monthly_cost_cents for drug_id, cost in priced.items()}


@pytest.fixture(scope="module")
def comparisons(ej3_data: tuple[Formulary, Mapping[str, int]]) -> tuple[_Comparison, ...]:
    ej3_formulary, ej3_costs = ej3_data
    agent = CostAwareAgent(ej3_formulary, ej3_costs, DPLLSolver(), prune=False)
    formulary, costs = load_formulary(), load_drug_costs()
    return tuple(
        _Comparison(
            history,
            agent.decide(_encounter(ej3_formulary, history)),
            cheapest_safe_regimen(formulary, costs, history),
        )
        for history in _histories(ej3_formulary)
    )


def _cases(
    formulary: Formulary, comparisons: tuple[_Comparison, ...]
) -> dict[str, list[_Comparison]]:
    cases: dict[str, list[_Comparison]] = {"a": [], "b": [], "c": []}
    for comparison in comparisons:
        if comparison.decision.action is not Action.PRESCRIBE:
            cases["c"].append(comparison)
        elif _one_per_condition(formulary, comparison.history, comparison.decision.regimen):
            cases["a"].append(comparison)
        else:
            cases["b"].append(comparison)
    return cases


def test_every_fully_observed_history_is_compared(comparisons: tuple[_Comparison, ...]) -> None:
    assert len(comparisons) == 4032


def test_case_counts(
    ej3_data: tuple[Formulary, Mapping[str, int]], comparisons: tuple[_Comparison, ...]
) -> None:
    cases = _cases(ej3_data[0], comparisons)
    assert {name: len(items) for name, items in cases.items()} == {"a": 2720, "b": 32, "c": 1280}


def test_a_r0_is_ej3s_regimen_whenever_it_has_one_drug_per_condition(
    ej3_data: tuple[Formulary, Mapping[str, int]], comparisons: tuple[_Comparison, ...]
) -> None:
    formulary = ej3_data[0]
    differences = [
        (c.history, sorted(c.decision.regimen), c.start)
        for c in _cases(formulary, comparisons)["a"]
        if _one_per_condition(formulary, c.history, c.decision.regimen) != c.start
    ]
    assert differences == []


def test_b_the_other_prescriptions_are_exactly_the_companion_cases(
    ej3_data: tuple[Formulary, Mapping[str, int]], comparisons: tuple[_Comparison, ...]
) -> None:
    formulary = ej3_data[0]
    companion = [
        c
        for c in comparisons
        if c.decision.action is Action.PRESCRIBE
        and _uses_an_unneeded_companion(formulary, c.history, c.decision.regimen)
    ]
    assert _cases(formulary, comparisons)["b"] == companion
    for c in companion:
        # Today every one of them is ibuprofen + omeprazole under AGE65 without GERD, for a
        # patient whose other analgesics are excluded (LIVER and EPI): EJ4 has no safe regimen.
        assert {"ibuprofen", "omeprazole"} <= c.decision.regimen
        assert {"AGE65", "EPI", "LIVER"} <= set(c.history.risk_factors)
        assert "GERD" not in c.history.conditions
        assert c.start is None


def test_c_ej4_has_no_safe_regimen_when_ej3_refers(
    ej3_data: tuple[Formulary, Mapping[str, int]], comparisons: tuple[_Comparison, ...]
) -> None:
    referred = _cases(ej3_data[0], comparisons)["c"]
    assert all(c.decision.action is Action.REFER for c in referred)
    assert [c.history for c in referred if c.start is not None] == []


def test_pruning_does_not_change_ej3s_regimen(
    ej3_data: tuple[Formulary, Mapping[str, int]], comparisons: tuple[_Comparison, ...]
) -> None:
    ej3_formulary, ej3_costs = ej3_data
    default_agent = CostAwareAgent(ej3_formulary, ej3_costs, DPLLSolver())
    for c in comparisons[::_PRUNED_STRIDE]:
        decision = default_agent.decide(_encounter(ej3_formulary, c.history))
        assert (decision.action, decision.regimen) == (c.decision.action, c.decision.regimen)
