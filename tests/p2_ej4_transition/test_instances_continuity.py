"""Continuity with P1: EJ4's starting regimen ``R_0`` against what P1's EJ3 agent prescribes.

``continuity_with_ej3`` runs the EJ3 agent (``CostAwareAgent``, data 1.2.0, the costs EJ3 uses)
on each of the 4032 fully observed histories and compares its regimen with ``R_0``:

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

from collections.abc import Mapping

import pytest

from symbolic_ai.dataloader import (
    DrugCost,
    Formulary,
    load_drug_costs,
    load_formulary,
)
from symbolic_ai.p1_ej1_logic import DPLLSolver
from symbolic_ai.p1_ej3_search import EJ3_DATA_VERSION, CostAwareAgent
from symbolic_ai.p2_ej4_transition.instances import (
    ContinuityReport,
    History,
    _all_histories,
    _observed_encounter,
    cheapest_safe_regimen,
    continuity_with_ej3,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: One history in 64 is re-decided by the default (pruned) agent: 63 histories, about 16 s.
_PRUNED_STRIDE = 64

#: The 32 companion histories: PAIN under AGE65 + EPI + LIVER (no CKD, no GERD), with any of
#: ASTHMA and PREG, alone or with any of DEP, HTN and T2D.
_COMPANION_COUNT = 32


@pytest.fixture(scope="module")
def ej3_data() -> tuple[Formulary, Mapping[str, int]]:
    """The formulary and costs exactly as EJ3's CLI loads them (data 1.2.0)."""
    formulary = load_formulary(version=EJ3_DATA_VERSION)
    priced = load_drug_costs(version=EJ3_DATA_VERSION)
    return formulary, {drug_id: cost.monthly_cost_cents for drug_id, cost in priced.items()}


@pytest.fixture(scope="module")
def ej4_data() -> tuple[Formulary, Mapping[str, DrugCost]]:
    return load_formulary(), load_drug_costs()


@pytest.fixture(scope="module")
def report(
    ej4_data: tuple[Formulary, Mapping[str, DrugCost]],
    ej3_data: tuple[Formulary, Mapping[str, int]],
) -> ContinuityReport:
    return continuity_with_ej3(*ej4_data, *ej3_data)


def test_case_counts(report: ContinuityReport) -> None:
    assert (report.histories, report.equal, len(report.companion), report.refer) == (
        4032,
        2720,
        _COMPANION_COUNT,
        1280,
    )


def test_a_r0_is_ej3s_regimen_whenever_it_has_one_drug_per_condition(
    report: ContinuityReport,
) -> None:
    assert report.different == ()


def test_no_unexpected_case(report: ContinuityReport) -> None:
    # Also covers (c): a referral with an R_0, or a test request, would be listed here.
    assert report.unexpected == ()


def test_b_companion_cases_are_ibuprofen_with_omeprazole_without_gerd(
    ej4_data: tuple[Formulary, Mapping[str, DrugCost]], report: ContinuityReport
) -> None:
    formulary, costs = ej4_data
    for history_id in report.companion:
        conditions, risk_factors = (part.split("+") for part in history_id.split("|"))
        history = History(tuple(conditions), tuple(risk_factors))
        assert "PAIN" in history.conditions
        assert set(history.conditions) - {"PAIN"} <= {"DEP", "HTN", "T2D"}
        assert {"AGE65", "EPI", "LIVER"} <= set(history.risk_factors)
        assert set(history.risk_factors) - {"AGE65", "EPI", "LIVER"} <= {"ASTHMA", "PREG"}
        assert cheapest_safe_regimen(formulary, costs, history) is None


def test_the_report_names_one_companion_history(report: ContinuityReport) -> None:
    assert "PAIN|AGE65+EPI+LIVER" in report.companion


def test_pruning_does_not_change_ej3s_regimen(
    ej3_data: tuple[Formulary, Mapping[str, int]],
) -> None:
    ej3_formulary, ej3_costs = ej3_data
    agents = [
        CostAwareAgent(ej3_formulary, ej3_costs, DPLLSolver(), prune=prune)
        for prune in (True, False)
    ]
    for history in list(_all_histories(ej3_formulary))[::_PRUNED_STRIDE]:
        encounter = _observed_encounter(ej3_formulary, history)
        pruned, plain = (agent.decide(encounter) for agent in agents)
        assert (pruned.action, pruned.regimen) == (plain.action, plain.regimen)


def test_a_subset_report_counts_only_its_histories(
    ej4_data: tuple[Formulary, Mapping[str, DrugCost]],
    ej3_data: tuple[Formulary, Mapping[str, int]],
) -> None:
    histories = [
        History(("GERD", "PAIN"), ("AGE65", "LIVER")),
        History(("PAIN",), ("AGE65", "EPI", "LIVER")),
    ]
    subset = continuity_with_ej3(*ej4_data, *ej3_data, prune=True, histories=histories)
    assert subset == ContinuityReport(
        histories=2,
        equal=1,
        different=(),
        companion=("PAIN|AGE65+EPI+LIVER",),
        refer=0,
        unexpected=(),
    )
