"""Tests for symbolic_ai.p1_ej1_logic.experiments: P1-P3 and the untreatable patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date

import pytest

from symbolic_ai.dataloader.models import (
    Condition,
    Drug,
    DrugClass,
    Encounter,
    Formulary,
    RiskStatus,
)
from symbolic_ai.p1_ej1_logic import experiments
from symbolic_ai.p1_ej1_logic.agent import Action, Decision, PrescribingAgent
from symbolic_ai.p1_ej1_logic.encoding import gamma
from symbolic_ai.p1_ej1_logic.experiments import (
    as_jsonable,
    run_scenarios,
    run_value_ordering,
    run_verification,
    untreatable_patterns,
)
from symbolic_ai.p1_ej1_logic.semantics import RegimenSpace
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver


def _multi_indication_formulary() -> Formulary:
    """Drug ``a`` treats X and Y and shares a family with ``b``; ``c`` treats Y only."""
    return Formulary(
        version="counterexample",
        conditions=(Condition("X", "x", "x"), Condition("Y", "y", "y")),
        drugs=tuple(Drug(d, d, d) for d in ("a", "b", "c")),
        risk_factors=(),
        candidates={"X": frozenset({"a", "b"}), "Y": frozenset({"a", "c"})},
        interactions=(),
        contraindications=(),
        coprescriptions=(),
        drug_classes=(DrugClass("g", "g", "g", exclusive=True),),
        class_members={"g": frozenset({"a", "b"})},
    )


class TestScenarios:
    @pytest.fixture
    def results(
        self, formulary: Formulary, encounters: Mapping[str, Encounter]
    ) -> dict[str, experiments.ScenarioResult]:
        ordered = [encounters[k] for k in sorted(encounters)]
        return {
            r.encounter_id: r for r in run_scenarios(formulary, ordered, RegimenSpace(formulary))
        }

    @pytest.mark.parametrize(
        ("encounter_id", "action", "plus", "minus", "output", "conflict_size"),
        [
            (
                "E001",
                "prescribe",
                115,
                1645,
                ("amlodipine", "linagliptin", "mirtazapine", "tramadol", "warfarin"),
                0,
            ),
            ("E002", "request_test", 0, 94, ("PREG",), 5),
            ("E003", "prescribe", 94, 94, ("amlodipine", "warfarin"), 0),
            ("E004", "prescribe", 1, 1, ("ibuprofen", "mirtazapine", "omeprazole"), 0),
            ("E005", "refer", 0, 0, (), 8),
        ],
    )
    def test_reference_row(
        self,
        results: dict[str, experiments.ScenarioResult],
        encounter_id: str,
        action: str,
        plus: int,
        minus: int,
        output: tuple[str, ...],
        conflict_size: int,
    ) -> None:
        row = results[encounter_id]
        assert row.action == action
        assert (row.gamma_plus_models, row.gamma_minus_models) == (plus, minus)
        assert (row.regimen or row.tests) == output
        assert len(row.conflict) == conflict_size

    def test_true_first_adds_losartan_on_e001(
        self, results: dict[str, experiments.ScenarioResult]
    ) -> None:
        row = results["E001"]
        assert row.minimum_regimen_size == 5
        assert row.regimen_true_first == (
            "amlodipine",
            "apixaban",
            "linagliptin",
            "losartan",
            "mirtazapine",
            "paracetamol",
        )

    def test_results_serialise_to_json(
        self, results: dict[str, experiments.ScenarioResult]
    ) -> None:
        text = json.dumps(as_jsonable(tuple(results.values())), sort_keys=True)
        assert json.loads(text)[0]["encounter_id"] == "E001"


class TestVerification:
    def test_agent_agrees_with_oracle_on_every_mini_record(
        self, elena_formulary: Formulary
    ) -> None:
        result = run_verification(elena_formulary, RegimenSpace(elena_formulary))
        assert result.n_records == (2**3 - 1) * 3**2
        assert result.disagreements == ()
        assert sum(result.actions.values()) == result.n_records

    def test_a_wrong_decision_is_reported(
        self, elena_formulary: Formulary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ReferringAgent(PrescribingAgent):
            def decide(self, encounter: Encounter) -> Decision:
                decision = super().decide(encounter)
                return Decision(Action.REFER, trace=decision.trace)

        monkeypatch.setattr(experiments, "PrescribingAgent", ReferringAgent)
        result = run_verification(elena_formulary, RegimenSpace(elena_formulary))
        checks = {d.check for d in result.disagreements}
        assert {"C2", "C4"} <= checks

    def test_workers_below_one_raise(self, elena_formulary: Formulary) -> None:
        with pytest.raises(ValueError, match="workers"):
            run_verification(elena_formulary, RegimenSpace(elena_formulary), workers=0)


class TestValueOrdering:
    def test_result_does_not_depend_on_workers(self, elena_formulary: Formulary) -> None:
        space = RegimenSpace(elena_formulary)
        sequential = run_value_ordering(elena_formulary, space, workers=1)
        parallel = run_value_ordering(elena_formulary, space, workers=2)
        assert as_jsonable(sequential) == as_jsonable(parallel)
        assert sequential.verdicts_agree_with_oracle

    def test_false_first_is_not_minimal_in_general(self) -> None:
        """Counterexample of results-design.md §3-P3: false-first returns {b, c}, {a} suffices."""
        formulary = _multi_indication_formulary()
        result = run_value_ordering(formulary, RegimenSpace(formulary))
        both = next(i for i in result.instances if i.conditions == ("X", "Y"))
        assert both.minimum_size == 1
        assert both.size_false_first == 2
        assert result.orderings[0].n_minimal < result.orderings[0].n_satisfiable

    @pytest.mark.slow
    def test_headline_numbers_of_the_report(self, formulary: Formulary) -> None:
        result = run_value_ordering(formulary, RegimenSpace(formulary), workers=4)
        false_first, true_first = result.orderings
        assert (result.n_instances, result.n_satisfiable) == (4032, 2752)
        assert false_first.n_minimal == 2752
        assert false_first.satisfiable_with_backtracking == 0
        assert false_first.unsatisfiable_max_decisions == 0
        assert true_first.n_minimal == 1928
        assert true_first.excess_histogram == {"0": 1928, "1": 448, "2": 256, "3": 32, "4": 88}


def test_minimality_of_false_first_depends_on_the_symbol_order(formulary: Formulary) -> None:
    """Pain at 65 or older: alphabetical order drops ibuprofen first, the reverse order keeps it."""
    statuses = {
        r: RiskStatus.PRESENT if r == "AGE65" else RiskStatus.ABSENT
        for r in formulary.risk_factor_ids
    }
    encounter = Encounter("X", "P", 1, date(2026, 1, 1), 70, frozenset({"PAIN"}), statuses)
    clauses = gamma(formulary, encounter)

    def regimen(solver: DPLLSolver) -> set[str]:
        model = solver.solve(clauses).model
        assert model is not None
        return {s[2:] for s, value in model.items() if value and s.startswith("T_")}

    assert regimen(DPLLSolver()) == {"tramadol"}
    assert regimen(DPLLSolver(reverse_symbol_order=True)) == {"ibuprofen", "omeprazole"}


def test_untreatable_patterns_of_the_formulary(formulary: Formulary) -> None:
    result = untreatable_patterns(formulary, RegimenSpace(formulary))
    assert (result.n_records, result.n_untreatable) == (4032, 1280)
    assert result.upward_closed
    assert [(p.conditions, p.present_risk_factors) for p in result.patterns] == [
        (("AF",), ("PREG",)),
        (("AF", "PAIN"), ("EPI", "LIVER")),
        (("PAIN",), ("CKD", "EPI", "LIVER")),
    ]


def test_as_jsonable_rejects_other_objects() -> None:
    with pytest.raises(TypeError, match="cannot serialise"):
        as_jsonable(date(2026, 1, 1))
