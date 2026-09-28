"""Tests for symbolic_ai.p1_ej1_logic.encoding: axiom grounding, percepts, Γ builders, the lemma."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic.encoding import (
    AxiomTag,
    Clause,
    Literal,
    axiom_clauses,
    check_lemma_precondition,
    clause_to_expr,
    gamma,
    gamma_minus,
    gamma_plus,
    gamma_u,
    percept_clauses,
    symbol_for_condition,
    symbol_for_drug,
    symbol_for_risk_factor,
    unknown_risk_factors,
)
from symbolic_ai.p1_ej1_logic.errors import EncodingError, LemmaPreconditionError


def _count_by_axiom(clauses: Sequence[Clause], tag: AxiomTag) -> int:
    return sum(1 for clause in clauses if clause.axiom is tag)


class TestAxiomClauseCounts:
    """The 50 rule clauses of the full 1.0.0 formulary, by axiom (01-database.md §6)."""

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            (AxiomTag.A1, 6),
            (AxiomTag.A2, 18),
            (AxiomTag.A3, 7),
            (AxiomTag.A4, 11),
            (AxiomTag.A5, 1),
            (AxiomTag.A6, 7),
        ],
    )
    def test_axiom_clause_count(self, formulary: Formulary, tag: AxiomTag, expected: int) -> None:
        assert _count_by_axiom(axiom_clauses(formulary), tag) == expected

    def test_total_clause_count(self, formulary: Formulary) -> None:
        assert len(axiom_clauses(formulary)) == 50

    def test_elena_mini_world_rule_clause_count(self, elena_formulary: Formulary) -> None:
        assert len(axiom_clauses(elena_formulary)) == 14


class TestPerceptClauses:
    def test_closed_world_conditions(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        clauses = percept_clauses(e001, formulary)
        by_condition = {
            clause.source: clause
            for clause in clauses
            if clause.axiom is AxiomTag.PERCEPT and clause.source in formulary.condition_ids
        }
        assert len(by_condition) == len(formulary.condition_ids)
        for condition_id in formulary.condition_ids:
            literal = by_condition[condition_id].literals[0]
            assert literal.symbol == symbol_for_condition(condition_id)
            assert literal.positive is (condition_id in e001.conditions)

    def test_unknown_risk_factor_gets_no_clause(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        sources = {clause.source for clause in percept_clauses(e001, formulary)}
        assert "CKD" not in sources

    def test_present_and_absent_risk_factors(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        by_risk_factor = {
            clause.source: clause
            for clause in percept_clauses(e001, formulary)
            if clause.source in formulary.risk_factor_ids
        }
        assert by_risk_factor["AGE65"].literals[0].positive is True
        assert by_risk_factor["ASTHMA"].literals[0].positive is False

    def test_condition_outside_formulary_is_ignored_and_warns(
        self,
        formulary: Formulary,
        encounters: dict[str, Encounter],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        e001 = encounters["E001"]
        bogus = replace(e001, conditions=e001.conditions | {"NOT_A_CONDITION"})
        with caplog.at_level(logging.WARNING):
            clauses = percept_clauses(bogus, formulary)
        assert not any(clause.source == "NOT_A_CONDITION" for clause in clauses)
        assert any("NOT_A_CONDITION" in message for message in caplog.messages)

    def test_risk_factor_outside_formulary_is_ignored_silently(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        bogus = replace(
            e001, risk_factors={**e001.risk_factors, "NOT_A_RISK_FACTOR": RiskStatus.PRESENT}
        )
        clauses = percept_clauses(bogus, formulary)
        assert not any(clause.source == "NOT_A_RISK_FACTOR" for clause in clauses)


class TestUnknownRiskFactors:
    def test_e001_unknown_is_ckd(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        assert unknown_risk_factors(encounters["E001"], formulary) == frozenset({"CKD"})

    def test_e003_no_unknowns(self, formulary: Formulary, encounters: dict[str, Encounter]) -> None:
        assert unknown_risk_factors(encounters["E003"], formulary) == frozenset()


class TestGammaBuilders:
    def test_gamma_plus_assumes_unknowns_present(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        base = gamma(formulary, e001)
        unknowns = unknown_risk_factors(e001, formulary)
        assumption = next(
            clause for clause in gamma_plus(base, unknowns) if clause.axiom is AxiomTag.ASSUMPTION
        )
        assert assumption.literals == (Literal(symbol_for_risk_factor("CKD"), positive=True),)

    def test_gamma_minus_assumes_unknowns_absent(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        base = gamma(formulary, e001)
        unknowns = unknown_risk_factors(e001, formulary)
        assumption = next(
            clause for clause in gamma_minus(base, unknowns) if clause.axiom is AxiomTag.ASSUMPTION
        )
        assert assumption.literals == (Literal(symbol_for_risk_factor("CKD"), positive=False),)

    def test_gamma_u_rejects_non_member(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        base = gamma(formulary, e001)
        unknowns = unknown_risk_factors(e001, formulary)
        with pytest.raises(EncodingError):
            gamma_u(base, unknowns, "PREG")

    def test_gamma_u_negates_u_and_assumes_the_rest_present(self) -> None:
        unknowns = frozenset({"CKD", "PREG"})
        clauses = gamma_u((), unknowns, "CKD")
        literals = {clause.literals[0] for clause in clauses}
        assert Literal(symbol_for_risk_factor("CKD"), positive=False) in literals
        assert Literal(symbol_for_risk_factor("PREG"), positive=True) in literals


class TestLemmaPrecondition:
    def test_all_real_axioms_satisfy_the_precondition(
        self, formulary: Formulary, encounters: dict[str, Encounter]
    ) -> None:
        e001 = encounters["E001"]
        base = gamma(formulary, e001)
        unknowns = unknown_risk_factors(e001, formulary)
        check_lemma_precondition(base, unknowns)  # must not raise

    def test_positive_unknown_risk_factor_raises(self) -> None:
        unknowns = frozenset({"AGE65"})
        bad_clause = Clause(
            (
                Literal(symbol_for_risk_factor("AGE65"), positive=True),
                Literal(symbol_for_drug("omeprazole")),
            ),
            AxiomTag.A2,
            "omeprazole",
        )
        with pytest.raises(LemmaPreconditionError):
            check_lemma_precondition((bad_clause,), unknowns)


class TestClauseToExpr:
    def test_empty_clause_raises(self) -> None:
        with pytest.raises(EncodingError):
            clause_to_expr(Clause((), AxiomTag.A1, "x"))

    def test_negation_and_positivity(self) -> None:
        clause = Clause(
            (Literal("H_AF", positive=False), Literal("T_warfarin", positive=True)),
            AxiomTag.A1,
            "AF",
        )
        assert str(clause_to_expr(clause)) == "(~H_AF | T_warfarin)"
