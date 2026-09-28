"""Tests for symbolic_ai.p1_ej1_logic.solver.DPLLSolver."""

from __future__ import annotations

import random

from aima.utils import Expr
from symbolic_ai.p1_ej1_logic.encoding import AxiomTag, Clause, Literal
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver


def _clause(*literals: tuple[str, bool]) -> Clause:
    return Clause(
        tuple(Literal(symbol, positive) for symbol, positive in literals), AxiomTag.A1, "test"
    )


class TestSolveBasics:
    def test_satisfiable_returns_certified_model(self) -> None:
        result = DPLLSolver().solve([_clause(("A", True), ("B", True))])
        assert result.satisfiable is True
        assert result.model is not None
        assert result.model["A"] or result.model["B"]

    def test_unsatisfiable_returns_no_model(self) -> None:
        result = DPLLSolver().solve([_clause(("A", True)), _clause(("A", False))])
        assert result.satisfiable is False
        assert result.model is None

    def test_unassigned_symbols_default_to_false(self) -> None:
        result = DPLLSolver().solve([_clause(("A", True))])
        assert result.model == {"A": True}

    def test_complete_model_includes_every_clause_symbol(self) -> None:
        clauses = [_clause(("A", True)), _clause(("B", True), ("C", True))]
        result = DPLLSolver().solve(clauses)
        assert result.model is not None
        assert set(result.model) == {"A", "B", "C"}

    def test_recursive_calls_are_counted(self) -> None:
        clauses = [_clause(("A", True), ("B", True)), _clause(("A", False), ("B", False))]
        result = DPLLSolver().solve(clauses)
        assert result.calls >= 1

    def test_result_independent_of_clause_order(self) -> None:
        clauses = [
            _clause(("H_AF", False), ("T_warfarin", True), ("T_apixaban", True)),
            _clause(("T_warfarin", False), ("H_AF", True)),
            _clause(("T_apixaban", False), ("H_AF", True)),
            _clause(("T_warfarin", False), ("T_apixaban", False)),
            _clause(("H_AF", True)),
        ]
        shuffled = list(clauses)
        random.Random(0).shuffle(shuffled)

        result_a = DPLLSolver().solve(clauses)
        result_b = DPLLSolver().solve(shuffled)

        assert result_a.satisfiable == result_b.satisfiable
        assert result_a.model == result_b.model


class TestBranchingHeuristic:
    """Direct tests of the value ordering, without a full solve (02-code-architecture.md §8)."""

    def test_default_tries_decision_symbol_false_first(self) -> None:
        solver = DPLLSolver(decision_first_value=False)
        assert solver.branching_heuristic([Expr("T_warfarin")], []) == (Expr("T_warfarin"), False)

    def test_true_first_flag_changes_the_heuristic(self) -> None:
        solver = DPLLSolver(decision_first_value=True)
        assert solver.branching_heuristic([Expr("T_warfarin")], []) == (Expr("T_warfarin"), True)

    def test_non_decision_symbols_always_try_true_first(self) -> None:
        for decision_first_value in (False, True):
            solver = DPLLSolver(decision_first_value=decision_first_value)
            assert solver.branching_heuristic([Expr("H_AF")], []) == (Expr("H_AF"), True)

    def test_custom_decision_prefix(self) -> None:
        solver = DPLLSolver(decision_first_value=False, decision_prefix="X_")
        assert solver.branching_heuristic([Expr("X_foo")], [])[1] is False
        assert solver.branching_heuristic([Expr("T_foo")], [])[1] is True
