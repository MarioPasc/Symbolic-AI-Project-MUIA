"""Agreement between DPLLSolver and truth-table brute force on random small CNFs (harness H-EJ1)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from symbolic_ai.p1_ej1_logic.encoding import AxiomTag, Clause, Literal
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import oracle  # noqa: E402


def _random_cnf(rng: random.Random, num_symbols: int, num_clauses: int) -> tuple[Clause, ...]:
    symbols = [f"X{i}" for i in range(num_symbols)]
    clauses = []
    for _ in range(num_clauses):
        width = rng.randint(1, min(3, num_symbols))
        chosen = rng.sample(symbols, k=width)
        literals = tuple(Literal(symbol, positive=rng.choice([True, False])) for symbol in chosen)
        clauses.append(Clause(literals, AxiomTag.A1, "random"))
    return tuple(clauses)


@pytest.mark.parametrize("seed", range(200))
def test_dpll_agrees_with_truth_table(seed: int) -> None:
    rng = random.Random(seed)
    num_symbols = rng.randint(1, 8)
    num_clauses = rng.randint(1, 3 * num_symbols)
    clauses = _random_cnf(rng, num_symbols, num_clauses)

    expected = oracle.is_satisfiable_clauses(clauses)
    result = DPLLSolver().solve(clauses)

    assert result.satisfiable == expected
