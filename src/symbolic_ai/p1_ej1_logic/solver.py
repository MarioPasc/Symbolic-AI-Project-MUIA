"""A DPLL-backed SAT solver wrapping ``aima.logic.dpll`` directly (never ``dpll_satisfiable``).

``dpll_satisfiable`` collects its symbols from ``prop_symbols(sentence)``, a Python ``set``, so
its branching order depends on ``Expr.__hash__`` and therefore on hash randomisation
(``PYTHONHASHSEED``): the same encounter could yield a different prescription on every run. This
module instead calls ``aima.logic.dpll`` with symbols sorted by name and a branching heuristic
that tries ``T_d = False`` first by default (AIMA 4th ed. §7.6.1, value ordering), which is
process-independent (``EJ1-sat/solver-choice.md`` §5, §7).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from aima import logic as aima_logic
from aima.utils import Expr
from symbolic_ai.p1_ej1_logic.encoding import Clause, clause_to_expr
from symbolic_ai.p1_ej1_logic.errors import CertificateError

BranchingHeuristic = Callable[[list[Expr], list[Expr]], tuple[Expr, bool]]

__all__ = ["DPLLSolver", "SolveResult"]


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Outcome of one call to :meth:`DPLLSolver.solve`.

    Parameters
    ----------
    satisfiable : bool
        Whether a model was found.
    model : dict[str, bool] | None
        A complete model (every symbol of the input clauses, unassigned ones set to ``False``),
        or ``None`` when unsatisfiable.
    calls : int
        Number of recursive invocations of ``aima.logic.dpll`` this solve took.
    """

    satisfiable: bool
    model: dict[str, bool] | None
    calls: int


class DPLLSolver:
    """Complete SAT solver over :class:`~symbolic_ai.p1_ej1_logic.encoding.Clause` lists.

    Parameters
    ----------
    decision_first_value : bool
        The value tried first when the branching heuristic must choose for a decision symbol
        (one whose name starts with ``decision_prefix``): ``False`` (default) reproduces the
        minimum-size regimens of ``solver-choice.md`` §5; ``True`` reproduces the textbook order
        (AIMA 4th ed. §7.6.1 item 2), which tends toward polypharmacy.
    decision_prefix : str
        Prefix identifying a decision symbol (``T_`` for drugs); every other symbol is tried
        ``True`` first, matching the pure-symbol rule's usual bias for facts already pinned by
        unit propagation.
    """

    def __init__(self, decision_first_value: bool = False, decision_prefix: str = "T_") -> None:
        self.decision_first_value = decision_first_value
        self.decision_prefix = decision_prefix

    def branching_heuristic(
        self, symbols: Sequence[Expr], clauses: Sequence[Expr]
    ) -> tuple[Expr, bool]:
        """Return ``(first(symbols), value)``: the value ordering of this solver.

        Parameters
        ----------
        symbols : Sequence[Expr]
            The remaining, still-unassigned symbols, in sorted order.
        clauses : Sequence[Expr]
            The clauses not yet known true (unused; required by ``aima.logic.dpll``'s interface).

        Returns
        -------
        tuple[Expr, bool]
            The first remaining symbol and the value to try first for it.
        """
        del clauses  # unused: aima.logic.dpll's branching_heuristic interface requires it
        symbol = symbols[0]
        if str(symbol).startswith(self.decision_prefix):
            return symbol, self.decision_first_value
        return symbol, True

    def solve(self, clauses: Sequence[Clause]) -> SolveResult:
        """Decide satisfiability of ``clauses`` and, if satisfiable, return a certified model.

        Parameters
        ----------
        clauses : Sequence[Clause]
            The clauses to solve, e.g. Γ, Γ⁺, Γ⁻ or Γ_u.

        Returns
        -------
        SolveResult
            ``satisfiable=False, model=None`` if no model exists; otherwise a complete model
            that has passed the certificate check against every input clause.

        Raises
        ------
        CertificateError
            If ``aima.logic.dpll`` returns a model that fails to satisfy some clause (should
            never happen for a sound DPLL implementation; a defensive, solver-independent check).
        """
        symbol_names = sorted(_symbols_of(clauses))
        expr_clauses = [clause_to_expr(clause) for clause in clauses]
        expr_symbols = [Expr(name) for name in symbol_names]

        raw_model, calls = _run_counted_dpll(expr_clauses, expr_symbols, self.branching_heuristic)
        if raw_model is None:
            return SolveResult(satisfiable=False, model=None, calls=calls)

        model = _complete_model(raw_model, symbol_names)
        _certify(model, clauses)
        return SolveResult(satisfiable=True, model=model, calls=calls)


def _symbols_of(clauses: Sequence[Clause]) -> set[str]:
    """Collect every literal symbol name appearing in ``clauses``."""
    return {literal.symbol for clause in clauses for literal in clause.literals}


def _run_counted_dpll(
    clauses: list[Expr], symbols: list[Expr], heuristic: BranchingHeuristic
) -> tuple[dict[Expr, bool] | None, int]:
    """Call ``aima.logic.dpll``, counting every recursive invocation.

    ``dpll`` recurses through the module-level name ``aima.logic.dpll``, so replacing that
    attribute with a counting wrapper for the duration of one solve counts every recursive call,
    including nested ones; the original is always restored, even on an exception.

    Parameters
    ----------
    clauses : list[Expr]
        The clauses to solve, as aima expressions.
    symbols : list[Expr]
        The symbols to branch over, sorted by name.
    heuristic : BranchingHeuristic
        The branching heuristic to pass to ``aima.logic.dpll``.

    Returns
    -------
    tuple[dict[Expr, bool] | None, int]
        The raw (possibly partial) model, or ``None`` if unsatisfiable, and the call count.
    """
    counter = {"calls": 0}
    original = aima_logic.dpll

    def counting_dpll(
        clauses: list[Expr],
        symbols: list[Expr],
        model: dict[Expr, bool],
        branching_heuristic: BranchingHeuristic = heuristic,
    ) -> dict[Expr, bool] | bool:
        counter["calls"] += 1
        return cast(
            "dict[Expr, bool] | bool", original(clauses, symbols, model, branching_heuristic)
        )

    aima_logic.dpll = counting_dpll
    try:
        result = counting_dpll(clauses, symbols, {}, heuristic)
    finally:
        aima_logic.dpll = original

    if result is False:
        return None, counter["calls"]
    return cast(dict[Expr, bool], result), counter["calls"]


def _complete_model(raw_model: dict[Expr, bool], symbol_names: list[str]) -> dict[str, bool]:
    """Complete a (possibly partial) model: unassigned symbols count as ``False``."""
    by_name = {str(symbol): value for symbol, value in raw_model.items()}
    return {name: by_name.get(name, False) for name in symbol_names}


def _certify(model: dict[str, bool], clauses: Sequence[Clause]) -> None:
    """Check ``model`` against every clause; raise :class:`CertificateError` if any fails."""
    unsatisfied = tuple(
        f"{clause.axiom}:{clause.source} ({clause})"
        for clause in clauses
        if not any(model[literal.symbol] == literal.positive for literal in clause.literals)
    )
    if unsatisfied:
        raise CertificateError(unsatisfied)
