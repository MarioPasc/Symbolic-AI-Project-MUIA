"""A brute-force, DPLL-independent SAT oracle used only to verify the solver under test.

Never imported by ``src/``: this is the harness's independent truth-table check
(``docs/HARNESSES/README.md``, H-EJ1), so it must not share any code with
:mod:`symbolic_ai.p1_ej1_logic.solver`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product

from symbolic_ai.p1_ej1_logic.encoding import Clause

RawClause = tuple[tuple[str, bool], ...]


def clause_to_raw(clause: Clause) -> RawClause:
    """Convert a :class:`Clause` into a plain ``((symbol, positive), ...)`` tuple."""
    return tuple((literal.symbol, literal.positive) for literal in clause.literals)


def _symbols_of(clauses: Iterable[RawClause]) -> set[str]:
    return {symbol for clause in clauses for symbol, _ in clause}


def _fixed_assignment(clauses: Iterable[RawClause]) -> dict[str, bool]:
    """Return the symbols pinned by some unit clause (percepts, Γ⁺/Γ⁻/Γ_u assumptions)."""
    fixed: dict[str, bool] = {}
    for clause in clauses:
        if len(clause) == 1:
            symbol, positive = clause[0]
            fixed[symbol] = positive
    return fixed


def is_satisfied(clause: RawClause, model: Mapping[str, bool]) -> bool:
    """Return whether ``clause`` is true under ``model``."""
    return any(model[symbol] == positive for symbol, positive in clause)


def count_models(clauses: Sequence[RawClause]) -> int:
    """Brute-force model count over ``clauses``: unit-clause symbols fixed, the rest enumerated.

    Parameters
    ----------
    clauses : Sequence[RawClause]
        The clauses to count models of.

    Returns
    -------
    int
        The number of assignments of the free symbols (those with no unit clause) that satisfy
        every clause, holding the unit-clause-fixed symbols at their forced value.
    """
    fixed = _fixed_assignment(clauses)
    free_symbols = sorted(_symbols_of(clauses) - fixed.keys())
    count = 0
    for combo in product((False, True), repeat=len(free_symbols)):
        model = {**fixed, **dict(zip(free_symbols, combo, strict=True))}
        if all(is_satisfied(clause, model) for clause in clauses):
            count += 1
    return count


def satisfying_models(clauses: Sequence[RawClause]) -> list[dict[str, bool]]:
    """Return every model of ``clauses`` (restricted to the free, non-unit-clause symbols)."""
    fixed = _fixed_assignment(clauses)
    free_symbols = sorted(_symbols_of(clauses) - fixed.keys())
    models = []
    for combo in product((False, True), repeat=len(free_symbols)):
        model = {**fixed, **dict(zip(free_symbols, combo, strict=True))}
        if all(is_satisfied(clause, model) for clause in clauses):
            models.append(model)
    return models


def is_satisfiable(clauses: Sequence[RawClause]) -> bool:
    """Return whether ``clauses`` has at least one model."""
    return count_models(clauses) > 0


def count_models_of_clauses(clauses: Sequence[Clause]) -> int:
    """:func:`count_models` over a sequence of :class:`Clause` objects."""
    return count_models([clause_to_raw(clause) for clause in clauses])


def is_satisfiable_clauses(clauses: Sequence[Clause]) -> bool:
    """:func:`is_satisfiable` over a sequence of :class:`Clause` objects."""
    return is_satisfiable([clause_to_raw(clause) for clause in clauses])


def models_of_clauses(clauses: Sequence[Clause], free_prefix: str) -> set[frozenset[str]]:
    """Return every model of ``clauses`` as the set of ``free_prefix``-symbols true in it.

    Parameters
    ----------
    clauses : Sequence[Clause]
        The clauses to enumerate models of.
    free_prefix : str
        Only symbols starting with this prefix (e.g. ``"T_"``) are reported per model, with the
        prefix stripped (so a model becomes a set of drug identifiers).

    Returns
    -------
    set[frozenset[str]]
        One ``frozenset`` of identifiers per model.
    """
    raw = [clause_to_raw(clause) for clause in clauses]
    result = set()
    for model in satisfying_models(raw):
        result.add(
            frozenset(
                symbol[len(free_prefix) :]
                for symbol, value in model.items()
                if value and symbol.startswith(free_prefix)
            )
        )
    return result
