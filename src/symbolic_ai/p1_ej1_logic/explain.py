"""Explain unsatisfiable answers with a minimal unsatisfiable subset of the clauses (deletion).

When Γ⁺ or Γ⁻ is unsatisfiable the agent requests a test or refers the patient; the verdict alone
does not say why. A minimal unsatisfiable subset (MUS) is a set of clauses that is unsatisfiable
while every proper subset is satisfiable: the axiom instances and percepts that conflict. It is
extracted by deletion (J. Marques-Silva, "Minimal unsatisfiability: models, algorithms and
applications", ISMVL 2010): try to drop each clause in turn and keep it dropped if the rest is still
unsatisfiable. Each test is one call to the same complete solver the agent uses.
"""

from __future__ import annotations

from collections.abc import Sequence

from symbolic_ai.p1_ej1_logic.encoding import Clause
from symbolic_ai.p1_ej1_logic.errors import ExplanationError
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver

__all__ = ["minimal_unsatisfiable_subset"]


def minimal_unsatisfiable_subset(
    clauses: Sequence[Clause], solver: DPLLSolver
) -> tuple[Clause, ...]:
    """Return a minimal unsatisfiable subset of ``clauses`` (deletion-based extraction).

    The result is unsatisfiable, and removing any one of its clauses makes it satisfiable: when a
    clause was kept, the remaining set without it was satisfiable, and later deletions only shrink
    that set, so it stays satisfiable. The subset found depends on the order of ``clauses``; it is
    minimal (no clause can be dropped), not necessarily the smallest one.

    Parameters
    ----------
    clauses : Sequence[Clause]
        An unsatisfiable set of clauses, e.g. Γ⁺ or Γ⁻ of a patient the agent cannot treat.
    solver : DPLLSolver
        The complete solver used for every satisfiability test (``len(clauses) + 1`` calls).

    Returns
    -------
    tuple[Clause, ...]
        The kept clauses, in their original order.

    Raises
    ------
    ExplanationError
        If ``clauses`` is satisfiable (there is no conflict to explain).
    """
    if solver.solve(clauses).satisfiable:
        raise ExplanationError(f"the {len(clauses)} clauses are satisfiable: nothing to explain")
    kept = list(range(len(clauses)))
    for index in range(len(clauses)):
        trial = [k for k in kept if k != index]
        if not solver.solve([clauses[k] for k in trial]).satisfiable:
            kept = trial
    return tuple(clauses[k] for k in kept)
