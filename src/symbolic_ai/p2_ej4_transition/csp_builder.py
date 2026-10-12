"""Build the time-indexed CSP of EJ4 for one horizon, as an ``aima.csp.CSP`` (design §4).

A variable ``(condition_id, day)`` says what is given for a condition on a day ``0..H``; its values
are those of the frozen contract (``problem.Value``). Node consistency is applied when the domains
are built (AIMA 4th ed. §5.2.1): the start regimen on day 0, no newly contraindicated drug on the
days in between, an acceptable drug at full dose on day ``H``. Every other rule is one of two
binary constraint families:

- **T** (transition), between ``(c, t-1)`` and ``(c, t)``: keep the drug, start or continue its
  taper, switch to another drug only after the whole taper, or start a first drug for a newly
  diagnosed condition; a switch or a start goes to an acceptable drug only (design v2.1). A
  treated condition is never left untreated.
- **I** (interaction with presence), between any ``(c, t)`` and ``(c', t')``, also ``c = c'``:
  drugs ``a`` and ``b`` that interact may not be given on days ``t`` and ``t'`` with
  ``t' - t`` in ``[-w_b, w_a]``, because a drug given on day ``t`` stays in the body until day
  ``t + w_a`` (inclusive).

The CSP is handed to the solvers of the vendored aima-python (``src/aima/csp.py``, commit
f104e03, MIT), which take the constraints as one function ``f(A, a, B, b)`` and a neighbour list per
variable. Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §4.2 and §4.3 (private
docs).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from aima import csp as aima_csp
from symbolic_ai.p2_ej4_transition.errors import ModelError
from symbolic_ai.p2_ej4_transition.problem import (
    Day,
    TransitionProblem,
    Value,
    drug_of,
    is_full_dose,
)

__all__ = [
    "Constraint",
    "Domains",
    "Variable",
    "build_csp",
    "condition_values",
    "constraint_function",
    "interaction_forbids",
    "node_consistent_domains",
    "transition_allowed",
]

#: A variable of the CSP: what is given for ``condition_id`` on ``day``.
Variable: TypeAlias = tuple[str, Day]

#: The domain of every variable, in static order (by day, then condition).
Domains: TypeAlias = dict[Variable, tuple[Value, ...]]

#: The constraint function aima expects: ``f(A, a, B, b)`` is true when ``A = a`` and ``B = b``
#: satisfy every constraint between the two variables.
Constraint: TypeAlias = Callable[[Variable, Value, Variable, Value], bool]


def condition_values(problem: TransitionProblem, condition: str) -> tuple[Value, ...]:
    """Return every value of a condition, ``V_c``, in keep-first order (design §4.2).

    Keep-first order is the start drug at full dose and then its taper days, then every other
    candidate in alphabetical order, each at full dose and then its taper days. A newly diagnosed
    condition (no start drug) has the untreated value ``None`` first.

    Parameters
    ----------
    problem
        The transition problem.
    condition
        One of ``problem.conditions``.

    Returns
    -------
    tuple of Value
        The values ``(drug, 0)``, ``(drug, k)`` for ``1 <= k <= taper_days[drug]``, and ``None``
        when the condition starts untreated.
    """
    start = problem.start[condition]
    others = [drug for drug in problem.candidates[condition] if drug != start]
    drugs = others if start is None else [start, *others]
    values: list[Value] = [None] if start is None else []
    for drug in drugs:
        values.extend((drug, phase) for phase in range(problem.taper_days[drug] + 1))
    return tuple(values)


def node_consistent_domains(problem: TransitionProblem, horizon: int) -> Domains:
    """Return the domain of every variable after node consistency (design §4.2).

    Day 0 holds the start value (the start drug at full dose, or ``None``). A day ``0 < t < H``
    holds every value whose drug is acceptable or is the start drug (which may stay while it is
    withdrawn), plus ``None`` for a newly diagnosed condition. Day ``H`` holds the acceptable drugs
    at full dose; with ``H = 0`` that is intersected with the start value.

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H``, the last day of the transition (``>= 0``).

    Returns
    -------
    Domains
        The domain of each variable ``(condition, day)``, keyed in static order (by day, then
        condition); each domain in keep-first order (see :func:`condition_values`). A domain may be
        empty (e.g. on day ``H = 0`` when the start drug is no longer acceptable).

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    if horizon < 0:
        raise ModelError(f"the horizon must be >= 0, got {horizon}")
    return {
        (condition, day): _domain(problem, condition, day, horizon)
        for day in range(horizon + 1)
        for condition in problem.conditions
    }


def _domain(
    problem: TransitionProblem, condition: str, day: Day, horizon: int
) -> tuple[Value, ...]:
    """Return the node-consistent domain of ``(condition, day)`` (see node_consistent_domains)."""
    start = problem.start[condition]
    acceptable = problem.acceptable[condition]
    values: tuple[Value, ...]
    if day == 0:
        values = (None,) if start is None else ((start, 0),)
    else:
        # ``None`` is in ``allowed`` exactly when the condition starts untreated, so the untreated
        # value survives only for the newly diagnosed condition.
        allowed = {start, *acceptable}
        values = tuple(v for v in condition_values(problem, condition) if drug_of(v) in allowed)
    if day == horizon:
        values = tuple(v for v in values if is_full_dose(v) and drug_of(v) in acceptable)
    return values


def transition_allowed(
    problem: TransitionProblem, condition: str, before: Value, after: Value
) -> bool:
    """Return whether constraint T allows ``before`` on day ``t - 1`` and ``after`` on day ``t``.

    The allowed pairs (design §4.3, v2.1), for drugs ``d`` and ``e != d`` with ``e`` acceptable:
    ``((d, 0), (d, 0))`` keep; ``((d, 0), (d, 1))`` start the taper (``taper_days[d] >= 1``);
    ``((d, k), (d, k + 1))`` continue it; ``((d, taper_days[d]), (e, 0))`` switch after the whole
    taper (with ``taper_days[d] = 0`` that is a switch from full dose); ``(None, None)`` wait and
    ``(None, (e, 0))`` start a first drug. Nothing else: a treated condition is never left
    untreated, a taper is never aborted, no drug starts at a taper day, and a drug that is no
    longer acceptable (the old one, kept or tapered while it is withdrawn) is never started again.

    Parameters
    ----------
    problem
        The transition problem (taper days and acceptable drugs).
    condition
        The condition both values belong to.
    before
        The value of the condition on day ``t - 1``.
    after
        The value of the same condition on day ``t``.

    Returns
    -------
    bool
        ``True`` if the pair is allowed.
    """
    if after is None:
        return before is None
    if before is not None:
        drug, phase = before
        if after == before and phase == 0:
            return True
        if after == (drug, phase + 1) and phase < problem.taper_days[drug]:
            return True
        if after[0] == drug or phase != problem.taper_days[drug]:
            return False
    # What remains is a start: a first drug, or a switch after the whole taper of the old one.
    return is_full_dose(after) and after[0] in problem.acceptable[condition]


def interaction_forbids(
    problem: TransitionProblem,
    drug_a: str | None,
    day_a: Day,
    drug_b: str | None,
    day_b: Day,
) -> bool:
    """Return whether constraint I forbids ``drug_a`` on ``day_a`` with ``drug_b`` on ``day_b``.

    Two drugs that interact may not be present together: ``drug_a`` given on ``day_a`` is present
    until ``day_a + w_a``, so the pair is forbidden when ``day_b - day_a`` lies in
    ``[-w_b, w_a]`` (design §4.3). The test is symmetric in its two (drug, day) arguments.

    Parameters
    ----------
    problem
        The transition problem (interactions and washout days).
    drug_a, drug_b
        The drugs given (``None``: nothing given).
    day_a, day_b
        The days on which they are given.

    Returns
    -------
    bool
        ``True`` if the two drugs interact and their presence overlaps.
    """
    if drug_a is None or drug_b is None:
        return False
    if frozenset((drug_a, drug_b)) not in problem.interactions:
        return False
    return -problem.washout_days[drug_b] <= day_b - day_a <= problem.washout_days[drug_a]


def constraint_function(problem: TransitionProblem) -> Constraint:
    """Return the constraint function of the CSP: the conjunction of T and I between two variables.

    aima-python calls it as ``f(A, a, B, b)`` for neighbours ``A`` and ``B`` (``csp.py`` l.73) and
    requires it to be symmetric, which it is: T is checked in day order whichever variable comes
    first, and I is symmetric.

    Parameters
    ----------
    problem
        The transition problem.

    Returns
    -------
    Constraint
        A function that is ``True`` when ``A = a`` and ``B = b`` satisfy T (if ``A`` and ``B`` are
        the same condition on consecutive days) and I.
    """

    def constraints(
        first: Variable, first_value: Value, second: Variable, second_value: Value
    ) -> bool:
        (condition_1, day_1), (condition_2, day_2) = first, second
        if condition_1 == condition_2 and abs(day_1 - day_2) == 1:
            # T reads in day order, whichever of the two variables aima passes first.
            before, after = (first_value, second_value)
            if day_1 > day_2:
                before, after = after, before
            if not transition_allowed(problem, condition_1, before, after):
                return False
        return not interaction_forbids(
            problem, drug_of(first_value), day_1, drug_of(second_value), day_2
        )

    return constraints


def build_csp(problem: TransitionProblem, horizon: int) -> aima_csp.CSP:
    """Return the CSP of the transition with horizon ``H``, as an ``aima.csp.CSP``.

    Variables are listed in static order (by day, then condition) and each domain in keep-first
    order (:func:`condition_values`). Two variables are neighbours when T or I can relate them: the
    same condition on consecutive days, or a pair of drugs in their domains that interact within
    the washout window. Neighbour lists follow the static order too, so every run visits them in
    the same order.

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H``, the last day of the transition (``>= 0``).

    Returns
    -------
    aima.csp.CSP
        The CSP, with the node-consistent domains of :func:`node_consistent_domains` and the
        constraint function of :func:`constraint_function`.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    domains = node_consistent_domains(problem, horizon)
    variables = list(domains)
    drugs = {v: sorted({d for d in map(drug_of, domains[v]) if d is not None}) for v in variables}
    neighbours = {
        a: [b for b in variables if b != a and _related(problem, a, b, drugs)] for a in variables
    }
    return aima_csp.CSP(
        variables,
        {v: list(values) for v, values in domains.items()},
        neighbours,
        constraint_function(problem),
    )


def _related(
    problem: TransitionProblem, first: Variable, second: Variable, drugs: dict[Variable, list[str]]
) -> bool:
    """Return whether T or I can relate two distinct variables (whether they are neighbours)."""
    (condition_1, day_1), (condition_2, day_2) = first, second
    if condition_1 == condition_2 and abs(day_1 - day_2) == 1:
        return True
    return any(
        interaction_forbids(problem, drug_1, day_1, drug_2, day_2)
        for drug_1 in drugs[first]
        for drug_2 in drugs[second]
    )
