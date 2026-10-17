"""Independent oracle of EJ4: the minimum horizon H* by breadth-first search over daily states.

The oracle does not build the CSP. It treats the transition as a search problem in the sense of
AIMA (4th ed. §3.1): a state is one *day* (what each condition gets today and how many more days
each drug in the body stays there), an action is what to give tomorrow, and the goal is a safe
regimen. Every action costs one day, so breadth-first search (AIMA 4th ed. §3.4.1, Fig. 3.9; 3rd
ed. Fig. 3.11) reaches a goal in the fewest days: the depth of the first goal found is H*. If the
reachable states run out without a goal, no transition exists at any horizon. The rules come from
the design in words (§1, §4.2, §4.3 "equivalent reading" of I), never from the CSP model (§6.2).
The search problem has aima's ``Problem`` interface; the search loop is the oracle's own.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

from symbolic_ai.p2_ej4_transition.errors import VerificationError
from symbolic_ai.p2_ej4_transition.problem import (
    Day,
    TransitionProblem,
    Value,
    drug_of,
    is_full_dose,
)

__all__ = [
    "H_MAX",
    "MAX_STATES",
    "DailyState",
    "DailyStateProblem",
    "OracleResult",
    "OracleStatus",
    "oracle_min_horizon",
]

#: ``found``: H* is known; ``impossible``: every reachable state was seen and none is a goal;
#: ``budget_exhausted``: the search stopped at ``max_states`` expansions or at depth ``h_max``.
OracleStatus: TypeAlias = Literal["found", "impossible", "budget_exhausted"]

#: The values given to every condition on one day, in ``problem.conditions`` order.
DayValues: TypeAlias = tuple[Value, ...]

#: The deepest day the oracle searches by default: the CSP's cap of design §4.4, kept here as a
#: number of its own so that the oracle does not import the CSP side.
H_MAX = 60
#: The most daily states the oracle expands by default (no real or synthetic problem of R2 reaches
#: it: ``budget_exhausted`` is 0 in ``results.json``).
MAX_STATES = 200_000


class DailyState(NamedTuple):
    """One day of the transition, as a hashable search state.

    Parameters
    ----------
    values
        What each condition gets today, in ``problem.conditions`` order.
    days_left
        Every drug in the body today with the number of days after today on which it is still
        present, as sorted ``(drug_id, days)`` pairs. A drug given today has its washout ``w_d``;
        each day every count goes down by one. Only counts ``>= 1`` are listed, so the listed
        drugs are exactly those still present tomorrow.
    """

    values: DayValues
    days_left: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class OracleResult:
    """The answer of the oracle for one problem.

    Parameters
    ----------
    h_star
        The minimum horizon, or ``None`` unless ``status`` is ``found``.
    status
        ``found``, ``impossible`` or ``budget_exhausted`` (see :data:`OracleStatus`).
    states_expanded
        Daily states whose successors were generated.
    schedule
        A witness schedule for days ``0..h_star`` keyed ``(condition_id, day)``, or ``None``.
    """

    h_star: int | None
    status: OracleStatus
    states_expanded: int
    schedule: dict[tuple[str, Day], Value] | None


class DailyStateProblem:
    """The transition as a search problem over daily states (AIMA 4th ed. §3.1).

    It implements the interface of ``aima.search.Problem`` (``initial``, ``actions``,
    ``result``, ``goal_test``, ``path_cost``) without subclassing it, because the vendored copy is
    untyped. :func:`oracle_min_horizon` searches it with its own budgeted loop; aima's
    ``breadth_first_graph_search`` also accepts it unchanged, which the tests use to check the
    depth of the goals found.

    - Initial state: day 0, the regimen before the event; nothing else is in the body.
    - Actions: the values of every condition tomorrow allowed by the transition and domain rules
      for each condition, whose drugs do not interact with what is present tomorrow.
    - Result: tomorrow's state; every count of days left goes down by one, and each drug given
      tomorrow is set to its washout.
    - Goal test: every condition on an acceptable drug at full dose.
    - Path cost: one per day.

    Parameters
    ----------
    problem
        The transition problem.
    """

    def __init__(self, problem: TransitionProblem) -> None:
        self.problem = problem
        starts = [problem.start[condition] for condition in problem.conditions]
        day0 = tuple(None if drug is None else (drug, 0) for drug in starts)
        self.initial = DailyState(day0, self._count_down((), day0))

    def actions(self, state: DailyState) -> list[DayValues]:
        """Return tomorrow's allowed values for every condition, in a fixed order."""
        options = [
            self._next_values(condition, value)
            for condition, value in zip(self.problem.conditions, state.values, strict=True)
        ]
        still_present = [drug for drug, _ in state.days_left]
        return [v for v in itertools.product(*options) if self.is_safe(v, still_present)]

    def result(self, state: DailyState, action: DayValues) -> DailyState:
        """Return tomorrow's state when ``action`` is given tomorrow."""
        return DailyState(action, self._count_down(state.days_left, action))

    def goal_test(self, state: DailyState) -> bool:
        """Rule *final*: every condition on an acceptable drug at full dose."""
        return all(
            is_full_dose(value) and drug_of(value) in self.problem.acceptable[condition]
            for condition, value in zip(self.problem.conditions, state.values, strict=True)
        )

    def path_cost(self, cost: int, state: DailyState, action: DayValues, next_: DailyState) -> int:
        """Every action is one day."""
        return cost + 1

    def _next_values(self, condition: str, value: Value) -> list[Value]:
        """Values of ``condition`` tomorrow: rule *transition*, then rule *domain*.

        Transition (design v2.1 §4.3, T): a full-dose drug is kept, tapered (phase 1) if it has a
        taper, or switched to another candidate at full dose if it has none; a taper advances one
        phase a day and, after its last phase, switches to another candidate at full dose; an
        untreated new condition stays untreated or starts a candidate at full dose. The target of
        a switch or a start must be acceptable. Domain (§4.2, days ``0 < t < H``): a drug other
        than the start drug must be acceptable (implied here by the transition rule; kept as the
        rule's own filter).
        """
        problem = self.problem
        others = [(d, 0) for d in problem.acceptable[condition] if d != drug_of(value)]
        nexts: list[Value]
        if value is None:
            nexts = [None, *others]
        else:
            drug, phase = value
            taper = problem.taper_days[drug]
            if phase == 0:
                nexts = [value, (drug, 1)] if taper >= 1 else [value, *others]
            else:
                nexts = [(drug, phase + 1)] if phase < taper else list(others)
        start = problem.start[condition]
        allowed = set(problem.acceptable[condition]) | {start}
        return [v for v in nexts if v is None or v[0] in allowed]

    def _count_down(
        self, days_left: tuple[tuple[str, int], ...], values: DayValues
    ) -> tuple[tuple[str, int], ...]:
        """Move the counts of days left one day forward, to the day on which ``values`` is given.

        Every count goes down by one; each drug given that day is then set to its washout ``w_d``
        (last given on day ``t``, a drug is present through day ``t + w_d`` inclusive). Counts
        that reach zero are dropped.
        """
        left = {drug: days - 1 for drug, days in days_left}
        for drug in map(drug_of, values):
            if drug is not None:
                left[drug] = self.problem.washout_days[drug]
        return tuple(sorted((drug, days) for drug, days in left.items() if days >= 1))

    def is_safe(self, values: DayValues, still_present: Collection[str]) -> bool:
        """Rule *interaction* on a day: no drug given that day interacts with a drug present.

        Parameters
        ----------
        values
            What each condition gets that day.
        still_present
            The drugs given on earlier days that are still in the body that day (none on day 0:
            nothing but the regimen before the event is in the body).

        Returns
        -------
        bool
            Whether no drug given that day interacts with a drug present that day: one given
            that day or one still present from earlier days.
        """
        given = {drug for drug in map(drug_of, values) if drug is not None}
        present = given | set(still_present)
        return not any(
            a != b and frozenset((a, b)) in self.problem.interactions
            for a in given
            for b in present
        )


def oracle_min_horizon(
    problem: TransitionProblem, h_max: int = H_MAX, max_states: int = MAX_STATES
) -> OracleResult:
    """Return the minimum horizon H* of ``problem`` by breadth-first search over daily states.

    Breadth-first graph search with the goal test applied when a state is generated (AIMA 4th ed.
    §3.4.1, Fig. 3.9; ``aima.search.breadth_first_graph_search``, 3rd ed. Fig. 3.11): the first goal
    generated is at the smallest depth, and depth is days, so it is H*. This is the oracle's own
    loop, written after aima's function and not a call to it, because aima's has no budget: it
    can neither stop at ``max_states`` expansions or at depth ``h_max`` nor tell an exhausted
    search from a cut one. It differs from aima's in two ways: it stops at the budget, and the
    states already reached are kept in a dictionary (``state -> parent``), so a membership test
    is constant-time instead of a scan of the frontier. Aima's function accepts
    :class:`DailyStateProblem` unchanged, and the tests check that it finds goals at the same
    depth. States are expanded and actions generated in a fixed order, so the result and the
    witness are deterministic.

    Parameters
    ----------
    problem
        The transition problem.
    h_max
        The deepest day searched: a goal at depth ``h_max`` is found, deeper ones are not.
    max_states
        The most daily states the search may expand.

    Returns
    -------
    OracleResult
        ``found`` with H* and a witness schedule; ``impossible`` when day 0 itself breaks rule
        *interaction* or every reachable state was expanded and none is a goal;
        ``budget_exhausted`` when a state at depth ``h_max`` was left unexpanded or the
        expansion budget ran out.

    Raises
    ------
    VerificationError
        If ``h_max`` or ``max_states`` is negative.
    """
    if h_max < 0 or max_states < 0:
        raise VerificationError(f"h_max and max_states must be >= 0, got {h_max}, {max_states}")
    search = DailyStateProblem(problem)
    root = search.initial
    if not search.is_safe(root.values, ()):
        # Day 0 is fixed, and rule interaction holds on every day, day 0 included.
        return OracleResult(None, "impossible", 0, None)
    if search.goal_test(root):
        return OracleResult(0, "found", 0, _witness(problem, [root]))
    parent: dict[DailyState, DailyState | None] = {root: None}
    frontier: deque[tuple[DailyState, Day]] = deque([(root, 0)])
    expanded, cut_at_h_max = 0, False
    while frontier:
        state, day = frontier.popleft()
        if day == h_max:
            cut_at_h_max = True
            continue
        if expanded == max_states:
            return OracleResult(None, "budget_exhausted", expanded, None)
        expanded += 1
        children = _new_children(search, state, parent)
        goal = next((child for child in children if search.goal_test(child)), None)
        if goal is not None:
            path = _path_to(parent, goal)
            return OracleResult(len(path) - 1, "found", expanded, _witness(problem, path))
        frontier.extend((child, day + 1) for child in children)
    status: OracleStatus = "budget_exhausted" if cut_at_h_max else "impossible"
    return OracleResult(None, status, expanded, None)


def _new_children(
    search: DailyStateProblem,
    state: DailyState,
    parent: dict[DailyState, DailyState | None],
) -> list[DailyState]:
    """Expand ``state``: return its successors not reached before, recording ``state`` as parent."""
    children: list[DailyState] = []
    for action in search.actions(state):
        child = search.result(state, action)
        if child not in parent:
            parent[child] = state
            children.append(child)
    return children


def _path_to(parent: dict[DailyState, DailyState | None], goal: DailyState) -> list[DailyState]:
    """Return the states from the initial state to ``goal`` by following the parent links."""
    path = [goal]
    while (previous := parent[path[-1]]) is not None:
        path.append(previous)
    return path[::-1]


def _witness(problem: TransitionProblem, path: list[DailyState]) -> dict[tuple[str, Day], Value]:
    """Turn a path of daily states (day 0 first) into a schedule keyed ``(condition, day)``."""
    return {
        (condition, day): value
        for day, state in enumerate(path)
        for condition, value in zip(problem.conditions, state.values, strict=True)
    }
