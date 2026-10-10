"""Search algorithms of EJ3, run on a :class:`RegimenProblem` and reported with their effort.

The algorithms are those of the vendored aima-python (``aima.search``, called unchanged): this
module chooses the evaluation function of each, counts what the search asks of the problem with
aima's own ``InstrumentedProblem``, and turns the goal node into a typed result. A*, greedy
best-first and uniform-cost search are best-first graph search with f = g + h, f = h and f = g
(AIMA 4th ed. Fig. 3.7, §3.5.2, §3.5.1 and §3.4.2); IDA* is iterative deepening on the f-cost
(§3.5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from aima import search as aima_search
from symbolic_ai.p1_ej3_search.problem import RegimenProblem, State

__all__ = ["Algorithm", "SearchResult", "search"]


class Algorithm(StrEnum):
    """The search algorithms the agent can run."""

    ASTAR = "astar"
    IDASTAR = "idastar"
    GREEDY = "greedy"
    UNIFORM_COST = "uniform_cost"

    @property
    def optimal(self) -> bool:
        """Whether the algorithm returns a cheapest regimen (all but greedy best-first)."""
        return self is not Algorithm.GREEDY


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of one search.

    Parameters
    ----------
    algorithm : Algorithm
        The algorithm that ran.
    regimen : frozenset[str] | None
        The goal state reached, or ``None`` when the problem has no regimen.
    cost : int | None
        The cost of ``regimen`` in euro cents per month, or ``None`` without one.
    chosen : tuple[str, ...]
        The actions of the solution path: the drug chosen for each condition, in order
        (companions are added by the transition model and are not listed).
    expanded : int
        Nodes expanded (calls to ``actions``). IDA* counts a node once per iteration.
    generated : int
        Successor nodes generated (calls to ``result``).
    goal_tests : int
        Goal tests applied.
    """

    algorithm: Algorithm
    regimen: frozenset[str] | None
    cost: int | None
    chosen: tuple[str, ...]
    expanded: int
    generated: int
    goal_tests: int


def search(problem: RegimenProblem, algorithm: Algorithm = Algorithm.ASTAR) -> SearchResult:
    """Run ``algorithm`` on ``problem`` and return the regimen found with the effort it took.

    Ties between nodes of equal f are broken by insertion order, and actions are generated in
    alphabetical order, so the result is the same on every run.

    Parameters
    ----------
    problem : RegimenProblem
        The search problem of one record.
    algorithm : Algorithm
        The algorithm to run (default: A*).

    Returns
    -------
    SearchResult
        The goal state and its cost (``None`` if the search space has no goal), the solution
        path and the effort counts.
    """
    counted = aima_search.InstrumentedProblem(problem)
    node = _goal_node(algorithm, counted)
    effort = {
        "expanded": cast("int", counted.succs),
        "generated": cast("int", counted.states),
        "goal_tests": cast("int", counted.goal_tests),
    }
    if node is None:
        return SearchResult(algorithm, regimen=None, cost=None, chosen=(), **effort)
    return SearchResult(
        algorithm,
        regimen=cast("State", node.state),
        cost=cast("int", node.path_cost),
        chosen=tuple(cast("list[str]", node.solution())),
        **effort,
    )


def _goal_node(algorithm: Algorithm, problem: Any) -> Any:
    """Call the ``aima.search`` function of ``algorithm``; return its goal ``Node`` or ``None``.

    ``problem`` and the node are ``aima.search`` objects, which are untyped: this is the one
    place where they are handled as such.
    """
    if algorithm is Algorithm.ASTAR:
        return aima_search.astar_search(problem)
    if algorithm is Algorithm.IDASTAR:
        return aima_search.iterative_deepening_astar_search(problem)
    if algorithm is Algorithm.GREEDY:
        return aima_search.best_first_graph_search(problem, problem.h)
    return aima_search.uniform_cost_search(problem)
