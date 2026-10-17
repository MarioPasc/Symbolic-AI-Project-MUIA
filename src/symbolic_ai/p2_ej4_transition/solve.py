"""Solve the transition CSP of EJ4 with aima-python's own algorithms, configured, not rewritten.

Design decision D22: every algorithm comes from the vendored ``src/aima/csp.py`` (aima-python
commit f104e03, MIT); this module only chooses and wraps them (design §6.1):

- consistency before search: AC-3 (AIMA 4th ed. §5.2.2 "Arc consistency"; ``csp.py``
  l.200-238; 3rd ed. Fig. 6.3);
- search: chronological backtracking (§5.3 "Backtracking search for CSPs"; l.449-475; 3rd ed.
  Fig. 6.5) with no inference, forward checking or MAC (§5.3.2 "Interleaving search and
  inference"; l.423-443), a static or MRV variable order and a keep-first or LCV value order
  (§5.3.1 "Variable and value ordering"; l.386-420);
- local search: min-conflicts (§5.4 "Local search for CSPs"; l.478-503; 3rd ed. Fig. 6.8), which
  may find a solution but can never prove that there is none.

The wrappers add, without editing aima: deterministic tie-breaks (MRV and AC-3's arc order follow
the static variable order instead of chance or hash order), a per-run timeout, the trace of the
worked example, and the repair of how aima's ``mac`` reports a wipe-out (see :func:`_mac`). The
horizon loop :func:`find_min_horizon` returns the minimum horizon H*: by monotonicity (design §4.4,
L2) the first solvable horizon is the minimum, and the infeasible run at ``H* - 1`` is its proof.
Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §4.4, §6.1, §7 R1/R3 (private docs).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, TypeAlias, get_args

from sortedcontainers import SortedSet  # type: ignore[import-untyped]

from aima import csp as aima_csp
from symbolic_ai.p2_ej4_transition.csp_builder import Domains, Variable, build_csp
from symbolic_ai.p2_ej4_transition.errors import SolverError, SolverTimeoutError
from symbolic_ai.p2_ej4_transition.problem import TransitionProblem, Value

__all__ = [
    "ALL_CONFIGS",
    "H_MAX",
    "MAIN_CONFIG",
    "ArcConstraint",
    "HorizonRun",
    "Inference",
    "ResultStatus",
    "RunStats",
    "RunStatus",
    "SolverConfig",
    "TraceEvent",
    "TraceKind",
    "TransitionResult",
    "ValueOrder",
    "VariableOrder",
    "ac3_domains",
    "arc_constraint",
    "find_min_horizon",
    "solve_horizon",
    "solve_min_conflicts",
]

#: Inference after each assignment: none (plain backtracking), forward checking, or MAC with AC-3.
Inference: TypeAlias = Literal["none", "fc", "mac"]
#: Variable order: static (by day, then condition) or minimum remaining values.
VariableOrder: TypeAlias = Literal["static", "mrv"]
#: Value order: keep-first (the static domain order, design §6.1) or least constraining value.
ValueOrder: TypeAlias = Literal["keep_first", "lcv"]
#: Outcome of one horizon. ``not_found`` is only reported by min-conflicts, which cannot prove
#: infeasibility; ``timeout`` (the clock) and ``budget`` (the assignment budget) mean unknown.
RunStatus: TypeAlias = Literal["solved", "infeasible", "timeout", "budget", "not_found"]
#: Outcome of the horizon loop.
ResultStatus: TypeAlias = Literal["solved", "no_transition_within_h_max", "timeout", "budget"]
#: What a trace event records (see :class:`TraceEvent`).
TraceKind: TypeAlias = Literal["ac3", "select", "try", "prune", "wipeout", "backtrack", "solution"]
#: The constraint family of an arc: T (same condition, consecutive days) or I (interaction window).
ArcConstraint: TypeAlias = Literal["T", "I"]

_Assignment: TypeAlias = dict[Variable, Value]
_Removals: TypeAlias = list[tuple[Variable, Value]]
#: An arc ``(Xi, Xj)`` of AC-3: the values of ``Xi`` are revised against the domain of ``Xj``.
_Arc: TypeAlias = tuple[Variable, Variable]
#: The signature of aima's inference functions: ``(csp, var, value, assignment, removals)``.
_InferenceFunction: TypeAlias = Callable[[Any, Variable, Value, _Assignment, _Removals], bool]


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """One configuration of aima's solver (design §6.1).

    Parameters
    ----------
    inference
        ``"none"`` (chronological backtracking only), ``"fc"`` (forward checking) or ``"mac"``
        (maintaining arc consistency with AC-3).
    variable_order
        ``"static"`` (aima's ``first_unassigned_variable``: by day, then condition) or ``"mrv"``
        (minimum remaining values, ties broken by the static order).
    value_order
        ``"keep_first"`` (the static domain order of ``csp_builder.condition_values``: the start
        drug if still acceptable, the other acceptable drugs, the withdrawn start drug, ``None``
        last; kept stable after backtracking) or ``"lcv"`` (aima's least-constraining-value).
    ac3_preprocess
        Whether AC-3 runs on the node-consistent domains before the search.
    timeout_s
        Wall-clock budget of one horizon, in seconds (``> 0``).
    max_assignments
        Deterministic budget of one horizon: the run stops with status ``"budget"`` once it has
        made more than this many assignments (``None``: no budget). Unlike the clock, it stops a
        run at the same point on every machine, which makes a table of costly runs reproducible.

    Raises
    ------
    SolverError
        If a field has a value outside the ones listed above.
    """

    inference: Inference = "fc"
    variable_order: VariableOrder = "mrv"
    value_order: ValueOrder = "keep_first"
    ac3_preprocess: bool = True
    timeout_s: float = 30.0
    max_assignments: int | None = None

    def __post_init__(self) -> None:
        """Reject values the type hints cannot enforce at run time."""
        for name, allowed in (
            ("inference", get_args(Inference)),
            ("variable_order", get_args(VariableOrder)),
            ("value_order", get_args(ValueOrder)),
        ):
            if getattr(self, name) not in allowed:
                raise SolverError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if not self.timeout_s > 0:
            raise SolverError(f"timeout_s must be > 0, got {self.timeout_s}")
        if self.max_assignments is not None and self.max_assignments < 0:
            raise SolverError(f"max_assignments must be >= 0 or None, got {self.max_assignments}")

    @property
    def label(self) -> str:
        """A short name, e.g. ``"AC-3 + BT + FC + MRV + keep-first"``."""
        parts = ["AC-3"] if self.ac3_preprocess else []
        parts.append("BT")
        parts.extend({"none": [], "fc": ["FC"], "mac": ["MAC"]}[self.inference])
        parts.append("MRV" if self.variable_order == "mrv" else "static")
        parts.append("LCV" if self.value_order == "lcv" else "keep-first")
        return " + ".join(parts)


#: The main configuration: consistency, then search (deck 04 l.43-57): AC-3 + BT + FC + MRV.
MAIN_CONFIG = SolverConfig()

#: The last horizon the horizon loop tries by default (design §4.4): a cap, not a bound. No real
#: instance comes near it (their largest H* is 12); beyond it the answer is "not found within
#: H_MAX days", never "impossible".
H_MAX = 60

#: Every complete configuration (3 inferences x 2 variable orders x 2 value orders x AC-3 or not).
ALL_CONFIGS: tuple[SolverConfig, ...] = tuple(
    SolverConfig(inference, variable_order, value_order, ac3_preprocess)
    for inference, variable_order, value_order, ac3_preprocess in product(
        get_args(Inference), get_args(VariableOrder), get_args(ValueOrder), (False, True)
    )
)


@dataclass(frozen=True, slots=True)
class RunStats:
    """The effort of one run.

    Parameters
    ----------
    assignments
        Assignments made (aima's ``csp.nassigns``).
    checks
        Calls of the constraint function, AC-3's included (aima's ``csp.nchecks``, which counts
        every call of ``csp.constraints``). A solved backtracking run also includes the goal test
        that aima's ``backtracking_search`` asserts on its result.
    ac3_removed
        Values removed by the AC-3 preprocessing (0 without it).
    seconds
        Wall-clock time of the run.
    """

    assignments: int
    checks: int
    ac3_removed: int
    seconds: float


@dataclass(frozen=True, slots=True)
class HorizonRun:
    """The outcome of solving the CSP of one horizon.

    Parameters
    ----------
    horizon
        ``H``.
    status
        ``"solved"``, ``"infeasible"`` (proved by AC-3 or by exhausting the search),
        ``"timeout"`` (unknown), ``"budget"`` (the assignment budget ran out; unknown) or
        ``"not_found"`` (min-conflicts gave up; unknown).
    schedule
        The solution, keyed ``(condition_id, day)`` in static order, or ``None``.
    stats
        The effort of the run.
    seed
        The seed of a min-conflicts run; ``None`` for the complete solvers.
    """

    horizon: int
    status: RunStatus
    schedule: dict[Variable, Value] | None
    stats: RunStats
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """The outcome of the horizon loop.

    Parameters
    ----------
    h_star
        The minimum horizon, or ``None`` if none was found.
    status
        ``"solved"``, ``"no_transition_within_h_max"`` (every horizon ``0..h_max`` is infeasible),
        ``"timeout"`` or ``"budget"`` (a horizon ran out of time or of assignments, so the minimum
        is unknown).
    runs
        One run per horizon tried, from 0 to the last one.
    schedule
        The solution at ``h_star``, or ``None``.
    """

    h_star: int | None
    status: ResultStatus
    runs: tuple[HorizonRun, ...]
    schedule: dict[Variable, Value] | None


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One step of a solver run, for the worked example (design §7, R1).

    Parameters
    ----------
    step
        Position of the event in the trace (0, 1, 2, ...).
    kind
        ``"ac3"``: the AC-3 preprocessing removed ``value`` from ``variable`` while revising the
        arc towards ``neighbour``. ``"select"``: the
        search chose ``variable``. ``"try"``: the search assigned ``value`` to ``variable`` (the
        value passed aima's check against the assigned neighbours; with forward checking or MAC
        every value left in a domain passes it). ``"prune"``: forward checking or MAC removed
        ``value`` from ``variable``.
        ``"wipeout"``: the domain of ``variable`` became empty. ``"backtrack"``: every value of
        ``variable`` failed, so the search goes back. ``"solution"``: every variable is assigned.
    variable
        The variable concerned, or ``None``.
    value
        The value concerned, or ``None``.
    detail
        A short explanation in words.
    neighbour
        For an ``"ac3"`` event, the other end ``Xj`` of the arc ``(variable, Xj)`` whose revision
        removed ``value``: no value left in its domain supports it (see :func:`arc_constraint`
        for the constraint between them). ``None`` for every other kind.
    """

    step: int
    kind: TraceKind
    variable: Variable | None
    value: Value
    detail: str
    neighbour: Variable | None = None


def arc_constraint(variable: Variable, neighbour: Variable) -> ArcConstraint:
    """Return the constraint family between two neighbours of the CSP (design §4.3).

    ``"T"`` when they are the same condition on consecutive days (the constraint function then
    checks T and also I, which only matters if two drugs of one condition interact); ``"I"``
    otherwise (the interaction window is the only reason two such variables are neighbours, see
    ``csp_builder.build_csp``).

    Parameters
    ----------
    variable, neighbour
        Two variables ``(condition_id, day)`` of the CSP.

    Returns
    -------
    ArcConstraint
        ``"T"`` or ``"I"``.
    """
    (condition_1, day_1), (condition_2, day_2) = variable, neighbour
    return "T" if condition_1 == condition_2 and abs(day_1 - day_2) == 1 else "I"


def solve_horizon(
    problem: TransitionProblem,
    horizon: int,
    config: SolverConfig = MAIN_CONFIG,
    trace: list[TraceEvent] | None = None,
) -> HorizonRun:
    """Solve the CSP of one horizon with one configuration of aima's backtracking search.

    The run is: build the CSP (node-consistent domains), optionally AC-3 to a fixpoint (an empty
    domain proves the horizon infeasible with no search), then aima's ``backtracking_search`` with
    the configured variable order, value order and inference (AIMA 4th ed. §5.2.2 and §5.3).

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).
    config
        The solver configuration (default: :data:`MAIN_CONFIG`).
    trace
        If given, the events of the run are appended to it (see :class:`TraceEvent`).

    Returns
    -------
    HorizonRun
        Status ``"solved"``, ``"infeasible"``, ``"timeout"`` or ``"budget"``, the schedule and
        the effort.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    csp = build_csp(problem, horizon)
    started = time.perf_counter()
    status, assignment, ac3_removed = _search(csp, config, trace, started + config.timeout_s)
    stats = RunStats(
        assignments=csp.nassigns,
        checks=csp.nchecks,
        ac3_removed=ac3_removed,
        seconds=time.perf_counter() - started,
    )
    return HorizonRun(horizon, status, _schedule(csp, assignment), stats)


def find_min_horizon(
    problem: TransitionProblem, config: SolverConfig = MAIN_CONFIG, h_max: int = H_MAX
) -> TransitionResult:
    """Return the minimum horizon H* by solving H = 0, 1, 2, ... (the horizon loop, design §4.4).

    The first solvable horizon is the minimum because a solution for H extends to H + 1 by
    repeating its last day (L2). The loop stops at the first solution, at the first timeout or
    exhausted assignment budget (the minimum can then no longer be certified), or after ``h_max``.

    Parameters
    ----------
    problem
        The transition problem.
    config
        The solver configuration (default: :data:`MAIN_CONFIG`).
    h_max
        The last horizon tried (``>= 0``).

    Returns
    -------
    TransitionResult
        H* and its schedule, or the reason there is none, with every run made.

    Raises
    ------
    SolverError
        If ``h_max`` is negative.
    """
    if h_max < 0:
        raise SolverError(f"h_max must be >= 0, got {h_max}")
    runs: list[HorizonRun] = []
    for horizon in range(h_max + 1):
        run = solve_horizon(problem, horizon, config)
        runs.append(run)
        if run.status == "solved":
            return TransitionResult(horizon, "solved", tuple(runs), run.schedule)
        if run.status == "timeout":
            return TransitionResult(None, "timeout", tuple(runs), None)
        if run.status == "budget":
            return TransitionResult(None, "budget", tuple(runs), None)
    return TransitionResult(None, "no_transition_within_h_max", tuple(runs), None)


def ac3_domains(problem: TransitionProblem, horizon: int) -> tuple[bool, Domains]:
    """Return the domains after node consistency and then aima's AC-3 (AIMA 4th ed. §5.2.2).

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).

    Returns
    -------
    tuple of (bool, Domains)
        ``False`` if AC-3 emptied a domain (the horizon is infeasible, with no search); the domains
        are then those at the moment AC-3 stopped, one of them empty. ``True`` otherwise, with the
        arc-consistent domains in static order, each in keep-first order.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    csp = build_csp(problem, horizon)
    consistent, _removed = _preprocess(csp, None)
    return consistent, {v: tuple(csp.curr_domains[v]) for v in csp.variables}


def solve_min_conflicts(
    problem: TransitionProblem, horizon: int, max_steps: int = 10_000, seed: int = 0
) -> HorizonRun:
    """Run aima's min-conflicts local search on the CSP of one horizon (AIMA 4th ed. §5.4).

    min-conflicts starts from a complete assignment and repeatedly gives a random conflicted
    variable the value with the fewest conflicts. It is random: the global ``random`` module is
    seeded with ``seed`` for the run and restored afterwards. It cannot prove infeasibility, so a
    run that does not find a solution is ``"not_found"``, never ``"infeasible"``.

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).
    max_steps
        The step budget (``>= 0``).
    seed
        The seed of the run (recorded in the result).

    Returns
    -------
    HorizonRun
        Status ``"solved"`` or ``"not_found"``, the schedule and the effort (no AC-3).

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    SolverError
        If ``max_steps`` is negative.
    """
    if max_steps < 0:
        raise SolverError(f"max_steps must be >= 0, got {max_steps}")
    csp = build_csp(problem, horizon)
    started = time.perf_counter()
    assignment: _Assignment | None = None
    # min-conflicts needs a value for every variable to start from.
    if all(csp.domains[v] for v in csp.variables):
        assignment = _seeded_min_conflicts(csp, max_steps, seed)
    stats = RunStats(csp.nassigns, csp.nchecks, 0, time.perf_counter() - started)
    status: RunStatus = "not_found" if assignment is None else "solved"
    return HorizonRun(horizon, status, _schedule(csp, assignment), stats, seed=seed)


def _search(
    csp: Any, config: SolverConfig, trace: list[TraceEvent] | None, deadline: float
) -> tuple[RunStatus, _Assignment | None, int]:
    """Run AC-3 (if configured), then aima's backtracking; return status, solution, removals."""
    ac3_removed = 0
    if config.ac3_preprocess:
        consistent, ac3_removed = _preprocess(csp, trace)
        if not consistent:
            return "infeasible", None, ac3_removed
    try:
        assignment = aima_csp.backtracking_search(
            csp,
            select_unassigned_variable=_variable_selector(config, trace, deadline),
            order_domain_values=_value_orderer(config, trace),
            inference=_inference(config, trace),
        )
    except SolverTimeoutError:
        return "timeout", None, ac3_removed
    except _AssignmentBudgetError:
        return "budget", None, ac3_removed
    if assignment is None:
        return "infeasible", None, ac3_removed
    _record(trace, "solution", None, None, f"all {len(assignment)} variables assigned")
    return "solved", assignment, ac3_removed


def _schedule(csp: Any, assignment: _Assignment | None) -> dict[Variable, Value] | None:
    """Return the assignment as a schedule in static order (or ``None``)."""
    if assignment is None:
        return None
    return {v: assignment[v] for v in csp.variables}


def _record(
    trace: list[TraceEvent] | None,
    kind: TraceKind,
    variable: Variable | None,
    value: Value,
    detail: str,
    neighbour: Variable | None = None,
) -> None:
    """Append one event to ``trace``; do nothing when the run is not traced."""
    if trace is not None:
        trace.append(TraceEvent(len(trace), kind, variable, value, detail, neighbour))


def _preprocess(csp: Any, trace: list[TraceEvent] | None) -> tuple[bool, int]:
    """Run aima's AC-3 on the whole CSP; return whether it is consistent and the values removed.

    The arc queue is :func:`_arc_order`'s, watched by an :class:`_ArcLog` so that the trace can
    name the arc behind each removal; the arcs, their order and every count are unchanged.
    """
    removals: _Removals = []
    log = _ArcLog(removals)
    consistent, _checks = aima_csp.AC3(csp, None, removals, arc_heuristic=log.order)
    if trace is not None:
        for index, (variable, value) in enumerate(removals):
            neighbour = log.arc_of(index)[1]
            condition, day = neighbour
            detail = (
                f"no support left in {condition} day {day} "
                f"(constraint {arc_constraint(variable, neighbour)})"
            )
            _record(trace, "ac3", variable, value, detail, neighbour)
    if not consistent:
        _record(trace, "wipeout", _emptied(csp), None, "AC-3 emptied this domain: no solution")
    return bool(consistent), len(removals)


class _ArcLog:
    """AC-3's arc queue in :func:`_arc_order`, logging each arc it hands to aima's ``AC3``.

    aima's ``AC3`` (``csp.py`` l.200) pops an arc ``(Xi, Xj)`` and revises it at once, and
    ``revise`` (l.219) only removes values of ``Xi``, appending them to ``removals``. So every
    removal made between two pops comes from the arc popped first: logging, at each pop, how many
    removals there were before it is enough to name the arc of each removal. The log only
    watches the queue; it pops and adds the same arcs as the plain :func:`_arc_order` queue.

    Parameters
    ----------
    removals
        The list aima's ``AC3`` appends its removals to.
    """

    def __init__(self, removals: _Removals) -> None:
        self._removals = removals
        self._queue: Any = None
        self._pops: list[tuple[int, _Arc]] = []  # (removals before the pop, arc popped)

    def order(self, csp: Any, queue: Any) -> _ArcLog:
        """Order the arcs (aima's ``arc_heuristic`` of ``AC3``); the log then acts as the queue."""
        self._queue = _arc_order(csp, queue)
        return self

    def __len__(self) -> int:
        """Arcs left in the queue (aima's loop is ``while queue``)."""
        return len(self._queue)

    def add(self, arc: _Arc) -> None:
        """Put an arc back in the queue."""
        self._queue.add(arc)

    def pop(self) -> _Arc:
        """Take the next arc, logging it with the number of removals made so far."""
        arc: _Arc = self._queue.pop()
        self._pops.append((len(self._removals), arc))
        return arc

    def arc_of(self, removal: int) -> _Arc:
        """Return the arc whose revision made removal number ``removal`` (from 0)."""
        return next(arc for before, arc in reversed(self._pops) if before <= removal)


def _ac3(csp: Any, queue: Any = None, removals: _Removals | None = None) -> tuple[bool, int]:
    """Call aima's ``AC3`` with a reproducible arc order (see :func:`_arc_order`)."""
    consistent, checks = aima_csp.AC3(csp, queue, removals, arc_heuristic=_arc_order)
    return bool(consistent), int(checks)


def _arc_order(csp: Any, queue: Any) -> Any:
    """Return aima's ``dom_j_up`` arc queue (``csp.py`` l.194), with ties broken by static order.

    ``dom_j_up`` keeps the arcs in a ``SortedSet`` keyed by the size of the second variable's
    domain; arcs with equal keys come out in the order of a Python ``set``, which depends on string
    hashing and changes from one process to the next. Adding both variables, day first, to the key
    makes every process revise the arcs in the same order, hence the same counts and, after a
    backtrack, the same order of restored values.
    """
    return SortedSet(
        queue,
        key=lambda arc: (-len(csp.curr_domains[arc[1]]), _static(arc[0]), _static(arc[1])),
    )


def _static(variable: Variable) -> tuple[int, str]:
    """Return the static-order key of a variable: day, then condition."""
    condition, day = variable
    return day, condition


def _emptied(csp: Any) -> Variable:
    """Return the variable whose current domain is empty after a wipe-out."""
    variables: list[Variable] = csp.variables
    return next(v for v in variables if not csp.curr_domains[v])


class _AssignmentBudgetError(SolverError):
    """Raised inside aima's search when a run exceeds ``max_assignments``; caught by _search."""


def _variable_selector(
    config: SolverConfig, trace: list[TraceEvent] | None, deadline: float
) -> Callable[[_Assignment, Any], Variable]:
    """Return the variable-selection function of the configuration, with the timeout check."""

    def select(assignment: _Assignment, csp: Any) -> Variable:
        # The budget is checked before the clock, so a run it stops ends the same way everywhere.
        if config.max_assignments is not None and csp.nassigns > config.max_assignments:
            raise _AssignmentBudgetError(
                f"the run exceeded its budget of {config.max_assignments} assignments"
            )
        if time.perf_counter() > deadline:
            raise SolverTimeoutError(f"the run exceeded its budget of {config.timeout_s} s")
        variable: Variable
        if config.variable_order == "mrv":
            variable = _mrv(assignment, csp)
        else:
            variable = aima_csp.first_unassigned_variable(assignment, csp)
        _record(trace, "select", variable, None, f"{len(csp.choices(variable))} values left")
        return variable

    return select


def _mrv(assignment: _Assignment, csp: Any) -> Variable:
    """Choose a variable as aima's ``mrv`` (``csp.py`` l.391) does, with ties in static order.

    Minimum remaining values: AIMA 4th ed. §5.3.1 "Variable and value ordering".

    aima's version is ``argmin_random_tie(unassigned, key=num_legal_values)``; ``min`` with the same
    key returns the first minimum, and the unassigned variables are listed in static order.
    """
    unassigned: list[Variable] = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: aima_csp.num_legal_values(csp, v, assignment))


def _value_orderer(
    config: SolverConfig, trace: list[TraceEvent] | None
) -> Callable[[Variable, _Assignment, Any], Iterator[Value]]:
    """Return the value-order function of the configuration.

    It hands aima's search the values one at a time; when they run out (aima's ``backtrack``
    leaves its loop with no solution), it records the backtrack in the trace.
    """
    order = aima_csp.lcv if config.value_order == "lcv" else _keep_first

    def values(variable: Variable, assignment: _Assignment, csp: Any) -> Iterator[Value]:
        yield from order(variable, assignment, csp)
        _record(trace, "backtrack", variable, None, "every value failed: undo and go back")

    return values


def _keep_first(variable: Variable, assignment: _Assignment, csp: Any) -> list[Value]:
    """Return the current values of ``variable`` in keep-first order (design §6.1, v2.3).

    A value-ordering heuristic in the sense of AIMA 4th ed. §5.3.1 "Variable and value ordering".

    aima's ``unordered_domain_values`` (``csp.py`` l.410) returns ``csp.choices(var)`` as it is,
    and its order drifts: on a backtrack ``restore`` (l.172) appends the restored values at the
    end. Sorting by the position in the static domain, which ``csp_builder`` builds in keep-first
    order, gives the same order on every visit.
    """
    del assignment  # aima's value-order signature passes it; the order does not depend on it
    static: list[Value] = csp.domains[variable]
    return sorted(csp.choices(variable), key=static.index)


def _inference(config: SolverConfig, trace: list[TraceEvent] | None) -> _InferenceFunction:
    """Return the inference function of the configuration, which also traces its prunings."""
    infer: _InferenceFunction = {
        "none": aima_csp.no_inference,
        "fc": aima_csp.forward_checking,
        "mac": _mac,
    }[config.inference]

    def traced(
        csp: Any, variable: Variable, value: Value, assignment: _Assignment, removals: _Removals
    ) -> bool:
        _record(trace, "try", variable, value, "assigned")
        already_removed = len(removals)  # backtracking's suppose() put the other values there
        consistent = infer(csp, variable, value, assignment, removals)
        for pruned_variable, pruned_value in removals[already_removed:]:
            _record(trace, "prune", pruned_variable, pruned_value, f"inconsistent with {value}")
        if not consistent:
            _record(trace, "wipeout", _emptied(csp), None, "empty domain: try the next value")
        return bool(consistent)

    return traced


def _mac(
    csp: Any, variable: Variable, value: Value, assignment: _Assignment, removals: _Removals
) -> bool:
    """Run aima's ``mac`` (``csp.py`` l.441) with AC-3, the propagation the course teaches.

    aima's ``mac`` returns whatever its propagation returns, and ``AC3`` returns the pair
    ``(consistent, checks)``. A non-empty tuple is always true, so aima's ``backtracking_search``
    would never notice a wipe-out found by MAC. We return the first element of the pair.
    """
    consistent, _checks = aima_csp.mac(
        csp, variable, value, assignment, removals, constraint_propagation=_ac3
    )
    return bool(consistent)


def _seeded_min_conflicts(csp: Any, max_steps: int, seed: int) -> _Assignment | None:
    """Run aima's ``min_conflicts`` with the global ``random`` seeded, then restore its state."""
    state = random.getstate()
    random.seed(seed)
    try:
        solution: _Assignment | None = aima_csp.min_conflicts(csp, max_steps)
        return solution
    finally:
        random.setstate(state)
