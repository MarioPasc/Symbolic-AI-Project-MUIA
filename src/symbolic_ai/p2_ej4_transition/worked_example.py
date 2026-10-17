"""The worked example of EJ4 (design §7, R1): one patient, its domains, AC-3 and the search.

R1 is the centre of the talk. It follows one instance, the case study ``DEP|LIVER|+cond:PAIN``
(design §1), through the steps the course uses for map colouring (deck 04 l.133-165):

1. the instance as data rows (drugs with washout and taper days, the contraindication, the
   interactions, the start regimen, the acceptable drugs);
2. node consistency: the domains of every variable ``(condition, day)`` (design §4.2; AIMA 4th
   ed. §5.2.1 "Node consistency");
3. AC-3 at ``H* - 1``: an emptied domain proves that horizon infeasible with no search;
4. AC-3 at ``H*``: what is left for the search;
5. the search trace and the schedule at ``H*``, checked by the independent checker.

Everything is computed by the model, solver and verification modules (``csp_builder``,
``solve``, ``checker``, ``oracle``); this module only collects it into frozen records that
serialise to JSON. Design:
``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §1, §4.2, §7 R1 (private docs).

The second part of the module is the *miniature* of the slides (``slides-and-guide.md`` §3): the
same patient with every washout and taper shortened (:func:`miniature_problem`, H* = 3), small
enough to draw every domain, every AC-3 removal with its arc, the plain backtracking tree and the
oracle's graph of daily states. :func:`build_walkthrough` collects all of it and
:func:`walkthrough_text` renders it for ``--worked-example --miniature``.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, get_args

from symbolic_ai.dataloader import Contraindication
from symbolic_ai.p2_ej4_transition.checker import check_schedule
from symbolic_ai.p2_ej4_transition.csp_builder import (
    Constraint,
    Variable,
    build_csp,
    constraint_function,
    node_consistent_domains,
)
from symbolic_ai.p2_ej4_transition.errors import SolverError
from symbolic_ai.p2_ej4_transition.experiments import R3_CONFIGS
from symbolic_ai.p2_ej4_transition.oracle import DailyState, DailyStateProblem, oracle_min_horizon
from symbolic_ai.p2_ej4_transition.problem import Schedule, TransitionProblem, Value, drug_of
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    ArcConstraint,
    HorizonRun,
    SolverConfig,
    TraceEvent,
    TraceKind,
    ac3_domains,
    arc_constraint,
    find_min_horizon,
    solve_horizon,
)

__all__ = [
    "MINIATURE_START_TAPER_DAYS",
    "MINIATURE_SUFFIX",
    "MINIATURE_TAPER_DAYS",
    "MINIATURE_WASHOUT_DAYS",
    "Ac3Removal",
    "Ac3Run",
    "ConfigRow",
    "ContraindicationRow",
    "DomainCell",
    "DrugRow",
    "EdgeKind",
    "HorizonDomains",
    "HorizonStep",
    "InstanceData",
    "RunSummary",
    "ScheduleTable",
    "StateEdge",
    "StateGraph",
    "StateNode",
    "TraceStep",
    "TreeStep",
    "TreeStepKind",
    "Walkthrough",
    "WorkedExample",
    "ac3_run",
    "build_walkthrough",
    "build_worked_example",
    "daily_state_graph",
    "horizon_domains",
    "horizon_step",
    "instance_data",
    "miniature_problem",
    "search_tree",
    "value_text",
    "walkthrough_text",
]

#: The order in which trace kinds are counted in :attr:`WorkedExample.trace_counts` (as declared).
_TRACE_KINDS: tuple[TraceKind, ...] = get_args(TraceKind)

#: The miniature (``slides-and-guide.md`` §3): every drug's washout ``w_d``, in days.
MINIATURE_WASHOUT_DAYS = 1
#: The miniature: every drug's taper ``tau_d``, in days, except the start drugs'.
MINIATURE_TAPER_DAYS = 0
#: The miniature: the taper of each start drug (sertraline in the case study), in days.
MINIATURE_START_TAPER_DAYS = 1
#: Appended to the problem identifier of the miniature.
MINIATURE_SUFFIX = "|miniature"

#: A step of a search tree: an assignment, a value rejected by aima's check against the assigned
#: neighbours, a wipe-out found by the inference, a backtrack, or the solution.
TreeStepKind: TypeAlias = Literal["assign", "reject", "wipeout", "backtrack", "solution"]
#: An edge of the daily-state graph: to a state reached for the first time, to one reached
#: before (not expanded again), or to a goal state.
EdgeKind: TypeAlias = Literal["new", "repeated", "goal"]


@dataclass(frozen=True, slots=True)
class DrugRow:
    """One candidate drug of the instance with its pharmacokinetic values (design §8).

    Parameters
    ----------
    drug_id
        The drug.
    condition_id
        The condition it treats (each drug treats exactly one, design §4.1).
    washout_days
        Days it stays present after its last dose, ``w_d``.
    taper_days
        Days of reduced dose before it may be stopped, ``tau_d``.
    acceptable
        Whether it is acceptable after the event (not contraindicated).
    start
        Whether it is the start drug ``R_0`` of its condition.
    """

    drug_id: str
    condition_id: str
    washout_days: int
    taper_days: int
    acceptable: bool
    start: bool


@dataclass(frozen=True, slots=True)
class ContraindicationRow:
    """A formulary contraindication that excludes a candidate under the patient's risk factors.

    Parameters
    ----------
    risk_factor_id
        The risk factor.
    drug_id
        The excluded drug.
    reason
        The reason given by the formulary.
    """

    risk_factor_id: str
    drug_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class InstanceData:
    """The data rows of one instance, as shown on the first slide of R1 (design §7 R1.1).

    Parameters
    ----------
    problem_id
        The problem identifier, e.g. ``"DEP|LIVER|+cond:PAIN"``.
    event_kind
        ``"new_risk_factor"`` or ``"new_condition"``.
    event_target
        The risk factor or condition that appeared.
    conditions
        The conditions after the event.
    risk_factors
        The risk factors after the event.
    start
        ``R_0`` as ``(condition, drug or None)`` pairs.
    acceptable
        ``Acc_c`` as ``(condition, drugs)`` pairs.
    drugs
        Every candidate drug, by condition and then drug identifier.
    contraindications
        The formulary rows that exclude a candidate (empty if none were given).
    interactions
        The adverse interactions among the candidates, as sorted pairs.
    """

    problem_id: str
    event_kind: str
    event_target: str
    conditions: tuple[str, ...]
    risk_factors: tuple[str, ...]
    start: tuple[tuple[str, str | None], ...]
    acceptable: tuple[tuple[str, tuple[str, ...]], ...]
    drugs: tuple[DrugRow, ...]
    contraindications: tuple[ContraindicationRow, ...]
    interactions: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DomainCell:
    """The domain of one variable ``(condition, day)`` before and after AC-3.

    Parameters
    ----------
    condition_id
        The condition.
    day
        The day.
    node_consistent
        The domain after node consistency (design §4.2), in keep-first order.
    after_ac3
        The domain when AC-3 stopped (a fixpoint, or the moment a domain emptied).
    removed_by_ac3
        The values of ``node_consistent`` that AC-3 removed, in the same order.
    """

    condition_id: str
    day: int
    node_consistent: tuple[Value, ...]
    after_ac3: tuple[Value, ...]
    removed_by_ac3: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The deterministic part of one solver run (no wall-clock time).

    Parameters
    ----------
    horizon
        ``H``.
    status
        The run status (``solved``, ``infeasible``, ``timeout``, ``budget``, ``not_found``).
    assignments
        Assignments made.
    checks
        Constraint checks, AC-3's included.
    ac3_removed
        Values removed by the AC-3 preprocessing.
    """

    horizon: int
    status: str
    assignments: int
    checks: int
    ac3_removed: int

    @classmethod
    def from_run(cls, run: HorizonRun) -> RunSummary:
        """Return the summary of a :class:`HorizonRun` (its seconds are dropped)."""
        stats = run.stats
        return cls(run.horizon, run.status, stats.assignments, stats.checks, stats.ac3_removed)


@dataclass(frozen=True, slots=True)
class HorizonDomains:
    """Node consistency and AC-3 at one horizon, with the main configuration's run there.

    Parameters
    ----------
    horizon
        ``H``.
    ac3_consistent
        ``False`` if AC-3 emptied a domain (the horizon is infeasible with no search).
    emptied
        The variable whose domain AC-3 emptied, as recorded by the solver's trace, or ``None``.
    cells
        One cell per variable, in static order (day, then condition).
    removed_by_ac3
        The number of values AC-3 removed, over all cells.
    run
        The run of the solver configuration at this horizon.
    """

    horizon: int
    ac3_consistent: bool
    emptied: Variable | None
    cells: tuple[DomainCell, ...]
    removed_by_ac3: int
    run: RunSummary

    def cell(self, condition_id: str, day: int) -> DomainCell:
        """Return the cell of ``(condition_id, day)``.

        Raises
        ------
        KeyError
            If there is no such variable at this horizon.
        """
        for cell in self.cells:
            if (cell.condition_id, cell.day) == (condition_id, day):
                return cell
        raise KeyError((condition_id, day))


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One event of the search trace (see ``solve.TraceEvent``), as plain data.

    Parameters
    ----------
    step
        Position in the full trace (AC-3 events included).
    kind
        ``select``, ``try``, ``prune``, ``wipeout``, ``backtrack`` or ``solution``.
    condition_id
        The condition of the variable concerned, or ``None``.
    day
        The day of the variable concerned, or ``None``.
    value
        The value concerned, or ``None``.
    detail
        The explanation recorded by the solver.
    """

    step: int
    kind: str
    condition_id: str | None
    day: int | None
    value: Value
    detail: str


@dataclass(frozen=True, slots=True)
class ScheduleTable:
    """A schedule as one row of values per condition, days ``0..H`` (for tables and the Gantt).

    Parameters
    ----------
    problem_id
        The problem the schedule solves.
    conditions
        The conditions, in row order.
    horizon
        ``H``, the last day.
    rows
        For each condition, its value on days ``0..H``.
    washout_days
        ``w_d`` of every drug given in the schedule, as sorted ``(drug, days)`` pairs.
    taper_days
        ``tau_d`` of every drug given in the schedule, as sorted ``(drug, days)`` pairs.
    """

    problem_id: str
    conditions: tuple[str, ...]
    horizon: int
    rows: tuple[tuple[Value, ...], ...]
    washout_days: tuple[tuple[str, int], ...]
    taper_days: tuple[tuple[str, int], ...]

    @classmethod
    def from_schedule(
        cls, problem: TransitionProblem, schedule: Schedule, horizon: int
    ) -> ScheduleTable:
        """Return the table of ``schedule``, a complete schedule of ``problem`` over ``0..horizon``.

        Raises
        ------
        KeyError
            If a ``(condition, day)`` key is missing from ``schedule``.
        """
        rows = tuple(
            tuple(schedule[condition, day] for day in range(horizon + 1))
            for condition in problem.conditions
        )
        drugs = sorted({drug for row in rows for value in row if (drug := drug_of(value))})
        return cls(
            problem.problem_id,
            problem.conditions,
            horizon,
            rows,
            tuple((d, problem.washout_days[d]) for d in drugs),
            tuple((d, problem.taper_days[d]) for d in drugs),
        )

    def row(self, condition_id: str) -> tuple[Value, ...]:
        """Return the values of ``condition_id`` on days ``0..H``.

        Raises
        ------
        ValueError
            If the condition is not in the table.
        """
        return self.rows[self.conditions.index(condition_id)]


@dataclass(frozen=True, slots=True)
class WorkedExample:
    """Everything R1 shows about one instance (design §7 R1).

    Parameters
    ----------
    instance
        The data rows of the instance.
    config_label
        The label of the solver configuration used (the main one).
    h_star
        The minimum horizon found by the horizon loop.
    horizon_runs
        The runs of the horizon loop, ``H = 0..H*``.
    domains
        Node consistency and AC-3 at ``H* - 1`` (when ``H* >= 1``) and at ``H*``.
    schedule
        The schedule at ``H*``.
    trace
        The search part of the trace at ``H*`` (every event after the AC-3 preprocessing).
    trace_counts
        How many events of each kind the full trace at ``H*`` has, AC-3 removals included.
    violations
        What the independent checker reports on the schedule (empty: valid).
    """

    instance: InstanceData
    config_label: str
    h_star: int
    horizon_runs: tuple[RunSummary, ...]
    domains: tuple[HorizonDomains, ...]
    schedule: ScheduleTable
    trace: tuple[TraceStep, ...]
    trace_counts: tuple[tuple[str, int], ...]
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether the checker found no violation in the schedule."""
        return not self.violations

    def at(self, horizon: int) -> HorizonDomains:
        """Return the domains at ``horizon``.

        Raises
        ------
        KeyError
            If the worked example has no domains at that horizon.
        """
        for domains in self.domains:
            if domains.horizon == horizon:
                return domains
        raise KeyError(horizon)

    def to_dict(self) -> dict[str, object]:
        """Return the record as JSON-ready built-in types (tuples become lists when dumped)."""
        content = dataclasses.asdict(self)
        content["valid"] = self.valid
        return content


def value_text(value: Value) -> str:
    """Return a value in words: ``sertraline``, ``sertraline↓2`` (taper day 2) or ``∅``.

    Parameters
    ----------
    value
        A value of the contract.

    Returns
    -------
    str
        The text.
    """
    if value is None:
        return "∅"
    drug, phase = value
    return drug if phase == 0 else f"{drug}↓{phase}"


def instance_data(
    problem: TransitionProblem, contraindications: Sequence[Contraindication] = ()
) -> InstanceData:
    """Return the data rows of ``problem`` (design §7 R1.1).

    Parameters
    ----------
    problem
        The transition problem.
    contraindications
        The formulary's contraindications; only those that exclude a candidate of ``problem``
        under its risk factors are kept.

    Returns
    -------
    InstanceData
        The rows.
    """
    drugs = tuple(
        DrugRow(
            drug,
            condition,
            problem.washout_days[drug],
            problem.taper_days[drug],
            drug in problem.acceptable[condition],
            drug == problem.start[condition],
        )
        for condition in problem.conditions
        for drug in problem.candidates[condition]
    )
    candidates = {row.drug_id for row in drugs}
    kept = sorted(
        (c.risk_factor_id, c.drug_id, c.reason)
        for c in contraindications
        if c.risk_factor_id in problem.risk_factors and c.drug_id in candidates
    )
    return InstanceData(
        problem_id=problem.problem_id,
        event_kind=problem.event.kind.value,
        event_target=problem.event.target_id,
        conditions=problem.conditions,
        risk_factors=problem.risk_factors,
        start=tuple((c, problem.start[c]) for c in problem.conditions),
        acceptable=tuple((c, problem.acceptable[c]) for c in problem.conditions),
        drugs=drugs,
        contraindications=tuple(ContraindicationRow(*row) for row in kept),
        interactions=tuple(sorted(_pair(pair) for pair in problem.interactions)),
    )


def horizon_domains(
    problem: TransitionProblem, horizon: int, config: SolverConfig = MAIN_CONFIG
) -> HorizonDomains:
    """Return the domains of every variable before and after AC-3 at one horizon (design §7 R1).

    The node-consistent domains come from ``csp_builder.node_consistent_domains`` and the AC-3
    domains from ``solve.ac3_domains`` (aima's AC-3 with our deterministic arc order, AIMA 4th ed.
    §5.2.2 "Arc consistency"). The emptied variable is taken from the trace of the solver's own
    run: the wipe-out it records right after the AC-3 preprocessing.

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).
    config
        The configuration whose run is recorded (default: the main one).

    Returns
    -------
    HorizonDomains
        The domains, the emptied variable (if any) and the run.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    node_consistent = node_consistent_domains(problem, horizon)
    consistent, after = ac3_domains(problem, horizon)
    trace: list[TraceEvent] = []
    run = solve_horizon(problem, horizon, config, trace)
    cells = tuple(
        DomainCell(
            variable[0],
            variable[1],
            values,
            after[variable],
            tuple(v for v in values if v not in after[variable]),
        )
        for variable, values in node_consistent.items()
    )
    return HorizonDomains(
        horizon=horizon,
        ac3_consistent=consistent,
        emptied=_ac3_emptied(trace),
        cells=cells,
        removed_by_ac3=sum(len(cell.removed_by_ac3) for cell in cells),
        run=RunSummary.from_run(run),
    )


def build_worked_example(
    problem: TransitionProblem,
    contraindications: Sequence[Contraindication] = (),
    config: SolverConfig = MAIN_CONFIG,
) -> WorkedExample:
    """Run R1 on ``problem``: horizon loop, domains at ``H* - 1`` and ``H*``, trace, schedule.

    Parameters
    ----------
    problem
        The transition problem (the case study, ``instances.case_study_problem``).
    contraindications
        The formulary's contraindications (for the data rows).
    config
        The solver configuration (default: the main one, AC-3 + BT + FC + MRV).

    Returns
    -------
    WorkedExample
        The record of design §7 R1.

    Raises
    ------
    SolverError
        If the horizon loop does not find a minimum horizon.
    """
    result = find_min_horizon(problem, config)
    if result.h_star is None:
        raise SolverError(f"{problem.problem_id} has no transition: {result.status}")
    h_star = result.h_star
    horizons = [h_star - 1, h_star] if h_star >= 1 else [h_star]
    trace: list[TraceEvent] = []
    run = solve_horizon(problem, h_star, config, trace)
    if run.schedule is None:
        raise SolverError(f"{problem.problem_id}: the traced run at H* = {h_star} found nothing")
    violations = check_schedule(problem, run.schedule, h_star)
    counts = Counter(event.kind for event in trace)
    return WorkedExample(
        instance=instance_data(problem, contraindications),
        config_label=config.label,
        h_star=h_star,
        horizon_runs=tuple(RunSummary.from_run(r) for r in result.runs),
        domains=tuple(horizon_domains(problem, h, config) for h in horizons),
        schedule=ScheduleTable.from_schedule(problem, run.schedule, h_star),
        trace=tuple(_trace_step(e) for e in trace if e.kind != "ac3"),
        trace_counts=tuple((kind, counts[kind]) for kind in _TRACE_KINDS),
        violations=tuple(f"{v.rule} {v.condition} day {v.day}: {v.detail}" for v in violations),
    )


def _ac3_emptied(trace: Sequence[TraceEvent]) -> Variable | None:
    """Return the variable of the wipe-out recorded right after the AC-3 events, if any."""
    for event in trace:
        if event.kind != "ac3":
            return event.variable if event.kind == "wipeout" else None
    return None


def _trace_step(event: TraceEvent) -> TraceStep:
    """Return a trace event as a :class:`TraceStep`."""
    condition, day = event.variable if event.variable is not None else (None, None)
    return TraceStep(event.step, event.kind, condition, day, event.value, event.detail)


def _pair(pair: frozenset[str]) -> tuple[str, str]:
    """Return an interaction as a sorted pair."""
    first, second = sorted(pair)
    return first, second


# --- the miniature of the slides (slides-and-guide.md §3) ----------------------------------------


@dataclass(frozen=True, slots=True)
class HorizonStep:
    """One horizon of the horizon loop, with the reason it fails (design §4.4).

    Parameters
    ----------
    run
        The run of the configuration at this horizon.
    node_emptied
        The first variable (static order) whose domain node consistency left empty, or ``None``.
    ac3_emptied
        The variable whose domain the AC-3 preprocessing emptied, or ``None``.
    reason
        Why the horizon fails, in words (``"solved"`` when it does not).
    """

    run: RunSummary
    node_emptied: Variable | None
    ac3_emptied: Variable | None
    reason: str


@dataclass(frozen=True, slots=True)
class Ac3Removal:
    """One value removed by the AC-3 preprocessing, with the arc that removed it.

    Parameters
    ----------
    order
        Its position among the removals of the run (1, 2, ...).
    variable
        ``Xi``, the variable that lost the value.
    value
        The value removed.
    neighbour
        ``Xj``: no value left in its domain was compatible with ``Xi = value``.
    constraint
        The constraint between ``Xi`` and ``Xj``: ``"T"`` or ``"I"`` (``solve.arc_constraint``).
    """

    order: int
    variable: Variable
    value: Value
    neighbour: Variable
    constraint: ArcConstraint


@dataclass(frozen=True, slots=True)
class Ac3Run:
    """The AC-3 preprocessing at one horizon, removal by removal (AIMA 4th ed. §5.2.2).

    Parameters
    ----------
    horizon
        ``H``.
    removals
        The removals in the order aima's AC-3 made them (our deterministic arc order).
    emptied
        The variable whose domain AC-3 emptied, or ``None`` if it reached a fixpoint.
    """

    horizon: int
    removals: tuple[Ac3Removal, ...]
    emptied: Variable | None


@dataclass(frozen=True, slots=True)
class ConfigRow:
    """One solver configuration of R3 run at ``H* - 1`` and ``H*``.

    Parameters
    ----------
    key
        Its key in ``experiments.R3_CONFIGS`` (``bt``, ``bt_fc``, ``bt_fc_mrv``, ``main``).
    label
        Its label.
    runs
        Its runs, one per horizon, in increasing order.
    """

    key: str
    label: str
    runs: tuple[RunSummary, ...]


@dataclass(frozen=True, slots=True)
class TreeStep:
    """One step of a backtracking search, read from the solver's trace.

    Parameters
    ----------
    kind
        ``assign``, ``reject`` (aima's check against the assigned neighbours failed),
        ``wipeout`` (the inference emptied ``variable``'s domain), ``backtrack`` (every value of
        ``variable`` failed) or ``solution``.
    number
        For ``assign``, its number in visiting order (1, 2, ...); ``None`` otherwise.
    depth
        Its level in the search tree: how many variables were assigned when it happened, the
        variable of an ``assign``, ``reject`` or ``backtrack`` step itself excluded.
    variable
        The variable concerned (``None`` for ``solution``).
    value
        The value assigned or rejected, or ``None``.
    reason
        For ``reject``, the assigned neighbours it conflicts with; for ``wipeout``, the solver's
        explanation; empty otherwise.
    """

    kind: TreeStepKind
    number: int | None
    depth: int
    variable: Variable | None
    value: Value
    reason: str


@dataclass(frozen=True, slots=True)
class StateNode:
    """A daily state of the oracle, named by a letter in the order it was first reached.

    Parameters
    ----------
    label
        ``A``, ``B``, ... (``A`` is day 0).
    depth
        The day on which breadth-first search first reached it.
    state
        The oracle's state: what each condition gets that day and the drugs still present
        tomorrow, with their days left.
    """

    label: str
    depth: int
    state: DailyState


@dataclass(frozen=True, slots=True)
class StateEdge:
    """A successor generated while expanding a daily state.

    Parameters
    ----------
    source, target
        The labels of the expanded state and of its successor.
    kind
        ``new``, ``repeated`` (reached before, so not expanded again) or ``goal``.
    """

    source: str
    target: str
    kind: EdgeKind


@dataclass(frozen=True, slots=True)
class StateGraph:
    """The part of the oracle's daily-state graph that breadth-first search explores.

    Parameters
    ----------
    conditions
        The conditions, in the order of every state's values.
    states
        The states reached, in the order they were reached (so by depth).
    edges
        Every successor generated, expanded state by expanded state, in action order.
    states_expanded
        The states whose successors were generated (``oracle_min_horizon``'s count).
    h_star
        The depth of the first goal reached, or ``None``.
    path
        The labels of the witness path from ``A`` to that goal (empty without a goal).
    """

    conditions: tuple[str, ...]
    states: tuple[StateNode, ...]
    edges: tuple[StateEdge, ...]
    states_expanded: int
    h_star: int | None
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Walkthrough:
    """Everything the slides show about one small instance, step by step.

    Parameters
    ----------
    instance
        The data rows.
    config_label
        The label of the main configuration.
    h_star
        The minimum horizon of the horizon loop.
    loop
        The horizon loop, ``H = 0..H*``.
    domains
        Node consistency and AC-3 at ``H* - 1`` (when ``H* >= 1``) and ``H*``.
    ac3
        The AC-3 removals at the same horizons.
    configurations
        The configurations of R3 at the same horizons.
    bt_tree
        The plain backtracking search (R3's ``bt``) at ``H*``.
    main_path
        The main configuration's search at ``H*``.
    state_graph
        The oracle's breadth-first search over daily states.
    oracle_h_star
        H* according to the oracle.
    schedule
        The main configuration's schedule at ``H*``.
    violations
        What the independent checker reports on it (empty: valid).
    """

    instance: InstanceData
    config_label: str
    h_star: int
    loop: tuple[HorizonStep, ...]
    domains: tuple[HorizonDomains, ...]
    ac3: tuple[Ac3Run, ...]
    configurations: tuple[ConfigRow, ...]
    bt_tree: tuple[TreeStep, ...]
    main_path: tuple[TreeStep, ...]
    state_graph: StateGraph
    oracle_h_star: int | None
    schedule: ScheduleTable
    violations: tuple[str, ...]


def miniature_problem(problem: TransitionProblem) -> TransitionProblem:
    """Return the miniature of ``problem``: the didactic variant the slides use (H* = 3).

    Every washout becomes :data:`MINIATURE_WASHOUT_DAYS` (1) and every taper
    :data:`MINIATURE_TAPER_DAYS` (0), except the taper of the start drugs,
    :data:`MINIATURE_START_TAPER_DAYS` (1). Everything else (conditions, acceptable drugs,
    interactions) is unchanged, so the mechanism is the same; on the case study (start drug
    sertraline) the minimum horizon becomes ``H* = tau_S + w_S + 1 = 1 + 1 + 1 = 3`` instead of
    ``4 + 7 + 1 = 12``, small enough to draw every domain and the search.

    Parameters
    ----------
    problem
        The transition problem (the case study, ``instances.case_study_problem``).

    Returns
    -------
    TransitionProblem
        The same problem with the shortened times and ``problem_id`` ending in
        :data:`MINIATURE_SUFFIX`.
    """
    start_drugs = {drug for drug in problem.start.values() if drug is not None}
    taper = {
        drug: MINIATURE_START_TAPER_DAYS if drug in start_drugs else MINIATURE_TAPER_DAYS
        for drug in problem.taper_days
    }
    return dataclasses.replace(
        problem,
        problem_id=problem.problem_id + MINIATURE_SUFFIX,
        washout_days=dict.fromkeys(problem.washout_days, MINIATURE_WASHOUT_DAYS),
        taper_days=taper,
    )


def horizon_step(domains: HorizonDomains) -> HorizonStep:
    """Return one row of the horizon loop: the run at a horizon and why it fails.

    The reasons, in the order they are tested: node consistency left a domain empty (no value
    can be tried); AC-3 emptied a domain (infeasible with no search); the search tried every
    value. A run that ran out of time or of assignments says so.

    Parameters
    ----------
    domains
        Node consistency and AC-3 at the horizon (:func:`horizon_domains`).

    Returns
    -------
    HorizonStep
        The row.
    """
    run = domains.run
    node_emptied = next(
        ((c.condition_id, c.day) for c in domains.cells if not c.node_consistent), None
    )
    if run.status == "solved":
        reason = "solved"
    elif node_emptied is not None:
        reason = f"node consistency leaves {_variable_text(node_emptied)} empty"
    elif domains.emptied is not None:
        reason = f"AC-3 empties {_variable_text(domains.emptied)}"
    elif run.status == "infeasible":
        reason = "the search tried every value"
    else:
        reason = f"unknown ({run.status})"
    return HorizonStep(run, node_emptied, domains.emptied, reason)


def ac3_run(problem: TransitionProblem, horizon: int) -> Ac3Run:
    """Return the AC-3 preprocessing at ``horizon``, each removal with its arc (§5.2.2).

    The removals and their arcs are read from the trace of the main configuration's run, whose
    AC-3 events name the neighbour of each removal (``solve.TraceEvent.neighbour``).

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).

    Returns
    -------
    Ac3Run
        The removals and the emptied variable.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    trace: list[TraceEvent] = []
    solve_horizon(problem, horizon, MAIN_CONFIG, trace)
    removals = tuple(
        Ac3Removal(
            order,
            event.variable,
            event.value,
            event.neighbour,
            arc_constraint(event.variable, event.neighbour),
        )
        for order, event in enumerate((e for e in trace if e.kind == "ac3"), start=1)
        if event.variable is not None and event.neighbour is not None
    )
    return Ac3Run(horizon, removals, _ac3_emptied(trace))


def search_tree(
    problem: TransitionProblem, horizon: int, config: SolverConfig
) -> tuple[TreeStep, ...]:
    """Return the backtracking search at ``horizon`` as steps in visiting order (AIMA §5.3).

    The steps are read from the solver's own trace: each ``try`` event is an assignment (numbered
    as aima's ``nassigns`` counts them), each ``backtrack`` a dead end. Plain backtracking in
    keep-first order (no AC-3, no inference) tries the static domain in order, so a value it
    skipped between two tries is one that aima's check against the assigned neighbours rejected;
    for that configuration the rejected values are listed with the neighbours they conflict
    with, and the constraint (T or I) between them. With inference a skipped value may have been
    pruned instead, so only the assignments, wipe-outs and backtracks are listed.

    Parameters
    ----------
    problem
        The transition problem.
    horizon
        ``H`` (``>= 0``).
    config
        The solver configuration.

    Returns
    -------
    tuple of TreeStep
        The steps.

    Raises
    ------
    ModelError
        If ``horizon`` is negative.
    """
    trace: list[TraceEvent] = []
    solve_horizon(problem, horizon, config, trace)
    csp = build_csp(problem, horizon)
    check = constraint_function(problem)
    explain = (
        config.inference == "none"
        and not config.ac3_preprocess
        and config.value_order == "keep_first"
    )
    assignment: dict[Variable, Value] = {}
    next_value: dict[Variable, int] = {}  # position in the static domain of the next value
    steps: list[TreeStep] = []
    assigned = 0
    for event in trace:
        variable = event.variable
        if event.kind == "solution":
            steps.append(TreeStep("solution", None, len(assignment), None, None, ""))
        elif variable is None or event.kind in ("ac3", "prune"):
            continue
        elif event.kind == "select":
            next_value[variable] = 0
        elif event.kind == "wipeout":
            steps.append(TreeStep("wipeout", None, len(assignment), variable, None, event.detail))
        else:  # "try" or "backtrack"
            domain: list[Value] = csp.domains[variable]
            stop = domain.index(event.value) if event.kind == "try" else len(domain)
            assignment.pop(variable, None)
            if explain:
                skipped = domain[next_value[variable] : stop]
                steps += [_rejection(csp, check, variable, v, assignment) for v in skipped]
            next_value[variable] = stop + 1
            if event.kind == "try":
                assigned += 1
                steps.append(
                    TreeStep("assign", assigned, len(assignment), variable, event.value, "")
                )
                assignment[variable] = event.value
            else:
                steps.append(TreeStep("backtrack", None, len(assignment), variable, None, ""))
    return tuple(steps)


def _rejection(
    csp: Any, check: Constraint, variable: Variable, value: Value, assignment: dict[Variable, Value]
) -> TreeStep:
    """Return the step of a value rejected by aima's check, with the neighbours it conflicts with.

    The neighbours are those aima's ``nconflicts`` (``csp.py`` l.94) counts: assigned and not
    satisfying the constraint with ``variable = value``, in static order.
    """
    conflicts = ", ".join(
        f"{arc_constraint(variable, other)} with {_variable_text(other)} = "
        f"{value_text(assignment[other])}"
        for other in csp.neighbors[variable]
        if other in assignment and not check(variable, value, other, assignment[other])
    )
    return TreeStep("reject", None, len(assignment), variable, value, conflicts)


def daily_state_graph(problem: TransitionProblem) -> StateGraph:
    """Return the daily states the oracle's breadth-first search reaches, and how.

    The states, actions, results and goal test are those of ``oracle.DailyStateProblem`` (AIMA
    4th ed. §3.1); the loop is breadth-first graph search with the goal test on generation
    (§3.4.1, Fig. 3.9), as in ``oracle.oracle_min_horizon``, written out to record every edge.
    It stops after the expansion that generates the first goal, so the number of expanded states
    and the depth of that goal are the oracle's (the tests check both). Unlike the oracle it has
    no budget: it is meant for small problems.

    Parameters
    ----------
    problem
        The transition problem.

    Returns
    -------
    StateGraph
        The states (labelled ``A``, ``B``, ... in the order reached), the edges, the number of
        expanded states, the depth of the first goal and the witness path to it.
    """
    search = DailyStateProblem(problem)
    root = search.initial
    nodes = {root: StateNode("A", 0, root)}
    parent: dict[DailyState, DailyState] = {}
    edges: list[StateEdge] = []
    goal = root if search.goal_test(root) else None
    # As in the oracle, a day 0 that breaks rule interaction is not expanded.
    frontier = deque([root] if search.is_safe(root.values, ()) else [])
    expanded = 0
    while frontier and goal is None:
        state = frontier.popleft()
        expanded += 1
        for action in search.actions(state):
            child = search.result(state, action)
            kind: EdgeKind = "repeated"
            if child not in nodes:
                nodes[child] = StateNode(_state_label(len(nodes)), nodes[state].depth + 1, child)
                parent[child] = state
                frontier.append(child)
                kind = "goal" if search.goal_test(child) else "new"
                goal = child if goal is None and kind == "goal" else goal
            edges.append(StateEdge(nodes[state].label, nodes[child].label, kind))
    path: list[str] = []
    step = goal
    while step is not None:
        path.insert(0, nodes[step].label)
        step = parent.get(step)
    return StateGraph(
        conditions=problem.conditions,
        states=tuple(nodes.values()),
        edges=tuple(edges),
        states_expanded=expanded,
        h_star=None if goal is None else nodes[goal].depth,
        path=tuple(path),
    )


def build_walkthrough(
    problem: TransitionProblem, contraindications: Sequence[Contraindication] = ()
) -> Walkthrough:
    """Return the step-by-step walkthrough of a small instance (the miniature of the slides).

    It runs the main configuration's horizon loop, node consistency and AC-3 at ``H* - 1`` and
    ``H*``, the four configurations of R3 (``experiments.R3_CONFIGS``) at both horizons, the
    plain backtracking tree and the main configuration's search at ``H*``, the oracle and the
    checker. Every run is deterministic, so the walkthrough is too.

    Parameters
    ----------
    problem
        The transition problem (meant for :func:`miniature_problem`; every step is printed).
    contraindications
        The formulary's contraindications (for the data rows).

    Returns
    -------
    Walkthrough
        The record.

    Raises
    ------
    SolverError
        If the horizon loop does not find a minimum horizon.
    """
    result = find_min_horizon(problem, MAIN_CONFIG)
    if result.h_star is None or result.schedule is None:
        raise SolverError(f"{problem.problem_id} has no transition: {result.status}")
    h_star = result.h_star
    horizons = [h_star - 1, h_star] if h_star >= 1 else [h_star]
    every_horizon = [horizon_domains(problem, h) for h in range(h_star + 1)]
    configurations = tuple(
        ConfigRow(
            key,
            config.label,
            tuple(RunSummary.from_run(solve_horizon(problem, h, config)) for h in horizons),
        )
        for key, config in R3_CONFIGS
    )
    return Walkthrough(
        instance=instance_data(problem, contraindications),
        config_label=MAIN_CONFIG.label,
        h_star=h_star,
        loop=tuple(horizon_step(d) for d in every_horizon),
        domains=tuple(every_horizon[h] for h in horizons),
        ac3=tuple(ac3_run(problem, h) for h in horizons),
        configurations=configurations,
        bt_tree=search_tree(problem, h_star, dict(R3_CONFIGS)["bt"]),
        main_path=search_tree(problem, h_star, MAIN_CONFIG),
        state_graph=daily_state_graph(problem),
        oracle_h_star=oracle_min_horizon(problem).h_star,
        schedule=ScheduleTable.from_schedule(problem, result.schedule, h_star),
        violations=tuple(
            f"{v.rule} {v.condition} day {v.day}: {v.detail}"
            for v in check_schedule(problem, result.schedule, h_star)
        ),
    )


def walkthrough_text(walk: Walkthrough) -> str:
    """Return the walkthrough as plain text (``--worked-example --miniature``).

    Parameters
    ----------
    walk
        The walkthrough.

    Returns
    -------
    str
        Deterministic text, one section per step, no trailing newline.
    """
    lines = [f"=== miniature worked example {walk.instance.problem_id} ==="]
    lines += _instance_lines(walk.instance)
    lines += ["", f"horizon loop ({walk.config_label}):"]
    lines += [
        f"  H = {s.run.horizon}: {s.run.status}, {s.run.assignments} assignments, "
        f"{s.run.checks} checks, {s.run.ac3_removed} removed by AC-3; {s.reason}"
        for s in walk.loop
    ]
    for domains in walk.domains:
        lines += ["", f"node-consistent domains at H = {domains.horizon}:"]
        lines += _grid_lines(domains, after_ac3=False)
    for ac3, domains in zip(walk.ac3, walk.domains, strict=True):
        lines += ["", *_ac3_lines(ac3, domains)]
    lines += ["", *_configuration_lines(walk)]
    bt_label = dict(R3_CONFIGS)["bt"].label
    lines += ["", f"search tree at H = {walk.h_star}, {bt_label}:"]
    lines += _tree_lines(walk.bt_tree)
    lines += ["", f"search at H = {walk.h_star}, {walk.config_label}:"]
    lines += _tree_lines(walk.main_path)
    lines += ["", *_graph_lines(walk.state_graph)]
    verdict = "valid" if not walk.violations else "INVALID: " + "; ".join(walk.violations)
    lines += [
        "",
        f"H* = {walk.h_star} (CSP), {walk.oracle_h_star} (oracle); schedule (checker: {verdict}):",
    ]
    lines += _schedule_text(walk.schedule)
    return "\n".join(lines)


def _variable_text(variable: Variable) -> str:
    """Return a variable in words, e.g. ``DEP day 1``."""
    condition, day = variable
    return f"{condition} day {day}"


def _values_text(values: Sequence[Value]) -> str:
    """Return a domain in words, e.g. ``{sertraline, sertraline↓1}``."""
    return "{" + ", ".join(value_text(v) for v in values) + "}"


def _state_label(index: int) -> str:
    """Return the label of the ``index``-th state reached: ``A``, ``B``, ..., then ``S26``, ..."""
    return chr(ord("A") + index) if index < 26 else f"S{index}"


def _instance_lines(data: InstanceData) -> list[str]:
    """Return the data rows of the instance."""
    lines = [
        f"  {row.condition_id} {row.drug_id}: washout {row.washout_days} d, taper "
        f"{row.taper_days} d ({'acceptable' if row.acceptable else 'contraindicated'}"
        f"{', start drug' if row.start else ''})"
        for row in data.drugs
    ]
    lines += [
        f"  contraindication: {c.risk_factor_id} -> {c.drug_id} ({c.reason})"
        for c in data.contraindications
    ]
    lines += [f"  interaction: {a} - {b}" for a, b in data.interactions]
    start = ", ".join(f"{c} {d if d is not None else '∅'}" for c, d in data.start)
    acceptable = ", ".join(f"{c} {{{', '.join(drugs)}}}" for c, drugs in data.acceptable)
    return [*lines, f"  start: {start}", f"  acceptable: {acceptable}"]


def _grid_lines(domains: HorizonDomains, *, after_ac3: bool) -> list[str]:
    """Return the domains at one horizon, one line per condition and day."""
    width = max(len(cell.condition_id) for cell in domains.cells)
    return [
        f"  {cell.condition_id:<{width}} day {cell.day}: "
        f"{_values_text(cell.after_ac3 if after_ac3 else cell.node_consistent)}"
        for cell in sorted(domains.cells, key=lambda c: (c.condition_id, c.day))
    ]


def _ac3_lines(ac3: Ac3Run, domains: HorizonDomains) -> list[str]:
    """Return the AC-3 removals at one horizon, their arcs, and the domains left."""
    verdict = (
        "consistent"
        if ac3.emptied is None
        else f"empties {_variable_text(ac3.emptied)}: infeasible with no search"
    )
    lines = [f"AC-3 at H = {ac3.horizon}: {len(ac3.removals)} removals, {verdict}"]
    lines += [
        f"  {r.order:>2}. {_variable_text(r.variable)} - {value_text(r.value)}: no support in "
        f"{_variable_text(r.neighbour)} (constraint {r.constraint})"
        for r in ac3.removals
    ]
    return [
        *lines,
        "  domains left:",
        *("  " + line for line in _grid_lines(domains, after_ac3=True)),
    ]


def _configuration_lines(walk: Walkthrough) -> list[str]:
    """Return the configurations table: status, assignments / checks (AC-3 removals)."""
    width = max(len(row.label) for row in walk.configurations) + 2
    header = "".join(f"{f'H = {run.horizon}':<28}" for run in walk.configurations[0].runs)
    lines = [
        "configurations: status, assignments / checks (removed by AC-3)",
        f"  {'configuration':<{width}}{header}".rstrip(),
    ]
    for row in walk.configurations:
        cells = "".join(
            f"{f'{r.status} {r.assignments} / {r.checks} ({r.ac3_removed})':<28}" for r in row.runs
        )
        lines.append(f"  {row.label:<{width}}{cells}".rstrip())
    return lines


def _tree_lines(steps: Sequence[TreeStep]) -> list[str]:
    """Return a search as an indented list: assignments numbered, dead ends marked."""
    lines: list[str] = []
    for step in steps:
        indent = "  " * step.depth
        variable = _variable_text(step.variable) if step.variable is not None else ""
        if step.kind == "assign":
            lines.append(f"  {step.number:>3} {indent}{variable} = {value_text(step.value)}")
        elif step.kind == "reject":
            lines.append(f"      {indent}x {variable} = {value_text(step.value)}: {step.reason}")
        elif step.kind == "wipeout":
            lines.append(f"      {indent}x {variable} emptied: {step.reason}")
        elif step.kind == "backtrack":
            lines.append(f"      {indent}<- {variable}: no value left, backtrack")
        else:
            lines.append(f"      {indent}solution: every variable assigned")
    return lines


def _graph_lines(graph: StateGraph) -> list[str]:
    """Return the daily-state graph: the states by day, the edges of each expansion, the path."""
    goals = {edge.target for edge in graph.edges if edge.kind == "goal"}
    lines = [
        "oracle: breadth-first search over daily states",
        f"  state = ({', '.join(graph.conditions)}; drugs present tomorrow: days left)",
    ]
    for node in graph.states:
        values = ", ".join(value_text(v) for v in node.state.values)
        present = ", ".join(f"{d} {n}" for d, n in node.state.days_left) or "none"
        mark = "  goal" if node.label in goals else ""
        lines.append(f"  day {node.depth}: {node.label} = ({values}; {present}){mark}")
    for label in dict.fromkeys(edge.source for edge in graph.edges):
        targets = ", ".join(f"{e.target} {e.kind}" for e in graph.edges if e.source == label)
        lines.append(f"  expand {label}: {targets}")
    lines.append(
        f"  {graph.states_expanded} states expanded; first goal at depth {graph.h_star}; "
        f"witness path {' -> '.join(graph.path)}"
    )
    return lines


def _schedule_text(table: ScheduleTable) -> list[str]:
    """Return a schedule as one line per day."""
    width = max(14, *(len(value_text(v)) + 2 for row in table.rows for v in row))
    lines = ["  day " + "".join(f"{c:<{width}}" for c in table.conditions).rstrip()]
    for day in range(table.horizon + 1):
        values = "".join(f"{value_text(row[day]):<{width}}" for row in table.rows)
        lines.append(f"  {day:>3} {values}".rstrip())
    return lines
