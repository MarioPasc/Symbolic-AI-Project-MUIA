"""The worked example of EJ4 (design §7, R1): one patient, its domains, AC-3 and the search.

R1 is the centre of the talk. It follows one instance, the case study ``DEP|LIVER|+cond:PAIN``
(design §1), through the steps the course uses for map colouring (deck 04 l.133-165):

1. the instance as data rows (drugs with washout and taper days, the contraindication, the
   interactions, the start regimen, the acceptable drugs);
2. node consistency: the domains of every variable ``(condition, day)`` (design §4.2);
3. AC-3 at ``H* - 1``: an emptied domain proves that horizon infeasible with no search;
4. AC-3 at ``H*``: what is left for the search;
5. the search trace and the schedule at ``H*``, checked by the independent checker.

Everything is computed by the modules of wave 1 (``csp_builder``, ``solve``, ``checker``); this
module only collects it into frozen records that serialise to JSON. Design:
``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §1, §4.2, §7 R1 (private docs).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from symbolic_ai.dataloader import Contraindication
from symbolic_ai.p2_ej4_transition.checker import check_schedule
from symbolic_ai.p2_ej4_transition.csp_builder import Variable, node_consistent_domains
from symbolic_ai.p2_ej4_transition.errors import SolverError
from symbolic_ai.p2_ej4_transition.problem import Schedule, TransitionProblem, Value, drug_of
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    HorizonRun,
    SolverConfig,
    TraceEvent,
    TraceKind,
    ac3_domains,
    find_min_horizon,
    solve_horizon,
)

__all__ = [
    "ContraindicationRow",
    "DomainCell",
    "DrugRow",
    "HorizonDomains",
    "InstanceData",
    "RunSummary",
    "ScheduleTable",
    "TraceStep",
    "WorkedExample",
    "build_worked_example",
    "horizon_domains",
    "instance_data",
    "value_text",
]

#: The order in which trace kinds are counted in :attr:`WorkedExample.trace_counts` (as declared).
_TRACE_KINDS: tuple[TraceKind, ...] = get_args(TraceKind)


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
    §5.2.1). The emptied variable is taken from the trace of the solver's own run: the wipe-out
    it records right after the AC-3 preprocessing.

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
