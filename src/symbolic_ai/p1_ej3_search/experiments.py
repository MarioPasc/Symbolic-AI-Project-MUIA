"""EJ3 experiments: the encounters of the database (P6) and every fully observed record (P7).

P6 runs the cost-aware agent on the encounters and sets its regimen beside Agent 1's and beside
the cheapest one of the reference oracle. P7 takes every fully observed record with at least one
condition and, on the satisfiable ones, runs each search algorithm with and without the pruning
of Agent 1's classification: it checks the cost found against the oracle's minimum and compares
the effort, and it measures how far Agent 1's own regimen is from the cheapest. The oracle
(:class:`~symbolic_ai.p1_ej1_logic.RegimenSpace`) enumerates the regimens; the agent never uses
it. Every function is deterministic and returns frozen dataclasses; ``main.py`` loads the data,
calls them and writes the results.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from itertools import chain, combinations, repeat
from typing import TypeVar

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic import (
    Action,
    DPLLSolver,
    FullRecord,
    PrescribingAgent,
    RegimenSpace,
    unknown_risk_factors,
)
from symbolic_ai.p1_ej3_search.agent import CostAwareAgent, worst_case_record
from symbolic_ai.p1_ej3_search.errors import EJ3Error
from symbolic_ai.p1_ej3_search.problem import DomainPruning, RegimenProblem
from symbolic_ai.p1_ej3_search.search import Algorithm, search

__all__ = [
    "CONFIGURATIONS",
    "BaselineSummary",
    "ComparisonInstance",
    "ComparisonResult",
    "Configuration",
    "ConfigurationSummary",
    "EffortByConditions",
    "ScenarioResult",
    "SearchRun",
    "run_scenarios",
    "run_search_comparison",
]

_R = TypeVar("_R")

_SYNTHETIC_DATE = date(2026, 1, 1)


@dataclass(frozen=True, slots=True)
class Configuration:
    """One way of running the search in P7: an algorithm, with or without Agent 1's pruning."""

    name: str
    algorithm: Algorithm
    pruned: bool


#: P7's configurations: every algorithm, first on the plain problem and then pruned with the
#: essential and excluded drugs of Agent 1. The agent's default is ``astar_pruned``.
CONFIGURATIONS: tuple[Configuration, ...] = tuple(
    Configuration(f"{algorithm.value}_pruned" if pruned else algorithm.value, algorithm, pruned)
    for algorithm in Algorithm
    for pruned in (False, True)
)


# --- result records ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """P6: the cost-aware agent on one encounter, beside Agent 1 and the oracle.

    Costs are in euro cents per month. ``regimen``, ``cost``, ``saving`` and the search fields
    are set only when the action is ``prescribe``; ``minimum_cost`` is the oracle's cheapest
    model of Γ⁺ (``None`` when Γ⁺ has none).
    """

    encounter_id: str
    patient_id: str
    conditions: tuple[str, ...]
    present_risk_factors: tuple[str, ...]
    unknown_risk_factors: tuple[str, ...]
    action: str
    tests: tuple[str, ...]
    baseline_regimen: tuple[str, ...]
    baseline_cost: int | None
    regimen: tuple[str, ...]
    cost: int | None
    saving: int | None
    gamma_plus_models: int
    minimum_cost: int | None
    algorithm: str | None
    chosen: tuple[str, ...]
    expanded: int | None
    generated: int | None


@dataclass(frozen=True, slots=True)
class SearchRun:
    """P7: one configuration on one satisfiable record."""

    configuration: str
    cost: int
    expanded: int
    generated: int


@dataclass(frozen=True, slots=True)
class ComparisonInstance:
    """P7: one fully observed record.

    ``models``, ``minimum_cost`` and ``optimal_regimens`` (how many models cost the minimum) come
    from the oracle. ``search_finds_regimen`` is A*'s verdict on the plain problem.
    ``root_heuristic`` is h of the empty regimen, ``baseline_cost`` the cost of Agent 1's
    regimen; both, and ``runs``, are set only when the record is satisfiable.
    """

    conditions: tuple[str, ...]
    present_risk_factors: tuple[str, ...]
    models: int
    minimum_cost: int | None
    optimal_regimens: int
    search_finds_regimen: bool
    root_heuristic: float | None
    baseline_cost: int | None
    runs: tuple[SearchRun, ...]


@dataclass(frozen=True, slots=True)
class ConfigurationSummary:
    """P7: one configuration over the satisfiable records (excess is cost above the minimum)."""

    configuration: str
    algorithm: str
    pruned: bool
    n_optimal: int
    mean_excess_cost: float
    max_excess_cost: int
    mean_expanded: float
    max_expanded: int
    mean_generated: float
    max_generated: int


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    """P7: Agent 1's regimen against the cheapest one, over the satisfiable records.

    ``mean_excess_cost`` is what the search saves per record on average.
    """

    n_optimal: int
    mean_cost: float
    mean_minimum_cost: float
    mean_excess_cost: float
    max_excess_cost: int


@dataclass(frozen=True, slots=True)
class EffortByConditions:
    """P7: mean costs and mean nodes expanded per configuration, by number of conditions."""

    n_conditions: int
    n_satisfiable: int
    mean_minimum_cost: float
    mean_baseline_cost: float
    mean_expanded: dict[str, float]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """P7: every fully observed record, and the aggregates the report quotes.

    ``verdicts_agree_with_oracle`` says that A* finds a regimen exactly on the satisfiable
    records; ``heuristic_admissible_at_root`` that h of the empty regimen never exceeds the
    minimum cost, and ``n_root_heuristic_exact`` on how many records it equals it.
    """

    n_instances: int
    n_satisfiable: int
    verdicts_agree_with_oracle: bool
    n_unique_optimum: int
    heuristic_admissible_at_root: bool
    n_root_heuristic_exact: int
    baseline: BaselineSummary
    configurations: tuple[ConfigurationSummary, ...]
    by_n_conditions: tuple[EffortByConditions, ...]
    instances: tuple[ComparisonInstance, ...] = field(repr=False)


# --- enumeration helpers ----------------------------------------------------------------------
# The same enumeration as EJ1's P2 (its helpers are private to that module).


def _subsets(items: Iterable[str], *, non_empty: bool = False) -> list[frozenset[str]]:
    """Every subset of ``items``, by size then in lexicographic order."""
    ordered = sorted(items)
    start = 1 if non_empty else 0
    return [
        frozenset(combo)
        for size in range(start, len(ordered) + 1)
        for combo in combinations(ordered, size)
    ]


def _observed_encounter(formulary: Formulary, record: FullRecord) -> Encounter:
    """Build the encounter of a fully observed record: every risk factor present or absent."""
    statuses = {
        risk_factor_id: (
            RiskStatus.PRESENT
            if risk_factor_id in record.present_risk_factors
            else RiskStatus.ABSENT
        )
        for risk_factor_id in formulary.risk_factor_ids
    }
    return Encounter(
        encounter_id="synthetic",
        patient_id="synthetic",
        seq=1,
        visit_date=_SYNTHETIC_DATE,
        age_years=0,
        conditions=record.conditions,
        risk_factors=statuses,
        note="generated record (EJ3 experiments)",
    )


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean rounded to 4 decimals (stable JSON), ``0.0`` for an empty sequence."""
    return round(sum(values) / len(values), 4) if values else 0.0


def _minimum_cost(regimens: Sequence[frozenset[str]], costs: Mapping[str, int]) -> int | None:
    """Return the oracle's minimum: the cost of the cheapest of ``regimens``, ``None`` if empty."""
    return min((sum(costs[d] for d in regimen) for regimen in regimens), default=None)


# --- P6: scenarios ----------------------------------------------------------------------------


def run_scenarios(
    formulary: Formulary,
    costs: Mapping[str, int],
    encounters: Sequence[Encounter],
    space: RegimenSpace,
    algorithm: Algorithm = Algorithm.ASTAR,
) -> tuple[ScenarioResult, ...]:
    """P6: decide each encounter with the cost-aware agent and compare with the oracle.

    Parameters
    ----------
    formulary : Formulary
        The formulary both agents reason with.
    costs : Mapping[str, int]
        The cost of each drug, in euro cents per month.
    encounters : Sequence[Encounter]
        The encounters to decide, in the order to report them.
    space : RegimenSpace
        The oracle over ``formulary``.
    algorithm : Algorithm
        The agent's search algorithm (default: A*).

    Returns
    -------
    tuple[ScenarioResult, ...]
        One result per encounter, in the given order.
    """
    agent = CostAwareAgent(formulary, costs, DPLLSolver(), algorithm, prune=True)
    return tuple(_scenario(formulary, costs, space, agent, encounter) for encounter in encounters)


def _scenario(
    formulary: Formulary,
    costs: Mapping[str, int],
    space: RegimenSpace,
    agent: CostAwareAgent,
    encounter: Encounter,
) -> ScenarioResult:
    """P6 for one encounter."""
    costed = agent.decide(encounter)
    record = worst_case_record(encounter, formulary)
    unknowns = unknown_risk_factors(encounter, formulary)
    regimens = space.regimens(record)
    found = costed.search
    return ScenarioResult(
        encounter_id=encounter.encounter_id,
        patient_id=encounter.patient_id,
        conditions=tuple(sorted(record.conditions)),
        present_risk_factors=tuple(sorted(record.present_risk_factors - unknowns)),
        unknown_risk_factors=tuple(sorted(unknowns)),
        action=costed.action.value,
        tests=costed.decision.tests,
        baseline_regimen=tuple(sorted(costed.decision.regimen)),
        baseline_cost=costed.baseline_cost,
        regimen=tuple(sorted(costed.regimen)),
        cost=costed.cost,
        saving=costed.saving,
        gamma_plus_models=len(regimens),
        minimum_cost=_minimum_cost(regimens, costs),
        algorithm=None if found is None else found.algorithm.value,
        chosen=() if found is None else found.chosen,
        expanded=None if found is None else found.expanded,
        generated=None if found is None else found.generated,
    )


# --- P7: every fully observed record ----------------------------------------------------------


def run_search_comparison(
    formulary: Formulary, costs: Mapping[str, int], space: RegimenSpace, workers: int = 1
) -> ComparisonResult:
    """P7: run every configuration of :data:`CONFIGURATIONS` on every fully observed record.

    Parameters
    ----------
    formulary : Formulary
        The formulary both agents reason with.
    costs : Mapping[str, int]
        The cost of each drug, in euro cents per month.
    space : RegimenSpace
        The oracle over ``formulary`` (the regimens of each record).
    workers : int
        Number of processes; ``1`` runs sequentially. The result does not depend on it.

    Returns
    -------
    ComparisonResult
        One instance per record with at least one condition, and the aggregates.

    Raises
    ------
    ValueError
        If ``workers`` is smaller than 1.
    EJ3Error
        If Agent 1 does not prescribe on a record the oracle finds satisfiable.
    """
    tasks = _subsets(formulary.condition_ids, non_empty=True)
    chunks = _map(_compare_for_conditions, formulary, costs, space, tasks, workers)
    instances = tuple(chain.from_iterable(chunks))
    satisfiable = [i for i in instances if i.minimum_cost is not None]
    return ComparisonResult(
        n_instances=len(instances),
        n_satisfiable=len(satisfiable),
        verdicts_agree_with_oracle=all(i.search_finds_regimen == (i.models > 0) for i in instances),
        n_unique_optimum=sum(i.optimal_regimens == 1 for i in satisfiable),
        heuristic_admissible_at_root=all(
            _number(i.root_heuristic) <= _number(i.minimum_cost) for i in satisfiable
        ),
        n_root_heuristic_exact=sum(i.root_heuristic == i.minimum_cost for i in satisfiable),
        baseline=_summarise_baseline(satisfiable),
        configurations=tuple(_summarise(c, satisfiable) for c in CONFIGURATIONS),
        by_n_conditions=_effort_by_conditions(satisfiable),
        instances=instances,
    )


def _map(
    function: Callable[[Formulary, Mapping[str, int], RegimenSpace, frozenset[str]], _R],
    formulary: Formulary,
    costs: Mapping[str, int],
    space: RegimenSpace,
    tasks: Sequence[frozenset[str]],
    workers: int,
) -> list[_R]:
    """Apply ``function`` to every task, in order, sequentially or in ``workers`` processes."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if workers == 1:
        return [function(formulary, costs, space, task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(function, repeat(formulary), repeat(dict(costs)), repeat(space), tasks)
        )


def _compare_for_conditions(
    formulary: Formulary, costs: Mapping[str, int], space: RegimenSpace, conditions: frozenset[str]
) -> tuple[ComparisonInstance, ...]:
    """P7 for every set of present risk factors of one condition set (one worker task)."""
    prescriber = PrescribingAgent(formulary, DPLLSolver(), classify=True)
    return tuple(
        _compare(formulary, costs, space, prescriber, FullRecord(conditions, present))
        for present in _subsets(formulary.risk_factor_ids)
    )


def _compare(
    formulary: Formulary,
    costs: Mapping[str, int],
    space: RegimenSpace,
    prescriber: PrescribingAgent,
    record: FullRecord,
) -> ComparisonInstance:
    """P7 for one record: the oracle, Agent 1's regimen and every configuration."""
    regimens = space.regimens(record)
    minimum_cost = _minimum_cost(regimens, costs)
    plain = RegimenProblem(formulary, costs, record)
    verdict = search(plain, Algorithm.ASTAR).regimen is not None
    label = (tuple(sorted(record.conditions)), tuple(sorted(record.present_risk_factors)))
    if minimum_cost is None:
        return ComparisonInstance(*label, 0, None, 0, verdict, None, None, ())

    decision = prescriber.decide(_observed_encounter(formulary, record))
    if decision.action is not Action.PRESCRIBE or decision.classification is None:
        raise EJ3Error(f"Agent 1 does not prescribe on the satisfiable record {label}")
    pruned = RegimenProblem(
        formulary, costs, record, DomainPruning.from_classification(decision.classification)
    )
    return ComparisonInstance(
        *label,
        models=len(regimens),
        minimum_cost=minimum_cost,
        optimal_regimens=sum(plain.cost(regimen) == minimum_cost for regimen in regimens),
        search_finds_regimen=verdict,
        root_heuristic=float(plain.heuristic(plain.initial)),
        baseline_cost=plain.cost(decision.regimen),
        runs=tuple(_run(c, pruned if c.pruned else plain, label) for c in CONFIGURATIONS),
    )


def _run(
    configuration: Configuration,
    problem: RegimenProblem,
    label: tuple[tuple[str, ...], tuple[str, ...]],
) -> SearchRun:
    """Run one configuration on a satisfiable record."""
    result = search(problem, configuration.algorithm)
    if result.cost is None:
        raise EJ3Error(f"{configuration.name} finds no regimen on the satisfiable record {label}")
    return SearchRun(configuration.name, result.cost, result.expanded, result.generated)


def _number(value: float | None) -> float:
    """Read a number that is set on every satisfiable instance; ``None`` is an internal error."""
    if value is None:
        raise EJ3Error("a satisfiable instance lacks a cost or a heuristic value")
    return value


def _run_of(instance: ComparisonInstance, configuration: Configuration) -> SearchRun:
    """Return the run of ``configuration`` on ``instance`` (in :data:`CONFIGURATIONS` order)."""
    return instance.runs[CONFIGURATIONS.index(configuration)]


def _summarise(
    configuration: Configuration, satisfiable: Sequence[ComparisonInstance]
) -> ConfigurationSummary:
    """Aggregate one configuration over the satisfiable instances."""
    runs = [_run_of(i, configuration) for i in satisfiable]
    excess = [
        run.cost - int(_number(i.minimum_cost)) for run, i in zip(runs, satisfiable, strict=True)
    ]
    return ConfigurationSummary(
        configuration=configuration.name,
        algorithm=configuration.algorithm.value,
        pruned=configuration.pruned,
        n_optimal=sum(e == 0 for e in excess),
        mean_excess_cost=_mean(excess),
        max_excess_cost=max(excess, default=0),
        mean_expanded=_mean([run.expanded for run in runs]),
        max_expanded=max((run.expanded for run in runs), default=0),
        mean_generated=_mean([run.generated for run in runs]),
        max_generated=max((run.generated for run in runs), default=0),
    )


def _summarise_baseline(satisfiable: Sequence[ComparisonInstance]) -> BaselineSummary:
    """Aggregate Agent 1's regimen against the minimum over the satisfiable instances."""
    baseline = [int(_number(i.baseline_cost)) for i in satisfiable]
    minimum = [int(_number(i.minimum_cost)) for i in satisfiable]
    excess = [b - m for b, m in zip(baseline, minimum, strict=True)]
    return BaselineSummary(
        n_optimal=sum(e == 0 for e in excess),
        mean_cost=_mean(baseline),
        mean_minimum_cost=_mean(minimum),
        mean_excess_cost=_mean(excess),
        max_excess_cost=max(excess, default=0),
    )


def _effort_by_conditions(
    satisfiable: Sequence[ComparisonInstance],
) -> tuple[EffortByConditions, ...]:
    """Mean costs and mean nodes expanded per configuration, per number of present conditions."""
    rows = []
    for k in sorted({len(i.conditions) for i in satisfiable}):
        group = [i for i in satisfiable if len(i.conditions) == k]
        rows.append(
            EffortByConditions(
                n_conditions=k,
                n_satisfiable=len(group),
                mean_minimum_cost=_mean([_number(i.minimum_cost) for i in group]),
                mean_baseline_cost=_mean([_number(i.baseline_cost) for i in group]),
                mean_expanded={
                    c.name: _mean([_run_of(i, c).expanded for i in group]) for c in CONFIGURATIONS
                },
            )
        )
    return tuple(rows)
