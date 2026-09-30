"""EJ1 experiments of the report's *Resultados*: scenarios (P1), verdicts and value ordering (P2).

P1 runs the agent on the encounters of the database and counts the models of its bounds. P2
solves Γ of every fully observed record, checks DPLL's verdict against the reference oracle of
:mod:`symbolic_ai.p1_ej1_logic.semantics` (every base the agent queries is one of these records up
to clause order), and compares the regimen size under DPLL's two value orderings with the
oracle's minimum. :func:`untreatable_patterns` finds the minimal records the formulary cannot
treat. Every function is deterministic and returns frozen dataclasses; ``main.py`` loads the
data, calls them and writes the results.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from itertools import chain, combinations, repeat
from typing import TypeVar

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic.agent import Action, PrescribingAgent, SatCall
from symbolic_ai.p1_ej1_logic.encoding import gamma, unknown_risk_factors
from symbolic_ai.p1_ej1_logic.errors import EJ1Error
from symbolic_ai.p1_ej1_logic.semantics import FullRecord, RegimenSpace
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver

__all__ = [
    "CONFIGURATIONS",
    "OrderingSummary",
    "PatternsResult",
    "ScenarioResult",
    "SizeByConditions",
    "UntreatablePattern",
    "ValueOrderingInstance",
    "ValueOrderingResult",
    "as_jsonable",
    "run_scenarios",
    "run_value_ordering",
    "untreatable_patterns",
]

_R = TypeVar("_R")

_SYNTHETIC_DATE = date(2026, 1, 1)

#: P2's DPLL configurations: (name, value tried first for T_d, reverse the alphabetical symbol
#: order). The agent uses the first; the second is the textbook value ordering. Both keep the
#: agent's alphabetical symbol order.
CONFIGURATIONS: tuple[tuple[str, bool, bool], ...] = (
    ("false_first", False, False),
    ("true_first", True, False),
)


# --- result records ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """P1: the agent's decision on one encounter, with the oracle's model counts.

    Parameters
    ----------
    encounter_id, patient_id : str
        Identifiers of the encounter and its patient.
    seq, age_years : int
        Visit number and age at the visit.
    conditions, present_risk_factors, unknown_risk_factors : tuple[str, ...]
        The record: present conditions, present risk factors and the unknown set U (sorted).
    gamma_plus_models, gamma_minus_models : int
        Model counts of Γ⁺ and Γ⁻ over the decision symbols, from the oracle.
    action : str
        The agent's action (``prescribe``, ``request_test`` or ``refer``).
    regimen, tests : tuple[str, ...]
        The prescribed drugs (PRESCRIBE) or the requested tests (REQUEST_TEST), sorted.
    minimum_regimen_size : int | None
        The oracle's smallest model of Γ⁺ (PRESCRIBE only).
    regimen_true_first : tuple[str, ...]
        The regimen of the same agent with DPLL trying ``T_d = True`` first (PRESCRIBE only).
    trace : tuple[SatCall, ...]
        The agent's SAT calls with their DPLL effort counts.
    """

    encounter_id: str
    patient_id: str
    seq: int
    age_years: int
    conditions: tuple[str, ...]
    present_risk_factors: tuple[str, ...]
    unknown_risk_factors: tuple[str, ...]
    gamma_plus_models: int
    gamma_minus_models: int
    action: str
    regimen: tuple[str, ...]
    tests: tuple[str, ...]
    minimum_regimen_size: int | None
    regimen_true_first: tuple[str, ...]
    trace: tuple[SatCall, ...]


@dataclass(frozen=True, slots=True)
class ValueOrderingInstance:
    """P2: one fully observed record solved by DPLL in each configuration of :data:`CONFIGURATIONS`.

    Sizes are ``None`` when Γ is unsatisfiable. ``calls``, ``decisions`` and ``failures`` are the
    counts of :class:`~symbolic_ai.p1_ej1_logic.solver.SolveResult`.
    """

    conditions: tuple[str, ...]
    present_risk_factors: tuple[str, ...]
    models: int
    minimum_size: int | None
    size_false_first: int | None
    size_true_first: int | None
    calls_false_first: int
    calls_true_first: int
    decisions_false_first: int
    decisions_true_first: int
    failures_false_first: int
    failures_true_first: int


@dataclass(frozen=True, slots=True)
class OrderingSummary:
    """P2: aggregates of one DPLL configuration over the satisfiable instances (and its effort)."""

    ordering: str
    n_satisfiable: int
    n_minimal: int
    excess_histogram: dict[str, int]
    mean_size: float
    mean_excess: float
    max_excess: int
    mean_calls: float
    max_calls: int
    mean_decisions: float
    max_decisions: int
    satisfiable_with_backtracking: int
    unsatisfiable_max_decisions: int


@dataclass(frozen=True, slots=True)
class SizeByConditions:
    """P2: mean regimen size by number of present conditions, over satisfiable instances."""

    n_conditions: int
    n_satisfiable: int
    mean_minimum: float
    mean_false_first: float
    mean_true_first: float


@dataclass(frozen=True, slots=True)
class ValueOrderingResult:
    """P2: every fully observed record, and the aggregates the report quotes."""

    n_instances: int
    n_satisfiable: int
    verdicts_agree_with_oracle: bool
    orderings: tuple[OrderingSummary, ...]
    by_n_conditions: tuple[SizeByConditions, ...]
    instances: tuple[ValueOrderingInstance, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class UntreatablePattern:
    """A minimal untreatable record: removing any condition or risk factor makes it treatable."""

    conditions: tuple[str, ...]
    present_risk_factors: tuple[str, ...]
    n_records_containing: int


@dataclass(frozen=True, slots=True)
class PatternsResult:
    """The minimal untreatable patterns among the fully observed records.

    ``upward_closed`` is the monotonicity check: every record that contains a pattern is
    untreatable, so the patterns explain all ``n_untreatable`` records.
    """

    n_records: int
    n_untreatable: int
    upward_closed: bool
    patterns: tuple[UntreatablePattern, ...]


# --- enumeration helpers ----------------------------------------------------------------------


def _subsets(items: Iterable[str], *, non_empty: bool = False) -> list[frozenset[str]]:
    """Every subset of ``items``, by size then in lexicographic order."""
    ordered = sorted(items)
    start = 1 if non_empty else 0
    return [
        frozenset(combo)
        for size in range(start, len(ordered) + 1)
        for combo in combinations(ordered, size)
    ]


def _synthetic_encounter(conditions: frozenset[str], statuses: dict[str, RiskStatus]) -> Encounter:
    """Build an encounter for a generated record; only conditions and statuses matter."""
    return Encounter(
        encounter_id="synthetic",
        patient_id="synthetic",
        seq=1,
        visit_date=_SYNTHETIC_DATE,
        age_years=0,
        conditions=conditions,
        risk_factors=statuses,
        note="generated record (EJ1 experiments)",
    )


def _map(
    function: Callable[[Formulary, RegimenSpace, frozenset[str]], _R],
    formulary: Formulary,
    space: RegimenSpace,
    tasks: Sequence[frozenset[str]],
    workers: int,
) -> list[_R]:
    """Apply ``function`` to every task, in order, sequentially or in ``workers`` processes."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if workers == 1:
        return [function(formulary, space, task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, repeat(formulary), repeat(space), tasks))


# --- P1: scenarios ----------------------------------------------------------------------------


def run_scenarios(
    formulary: Formulary, encounters: Sequence[Encounter], space: RegimenSpace
) -> tuple[ScenarioResult, ...]:
    """P1: decide each encounter and count the models of its bounds with the oracle.

    Parameters
    ----------
    formulary : Formulary
        The formulary the agent reasons with.
    encounters : Sequence[Encounter]
        The encounters to decide, in the order to report them.
    space : RegimenSpace
        The oracle over ``formulary``.

    Returns
    -------
    tuple[ScenarioResult, ...]
        One result per encounter, in the given order.
    """
    agent = PrescribingAgent(formulary, DPLLSolver(decision_first_value=False), classify=False)
    textbook_agent = PrescribingAgent(
        formulary, DPLLSolver(decision_first_value=True), classify=False
    )
    return tuple(
        _scenario(formulary, space, agent, textbook_agent, encounter) for encounter in encounters
    )


def _scenario(
    formulary: Formulary,
    space: RegimenSpace,
    agent: PrescribingAgent,
    textbook_agent: PrescribingAgent,
    encounter: Encounter,
) -> ScenarioResult:
    """P1 for one encounter."""
    conditions = encounter.conditions & set(formulary.condition_ids)
    present = encounter.risk_factors_with(RiskStatus.PRESENT) & set(formulary.risk_factor_ids)
    unknowns = unknown_risk_factors(encounter, formulary)
    decision = agent.decide(encounter)
    textbook = textbook_agent.decide(encounter)
    plus_record = FullRecord(conditions, present | unknowns)

    prescribes = decision.action is Action.PRESCRIBE
    return ScenarioResult(
        encounter_id=encounter.encounter_id,
        patient_id=encounter.patient_id,
        seq=encounter.seq,
        age_years=encounter.age_years,
        conditions=tuple(sorted(conditions)),
        present_risk_factors=tuple(sorted(present)),
        unknown_risk_factors=tuple(sorted(unknowns)),
        gamma_plus_models=space.count(plus_record),
        gamma_minus_models=space.count(FullRecord(conditions, present)),
        action=decision.action.value,
        regimen=tuple(sorted(decision.regimen)),
        tests=decision.tests,
        minimum_regimen_size=space.minimum_size(plus_record) if prescribes else None,
        regimen_true_first=tuple(sorted(textbook.regimen)) if prescribes else (),
        trace=decision.trace,
    )


# --- P2: verdicts and value ordering --------------------------------------------------------------


def run_value_ordering(
    formulary: Formulary, space: RegimenSpace, workers: int = 1
) -> ValueOrderingResult:
    """P2: solve Γ of every fully observed record in each DPLL configuration.

    Parameters
    ----------
    formulary : Formulary
        The formulary the agent reasons with.
    space : RegimenSpace
        The oracle over ``formulary`` (model counts and minimum sizes).
    workers : int
        Number of processes; ``1`` runs sequentially. The result does not depend on it.

    Returns
    -------
    ValueOrderingResult
        One instance per record with at least one condition, and the aggregates.

    Raises
    ------
    ValueError
        If ``workers`` is smaller than 1.
    """
    tasks = _subsets(formulary.condition_ids, non_empty=True)
    chunks = _map(_ordering_for_conditions, formulary, space, tasks, workers)
    instances = tuple(chain.from_iterable(chunks))
    satisfiable = [i for i in instances if i.models > 0]
    agree = all(
        (i.models > 0) == (getattr(i, f"size_{name}") is not None)
        for i in instances
        for name, *_ in CONFIGURATIONS
    )
    return ValueOrderingResult(
        n_instances=len(instances),
        n_satisfiable=len(satisfiable),
        verdicts_agree_with_oracle=agree,
        orderings=tuple(_summarise_ordering(name, instances) for name, *_ in CONFIGURATIONS),
        by_n_conditions=_sizes_by_conditions(satisfiable),
        instances=instances,
    )


def _ordering_for_conditions(
    formulary: Formulary, space: RegimenSpace, conditions: frozenset[str]
) -> tuple[ValueOrderingInstance, ...]:
    """P2 for every set of present risk factors of one condition set (one worker task)."""
    solvers = [
        DPLLSolver(decision_first_value=first, reverse_symbol_order=reverse)
        for _, first, reverse in CONFIGURATIONS
    ]
    drug_symbols = [f"T_{d}" for d in formulary.drug_ids]
    instances = []
    for present in _subsets(formulary.risk_factor_ids):
        statuses = {
            r: RiskStatus.PRESENT if r in present else RiskStatus.ABSENT
            for r in formulary.risk_factor_ids
        }
        clauses = gamma(formulary, _synthetic_encounter(conditions, statuses))
        results = [solver.solve(clauses) for solver in solvers]
        sizes = [
            None if r.model is None else sum(r.model.get(s, False) for s in drug_symbols)
            for r in results
        ]
        record = FullRecord(conditions, present)
        instances.append(
            ValueOrderingInstance(
                conditions=tuple(sorted(conditions)),
                present_risk_factors=tuple(sorted(present)),
                models=space.count(record),
                minimum_size=space.minimum_size(record),
                size_false_first=sizes[0],
                size_true_first=sizes[1],
                calls_false_first=results[0].calls,
                calls_true_first=results[1].calls,
                decisions_false_first=results[0].decisions,
                decisions_true_first=results[1].decisions,
                failures_false_first=results[0].failures,
                failures_true_first=results[1].failures,
            )
        )
    return tuple(instances)


def _summarise_ordering(
    ordering: str, instances: Sequence[ValueOrderingInstance]
) -> OrderingSummary:
    """Aggregate the columns of configuration ``ordering`` (a name of :data:`CONFIGURATIONS`)."""
    sat = [i for i in instances if i.models > 0]
    unsat = [i for i in instances if i.models == 0]
    sizes = [_get_int(i, f"size_{ordering}") for i in sat]
    excess = [size - _get_int(i, "minimum_size") for size, i in zip(sizes, sat, strict=True)]
    calls = [_get_int(i, f"calls_{ordering}") for i in instances]
    decisions = [_get_int(i, f"decisions_{ordering}") for i in sat]
    return OrderingSummary(
        ordering=ordering,
        n_satisfiable=len(sat),
        n_minimal=sum(e == 0 for e in excess),
        excess_histogram={str(k): v for k, v in sorted(Counter(excess).items())},
        mean_size=_mean(sizes),
        mean_excess=_mean(excess),
        max_excess=max(excess, default=0),
        mean_calls=_mean(calls),
        max_calls=max(calls, default=0),
        mean_decisions=_mean(decisions),
        max_decisions=max(decisions, default=0),
        satisfiable_with_backtracking=sum(_get_int(i, f"failures_{ordering}") > 0 for i in sat),
        unsatisfiable_max_decisions=max(
            (_get_int(i, f"decisions_{ordering}") for i in unsat), default=0
        ),
    )


def _sizes_by_conditions(
    satisfiable: Sequence[ValueOrderingInstance],
) -> tuple[SizeByConditions, ...]:
    """Mean minimum, false-first and true-first sizes per number of present conditions."""
    rows = []
    for k in sorted({len(i.conditions) for i in satisfiable}):
        group = [i for i in satisfiable if len(i.conditions) == k]
        rows.append(
            SizeByConditions(
                n_conditions=k,
                n_satisfiable=len(group),
                mean_minimum=_mean([_get_int(i, "minimum_size") for i in group]),
                mean_false_first=_mean([_get_int(i, "size_false_first") for i in group]),
                mean_true_first=_mean([_get_int(i, "size_true_first") for i in group]),
            )
        )
    return tuple(rows)


def _get_int(instance: ValueOrderingInstance, name: str) -> int:
    """Read an integer field of ``instance`` by name; a missing size is an internal error."""
    value = getattr(instance, name)
    if not isinstance(value, int):
        raise EJ1Error(f"{name} is {value!r} on a satisfiable instance {instance.conditions}")
    return value


def _mean(values: Sequence[int]) -> float:
    """Arithmetic mean rounded to 4 decimals (stable JSON), ``0.0`` for an empty sequence."""
    return round(sum(values) / len(values), 4) if values else 0.0


# --- discussion: minimal untreatable patterns ---------------------------------------------------


def untreatable_patterns(formulary: Formulary, space: RegimenSpace) -> PatternsResult:
    """Find the minimal untreatable fully observed records (no regimen satisfies A1-A6).

    A record is untreatable when Γ has no model. Adding a condition or a present risk factor never
    makes an untreatable record treatable (checked here as ``upward_closed``), so the untreatable
    records are exactly those containing a minimal one.

    Parameters
    ----------
    formulary : Formulary
        The formulary whose records are enumerated (at least one condition present).
    space : RegimenSpace
        The oracle over ``formulary``.

    Returns
    -------
    PatternsResult
        The minimal patterns, how many records contain each, and the monotonicity check.
    """
    records = [
        FullRecord(conditions, present)
        for conditions in _subsets(formulary.condition_ids, non_empty=True)
        for present in _subsets(formulary.risk_factor_ids)
    ]
    untreatable = [r for r in records if space.count(r) == 0]
    minimal = [r for r in untreatable if not any(_contains(r, o) and o != r for o in untreatable)]
    covering = [r for r in records if any(_contains(r, m) for m in minimal)]
    patterns = sorted(
        (
            UntreatablePattern(
                conditions=tuple(sorted(m.conditions)),
                present_risk_factors=tuple(sorted(m.present_risk_factors)),
                n_records_containing=sum(_contains(r, m) for r in records),
            )
            for m in minimal
        ),
        key=lambda p: (
            len(p.conditions) + len(p.present_risk_factors),
            p.conditions,
            p.present_risk_factors,
        ),
    )
    return PatternsResult(
        n_records=len(records),
        n_untreatable=len(untreatable),
        upward_closed=len(covering) == len(untreatable),
        patterns=tuple(patterns),
    )


def _contains(record: FullRecord, pattern: FullRecord) -> bool:
    """Whether ``record`` has every condition and present risk factor of ``pattern``."""
    return (
        pattern.conditions <= record.conditions
        and pattern.present_risk_factors <= record.present_risk_factors
    )


# --- serialisation ----------------------------------------------------------------------------


def as_jsonable(result: object) -> object:
    """Convert a result dataclass (or a tuple of them) into JSON-ready built-in types.

    Parameters
    ----------
    result : object
        A result dataclass instance, or a tuple or list of them.

    Returns
    -------
    object
        Nested ``dict``/``list``/``str``/``int``/``float``/``bool``/``None`` values.

    Raises
    ------
    TypeError
        If ``result`` is neither a dataclass instance nor a sequence of them.
    """
    if isinstance(result, tuple | list):
        return [as_jsonable(item) for item in result]
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    raise TypeError(f"cannot serialise {type(result).__name__}")
