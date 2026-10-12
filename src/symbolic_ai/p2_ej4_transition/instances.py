"""EJ4's instances: every patient history times every event, as classified, deduplicated problems.

A *history* is a fully observed patient before the event: a non-empty set of conditions and a set
of risk factors. Its starting regimen ``R_0`` is the cheapest safe regimen (one acceptable drug per
condition, no interacting pair), recomputed here by brute force (design D21). Acceptability
applies P1's co-prescription rule exactly within one drug per condition (D23, see
:func:`acceptable_drugs`). Measured on the 4032 fully observed histories (tests in
``test_instances_continuity.py``): ``R_0`` equals the regimen P1's EJ3 agent prescribes on all
2720 histories where that regimen has one drug per condition; on the other 32 EJ3 adds omeprazole
as ibuprofen's companion under AGE65 without GERD, which one drug per condition cannot express, and
EJ4 has no safe regimen; on the 1280 histories EJ3 refers, EJ4 has none either
(:func:`continuity_with_ej3`). An *event* is a
risk factor or a condition the history lacks (D20). Each
(history, event) pair is classified before any solving:

- ``REFER``: no safe regimen exists after the event, so there is nothing to transition to;
- ``TRIVIAL``: a new risk factor leaves every drug of ``R_0`` acceptable (``H* = 0``);
- ``FORCED_ONLY``: some safe regimen after the event keeps every drug of ``R_0`` that is still
  acceptable, so only the drugs the event forbids (or the new condition) must change;
- ``KNOCK_ON``: every safe regimen after the event changes a drug that is still acceptable.

Pairs that give the same CSP (same conditions, start regimen and acceptable drugs; interactions and
pharmacokinetics are global) are merged into one ``ProblemEntry`` that lists all of them. Design:
``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §4.1 and §5 (private docs).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TypeAlias

from symbolic_ai.dataloader import DrugCost, Formulary, Pharmacokinetics
from symbolic_ai.dataloader.models import Encounter, RiskStatus
from symbolic_ai.p2_ej4_transition.errors import InstanceError
from symbolic_ai.p2_ej4_transition.problem import Event, EventKind, TransitionProblem

__all__ = [
    "Category",
    "CensusRow",
    "ContinuityReport",
    "History",
    "ProblemEntry",
    "acceptable_drugs",
    "case_study_problem",
    "census",
    "cheapest_safe_regimen",
    "continuity_with_ej3",
    "enumerate_problems",
]

#: A regimen as a tuple of drugs aligned with a sorted tuple of conditions.
_Regimen: TypeAlias = tuple[str, ...]
#: What the CSP sees of a problem: conditions, start regimen and acceptable drugs, as tuples.
_CspKey: TypeAlias = tuple[
    tuple[str, ...],
    tuple[tuple[str, str | None], ...],
    tuple[tuple[str, tuple[str, ...]], ...],
]
#: A history in the caches: (sorted conditions, sorted risk factors).
_State: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]

#: The worked example of the design (§1, R1): depression and severe liver disease, then pain.
_CASE_STUDY_HISTORY = (("DEP",), ("LIVER",))
_CASE_STUDY_EVENT = Event(EventKind.NEW_CONDITION, "PAIN")
#: The visit date of the generated records compared with EJ3 (any fixed date would do).
_SYNTHETIC_DATE = date(2026, 10, 12)


@dataclass(frozen=True, slots=True)
class History:
    """A fully observed patient before the event.

    Parameters
    ----------
    conditions
        The conditions treated before the event, sorted, non-empty, without repetition.
    risk_factors
        The risk factors present before the event, sorted, without repetition.

    Raises
    ------
    InstanceError
        If ``conditions`` is empty, or either tuple is unsorted or repeats an identifier.
    """

    conditions: tuple[str, ...]
    risk_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that both tuples are sorted and without repetition, and conditions non-empty."""
        if not self.conditions:
            raise InstanceError("a history needs at least one condition")
        for name, items in (("conditions", self.conditions), ("risk_factors", self.risk_factors)):
            if list(items) != sorted(set(items)):
                raise InstanceError(f"history {name} {items} must be sorted and without repetition")


class Category(Enum):
    """The classification of a (history, event) pair before solving (design §5)."""

    REFER = "refer"
    TRIVIAL = "trivial"
    FORCED_ONLY = "forced_only"
    KNOCK_ON = "knock_on"


@dataclass(frozen=True, slots=True)
class ProblemEntry:
    """One distinct CSP and every (history, event) pair that produces it.

    Parameters
    ----------
    problem
        The transition problem of the first pair in enumeration order; ``None`` only for
        ``REFER`` (no safe regimen exists after the event).
    category
        The category, identical for every pair of the entry.
    occurrences
        Every (history, event) pair that yields this problem, in enumeration order.
    """

    problem: TransitionProblem | None
    category: Category
    occurrences: tuple[tuple[History, Event], ...]


@dataclass(frozen=True, slots=True)
class CensusRow:
    """How many events of one kind fall in each category; counts are events, not entries."""

    event_kind: EventKind
    events: int
    refer: int
    trivial: int
    forced_only: int
    knock_on: int


@dataclass(frozen=True, slots=True)
class _Tables:
    """The formulary reduced to what EJ4 needs, with every set sorted.

    ``contraindicated_by[r]`` holds the drugs risk factor ``r`` contraindicates;
    ``companions_under[r]`` holds the ``(drug, companion)`` pairs of ``coprescriptions.csv`` for
    ``r`` (the drug may be given under ``r`` only together with the companion, D23).
    """

    conditions: tuple[str, ...]
    risk_factors: tuple[str, ...]
    candidates: Mapping[str, tuple[str, ...]]
    contraindicated_by: Mapping[str, frozenset[str]]
    companions_under: Mapping[str, tuple[tuple[str, str], ...]]
    interactions: frozenset[frozenset[str]]


#: The key of the acceptability cache: (condition, sorted risk factors, sorted treated conditions).
_AcceptableKey: TypeAlias = tuple[str, tuple[str, ...], tuple[str, ...]]
#: The key of the safe-regimen cache: (patient state, sorted conditions treated from day 0).
_SafeKey: TypeAlias = tuple[_State, tuple[str, ...]]


@dataclass(slots=True)
class _Enumeration:
    """The caches of one enumeration run (local to the run, never module-level)."""

    tables: _Tables
    costs: Mapping[str, int]
    acceptable: dict[_AcceptableKey, tuple[str, ...]] = field(default_factory=dict)
    safe: dict[_SafeKey, tuple[_Regimen, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Outcome:
    """The classified result of one (history, event) pair."""

    category: Category
    key: _CspKey
    conditions: tuple[str, ...]
    risk_factors: tuple[str, ...]
    start: Mapping[str, str | None]
    acceptable: Mapping[str, tuple[str, ...]]


def _tables(formulary: Formulary) -> _Tables:
    """Reduce the formulary to EJ4's tables: contraindications and co-prescriptions per factor."""
    contraindicated: dict[str, set[str]] = {r: set() for r in formulary.risk_factor_ids}
    for contraindication in formulary.contraindications:
        contraindicated[contraindication.risk_factor_id].add(contraindication.drug_id)
    companions: dict[str, set[tuple[str, str]]] = {r: set() for r in formulary.risk_factor_ids}
    for coprescription in formulary.coprescriptions:
        companions[coprescription.risk_factor_id].add(
            (coprescription.drug_id, coprescription.companion_drug_id)
        )
    return _Tables(
        conditions=formulary.condition_ids,
        risk_factors=formulary.risk_factor_ids,
        candidates={c: tuple(sorted(formulary.candidates[c])) for c in formulary.condition_ids},
        contraindicated_by={r: frozenset(drugs) for r, drugs in contraindicated.items()},
        companions_under={r: tuple(sorted(pairs)) for r, pairs in companions.items()},
        interactions=frozenset(frozenset((i.drug_a, i.drug_b)) for i in formulary.interactions),
    )


def _always_given(tables: _Tables, treated: Iterable[str]) -> frozenset[str]:
    """Return the drugs given on every day: the only candidate of a condition treated from day 0.

    A treated condition is never left untreated and a single-candidate condition can never switch,
    so its only candidate is given every day of the transition.
    """
    given: set[str] = set()
    for condition_id in treated:
        if condition_id not in tables.candidates:
            raise InstanceError(f"unknown condition {condition_id!r}")
        if len(tables.candidates[condition_id]) == 1:
            given.add(tables.candidates[condition_id][0])
    return frozenset(given)


def _acceptable(
    tables: _Tables,
    condition_id: str,
    risk_factors: Iterable[str],
    treated: Iterable[str] = (),
) -> tuple[str, ...]:
    """``Acc_c``: the candidates of a condition that no present risk factor excludes (D23 exact).

    A candidate is excluded by a contraindication of a present factor, or when it needs a companion
    under a present factor and no condition in ``treated`` (treated from day 0) has that companion
    as its only candidate (so the companion is not given on every day of the transition).
    """
    if condition_id not in tables.candidates:
        raise InstanceError(f"unknown condition {condition_id!r}")
    given = _always_given(tables, treated)
    excluded: set[str] = set()
    for risk_factor in risk_factors:
        if risk_factor not in tables.contraindicated_by:
            raise InstanceError(f"unknown risk factor {risk_factor!r}")
        excluded |= tables.contraindicated_by[risk_factor]
        excluded |= {d for d, k in tables.companions_under[risk_factor] if k not in given}
    return tuple(d for d in tables.candidates[condition_id] if d not in excluded)


def acceptable_drugs(
    formulary: Formulary,
    condition_id: str,
    risk_factors: Iterable[str],
    *,
    conditions: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return ``Acc_c``, the candidates of a condition acceptable for a patient.

    A candidate is excluded if a present risk factor contraindicates it. Design decision D23
    applies P1's co-prescription rule (axiom A5, ``coprescriptions.csv``) exactly within EJ4's
    one-drug-per-condition model: a drug that needs a companion ``k`` under a present risk factor
    is acceptable iff some condition treated from day 0 has ``k`` as its only candidate, so ``k``
    is given on every day of the transition and A5 holds on each day, not only on the last one
    (design v2.3). Today: ibuprofen under AGE65 is acceptable iff GERD (whose only candidate is
    omeprazole) is a condition of the history; a newly diagnosed GERD does not count, because it
    is untreated until omeprazole starts.

    Parameters
    ----------
    formulary
        The formulary.
    condition_id
        The condition.
    risk_factors
        The risk factors present.
    conditions
        The conditions treated from day 0: the history's conditions, never the newly diagnosed
        one (``condition_id`` may or may not be listed). The default, no condition, makes every
        co-prescribed drug unacceptable under its risk factor.

    Returns
    -------
    tuple[str, ...]
        The acceptable drugs, sorted (possibly empty).

    Raises
    ------
    InstanceError
        If the condition, a risk factor or one of ``conditions`` is not in the formulary.
    """
    return _acceptable(_tables(formulary), condition_id, risk_factors, conditions)


def _safe_regimens(
    options: Sequence[tuple[str, ...]], interactions: frozenset[frozenset[str]]
) -> Iterator[_Regimen]:
    """Yield every choice of one drug per option list with no interacting pair, in product order.

    A depth-first product that drops a prefix as soon as it contains an interacting pair; the
    regimens come out in the order of ``itertools.product``.
    """

    def extend(prefix: _Regimen) -> Iterator[_Regimen]:
        if len(prefix) == len(options):
            yield prefix
            return
        for drug in options[len(prefix)]:
            if all(frozenset((drug, other)) not in interactions for other in prefix):
                yield from extend((*prefix, drug))

    return extend(())


def _cheapest(regimens: Iterable[_Regimen], costs: Mapping[str, int]) -> _Regimen | None:
    """Return the cheapest regimen; ties go to the smaller sorted tuple of drug ids (design §5)."""
    return min(
        regimens,
        key=lambda regimen: (sum(costs[d] for d in regimen), tuple(sorted(regimen))),
        default=None,
    )


def _cost_table(tables: _Tables, costs: Mapping[str, DrugCost]) -> dict[str, int]:
    """Monthly cost of every candidate drug; raises ``InstanceError`` if one is missing."""
    drugs = sorted({d for candidates in tables.candidates.values() for d in candidates})
    missing = [d for d in drugs if d not in costs]
    if missing:
        raise InstanceError(f"no cost for the candidate drugs {missing}")
    return {d: costs[d].monthly_cost_cents for d in drugs}


def cheapest_safe_regimen(
    formulary: Formulary, costs: Mapping[str, DrugCost], history: History
) -> dict[str, str] | None:
    """Return ``R_0``, the cheapest safe regimen of a history, by brute force (design D21).

    A safe regimen gives each condition one of its acceptable drugs (D23 exact, judged on the
    history's conditions) with no interacting pair. The cheapest minimises the total monthly cost;
    ties are broken by the sorted tuple of drug identifiers.

    Parameters
    ----------
    formulary
        The formulary.
    costs
        The monthly cost of every drug.
    history
        The patient before the event.

    Returns
    -------
    dict[str, str] or None
        The drug of each condition, or ``None`` if the history has no safe regimen.

    Raises
    ------
    InstanceError
        If an identifier is not in the formulary or a candidate drug has no cost.
    """
    tables = _tables(formulary)
    options = [
        _acceptable(tables, c, history.risk_factors, history.conditions) for c in history.conditions
    ]
    best = _cheapest(_safe_regimens(options, tables.interactions), _cost_table(tables, costs))
    return None if best is None else dict(zip(history.conditions, best, strict=True))


def _cached_acceptable(
    run: _Enumeration, condition_id: str, risk_factors: tuple[str, ...], treated: tuple[str, ...]
) -> tuple[str, ...]:
    """``Acc_c`` of one condition under ``risk_factors``, with ``treated`` treated from day 0."""
    key = (condition_id, risk_factors, treated)
    if key not in run.acceptable:
        run.acceptable[key] = _acceptable(run.tables, condition_id, risk_factors, treated)
    return run.acceptable[key]


def _cached_safe(
    run: _Enumeration, state: _State, treated: tuple[str, ...]
) -> tuple[_Regimen, ...]:
    """Return the safe regimens of ``state`` when ``treated`` are treated from day 0."""
    key = (state, treated)
    if key not in run.safe:
        conditions, risk_factors = state
        options = [_cached_acceptable(run, c, risk_factors, treated) for c in conditions]
        run.safe[key] = tuple(_safe_regimens(options, run.tables.interactions))
    return run.safe[key]


def _state_after(history: History, event: Event) -> _State:
    """Return the conditions and risk factors after the event, which must add something new."""
    if event.kind is EventKind.NEW_RISK_FACTOR:
        if event.target_id in history.risk_factors:
            raise InstanceError(f"risk factor {event.target_id!r} is already present")
        return history.conditions, tuple(sorted((*history.risk_factors, event.target_id)))
    if event.target_id in history.conditions:
        raise InstanceError(f"condition {event.target_id!r} is already present")
    return tuple(sorted((*history.conditions, event.target_id))), history.risk_factors


def _classify(
    event: Event,
    start: Mapping[str, str | None],
    acceptable: Mapping[str, tuple[str, ...]],
    finals: tuple[_Regimen, ...],
    conditions: tuple[str, ...],
) -> Category:
    """Classify a pair from the start regimen and the safe regimens after the event (design §5)."""
    if not finals:
        return Category.REFER
    keep = {c: d for c, d in start.items() if d is not None and d in acceptable[c]}
    treated = sum(d is not None for d in start.values())
    if event.kind is EventKind.NEW_RISK_FACTOR and len(keep) == treated:
        return Category.TRIVIAL
    position = {c: i for i, c in enumerate(conditions)}
    if any(all(f[position[c]] == d for c, d in keep.items()) for f in finals):
        return Category.FORCED_ONLY
    return Category.KNOCK_ON


def _outcome(
    run: _Enumeration, history: History, start_regimen: Mapping[str, str], event: Event
) -> _Outcome:
    """Classify one (history, event) pair and compute what the CSP would see."""
    state = _state_after(history, event)
    conditions, risk_factors = state
    start = {c: start_regimen.get(c) for c in conditions}
    # D23: only the history's conditions are treated from day 0, not a newly diagnosed one.
    treated = history.conditions
    acceptable = {c: _cached_acceptable(run, c, risk_factors, treated) for c in conditions}
    finals = _cached_safe(run, state, treated)
    category = _classify(event, start, acceptable, finals, conditions)
    key: _CspKey = (
        conditions,
        tuple((c, start[c]) for c in conditions),
        tuple((c, acceptable[c]) for c in conditions),
    )
    return _Outcome(category, key, conditions, risk_factors, start, acceptable)


def _problem_id(history: History, event: Event) -> str:
    """Return a readable identifier such as ``DEP|LIVER|+cond:PAIN`` (history, then event)."""
    prefix = "+rf:" if event.kind is EventKind.NEW_RISK_FACTOR else "+cond:"
    risk_factors = "+".join(history.risk_factors) or "-"
    return f"{'+'.join(history.conditions)}|{risk_factors}|{prefix}{event.target_id}"


def _build_problem(
    tables: _Tables,
    pharmacokinetics: Mapping[str, Pharmacokinetics],
    history: History,
    event: Event,
    outcome: _Outcome,
) -> TransitionProblem:
    """Build the ``TransitionProblem`` of a non-REFER pair."""
    candidates = {c: tables.candidates[c] for c in outcome.conditions}
    drugs = {d for options in candidates.values() for d in options}
    missing = sorted(drugs - set(pharmacokinetics))
    if missing:
        raise InstanceError(f"no pharmacokinetics for the candidate drugs {missing}")
    return TransitionProblem(
        problem_id=_problem_id(history, event),
        event=event,
        conditions=outcome.conditions,
        risk_factors=outcome.risk_factors,
        start=dict(outcome.start),
        candidates=candidates,
        acceptable=dict(outcome.acceptable),
        interactions=frozenset(pair for pair in tables.interactions if pair <= drugs),
        washout_days={d: pharmacokinetics[d].washout_days for d in sorted(drugs)},
        taper_days={d: pharmacokinetics[d].taper_days for d in sorted(drugs)},
    )


def _subsets(items: tuple[str, ...], *, smallest: int) -> Iterator[tuple[str, ...]]:
    """Yield every subset of a sorted tuple with at least ``smallest`` items, by size then order."""
    for size in range(smallest, len(items) + 1):
        yield from itertools.combinations(items, size)


def _events(tables: _Tables, history: History) -> Iterator[Event]:
    """Yield the events of a history: each absent risk factor, then each absent condition."""
    for risk_factor in tables.risk_factors:
        if risk_factor not in history.risk_factors:
            yield Event(EventKind.NEW_RISK_FACTOR, risk_factor)
    for condition in tables.conditions:
        if condition not in history.conditions:
            yield Event(EventKind.NEW_CONDITION, condition)


def _start_regimens(run: _Enumeration) -> Iterator[tuple[History, dict[str, str]]]:
    """Every history with a safe regimen and its ``R_0``, in deterministic enumeration order."""
    for conditions in _subsets(run.tables.conditions, smallest=1):
        for risk_factors in _subsets(run.tables.risk_factors, smallest=0):
            state = (conditions, risk_factors)
            best = _cheapest(_cached_safe(run, state, conditions), run.costs)
            if best is not None:
                regimen = dict(zip(conditions, best, strict=True))
                yield History(conditions, risk_factors), regimen


@dataclass(slots=True)
class _EntryBuilder:
    """A ``ProblemEntry`` while the enumeration is still adding occurrences to it."""

    problem: TransitionProblem | None
    category: Category
    occurrences: list[tuple[History, Event]]


def enumerate_problems(
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    pharmacokinetics: Mapping[str, Pharmacokinetics],
) -> tuple[ProblemEntry, ...]:
    """Enumerate every (history, event) pair, classify it and merge pairs that give the same CSP.

    Histories are every non-empty set of conditions times every set of risk factors (subsets by
    size, then lexicographic) that has a safe regimen; ``R_0`` is the cheapest one. Each history
    gets one event per absent risk factor, then one per absent condition. Pairs are keyed by
    what the CSP sees (conditions, start regimen, acceptable drugs); ``REFER`` pairs use the same
    key and carry no problem.

    Parameters
    ----------
    formulary
        The formulary.
    costs
        The monthly cost of every drug (to choose ``R_0``).
    pharmacokinetics
        The washout and taper days of every drug.

    Returns
    -------
    tuple[ProblemEntry, ...]
        One entry per distinct CSP, in order of first occurrence.

    Raises
    ------
    InstanceError
        If a candidate drug has no cost or no pharmacokinetics, or two pairs with the same key
        get different categories (a broken invariant of the classification).
    """
    tables = _tables(formulary)
    run = _Enumeration(tables, _cost_table(tables, costs))
    entries: dict[_CspKey, _EntryBuilder] = {}
    for history, regimen in _start_regimens(run):
        for event in _events(tables, history):
            outcome = _outcome(run, history, regimen, event)
            entry = entries.get(outcome.key)
            if entry is None:
                problem = None
                if outcome.category is not Category.REFER:
                    problem = _build_problem(tables, pharmacokinetics, history, event, outcome)
                entries[outcome.key] = _EntryBuilder(problem, outcome.category, [(history, event)])
                continue
            if entry.category is not outcome.category:
                raise InstanceError(
                    f"{_problem_id(history, event)} is {outcome.category.value} but its CSP was "
                    f"classified {entry.category.value} before"
                )
            entry.occurrences.append((history, event))
    return tuple(
        ProblemEntry(e.problem, e.category, tuple(e.occurrences)) for e in entries.values()
    )


def census(entries: Sequence[ProblemEntry]) -> tuple[CensusRow, ...]:
    """Count the events of each kind in each category (the table of design §5).

    Parameters
    ----------
    entries
        The output of :func:`enumerate_problems`.

    Returns
    -------
    tuple[CensusRow, ...]
        One row per ``EventKind``, ``NEW_RISK_FACTOR`` first; counts are occurrences (events).
    """
    counts = {(kind, category): 0 for kind in EventKind for category in Category}
    for entry in entries:
        for _history, event in entry.occurrences:
            counts[event.kind, entry.category] += 1
    return tuple(
        CensusRow(
            event_kind=kind,
            events=sum(counts[kind, category] for category in Category),
            refer=counts[kind, Category.REFER],
            trivial=counts[kind, Category.TRIVIAL],
            forced_only=counts[kind, Category.FORCED_ONLY],
            knock_on=counts[kind, Category.KNOCK_ON],
        )
        for kind in (EventKind.NEW_RISK_FACTOR, EventKind.NEW_CONDITION)
    )


@dataclass(frozen=True, slots=True)
class ContinuityReport:
    """How EJ4's ``R_0`` compares with P1's EJ3 regimen over the fully observed histories.

    History identifiers have the form ``"PAIN|AGE65+EPI+LIVER"`` (conditions, then risk factors,
    ``-`` for none).

    Parameters
    ----------
    histories
        The number of histories compared.
    equal
        (a) EJ3 prescribes one drug per condition and ``R_0`` is that regimen.
    different
        (a') EJ3 prescribes one drug per condition and ``R_0`` differs or is missing (expected
        empty).
    companion
        (b) EJ3 prescribes a companion drug whose condition the history lacks, under the risk
        factor of a co-prescription row (one drug per condition cannot express it).
    refer
        (c) EJ3 refers and EJ4 has no safe regimen.
    unexpected
        Every other history (expected empty): EJ3 refers while EJ4 has a regimen, EJ3 requests a
        test, or EJ3's regimen breaks one drug per condition without a companion.
    """

    histories: int
    equal: int
    different: tuple[str, ...]
    companion: tuple[str, ...]
    refer: int
    unexpected: tuple[str, ...]


def _history_id(history: History) -> str:
    """Return a readable identifier such as ``PAIN|AGE65+LIVER``."""
    return f"{'+'.join(history.conditions)}|{'+'.join(history.risk_factors) or '-'}"


def _all_histories(formulary: Formulary) -> Iterator[History]:
    """Every non-empty set of conditions times every set of risk factors, in enumeration order."""
    for conditions in _subsets(tuple(sorted(formulary.condition_ids)), smallest=1):
        for risk_factors in _subsets(tuple(sorted(formulary.risk_factor_ids)), smallest=0):
            yield History(conditions, risk_factors)


def _observed_encounter(formulary: Formulary, history: History) -> Encounter:
    """Build the fully observed record of a history: each risk factor is present or absent."""
    return Encounter(
        encounter_id="synthetic",
        patient_id="synthetic",
        seq=1,
        visit_date=_SYNTHETIC_DATE,
        age_years=0,
        conditions=frozenset(history.conditions),
        risk_factors={
            r: RiskStatus.PRESENT if r in history.risk_factors else RiskStatus.ABSENT
            for r in formulary.risk_factor_ids
        },
        note="generated record (EJ4 continuity with EJ3)",
    )


def _one_per_condition(
    condition_of: Mapping[str, str], history: History, regimen: frozenset[str]
) -> dict[str, str] | None:
    """Return the regimen as condition -> drug if it gives exactly one drug per condition."""
    by_condition: dict[str, list[str]] = {}
    for drug in sorted(regimen):
        by_condition.setdefault(condition_of[drug], []).append(drug)
    if set(by_condition) != set(history.conditions):
        return None
    if any(len(drugs) != 1 for drugs in by_condition.values()):
        return None
    return {c: drugs[0] for c, drugs in by_condition.items()}


def _uses_a_companion(
    formulary: Formulary,
    condition_of: Mapping[str, str],
    history: History,
    regimen: frozenset[str],
) -> bool:
    """Whether a co-prescription row puts in ``regimen`` a companion no condition needs."""
    return any(
        row.drug_id in regimen
        and row.risk_factor_id in history.risk_factors
        and row.companion_drug_id in regimen
        and condition_of[row.companion_drug_id] not in history.conditions
        for row in formulary.coprescriptions
    )


def continuity_with_ej3(
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    ej3_formulary: Formulary,
    ej3_costs: Mapping[str, int],
    *,
    prune: bool = False,
    histories: Sequence[History] | None = None,
) -> ContinuityReport:
    """Compare ``R_0`` with the regimen P1's EJ3 agent prescribes on every fully observed history.

    The EJ3 agent (``CostAwareAgent`` with A*) decides each history as a fully observed encounter;
    its regimen is classified as (a) one drug per condition and equal to ``R_0``, (a') one drug
    per condition but different, (b) with a companion no condition needs, or (c) a referral with
    no ``R_0`` either. Agent 1's domain pruning (``prune``) only removes drugs Agent 1 proved
    essential or excluded; it is off by default because it makes each decision ~35 times slower.

    Parameters
    ----------
    formulary
        EJ4's formulary (for ``R_0``).
    costs
        EJ4's monthly cost of every drug.
    ej3_formulary
        The formulary as EJ3 loads it (data 1.2.0).
    ej3_costs
        EJ3's cost of every drug, in euro cents per month.
    prune
        Whether the EJ3 agent prunes its search with Agent 1's classification (EJ3's default).
    histories
        The histories to compare; ``None`` means every non-empty set of conditions times every set
        of risk factors of ``ej3_formulary`` (4032 today).

    Returns
    -------
    ContinuityReport
        The counts of each case and the identifiers of (a'), (b) and the unexpected histories.

    Raises
    ------
    InstanceError
        If an identifier is not in ``formulary`` or a candidate drug has no cost.
    """
    # Imported here so that importing this module does not load P1's agents.
    from symbolic_ai.p1_ej1_logic import Action, DPLLSolver
    from symbolic_ai.p1_ej3_search import CostAwareAgent

    agent = CostAwareAgent(ej3_formulary, ej3_costs, DPLLSolver(), prune=prune)
    tables = _tables(formulary)
    run = _Enumeration(tables, _cost_table(tables, costs))
    condition_of = {d: c for c, drugs in ej3_formulary.candidates.items() for d in drugs}
    counts = {"equal": 0, "refer": 0}
    lists: dict[str, list[str]] = {"different": [], "companion": [], "unexpected": []}
    chosen = list(_all_histories(ej3_formulary)) if histories is None else list(histories)
    for history in chosen:
        decision = agent.decide(_observed_encounter(ej3_formulary, history))
        state = (history.conditions, history.risk_factors)
        best = _cheapest(_cached_safe(run, state, history.conditions), run.costs)
        start = None if best is None else dict(zip(history.conditions, best, strict=True))
        if decision.action is not Action.PRESCRIBE:
            is_refer = decision.action is Action.REFER and start is None
            if is_refer:
                counts["refer"] += 1
            else:
                lists["unexpected"].append(_history_id(history))
            continue
        regimen = _one_per_condition(condition_of, history, decision.regimen)
        if regimen is not None:
            if regimen == start:
                counts["equal"] += 1
            else:
                lists["different"].append(_history_id(history))
        elif _uses_a_companion(ej3_formulary, condition_of, history, decision.regimen):
            lists["companion"].append(_history_id(history))
        else:
            lists["unexpected"].append(_history_id(history))
    return ContinuityReport(
        histories=len(chosen),
        equal=counts["equal"],
        different=tuple(lists["different"]),
        companion=tuple(lists["companion"]),
        refer=counts["refer"],
        unexpected=tuple(lists["unexpected"]),
    )


def _transition_for(
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    pharmacokinetics: Mapping[str, Pharmacokinetics],
    history: History,
    event: Event,
) -> tuple[TransitionProblem | None, Category]:
    """Return the problem (``None`` for ``REFER``) and category of one (history, event) pair.

    Raises ``InstanceError`` if the history has no safe regimen or the event adds nothing new.
    """
    tables = _tables(formulary)
    run = _Enumeration(tables, _cost_table(tables, costs))
    state = (history.conditions, history.risk_factors)
    best = _cheapest(_cached_safe(run, state, history.conditions), run.costs)
    if best is None:
        raise InstanceError(f"history {history} has no safe regimen to start from")
    regimen = dict(zip(history.conditions, best, strict=True))
    outcome = _outcome(run, history, regimen, event)
    if outcome.category is Category.REFER:
        return None, outcome.category
    return _build_problem(tables, pharmacokinetics, history, event, outcome), outcome.category


def case_study_problem(
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    pharmacokinetics: Mapping[str, Pharmacokinetics],
) -> TransitionProblem:
    """Return the worked example: depression and severe liver disease, then chronic pain.

    History ``({DEP}, {LIVER})`` with ``R_0 = {DEP: sertraline}``, event ``NEW_CONDITION PAIN``
    (design §1 and R1). It is a ``KNOCK_ON`` problem: both acceptable analgesics interact with
    sertraline, so the antidepressant must change first.

    Parameters
    ----------
    formulary
        The formulary.
    costs
        The monthly cost of every drug.
    pharmacokinetics
        The washout and taper days of every drug.

    Returns
    -------
    TransitionProblem
        The problem, with ``problem_id`` ``"DEP|LIVER|+cond:PAIN"``.

    Raises
    ------
    InstanceError
        If the data no longer give the case study a safe regimen after the event.
    """
    history = History(*_CASE_STUDY_HISTORY)
    problem, _category = _transition_for(
        formulary, costs, pharmacokinetics, history, _CASE_STUDY_EVENT
    )
    if problem is None:
        raise InstanceError("the case study has no safe regimen after the event")
    return problem
