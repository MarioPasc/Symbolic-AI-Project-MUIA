"""The experiments of EJ4 (design §7): validation R2, algorithm table R3 and the context numbers.

- **R2, "the model means what we say".** On every deduplicated problem that is not ``REFER``, the
  minimum horizon H* of the CSP (main configuration) is compared with the independent oracle's
  (breadth-first search over daily states), every CSP schedule is replayed by the independent
  checker, and each schedule is mutated in targeted ways that are invalid by construction: the
  checker must reject every mutant with the rule the mutation breaks (design §6.2, §7 R2).
- **R3, what consistency and search contribute.** Four configurations of aima's solver, from plain
  backtracking to the main configuration, at ``H* - 1`` and ``H*`` on two real instances, with a
  deterministic assignment budget; and min-conflicts, which can find a schedule but never prove
  that there is none (design §7 R3).
- **Context numbers.** The distribution of H* counted in events, by event kind and category.

Every function is pure and returns frozen records; ``main.py`` writes them. Wall-clock times are
kept apart from the deterministic numbers (:attr:`AlgorithmRun.seconds` is not serialised).
Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §5, §6.2, §7 (private docs).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from symbolic_ai.p2_ej4_transition.checker import Rule, check_schedule
from symbolic_ai.p2_ej4_transition.csp_builder import Variable
from symbolic_ai.p2_ej4_transition.instances import Category, CensusRow, ProblemEntry
from symbolic_ai.p2_ej4_transition.oracle import oracle_min_horizon
from symbolic_ai.p2_ej4_transition.problem import (
    Schedule,
    TransitionProblem,
    Value,
    drug_of,
)
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    HorizonRun,
    SolverConfig,
    TransitionResult,
    find_min_horizon,
    solve_horizon,
    solve_min_conflicts,
)

__all__ = [
    "DESIGN_CENSUS",
    "EXTENDED_CASE_ID",
    "MUTATIONS",
    "R3_BUDGET",
    "R3_CONFIGS",
    "R3_TIMEOUT_S",
    "AlgorithmRun",
    "CheckerSummary",
    "ContextNumbers",
    "Disagreement",
    "HStarCount",
    "Mutant",
    "Mutation",
    "MutationSummary",
    "OracleComparison",
    "ValidationResult",
    "census_matches_design",
    "context_numbers",
    "entries_in_category",
    "find_entry",
    "mutate_contraindicated_drug",
    "mutate_day0",
    "mutate_delete_key",
    "mutate_drop_taper_day",
    "mutate_final",
    "mutate_start_early",
    "mutate_untreated_day",
    "run_algorithm_table",
    "run_min_conflicts_rows",
    "run_validation",
]

logger = logging.getLogger(__name__)

#: The second instance of R3 (design §7 R3): a five-condition patient on sertraline and paracetamol
#: develops severe liver disease; the same knock-on mechanism as the worked example.
EXTENDED_CASE_ID = "DEP+GERD+HTN+PAIN+T2D|-|+rf:LIVER"

#: The census of design §5, per event kind: (events, refer, trivial, forced only, knock-on).
DESIGN_CENSUS: dict[str, tuple[int, int, int, int, int]] = {
    "new_risk_factor": (8896, 1664, 5472, 1600, 160),
    "new_condition": (8512, 1236, 0, 7084, 192),
}

#: R3's deterministic budget: a run stops after this many assignments (status ``budget``).
R3_BUDGET = 1_000_000
#: R3's wall-clock safety net; the budget, not the clock, is meant to stop plain backtracking.
R3_TIMEOUT_S = 60.0

#: The configurations of R3, in table order: each adds one technique of deck 04 to the previous.
R3_CONFIGS: tuple[tuple[str, SolverConfig], ...] = (
    ("bt", SolverConfig("none", "static", "keep_first", False, R3_TIMEOUT_S, R3_BUDGET)),
    ("bt_fc", SolverConfig("fc", "static", "keep_first", False, R3_TIMEOUT_S, R3_BUDGET)),
    ("bt_fc_mrv", SolverConfig("fc", "mrv", "keep_first", False, R3_TIMEOUT_S, R3_BUDGET)),
    (
        "main",
        dataclasses.replace(MAIN_CONFIG, timeout_s=R3_TIMEOUT_S, max_assignments=R3_BUDGET),
    ),
)

#: One mutant: where the mutation was applied (in words) and the mutated schedule.
Mutant: TypeAlias = tuple[str, dict[Variable, Value]]
#: A mutation generator: ``(problem, schedule, horizon) -> mutants`` (empty if not applicable).
MutationGenerator: TypeAlias = Callable[[TransitionProblem, Schedule, int], list[Mutant]]


# --- records ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Disagreement:
    """A problem on which the CSP and the oracle do not give the same answer.

    Parameters
    ----------
    problem_id
        The problem.
    csp_status
        The status of the CSP's horizon loop.
    csp_h_star
        The CSP's H*, or ``None``.
    oracle_status
        The oracle's status (``found``, ``impossible``, ``budget_exhausted``).
    oracle_h_star
        The oracle's H*, or ``None``.
    """

    problem_id: str
    csp_status: str
    csp_h_star: int | None
    oracle_status: str
    oracle_h_star: int | None


@dataclass(frozen=True, slots=True)
class OracleComparison:
    """The CSP's H* against the oracle's on every problem (design §7 R2).

    Parameters
    ----------
    problems
        Problems submitted to the oracle.
    compared
        Problems the oracle finished (``found`` or ``impossible``).
    agree
        Compared problems on which both give the same H*, or both say there is no transition.
    budget_exhausted
        Problems on which the oracle ran out of its state budget (not compared).
    disagreements
        Every compared problem on which they differ.
    """

    problems: int
    compared: int
    agree: int
    budget_exhausted: int
    disagreements: tuple[Disagreement, ...]


@dataclass(frozen=True, slots=True)
class CheckerSummary:
    """The independent checker on every CSP schedule.

    Parameters
    ----------
    schedules
        Schedules checked (one per solved problem).
    valid
        Schedules with no violation.
    invalid
        The problem identifiers of the schedules with a violation.
    """

    schedules: int
    valid: int
    invalid: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Mutation:
    """A targeted mutation of a valid schedule, invalid by construction.

    Parameters
    ----------
    name
        A short identifier, e.g. ``"drop_taper_day"``.
    expected_rule
        The checker rule the mutant breaks by construction.
    description
        What the mutation does, in words.
    generate
        The generator: every mutant of one schedule (empty where it does not apply).
    """

    name: str
    expected_rule: Rule
    description: str
    generate: MutationGenerator


@dataclass(frozen=True, slots=True)
class MutationSummary:
    """How the checker judged the mutants of one mutation.

    Parameters
    ----------
    name
        The mutation.
    expected_rule
        The rule it breaks by construction.
    description
        What it does.
    schedules
        Schedules on which it could be applied at least once.
    applied
        Mutants checked (a schedule may give several, one per place it applies).
    rejected
        Mutants with at least one violation.
    rejected_with_expected_rule
        Mutants whose violations include ``expected_rule``.
    rules_reported
        For every rule, how many mutants it was reported on (a mutant can break several rules).
    misses
        ``problem_id: place`` of every mutant not rejected with the expected rule.
    """

    name: str
    expected_rule: str
    description: str
    schedules: int
    applied: int
    rejected: int
    rejected_with_expected_rule: int
    rules_reported: tuple[tuple[str, int], ...]
    misses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HStarCount:
    """How many problems and events of one kind and category have a given H*.

    Parameters
    ----------
    h_star
        The minimum horizon (``None``: no transition found).
    event_kind
        ``new_risk_factor`` or ``new_condition``.
    category
        ``trivial``, ``forced_only`` or ``knock_on``.
    entries
        Deduplicated problems.
    events
        (history, event) pairs.
    """

    h_star: int | None
    event_kind: str
    category: str
    entries: int
    events: int


@dataclass(frozen=True, slots=True)
class ContextNumbers:
    """The distribution of H* over the population (design §7, context numbers).

    Parameters
    ----------
    histogram
        ``(H*, events)`` pairs, by H* (``None`` last).
    by_kind_and_category
        The same counts split by event kind and category.
    """

    histogram: tuple[tuple[int | None, int], ...]
    by_kind_and_category: tuple[HStarCount, ...]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """R2: the CSP against the oracle, the checker and the mutations (design §7 R2).

    Parameters
    ----------
    problems
        Deduplicated problems validated.
    events
        The (history, event) pairs they stand for.
    csp_status_counts
        How many problems ended with each status of the horizon loop.
    oracle
        The comparison with the oracle, or ``None`` when it was skipped.
    checker
        The checker on every CSP schedule.
    mutations
        One summary per mutation, in :data:`MUTATIONS` order.
    context
        The distribution of H* (design §7, context numbers).
    """

    problems: int
    events: int
    csp_status_counts: tuple[tuple[str, int], ...]
    oracle: OracleComparison | None
    checker: CheckerSummary
    mutations: tuple[MutationSummary, ...]
    context: ContextNumbers


@dataclass(frozen=True, slots=True)
class AlgorithmRun:
    """One row of R3: one configuration at one horizon on one instance.

    Parameters
    ----------
    instance_id
        The problem.
    configuration
        The configuration's key in :data:`R3_CONFIGS`, or ``"min_conflicts"``.
    label
        The configuration's label (``SolverConfig.label``, or min-conflicts with its seed).
    horizon
        ``H``.
    status
        ``solved``, ``infeasible``, ``budget``, ``timeout`` or ``not_found``.
    assignments
        Assignments made.
    checks
        Constraint checks.
    ac3_removed
        Values removed by AC-3 preprocessing.
    seconds
        Wall-clock time; machine-dependent, so left out of :meth:`to_dict`.
    """

    instance_id: str
    configuration: str
    label: str
    horizon: int
    status: str
    assignments: int
    checks: int
    ac3_removed: int
    seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic fields as a JSON-ready ``dict`` (no seconds)."""
        content = dataclasses.asdict(self)
        del content["seconds"]
        return content


# --- R2 -----------------------------------------------------------------------------------------


def run_validation(
    entries: Sequence[ProblemEntry],
    *,
    use_oracle: bool = True,
    config: SolverConfig = MAIN_CONFIG,
) -> ValidationResult:
    """Run R2 on every non-``REFER`` entry (design §7 R2).

    Parameters
    ----------
    entries
        Entries of ``instances.enumerate_problems`` (a subset is fine; ``REFER`` ones are skipped).
    use_oracle
        Whether to compare with the oracle (it dominates the run time).
    config
        The solver configuration (default: the main one).

    Returns
    -------
    ValidationResult
        The comparison, the checker, the mutations and the distribution of H*.
    """
    runs = [
        (entry, problem, find_min_horizon(problem, config))
        for entry in entries
        if (problem := entry.problem) is not None
    ]
    solved = [
        (problem, result.schedule, result.h_star)
        for _entry, problem, result in runs
        if result.schedule is not None and result.h_star is not None
    ]
    statuses = Counter(result.status for _entry, _problem, result in runs)
    return ValidationResult(
        problems=len(runs),
        events=sum(len(entry.occurrences) for entry, _problem, _result in runs),
        csp_status_counts=tuple(sorted(statuses.items())),
        oracle=_compare_with_oracle(runs) if use_oracle else None,
        checker=_check_all(solved),
        mutations=tuple(_run_mutation(m, solved) for m in MUTATIONS),
        context=context_numbers((entry, result) for entry, _problem, result in runs),
    )


def _compare_with_oracle(
    runs: Sequence[tuple[ProblemEntry, TransitionProblem, TransitionResult]],
) -> OracleComparison:
    """Compare each CSP H* with the oracle's (default state budget)."""
    compared = agree = exhausted = 0
    disagreements: list[Disagreement] = []
    for _entry, problem, result in runs:
        oracle = oracle_min_horizon(problem)
        if oracle.status == "budget_exhausted":
            exhausted += 1
            continue
        compared += 1
        both_found = oracle.status == "found" and result.status == "solved"
        both_none = oracle.status == "impossible" and result.status == "no_transition_within_h_max"
        if (both_found and oracle.h_star == result.h_star) or both_none:
            agree += 1
            continue
        disagreements.append(
            Disagreement(
                problem.problem_id, result.status, result.h_star, oracle.status, oracle.h_star
            )
        )
    return OracleComparison(len(runs), compared, agree, exhausted, tuple(disagreements))


def _check_all(solved: Sequence[tuple[TransitionProblem, Schedule, int]]) -> CheckerSummary:
    """Replay every CSP schedule with the checker."""
    invalid = tuple(
        problem.problem_id
        for problem, schedule, horizon in solved
        if check_schedule(problem, schedule, horizon)
    )
    return CheckerSummary(len(solved), len(solved) - len(invalid), invalid)


def _run_mutation(
    mutation: Mutation, solved: Sequence[tuple[TransitionProblem, Schedule, int]]
) -> MutationSummary:
    """Apply one mutation wherever it applies and check every mutant."""
    schedules = applied = rejected = with_expected = 0
    reported: Counter[str] = Counter()
    misses: list[str] = []
    for problem, schedule, horizon in solved:
        mutants = mutation.generate(problem, schedule, horizon)
        schedules += bool(mutants)
        for place, mutant in mutants:
            rules = {v.rule for v in check_schedule(problem, mutant, horizon)}
            applied += 1
            rejected += bool(rules)
            reported.update(rules)
            if mutation.expected_rule in rules:
                with_expected += 1
            else:
                misses.append(f"{problem.problem_id}: {place}")
    return MutationSummary(
        name=mutation.name,
        expected_rule=mutation.expected_rule,
        description=mutation.description,
        schedules=schedules,
        applied=applied,
        rejected=rejected,
        rejected_with_expected_rule=with_expected,
        rules_reported=tuple(sorted(reported.items())),
        misses=tuple(misses),
    )


def context_numbers(results: Iterable[tuple[ProblemEntry, TransitionResult]]) -> ContextNumbers:
    """Count the H* of every problem in events, overall and by event kind and category.

    Parameters
    ----------
    results
        ``(entry, horizon-loop result)`` pairs.

    Returns
    -------
    ContextNumbers
        The histogram and its split.
    """
    events: Counter[int | None] = Counter()
    split_entries: Counter[tuple[int | None, str, str]] = Counter()
    split_events: Counter[tuple[int | None, str, str]] = Counter()
    for entry, result in results:
        for _history, event in entry.occurrences:
            key = (result.h_star, event.kind.value, entry.category.value)
            events[result.h_star] += 1
            split_events[key] += 1
        kinds = {event.kind.value for _history, event in entry.occurrences}
        for kind in kinds:
            split_entries[result.h_star, kind, entry.category.value] += 1
    rows = tuple(
        HStarCount(h, kind, category, split_entries[h, kind, category], count)
        for (h, kind, category), count in sorted(split_events.items(), key=_h_star_key)
    )
    histogram = tuple(sorted(events.items(), key=lambda item: _none_last(item[0])))
    return ContextNumbers(histogram, rows)


def census_matches_design(rows: Sequence[CensusRow]) -> bool:
    """Return whether a census equals the table of design §5 (:data:`DESIGN_CENSUS`).

    Parameters
    ----------
    rows
        The output of ``instances.census``.

    Returns
    -------
    bool
        ``True`` if every row has the design's counts.
    """
    obtained = {
        row.event_kind.value: (row.events, row.refer, row.trivial, row.forced_only, row.knock_on)
        for row in rows
    }
    return obtained == DESIGN_CENSUS


def find_entry(entries: Sequence[ProblemEntry], problem_id: str) -> ProblemEntry | None:
    """Return the entry whose problem has ``problem_id``, or ``None``.

    Parameters
    ----------
    entries
        Entries of ``instances.enumerate_problems``.
    problem_id
        The identifier of the entry's problem (its first occurrence).

    Returns
    -------
    ProblemEntry or None
        The entry, or ``None`` if no problem has that identifier.
    """
    return next(
        (e for e in entries if e.problem is not None and e.problem.problem_id == problem_id),
        None,
    )


def _none_last(h_star: int | None) -> tuple[bool, int]:
    """Sort key that puts ``None`` (no transition) after every horizon."""
    return (h_star is None, h_star or 0)


def _h_star_key(
    item: tuple[tuple[int | None, str, str], int],
) -> tuple[tuple[bool, int], str, str]:
    """Sort key of a split count: H*, then event kind, then category."""
    (h_star, kind, category), _count = item
    return _none_last(h_star), kind, category


# --- mutations (R2) -----------------------------------------------------------------------------


def _with(schedule: Schedule, changes: dict[Variable, Value]) -> dict[Variable, Value]:
    """Return a copy of ``schedule`` with ``changes`` applied."""
    mutant = dict(schedule)
    mutant.update(changes)
    return mutant


def mutate_drop_taper_day(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> list[Mutant]:
    """Drop the last day of a taper: the next drug starts one day early, on that day.

    Where a condition is on ``d↓tau_d`` on day ``t`` (the last taper day) and switches on day
    ``t + 1``, day ``t`` gets the next day's value. The taper is then cut short, which rule
    *transition* forbids (``d↓(tau_d - 1)`` or ``d`` cannot be followed by another drug).
    """
    mutants: list[Mutant] = []
    for condition in problem.conditions:
        for day in range(1, horizon):
            value, after = schedule[condition, day], schedule[condition, day + 1]
            drug = drug_of(value)
            if value is None or drug is None or value[1] < 1:
                continue
            if value[1] == problem.taper_days[drug] and drug_of(after) != drug:
                mutants.append(
                    (f"{condition} day {day}", _with(schedule, {(condition, day): after}))
                )
    return mutants


def mutate_start_early(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> list[Mutant]:
    """Start a drug one day before the CSP starts it, where an interacting drug is present then.

    For every start or switch to ``e`` on day ``t >= 2`` (``x[c, t] = e`` at full dose and
    ``x[c, t-1]`` another drug or untreated), ``e`` is moved to day ``t - 1`` only if a drug that
    interacts with ``e`` is present on day ``t - 1`` in the rest of the schedule (given that day by
    another condition, or given on a day ``s`` with ``s <= t - 1 <= s + w``). The mutant breaks rule
    *interaction* on day ``t - 1``.
    """
    mutants: list[Mutant] = []
    for condition in problem.conditions:
        for day in range(2, horizon + 1):
            value = schedule[condition, day]
            if value is None or value[1] != 0 or drug_of(schedule[condition, day - 1]) == value[0]:
                continue
            present = _present_on(problem, schedule, day - 1, excluding=(condition, day - 1))
            if any(frozenset((value[0], other)) in problem.interactions for other in present):
                place = f"{condition} starts {value[0]} on day {day - 1}"
                mutants.append((place, _with(schedule, {(condition, day - 1): value})))
    return mutants


def _present_on(
    problem: TransitionProblem, schedule: Schedule, day: int, excluding: Variable
) -> set[str]:
    """Return the drugs present on ``day``: given on some ``s <= day`` with ``day <= s + w``."""
    present: set[str] = set()
    for (condition, given_day), value in schedule.items():
        drug = drug_of(value)
        if (condition, given_day) == excluding or drug is None:
            continue
        if given_day <= day <= given_day + problem.washout_days[drug]:
            present.add(drug)
    return present


def mutate_untreated_day(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> list[Mutant]:
    """Leave a treated condition untreated on one intermediate day (``0 < t < H``).

    Rule *transition* never lets a drug (full dose or tapered) be followed by "untreated".
    """
    return [
        (f"{condition} day {day}", _with(schedule, {(condition, day): None}))
        for condition in problem.conditions
        for day in range(1, horizon)
        if schedule[condition, day] is not None
    ]


def mutate_contraindicated_drug(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> list[Mutant]:
    """Give a contraindicated candidate (not acceptable, not the start drug) on an intermediate day.

    Node consistency (rule *domain*) allows a non-acceptable drug on ``0 < t < H`` only if it is
    the start drug, which may stay while it is withdrawn.
    """
    return [
        (f"{condition} day {day} {drug}", _with(schedule, {(condition, day): (drug, 0)}))
        for condition in problem.conditions
        for drug in problem.candidates[condition]
        if drug not in problem.acceptable[condition] and drug != problem.start[condition]
        for day in range(1, horizon)
    ]


def mutate_day0(problem: TransitionProblem, schedule: Schedule, horizon: int) -> list[Mutant]:
    """Change day 0 of each condition: another candidate, or untreated if there is none.

    Day 0 is the regimen before the event (rule *day0*): the start drug at full dose, or untreated
    for the newly diagnosed condition.
    """
    mutants: list[Mutant] = []
    for condition in problem.conditions:
        start = problem.start[condition]
        others = [d for d in problem.candidates[condition] if d != start]
        replacement: Value = (others[0], 0) if others else None
        if replacement == schedule[condition, 0]:
            continue
        mutants.append((f"{condition} day 0", _with(schedule, {(condition, 0): replacement})))
    return mutants


def mutate_final(problem: TransitionProblem, schedule: Schedule, horizon: int) -> list[Mutant]:
    """End a condition on a tapered value, or on a candidate that is not acceptable.

    Rule *final*: on day ``H`` every condition is on an acceptable drug at full dose.
    """
    mutants: list[Mutant] = []
    for condition in problem.conditions:
        drug = drug_of(schedule[condition, horizon])
        if drug is not None and problem.taper_days[drug] >= 1:
            tapered = _with(schedule, {(condition, horizon): (drug, 1)})
            mutants.append((f"{condition} ends on {drug}↓1", tapered))
        rejected = [
            d for d in problem.candidates[condition] if d not in problem.acceptable[condition]
        ]
        if rejected:
            excluded = _with(schedule, {(condition, horizon): (rejected[0], 0)})
            mutants.append((f"{condition} ends on {rejected[0]}", excluded))
    return mutants


def mutate_delete_key(problem: TransitionProblem, schedule: Schedule, horizon: int) -> list[Mutant]:
    """Delete the last day of each condition from the schedule (rule *shape*)."""
    mutants: list[Mutant] = []
    for condition in problem.conditions:
        mutant = dict(schedule)
        del mutant[condition, horizon]
        mutants.append((f"{condition} day {horizon} deleted", mutant))
    return mutants


#: The mutations of R2, one per rule of the checker (design §7 R2; ``checker.RULES``).
MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "drop_taper_day",
        "transition",
        "the last taper day is replaced by the next drug",
        mutate_drop_taper_day,
    ),
    Mutation(
        "start_early",
        "interaction",
        "a drug starts one day early, while an interacting drug is present",
        mutate_start_early,
    ),
    Mutation(
        "untreated_day",
        "transition",
        "a treated condition is left untreated on an intermediate day",
        mutate_untreated_day,
    ),
    Mutation(
        "contraindicated_drug",
        "domain",
        "a contraindicated candidate is given on an intermediate day",
        mutate_contraindicated_drug,
    ),
    Mutation("day0", "day0", "day 0 differs from the regimen before the event", mutate_day0),
    Mutation(
        "final",
        "final",
        "the last day is a taper day or a drug that is not acceptable",
        mutate_final,
    ),
    Mutation("delete_key", "shape", "the last day of a condition is missing", mutate_delete_key),
)


# --- R3 -----------------------------------------------------------------------------------------


def run_algorithm_table(
    instances: Sequence[tuple[TransitionProblem, int]],
    configs: Sequence[tuple[str, SolverConfig]] = R3_CONFIGS,
) -> tuple[AlgorithmRun, ...]:
    """Run every configuration at ``H* - 1`` and ``H*`` on every instance (design §7 R3).

    Parameters
    ----------
    instances
        ``(problem, H*)`` pairs, ``H* >= 1``, in table order.
    configs
        ``(key, configuration)`` pairs, in table order (default: :data:`R3_CONFIGS`).

    Returns
    -------
    tuple[AlgorithmRun, ...]
        One run per instance, configuration and horizon, in that nesting order.
    """
    runs: list[AlgorithmRun] = []
    for problem, h_star in instances:
        for key, config in configs:
            for horizon in (h_star - 1, h_star):
                run = solve_horizon(problem, horizon, config)
                logger.info("R3 %s %s H=%d: %s", problem.problem_id, key, horizon, run.status)
                runs.append(_algorithm_run(problem, key, config.label, run))
    return tuple(runs)


def run_min_conflicts_rows(
    problem: TransitionProblem, horizons: Sequence[int], max_steps: int = 10_000, seed: int = 0
) -> tuple[AlgorithmRun, ...]:
    """Run aima's min-conflicts at each horizon (design §7 R3, failure case 2).

    Parameters
    ----------
    problem
        The instance (the worked example).
    horizons
        The horizons, e.g. ``(H* - 1, H*)``.
    max_steps
        The step budget of each run.
    seed
        The seed of each run.

    Returns
    -------
    tuple[AlgorithmRun, ...]
        One run per horizon; ``not_found`` means unknown, never infeasible.
    """
    label = f"min-conflicts (seed {seed}, {max_steps} steps)"
    return tuple(
        _algorithm_run(
            problem, "min_conflicts", label, solve_min_conflicts(problem, h, max_steps, seed)
        )
        for h in horizons
    )


def _algorithm_run(
    problem: TransitionProblem, key: str, label: str, run: HorizonRun
) -> AlgorithmRun:
    """Return one R3 row from a solver run."""
    stats = run.stats
    return AlgorithmRun(
        problem.problem_id,
        key,
        label,
        run.horizon,
        run.status,
        stats.assignments,
        stats.checks,
        stats.ac3_removed,
        stats.seconds,
    )


def entries_in_category(
    entries: Iterable[ProblemEntry], category: Category
) -> tuple[ProblemEntry, ...]:
    """Return the entries of one category, in enumeration order.

    Parameters
    ----------
    entries
        Entries of ``instances.enumerate_problems``.
    category
        The category.

    Returns
    -------
    tuple[ProblemEntry, ...]
        The entries whose category is ``category``.
    """
    return tuple(e for e in entries if e.category is category)
