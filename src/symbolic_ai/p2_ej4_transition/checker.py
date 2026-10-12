"""Independent checker of EJ4: replay a day-by-day schedule against the rules in words.

The checker is written from the rules as the design states them in words (design §1, §4.2 and the
"equivalent reading" of constraint I in §4.3), not from the CSP encoding, so that it can catch a
CSP model that is too permissive or a solver that returns a wrong assignment (design §6.2, R2).
Each rule is one small function that walks the schedule day by day and returns the violations it
finds; :func:`check_schedule` runs them all and sorts the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from symbolic_ai.p2_ej4_transition.errors import VerificationError
from symbolic_ai.p2_ej4_transition.problem import (
    Day,
    Schedule,
    TransitionProblem,
    Value,
    drug_of,
    is_full_dose,
)

__all__ = ["RULES", "Rule", "Violation", "check_schedule"]

#: The rule a violation breaks (design §4.2-§4.3, stated in words in each rule's function).
Rule: TypeAlias = Literal["shape", "day0", "domain", "transition", "interaction", "final"]

#: The rules in the order they are reported when two violations share a day and a condition.
RULES: tuple[Rule, ...] = ("shape", "day0", "domain", "transition", "interaction", "final")

#: The well-formed cells of a schedule with a known drug (or untreated), keyed (condition, day).
_Grid: TypeAlias = dict[tuple[str, Day], Value]


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule in a schedule.

    Parameters
    ----------
    rule
        The rule broken (one of :data:`RULES`).
    condition
        The condition involved, or ``None`` when the violation names none (an unexpected key).
    day
        The day involved, or ``None`` when the violation names none.
    detail
        A human-readable explanation naming the values involved.
    """

    rule: Rule
    condition: str | None
    day: Day | None
    detail: str


def check_schedule(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> tuple[Violation, ...]:
    """Replay ``schedule`` over days ``0..horizon`` and return every rule it breaks.

    The rules are those of the design in words (§4.2, §4.3): ``shape`` (exactly one well-formed
    value per condition and day), ``day0`` (day 0 is the regimen before the event), ``domain``
    (candidate drugs, valid taper phases, untreated only for the new condition, no new
    contraindicated drug on intermediate days), ``transition`` (keep, switch, taper or start, one
    day at a time), ``interaction`` (no drug given on a day interacts with a drug present that day)
    and ``final`` (every condition on an acceptable drug at full dose on the last day). Cells that
    break ``shape`` are left out of the other rules, so one bad cell is reported once.

    Parameters
    ----------
    problem
        The transition problem the schedule claims to solve.
    schedule
        The values keyed ``(condition_id, day)``.
    horizon
        The last day ``H`` of the schedule (``>= 0``).

    Returns
    -------
    tuple[Violation, ...]
        Every violation, sorted by day, then condition, then rule (in :data:`RULES` order), then
        detail; empty when the schedule is valid.

    Raises
    ------
    VerificationError
        If ``horizon`` is not a non-negative integer.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise VerificationError(f"horizon must be a non-negative integer, got {horizon!r}")
    grid, violations = _check_shape(problem, schedule, horizon)
    violations += _check_day0(problem, grid)
    violations += _check_domain(problem, grid, horizon)
    violations += _check_transition(problem, grid, horizon)
    violations += _check_interaction(problem, grid, horizon)
    violations += _check_final(problem, grid, horizon)
    return tuple(sorted(violations, key=_report_order))


def _check_shape(
    problem: TransitionProblem, schedule: Schedule, horizon: int
) -> tuple[_Grid, list[Violation]]:
    """Rule *shape*: exactly the keys ``(c, t)``, ``t = 0..H``, each a well-formed value.

    A well-formed value is ``None`` or a pair ``(drug_id, phase)`` with a string and an integer;
    its drug must be a candidate of some condition of the problem. Returns the well-formed cells
    (the grid the other rules read) and the violations.
    """
    known_drugs = {
        drug for condition in problem.conditions for drug in problem.candidates[condition]
    }
    expected = {(c, t) for c in problem.conditions for t in range(horizon + 1)}
    grid: _Grid = {}
    violations: list[Violation] = []
    for condition, day in sorted(expected, key=lambda key: (key[1], key[0])):
        if (condition, day) not in schedule:
            violations.append(Violation("shape", condition, day, "missing value"))
    for key, value in schedule.items():
        if key not in expected:
            where, when = _parse_key(key)
            violations.append(Violation("shape", where, when, f"unexpected key {key!r}"))
        elif not _is_well_formed(value):
            violations.append(Violation("shape", key[0], key[1], f"malformed value {value!r}"))
        elif value is not None and value[0] not in known_drugs:
            violations.append(Violation("shape", key[0], key[1], f"unknown drug {value[0]!r}"))
        else:
            grid[key] = value
    return grid, violations


def _check_day0(problem: TransitionProblem, grid: _Grid) -> list[Violation]:
    """Rule *day0*: day 0 is the regimen before the event, ``(R_0(c), 0)`` or untreated."""
    violations: list[Violation] = []
    for condition in problem.conditions:
        start = problem.start[condition]
        expected: Value = None if start is None else (start, 0)
        if (condition, 0) in grid and grid[(condition, 0)] != expected:
            detail = f"day 0 is {_show(grid[(condition, 0)])}, the regimen before the event is "
            violations.append(Violation("day0", condition, 0, detail + _show(expected)))
    return violations


def _check_domain(problem: TransitionProblem, grid: _Grid, horizon: int) -> list[Violation]:
    """Rule *domain*: what each value may be, condition by condition and day by day.

    The drug is a candidate of its condition; a taper phase ``k`` satisfies ``1 <= k <= tau_d``;
    untreated appears only for the newly diagnosed condition; on days ``0 < t < H`` a drug other
    than the start drug must be acceptable (no *new* contraindicated drug; the start drug may stay
    while it is withdrawn). Day 0 and day ``H`` have their own rules (*day0*, *final*).
    """
    violations: list[Violation] = []
    for (condition, day), value in grid.items():
        for detail in _domain_problems(problem, condition, day, value, horizon):
            violations.append(Violation("domain", condition, day, detail))
    return violations


def _domain_problems(
    problem: TransitionProblem, condition: str, day: Day, value: Value, horizon: int
) -> list[str]:
    """Return what is wrong with ``value`` for ``condition`` on ``day`` (rule *domain*)."""
    start = problem.start[condition]
    if value is None:
        return (
            [] if start is None else ["untreated, but the condition was treated before the event"]
        )
    drug, phase = value
    if drug not in problem.candidates[condition]:
        return [f"{drug} is not a candidate of {condition}"]
    found: list[str] = []
    if not 0 <= phase <= problem.taper_days[drug]:
        found.append(f"phase {phase} outside 0..{problem.taper_days[drug]} (taper days of {drug})")
    if 0 < day < horizon and drug != start and drug not in problem.acceptable[condition]:
        found.append(f"{drug} is newly given but contraindicated (not acceptable)")
    return found


def _check_transition(problem: TransitionProblem, grid: _Grid, horizon: int) -> list[Violation]:
    """Rule *transition*: each condition goes from day ``t - 1`` to day ``t`` by an allowed step."""
    violations: list[Violation] = []
    for condition in problem.conditions:
        for day in range(1, horizon + 1):
            before, after = (condition, day - 1), (condition, day)
            if before not in grid or after not in grid:
                continue
            if not _is_allowed_step(problem, grid[before], grid[after]):
                detail = f"{_show(grid[before])} -> {_show(grid[after])} is not an allowed step"
                violations.append(Violation("transition", condition, day, detail))
    return violations


def _is_allowed_step(problem: TransitionProblem, before: Value, after: Value) -> bool:
    """Return whether one condition may go from ``before`` to ``after`` in one day.

    In words (design §4.3, T): a full-dose drug is kept, or switched to another drug at full dose
    if it needs no taper, or its taper starts (phase 1); a taper advances one phase per day; after
    the last taper phase the condition switches to another drug at full dose; an untreated new
    condition stays untreated or starts a drug at full dose. Nothing else is allowed. Whether the
    new drug is a candidate is rule *domain*'s concern.
    """
    if before is None:
        return after is None or after[1] == 0
    if after is None:
        return False  # a treated condition is never left untreated
    (drug, phase), (next_drug, next_phase) = before, after
    taper = problem.taper_days[drug]
    if phase == 0 and next_drug == drug:
        return next_phase == 0 or (next_phase == 1 and taper >= 1)
    if phase == 0:
        return next_phase == 0 and taper == 0
    if next_drug == drug:
        return 1 <= phase < taper and next_phase == phase + 1
    return phase == taper and next_phase == 0


def _check_interaction(problem: TransitionProblem, grid: _Grid, horizon: int) -> list[Violation]:
    """Rule *interaction*: on every day, no drug given that day interacts with a drug present.

    The equivalent reading of constraint I (design §4.3): a drug is present on every day it is
    given (any phase) and on the ``w_d`` days after its last dose, i.e. a drug last given on day
    ``s`` is present through day ``s + w_d`` inclusive. Two drugs of the same condition count (the
    old drug is still present after a switch). A pair given on the same day is reported once.
    """
    last_given: dict[str, Day] = {}
    violations: list[Violation] = []
    for day in range(horizon + 1):
        given = _given_on(problem, grid, day)
        last_given.update(dict.fromkeys(given, day))
        present = sorted(d for d, s in last_given.items() if day <= s + problem.washout_days[d])
        reported: set[frozenset[str]] = set()
        for drug, condition in given.items():
            for other in present:
                pair = frozenset((drug, other))
                if other == drug or pair not in problem.interactions or pair in reported:
                    continue
                reported.add(pair)
                how = (
                    "given"
                    if other in given
                    else f"present since its dose on day {last_given[other]}"
                )
                detail = f"{drug} interacts with {other} ({how})"
                violations.append(Violation("interaction", condition, day, detail))
    return violations


def _given_on(problem: TransitionProblem, grid: _Grid, day: Day) -> dict[str, str]:
    """Return the drugs given on ``day`` (any phase) with their condition, in condition order."""
    given: dict[str, str] = {}
    for condition in problem.conditions:
        drug = drug_of(grid.get((condition, day)))
        if drug is not None:
            given[drug] = condition
    return given


def _check_final(problem: TransitionProblem, grid: _Grid, horizon: int) -> list[Violation]:
    """Rule *final*: on day ``H`` every condition is on an acceptable drug at full dose."""
    violations: list[Violation] = []
    for condition in problem.conditions:
        key = (condition, horizon)
        if key not in grid:
            continue
        value = grid[key]
        if not (is_full_dose(value) and drug_of(value) in problem.acceptable[condition]):
            detail = f"ends on {_show(value)}, not an acceptable drug at full dose"
            violations.append(Violation("final", condition, horizon, detail))
    return violations


def _is_well_formed(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a ``(str, int)`` pair (a ``bool`` is no int here)."""
    if value is None:
        return True
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _parse_key(key: object) -> tuple[str | None, Day | None]:
    """Return the condition and day of a schedule key, or ``None`` for a part it does not have."""
    if not isinstance(key, tuple) or len(key) != 2:
        return None, None
    condition = key[0] if isinstance(key[0], str) else None
    day = key[1] if isinstance(key[1], int) and not isinstance(key[1], bool) else None
    return condition, day


def _show(value: Value) -> str:
    """Return a value as text: ``untreated``, ``x1`` (full dose) or ``x1 taper day 2``."""
    if value is None:
        return "untreated"
    drug, phase = value
    return drug if phase == 0 else f"{drug} taper day {phase}"


def _report_order(violation: Violation) -> tuple[int, str, int, str]:
    """Sort key: day, then condition, then rule, then detail (missing parts sort first)."""
    day = -1 if violation.day is None else violation.day
    condition = "" if violation.condition is None else violation.condition
    return day, condition, RULES.index(violation.rule), violation.detail
