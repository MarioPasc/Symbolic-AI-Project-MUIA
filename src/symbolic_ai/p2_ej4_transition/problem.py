"""The frozen contract of EJ4: one patient after one event and the day-by-day answer, as plain data.

A ``TransitionProblem`` is everything the transition CSP needs about one patient right after an
event (a new risk factor or a new condition): the conditions to treat, the regimen taken before
the event, the drugs acceptable afterwards, the interactions and the pharmacokinetic data. A
``Schedule`` is an answer: what is given for every condition on every day ``0..H``. Every module of
EJ4 (instances, CSP builder, solver, checker, oracle) exchanges only these types, which is what
lets the CSP model, the checker and the oracle be written independently of each other.

The model is a CSP indexed by day (AIMA 4th ed. ch. 5; bounded planning encoded as a CSP, 3rd ed.
§10.4.3). Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §4 and §9 (private docs).
This module is frozen: change it only through the orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from symbolic_ai.p2_ej4_transition.errors import ProblemError

__all__ = [
    "Day",
    "Event",
    "EventKind",
    "Schedule",
    "TransitionProblem",
    "Value",
    "drug_of",
    "is_full_dose",
]

#: A day of the transition: 0 is the day of the event; the horizon ``H`` is the first day on which
#: every condition is on an acceptable drug at full dose.
Day: TypeAlias = int

#: What is given for one condition on one day: ``(drug_id, 0)`` is the drug at full dose,
#: ``(drug_id, k)`` with ``k >= 1`` is day ``k`` of its taper, and ``None`` means untreated (allowed
#: only for the newly diagnosed condition, before its first drug).
Value: TypeAlias = tuple[str, int] | None

#: An answer: the value of every condition on every day ``0..H``, keyed by ``(condition_id, day)``.
Schedule: TypeAlias = Mapping[tuple[str, Day], Value]


class EventKind(Enum):
    """The two kinds of event that start a transition (design decision D20)."""

    NEW_RISK_FACTOR = "new_risk_factor"
    NEW_CONDITION = "new_condition"


@dataclass(frozen=True, slots=True)
class Event:
    """The change in the patient's record that makes the regimen change.

    Parameters
    ----------
    kind
        Whether a risk factor appeared or a condition was diagnosed.
    target_id
        The ``risk_factor_id`` or the ``condition_id`` that appeared.
    """

    kind: EventKind
    target_id: str


@dataclass(frozen=True, slots=True)
class TransitionProblem:
    """One patient right after one event: the data of the transition CSP.

    Parameters
    ----------
    problem_id
        A stable, human-readable identifier (unique within one enumeration).
    event
        The event that starts the transition.
    conditions
        The conditions to treat after the event, sorted, without repetition.
    risk_factors
        The patient's risk factors after the event, sorted, without repetition.
    start
        ``R_0``: the drug given for each condition before the event, or ``None`` for the newly
        diagnosed condition. Keys are exactly ``conditions``.
    candidates
        ``D_c``: the candidate drugs of each condition (sorted). Keys are exactly ``conditions``.
    acceptable
        ``Acc_c``: the candidates not contraindicated after the event (sorted, a subset of
        ``candidates[c]``; may be empty, in which case no transition exists).
    interactions
        The adverse interactions as unordered pairs of drug identifiers.
    washout_days
        Days a drug stays present after its last dose, for every candidate drug (``>= 0``).
    taper_days
        Days of reduced dose before a drug may be stopped, for every candidate drug (``0`` means it
        can be stopped at once).

    Raises
    ------
    ProblemError
        If any of the invariants above, or the consistency between ``event`` and ``start``, fails.
    """

    problem_id: str
    event: Event
    conditions: tuple[str, ...]
    risk_factors: tuple[str, ...]
    start: Mapping[str, str | None]
    candidates: Mapping[str, tuple[str, ...]]
    acceptable: Mapping[str, tuple[str, ...]]
    interactions: frozenset[frozenset[str]]
    washout_days: Mapping[str, int]
    taper_days: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate the invariants of the contract (see the class docstring)."""
        _check(bool(self.problem_id), "problem_id is empty")
        _check_sorted_unique(self.conditions, "conditions")
        _check(bool(self.conditions), "conditions is empty")
        _check_sorted_unique(self.risk_factors, "risk_factors")
        keys = set(self.conditions)
        for name, mapping in (
            ("start", self.start),
            ("candidates", self.candidates),
            ("acceptable", self.acceptable),
        ):
            _check(set(mapping) == keys, f"{name} keys differ from conditions")
        for condition in self.conditions:
            self._check_condition(condition)
        self._check_event()
        for pair in self.interactions:
            _check(len(pair) == 2, f"interaction {sorted(pair)} is not a pair of distinct drugs")

    def _check_condition(self, condition: str) -> None:
        """Validate candidates, acceptable drugs, start drug and pharmacokinetics of a condition."""
        candidates = self.candidates[condition]
        _check(bool(candidates), f"condition {condition} has no candidate")
        _check_sorted_unique(candidates, f"candidates[{condition}]")
        _check_sorted_unique(self.acceptable[condition], f"acceptable[{condition}]")
        _check(
            set(self.acceptable[condition]) <= set(candidates),
            f"acceptable[{condition}] is not a subset of its candidates",
        )
        drug = self.start[condition]
        _check(drug is None or drug in candidates, f"start[{condition}] is not a candidate")
        for candidate in candidates:
            for name, table in (
                ("washout_days", self.washout_days),
                ("taper_days", self.taper_days),
            ):
                days = table.get(candidate)
                _check(days is not None and days >= 0, f"{name}[{candidate}] missing or negative")

    def _check_event(self) -> None:
        """Validate that the event agrees with the conditions, risk factors and start regimen."""
        untreated = [c for c in self.conditions if self.start[c] is None]
        if self.event.kind is EventKind.NEW_CONDITION:
            _check(
                untreated == [self.event.target_id],
                "a new-condition event needs exactly its condition untreated at the start",
            )
        else:
            _check(self.event.target_id in self.risk_factors, "the new risk factor is not listed")
            _check(not untreated, "a new-risk-factor event leaves no condition untreated")


def drug_of(value: Value) -> str | None:
    """Return the drug of a value (full dose or tapered), or ``None`` for an untreated condition.

    Parameters
    ----------
    value
        A value of the contract.

    Returns
    -------
    str or None
        The drug identifier, or ``None``.
    """
    return None if value is None else value[0]


def is_full_dose(value: Value) -> bool:
    """Return whether a value is a drug at full dose (phase 0).

    Parameters
    ----------
    value
        A value of the contract.

    Returns
    -------
    bool
        ``True`` for ``(drug_id, 0)``, ``False`` for a taper day or ``None``.
    """
    return value is not None and value[1] == 0


def _check(condition: bool, message: str) -> None:
    """Raise ``ProblemError(message)`` unless ``condition`` holds."""
    if not condition:
        raise ProblemError(message)


def _check_sorted_unique(items: tuple[str, ...], name: str) -> None:
    """Raise ``ProblemError`` unless ``items`` is sorted and has no repetition."""
    _check(list(items) == sorted(set(items)), f"{name} must be sorted and without repetition")
