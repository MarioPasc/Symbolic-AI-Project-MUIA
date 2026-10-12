"""Tests of the frozen contract of EJ4 (``problem.py``)."""

from __future__ import annotations

import dataclasses
from typing import Any

import ej4_toys
import pytest

from symbolic_ai.p2_ej4_transition import (
    Event,
    EventKind,
    ProblemError,
    TransitionProblem,
    drug_of,
    is_full_dose,
)


def test_known_answers_are_valid_problems(known_answers: tuple[ej4_toys.KnownAnswer, ...]) -> None:
    assert [k.problem.problem_id for k in known_answers] == [f"T{i}" for i in range(7)]


@pytest.mark.parametrize(
    ("value", "drug", "full"),
    [(("x1", 0), "x1", True), (("x1", 2), "x1", False), (None, None, False)],
)
def test_value_helpers(value: tuple[str, int] | None, drug: str | None, full: bool) -> None:
    assert drug_of(value) == drug
    assert is_full_dose(value) is full


def _replace(problem: TransitionProblem, **changes: Any) -> TransitionProblem:
    return dataclasses.replace(problem, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"problem_id": ""},
        {"conditions": ()},
        {"conditions": ("X", "X")},
        {"risk_factors": ("S", "R")},
        {"start": {"X": "y1"}},
        {"acceptable": {"X": ("x1", "y1")}},
        {"candidates": {"X": ()}},
        {"washout_days": {"x1": -1, "x2": 0}},
        {"taper_days": {"x1": 0}},
        {"interactions": frozenset({frozenset({"x1"})})},
        {"event": Event(EventKind.NEW_RISK_FACTOR, "S")},
        {"event": Event(EventKind.NEW_CONDITION, "X")},
    ],
)
def test_invalid_problems_raise(changes: dict[str, Any]) -> None:
    with pytest.raises(ProblemError):
        _replace(ej4_toys.T1.problem, **changes)
