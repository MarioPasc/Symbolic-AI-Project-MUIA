"""Seven toy transition problems with hand-computed minimum horizons (frozen known answers).

Toy drugs: ``x1, x2, x3`` treat condition X, ``y1, y2`` treat Y, ``z1`` treats Z. Unlisted washout
and taper days are 0. Each docstring line gives the reasoning a person can check on paper; the CSP
solver, the checker and the oracle of EJ4 are all tested against these answers (design §9.3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symbolic_ai.p2_ej4_transition.problem import Event, EventKind, TransitionProblem

DRUGS: tuple[str, ...] = ("x1", "x2", "x3", "y1", "y2", "z1")


@dataclass(frozen=True)
class KnownAnswer:
    """A toy problem and its minimum horizon (``None``: no safe transition exists)."""

    problem: TransitionProblem
    h_star: int | None
    why: str


def _pk(overrides: Mapping[str, int]) -> dict[str, int]:
    """Return a table with 0 days for every toy drug except the overrides."""
    table = dict.fromkeys(DRUGS, 0)
    table.update(overrides)
    return table


def _pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _x_only(
    problem_id: str,
    acceptable: tuple[str, ...],
    *,
    candidates: tuple[str, ...] = ("x1", "x2"),
    interactions: frozenset[frozenset[str]] = frozenset(),
    washout: Mapping[str, int] | None = None,
    taper: Mapping[str, int] | None = None,
) -> TransitionProblem:
    """A one-condition problem: X treated with x1, after the new risk factor R."""
    return TransitionProblem(
        problem_id=problem_id,
        event=Event(EventKind.NEW_RISK_FACTOR, "R"),
        conditions=("X",),
        risk_factors=("R",),
        start={"X": "x1"},
        candidates={"X": candidates},
        acceptable={"X": acceptable},
        interactions=interactions,
        washout_days=_pk(washout or {}),
        taper_days=_pk(taper or {}),
    )


T0 = KnownAnswer(
    _x_only("T0", ("x1", "x2")),
    0,
    "x1 is still acceptable and nothing else changed: day 0 already is a safe regimen.",
)
T1 = KnownAnswer(
    _x_only("T1", ("x2",)),
    1,
    "x1 is no longer acceptable and needs no taper: switch to x2 on day 1.",
)
T2 = KnownAnswer(
    _x_only("T2", ("x2",), taper={"x1": 2}),
    3,
    "x1 needs a 2-day taper: x1 taper day 1 (day 1), taper day 2 (day 2), x2 on day 3.",
)
T3 = KnownAnswer(
    TransitionProblem(
        problem_id="T3",
        event=Event(EventKind.NEW_CONDITION, "Y"),
        conditions=("X", "Y"),
        risk_factors=(),
        start={"X": "x1", "Y": None},
        candidates={"X": ("x1", "x2"), "Y": ("y1",)},
        acceptable={"X": ("x1", "x2"), "Y": ("y1",)},
        interactions=frozenset({_pair("x1", "y1")}),
        washout_days=_pk({"x1": 2}),
        taper_days=_pk({"x1": 1}),
    ),
    4,
    "y1 interacts with x1, so X must change first (knock-on): x1 taper day 1 on day 1, x2 on "
    "day 2; x1 was last given on day 1 and is present until day 3; y1 starts on day 4.",
)
T4 = KnownAnswer(
    _x_only(
        "T4",
        ("x2",),
        interactions=frozenset({_pair("x1", "x2")}),
        washout={"x1": 1},
    ),
    None,
    "x2 is the only acceptable drug and interacts with x1, which stays present for 1 day after its "
    "last dose; X can never be left untreated, so x2 can never start: no transition exists.",
)
T5 = KnownAnswer(
    TransitionProblem(
        problem_id="T5",
        event=Event(EventKind.NEW_CONDITION, "Z"),
        conditions=("X", "Y", "Z"),
        risk_factors=(),
        start={"X": "x1", "Y": "y1", "Z": None},
        candidates={"X": ("x1", "x2"), "Y": ("y1", "y2"), "Z": ("z1",)},
        acceptable={"X": ("x1", "x2"), "Y": ("y1", "y2"), "Z": ("z1",)},
        interactions=frozenset({_pair("x1", "z1"), _pair("y1", "z1")}),
        washout_days=_pk({"x1": 3, "y1": 2}),
        taper_days=_pk({}),
    ),
    4,
    "z1 interacts with x1 and y1: X and Y switch on day 1 and both washouts run at the same time "
    "(x1 present until day 3, y1 until day 2), so z1 starts on day 4 (not 3 + 2 + 1 days later).",
)
T6 = KnownAnswer(
    _x_only(
        "T6",
        ("x2",),
        candidates=("x1", "x2", "x3"),
        interactions=frozenset({_pair("x1", "x2")}),
        washout={"x1": 1},
    ),
    None,
    "As T4, plus x3, which is contraindicated (not acceptable): it may not be started even as a "
    "bridge, so no transition exists.",
)

KNOWN_ANSWERS: tuple[KnownAnswer, ...] = (T0, T1, T2, T3, T4, T5, T6)
