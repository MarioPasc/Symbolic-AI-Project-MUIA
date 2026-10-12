"""Deterministic random transition problems that exercise what the real formulary never does.

The real instances of EJ4 (``instances.py``) cover few of the model's paths: every H* there is
0, 1, 2 or 12, no contraindicated start drug needs a taper, and no interaction inside one
condition forces a bridge drug. The cross-check of the CSP against the independent oracle and
checker (design §6.2, §7 R2) therefore also runs on the problems generated here, which mix on
purpose:

- interactions between two drugs of the *same* condition, which can force a bridge drug or make a
  transition impossible (toys T4 and T6 of design §9.3, at random);
- new-risk-factor events whose start drugs become contraindicated and need a taper;
- washouts and tapers of 0 to a few days, 2 to 5 conditions, and both kinds of event.

Every problem satisfies the frozen contract (``problem.py``) and has a safe starting regimen (no
interacting pair is given on day 0); it may have no transition at all. The generator draws from
an explicit ``random.Random``, so a seed fixes the whole batch.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

from symbolic_ai.p2_ej4_transition.errors import InstanceError
from symbolic_ai.p2_ej4_transition.problem import Event, EventKind, TransitionProblem

__all__ = [
    "Coverage",
    "coverage",
    "has_contraindicated_taper_start",
    "has_same_condition_interaction",
    "random_problem",
    "random_problems",
]

#: Probability that a candidate other than the start drug is acceptable after the event.
P_ACCEPTABLE = 0.7
#: Probability that a start drug becomes contraindicated after a new-risk-factor event.
P_CONTRAINDICATED_START = 0.5
#: The fewest candidate drugs of a condition (one drug cannot switch to anything).
MIN_CANDIDATES = 2
#: The risk factor that appears in every new-risk-factor event.
NEW_RISK_FACTOR = "R1"


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a batch of problems exercises, counted in problems.

    Parameters
    ----------
    problems
        Problems in the batch.
    same_condition_interactions
        Problems with at least one interaction between two candidates of the same condition.
    contraindicated_taper_starts
        Problems in which some start drug is not acceptable after the event and has a taper.
    new_condition_events
        Problems started by a new condition.
    new_risk_factor_events
        Problems started by a new risk factor.
    """

    problems: int
    same_condition_interactions: int
    contraindicated_taper_starts: int
    new_condition_events: int
    new_risk_factor_events: int


def random_problem(
    rng: random.Random,
    n_conditions: int,
    *,
    p_interaction: float = 0.2,
    p_same_condition_interaction: float = 0.3,
    max_washout: int = 3,
    max_taper: int = 3,
    max_candidates: int = 3,
    p_bridge: float = 0.2,
    problem_id: str = "synthetic",
) -> TransitionProblem:
    """Draw one valid transition problem.

    The conditions are ``C1 .. Cn`` and the candidates of ``Ck`` are ``cka, ckb, ...``. The event
    is a new condition (a random one, untreated at the start; the start drugs of the others stay
    acceptable, since a new condition brings no risk factor) or the new risk factor ``R1`` (each
    start drug becomes contraindicated with probability :data:`P_CONTRAINDICATED_START`). Any
    other candidate is acceptable with probability :data:`P_ACCEPTABLE`, so a condition may have
    no acceptable drug. Each pair of drugs interacts with probability ``p_interaction`` (different
    conditions) or ``p_same_condition_interaction`` (same condition), except that two start drugs
    never interact, so day 0 is safe.

    A bridge drug is rarely forced at random, because a bridge must be acceptable (design v2.1)
    and then usually serves as the final drug too. So, with probability ``p_bridge``, a
    new-condition problem gets a planted one (see :func:`_plant_bridge`).

    Parameters
    ----------
    rng
        The source of randomness; the problem depends only on its state.
    n_conditions
        Number of conditions, 1 to 9.
    p_interaction
        Probability that two drugs of different conditions interact.
    p_same_condition_interaction
        Probability that two drugs of the same condition interact.
    max_washout
        Largest washout, in days (each drug gets one uniformly in ``0..max_washout``).
    max_taper
        Largest taper, in days (each drug gets one uniformly in ``0..max_taper``).
    max_candidates
        Largest number of candidates of a condition (each gets ``2..max_candidates``, at most 26).
    p_bridge
        Probability that a new-condition problem gets a planted bridge.
    problem_id
        The identifier of the problem.

    Returns
    -------
    TransitionProblem
        A problem that satisfies the contract, with a safe starting regimen.

    Raises
    ------
    InstanceError
        If a count or a probability is outside its range.
    """
    _check_arguments(
        n_conditions,
        (p_interaction, p_same_condition_interaction, p_bridge),
        max_washout,
        max_taper,
    )
    if not MIN_CANDIDATES <= max_candidates <= 26:
        raise InstanceError(f"max_candidates must be in 2..26, got {max_candidates}")
    conditions = tuple(f"C{i}" for i in range(1, n_conditions + 1))
    candidates = {
        c: tuple(
            f"{c.lower()}{chr(ord('a') + j)}"
            for j in range(rng.randint(MIN_CANDIDATES, max_candidates))
        )
        for c in conditions
    }
    start: dict[str, str | None] = {c: rng.choice(candidates[c]) for c in conditions}
    risk_factors: tuple[str, ...] = ()
    p_start_ok = 1.0
    if rng.random() < 0.5:
        new_condition = rng.choice(conditions)
        start[new_condition] = None
        event = Event(EventKind.NEW_CONDITION, new_condition)
    else:
        event = Event(EventKind.NEW_RISK_FACTOR, NEW_RISK_FACTOR)
        risk_factors, p_start_ok = (NEW_RISK_FACTOR,), 1.0 - P_CONTRAINDICATED_START
    acceptable = {
        c: tuple(
            d
            for d in candidates[c]
            if rng.random() < (p_start_ok if d == start[c] else P_ACCEPTABLE)
        )
        for c in conditions
    }
    drugs = sorted(d for c in conditions for d in candidates[c])
    interactions = _random_interactions(
        rng, candidates, set(start.values()), p_interaction, p_same_condition_interaction
    )
    washout_days = {d: rng.randint(0, max_washout) for d in drugs}
    if event.kind is EventKind.NEW_CONDITION and rng.random() < p_bridge:
        _plant_bridge(
            rng, event.target_id, start, candidates, acceptable, interactions, washout_days
        )
    return TransitionProblem(
        problem_id=problem_id,
        event=event,
        conditions=conditions,
        risk_factors=risk_factors,
        start=start,
        candidates=candidates,
        acceptable=acceptable,
        interactions=frozenset(interactions),
        washout_days=washout_days,
        taper_days={d: rng.randint(0, max_taper) for d in drugs},
    )


def random_problems(
    seed: int, count: int, *, min_conditions: int = 2, max_conditions: int = 5
) -> tuple[TransitionProblem, ...]:
    """Draw a reproducible batch of problems with the default parameters of :func:`random_problem`.

    Parameters
    ----------
    seed
        The seed of the batch: the same seed gives the same problems.
    count
        Number of problems (``>= 0``); their identifiers are ``S0001``, ``S0002``, ...
    min_conditions, max_conditions
        Each problem has a uniformly drawn number of conditions in this range.

    Returns
    -------
    tuple[TransitionProblem, ...]
        The problems, in generation order.

    Raises
    ------
    InstanceError
        If ``count`` is negative or the range of conditions is empty or not inside ``1..9``.
    """
    if count < 0 or not 1 <= min_conditions <= max_conditions <= 9:
        raise InstanceError(
            f"need count >= 0 and 1 <= min_conditions <= max_conditions, "
            f"got {count}, {min_conditions}, {max_conditions}"
        )
    rng = random.Random(seed)
    return tuple(
        random_problem(rng, rng.randint(min_conditions, max_conditions), problem_id=f"S{index:04d}")
        for index in range(1, count + 1)
    )


def coverage(problems: Sequence[TransitionProblem]) -> Coverage:
    """Count what a batch of problems exercises (see :class:`Coverage`).

    Parameters
    ----------
    problems
        The batch.

    Returns
    -------
    Coverage
        The counts, in problems.
    """
    return Coverage(
        problems=len(problems),
        same_condition_interactions=sum(map(has_same_condition_interaction, problems)),
        contraindicated_taper_starts=sum(map(has_contraindicated_taper_start, problems)),
        new_condition_events=sum(p.event.kind is EventKind.NEW_CONDITION for p in problems),
        new_risk_factor_events=sum(p.event.kind is EventKind.NEW_RISK_FACTOR for p in problems),
    )


def has_same_condition_interaction(problem: TransitionProblem) -> bool:
    """Return whether two candidates of one condition interact.

    Parameters
    ----------
    problem
        The transition problem.

    Returns
    -------
    bool
        ``True`` if some interacting pair lies inside one condition's candidates.
    """
    return any(
        pair <= set(problem.candidates[c])
        for c in problem.conditions
        for pair in problem.interactions
    )


def has_contraindicated_taper_start(problem: TransitionProblem) -> bool:
    """Return whether some start drug is no longer acceptable and must be tapered before a switch.

    Parameters
    ----------
    problem
        The transition problem.

    Returns
    -------
    bool
        ``True`` if a condition starts on a drug outside its acceptable drugs whose taper is
        at least one day.
    """
    return any(
        drug is not None and drug not in problem.acceptable[c] and problem.taper_days[drug] >= 1
        for c, drug in problem.start.items()
    )


def _random_interactions(
    rng: random.Random,
    candidates: dict[str, tuple[str, ...]],
    start_drugs: set[str | None],
    p_interaction: float,
    p_same_condition_interaction: float,
) -> set[frozenset[str]]:
    """Draw the interacting pairs; two start drugs never interact, so day 0 is safe."""
    condition_of = {d: c for c, drugs in candidates.items() for d in drugs}
    pairs: set[frozenset[str]] = set()
    for a, b in itertools.combinations(sorted(condition_of), 2):
        same = condition_of[a] == condition_of[b]
        drawn = rng.random() < (p_same_condition_interaction if same else p_interaction)
        if drawn and not (a in start_drugs and b in start_drugs):
            pairs.add(frozenset((a, b)))
    return pairs


def _plant_bridge(
    rng: random.Random,
    new_condition: str,
    start: dict[str, str | None],
    candidates: dict[str, tuple[str, ...]],
    acceptable: dict[str, tuple[str, ...]],
    interactions: set[frozenset[str]],
    washout_days: dict[str, int],
) -> None:
    """Edit ``acceptable``, ``interactions`` and ``washout_days`` so that a bridge drug is needed.

    A treated condition ``X`` with at least three candidates (none: nothing is planted) starts on
    ``old`` and gets an acceptable ``target`` and ``bridge``. ``old`` and ``bridge`` interact with
    every acceptable drug of the new condition, so neither can be final; ``old`` interacts with
    ``target`` and stays in the body at least one day, so ``X`` cannot switch from ``old`` to
    ``target`` directly; ``bridge`` interacts with neither. With exactly three candidates, the only
    route is ``old -> bridge -> target``, unless the other random interactions make the transition
    impossible. Day 0 stays safe: neither ``target``, ``bridge`` nor the new condition's drugs are
    given on it.
    """
    hosts = [(c, d) for c, d in sorted(start.items()) if d is not None and len(candidates[c]) >= 3]
    if not hosts:
        return
    host, old = rng.choice(hosts)
    others = [d for d in candidates[host] if d != old]
    target, bridge = rng.sample(others, 2)
    acceptable[host] = tuple(sorted({*acceptable[host], target, bridge}))
    interactions.difference_update({frozenset((old, bridge)), frozenset((target, bridge))})
    interactions.add(frozenset((old, target)))
    interactions.update(
        frozenset((drug, d)) for drug in (old, bridge) for d in acceptable[new_condition]
    )
    washout_days[old] = max(washout_days[old], 1)


def _check_arguments(
    n_conditions: int, probabilities: tuple[float, ...], max_washout: int, max_taper: int
) -> None:
    """Raise ``InstanceError`` unless the arguments of :func:`random_problem` are in range."""
    if not 1 <= n_conditions <= 9:
        raise InstanceError(f"n_conditions must be in 1..9, got {n_conditions}")
    if not all(0.0 <= p <= 1.0 for p in probabilities):
        raise InstanceError(f"probabilities must be in [0, 1], got {probabilities}")
    if max_washout < 0 or max_taper < 0:
        raise InstanceError(
            f"max_washout and max_taper must be >= 0, got {max_washout}, {max_taper}"
        )
