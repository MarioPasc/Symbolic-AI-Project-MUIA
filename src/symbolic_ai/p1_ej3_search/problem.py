"""The search problem of EJ3: build the cheapest regimen of a record one drug at a time.

A state is a partial regimen, the set of drugs prescribed so far. It never breaks an axiom that
more drugs cannot repair: no interacting pair (A3), no contraindicated drug (A4), every required
companion present (A5), at most one drug per family (A6), and every drug either treats a present
condition or accompanies another (A2). What a state may still lack is coverage (A1), so the goal
test is that every present condition has one of its candidates, and a goal state is a model of Γ
for the record. The cost of a path is the cost of the drugs it added, and the heuristic is the
cost of a relaxed problem in which the remaining conditions no longer constrain each other
(AIMA 4th ed. §3.1 for the formulation, §3.6.2 for heuristics from relaxed problems).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Protocol, TypeAlias

from symbolic_ai.dataloader.models import Formulary
from symbolic_ai.p1_ej1_logic import DrugClassification, FullRecord
from symbolic_ai.p1_ej3_search.errors import CostError, RecordError

__all__ = ["DomainPruning", "HeuristicValue", "RegimenProblem", "SearchNode", "State"]

#: A partial regimen: the identifiers of the drugs prescribed so far.
State: TypeAlias = frozenset[str]
#: A heuristic value: an exact fraction of a cost, or ``math.inf`` for a state with no completion.
HeuristicValue: TypeAlias = Fraction | float


class SearchNode(Protocol):
    """What the heuristic reads from a node of ``aima.search`` (its ``Node`` is untyped)."""

    @property
    def state(self) -> State:
        """The state of the node."""
        ...


@dataclass(frozen=True, slots=True)
class DomainPruning:
    """What Agent 1 proved about every model of Γ⁺, used to prune the search.

    Parameters
    ----------
    essential : frozenset[str]
        Drugs present in every model: when one is a candidate for the condition being covered,
        it is the only action offered.
    excluded : frozenset[str]
        Drugs present in no model: never offered, alone or as a companion.
    """

    essential: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset()

    @classmethod
    def from_classification(cls, classification: Mapping[str, DrugClassification]) -> DomainPruning:
        """Build the pruning from the classification of Agent 1's decision.

        Parameters
        ----------
        classification : Mapping[str, DrugClassification]
            Every drug's status in the models of Γ⁺ (``Decision.classification`` of EJ1).

        Returns
        -------
        DomainPruning
            Its essential and excluded drugs; the optional ones stay free for the search.
        """
        return cls(
            essential=_with_status(classification, DrugClassification.ESSENTIAL),
            excluded=_with_status(classification, DrugClassification.EXCLUDED),
        )


def _with_status(
    classification: Mapping[str, DrugClassification], status: DrugClassification
) -> frozenset[str]:
    return frozenset(drug_id for drug_id, found in classification.items() if found is status)


class RegimenProblem:
    """The cheapest regimen for one fully observed record, as a state-space search problem.

    It has the interface of ``aima.search.Problem`` (``initial``, ``actions``, ``result``,
    ``goal_test``, ``path_cost``, ``h``) without subclassing it, because the vendored copy is
    untyped; the search functions of ``aima.search`` only call these members.

    Only the first uncovered condition (in alphabetical order) is branched on. The order in which
    conditions are covered does not change the regimen reached, so one condition per node is
    enough, as for the variables of a constraint satisfaction problem (AIMA 4th ed. §6.3).

    For every regimen that satisfies A1-A6 some goal state is a subset of it, and with it at most
    as dear, because a path may always pick its drugs inside that regimen. The cheapest goal
    state is therefore a cheapest regimen, which is what makes the search complete for the task.

    Parameters
    ----------
    formulary : Formulary
        The stable knowledge of the environment (axioms A1-A6).
    costs : Mapping[str, int]
        The cost of each drug, in euro cents per month of treatment.
    record : FullRecord
        The present conditions and the risk factors taken as present (for Γ⁺, the observed ones
        and the unknown ones).
    pruning : DomainPruning | None
        What Agent 1 proved about the drugs, or ``None`` to search without it.

    Raises
    ------
    CostError
        If a drug of the formulary has no cost, or a negative one.
    RecordError
        If the record or the pruning names an identifier outside the formulary.
    """

    def __init__(
        self,
        formulary: Formulary,
        costs: Mapping[str, int],
        record: FullRecord,
        pruning: DomainPruning | None = None,
    ) -> None:
        pruning = pruning if pruning is not None else DomainPruning()
        _check_costs(formulary, costs)
        _check_identifiers(formulary, record, pruning)

        self.initial: State = frozenset()
        self._costs = {drug_id: costs[drug_id] for drug_id in formulary.drug_ids}
        self._conditions = tuple(sorted(record.conditions))
        self._essential = pruning.essential
        self._candidates = {c: formulary.candidates.get(c, frozenset()) for c in self._conditions}
        self._treats = {
            drug_id: formulary.indications(drug_id) & record.conditions
            for drug_id in formulary.drug_ids
        }

        banned = pruning.excluded | {
            x.drug_id
            for x in formulary.contraindications
            if x.risk_factor_id in record.present_risk_factors
        }
        conflicts = _conflicts(formulary)
        self._bundle = _bundles(formulary, record)
        # Drugs that cannot share a regimen with some drug of the bundle (A3, A6).
        self._bundle_conflicts = {
            drug_id: frozenset[str]().union(*(conflicts[member] for member in bundle))
            for drug_id, bundle in self._bundle.items()
        }
        # A bundle is offered only if no risk factor bans a member (A4) and it is free of
        # internal conflicts; neither depends on the state, so both are decided once.
        self._options = {
            condition_id: tuple(
                drug_id
                for drug_id in sorted(candidates)
                if not self._bundle[drug_id] & (banned | self._bundle_conflicts[drug_id])
            )
            for condition_id, candidates in self._candidates.items()
        }

    def uncovered(self, state: State) -> tuple[str, ...]:
        """Return the present conditions with no candidate drug in ``state``, sorted (A1).

        Parameters
        ----------
        state : State
            A partial regimen.

        Returns
        -------
        tuple[str, ...]
            Condition identifiers; empty exactly when ``state`` is a goal.
        """
        return tuple(c for c in self._conditions if not self._candidates[c] & state)

    def actions(self, state: State) -> tuple[str, ...]:
        """Return the drugs that may be prescribed next, for the first uncovered condition.

        A candidate is offered if its bundle (the drug and the companions it requires under the
        record's risk factors) keeps the state free of the conflicts of A3, A4 and A6. If Agent 1
        proved one of the offered candidates essential, it is the only one offered: it belongs
        to every regimen, so choosing it loses none.

        Parameters
        ----------
        state : State
            A partial regimen.

        Returns
        -------
        tuple[str, ...]
            Drug identifiers, sorted; empty for a goal state and for a dead end.
        """
        uncovered = self.uncovered(state)
        if not uncovered:
            return ()
        options = self._compatible(state, uncovered[0])
        essential = tuple(drug_id for drug_id in options if drug_id in self._essential)
        return essential[:1] or options

    def result(self, state: State, action: str) -> State:
        """Return ``state`` with the drug ``action`` and its required companions added (A5).

        Parameters
        ----------
        state : State
            A partial regimen.
        action : str
            A drug of ``self.actions(state)``.

        Returns
        -------
        State
            The partial regimen after prescribing the bundle of ``action``.
        """
        return state | self._bundle[action]

    def goal_test(self, state: State) -> bool:
        """Return whether ``state`` covers every present condition (A1), i.e. is a regimen.

        Parameters
        ----------
        state : State
            A partial regimen.

        Returns
        -------
        bool
            ``True`` when no condition of the record is uncovered.
        """
        return not self.uncovered(state)

    def path_cost(self, cost_so_far: int, state: State, action: str, next_state: State) -> int:
        """Return the cost of reaching ``next_state``: the path so far plus the drugs just added.

        A companion already in ``state`` is not paid twice, so the cost of a path is the cost of
        the state it reaches.

        Parameters
        ----------
        cost_so_far : int
            The cost of the path up to ``state``.
        state : State
            The partial regimen before the action.
        action : str
            The drug prescribed (unused: the cost is read from the two states).
        next_state : State
            The partial regimen after the action.

        Returns
        -------
        int
            The cost of the path up to ``next_state``, in euro cents per month.
        """
        del action  # the interface of aima.search.Problem passes it; the states carry the cost
        return cost_so_far + self.cost(next_state - state)

    def cost(self, drugs: frozenset[str]) -> int:
        """Return the monthly cost of a set of drugs, in euro cents.

        Parameters
        ----------
        drugs : frozenset[str]
            Drug identifiers of the formulary.

        Returns
        -------
        int
            The sum of their costs.

        Raises
        ------
        RecordError
            If a drug is outside the formulary.
        """
        unknown = sorted(drugs - self._costs.keys())
        if unknown:
            raise RecordError(f"drugs outside the formulary: {unknown}")
        return sum(self._costs[drug_id] for drug_id in drugs)

    def heuristic(self, state: State) -> HeuristicValue:
        """Return h(state): for each uncovered condition, its cheapest drug still compatible.

        The relaxed problem lets every uncovered condition pick its drug independently of the
        others and charges nothing for companions (AIMA 4th ed. §3.6.2). A drug that treats k
        uncovered conditions is charged cost/k to each, so that a regimen using it for all of them
        is not overestimated. A condition with no compatible drug left has no completion, and the
        value is infinite.

        The heuristic is consistent (AIMA 4th ed. §3.5.2): an action only removes compatible drugs
        and uncovered conditions, so the term of a condition still uncovered never decreases, and
        the terms of the conditions it covers add up to at most the cost of the drugs it adds. It
        is therefore admissible, and A* graph search with it returns a cheapest regimen.

        Parameters
        ----------
        state : State
            A partial regimen.

        Returns
        -------
        HeuristicValue
            A lower bound of the cost still to pay; ``math.inf`` for a dead end.
        """
        uncovered = frozenset(self.uncovered(state))
        total = Fraction(0)
        for condition_id in sorted(uncovered):
            shares = [
                Fraction(self._costs[drug_id], len(self._treats[drug_id] & uncovered))
                for drug_id in self._compatible(state, condition_id)
            ]
            if not shares:
                return math.inf
            total += min(shares)
        return total

    def h(self, node: SearchNode) -> HeuristicValue:
        """Return the heuristic of a search node, as ``aima.search`` asks for it.

        Parameters
        ----------
        node : SearchNode
            A node of the search tree.

        Returns
        -------
        HeuristicValue
            ``self.heuristic(node.state)``.
        """
        return self.heuristic(node.state)

    def _compatible(self, state: State, condition_id: str) -> tuple[str, ...]:
        """Candidates of ``condition_id`` whose bundle conflicts with no drug of ``state``."""
        return tuple(
            drug_id
            for drug_id in self._options[condition_id]
            if not self._bundle_conflicts[drug_id] & state
        )


def _check_costs(formulary: Formulary, costs: Mapping[str, int]) -> None:
    """Raise :class:`CostError` unless every drug of ``formulary`` has a non-negative cost."""
    missing = sorted(set(formulary.drug_ids) - costs.keys())
    if missing:
        raise CostError(f"drugs of formulary {formulary.version} without a cost: {missing}")
    negative = sorted(d for d in formulary.drug_ids if costs[d] < 0)
    if negative:
        raise CostError(f"drugs with a negative cost: {[(d, costs[d]) for d in negative]}")


def _check_identifiers(formulary: Formulary, record: FullRecord, pruning: DomainPruning) -> None:
    """Raise :class:`RecordError` if ``record`` or ``pruning`` names an unknown identifier."""
    unknown = {
        "conditions": record.conditions - set(formulary.condition_ids),
        "risk factors": record.present_risk_factors - set(formulary.risk_factor_ids),
        "drugs": (pruning.essential | pruning.excluded) - set(formulary.drug_ids),
    }
    problems = [f"{kind} {sorted(ids)}" for kind, ids in unknown.items() if ids]
    if problems:
        raise RecordError(f"outside formulary {formulary.version}: {'; '.join(problems)}")


def _conflicts(formulary: Formulary) -> dict[str, frozenset[str]]:
    """Map each drug to the drugs it must not be prescribed with (A3 pairs and A6 families)."""
    pairs = [(i.drug_a, i.drug_b) for i in formulary.interactions]
    for family in formulary.exclusive_families():
        pairs.extend(combinations(sorted(family), 2))
    conflicts: dict[str, set[str]] = {drug_id: set() for drug_id in formulary.drug_ids}
    for drug_a, drug_b in pairs:
        conflicts[drug_a].add(drug_b)
        conflicts[drug_b].add(drug_a)
    return {drug_id: frozenset(others) for drug_id, others in conflicts.items()}


def _bundles(formulary: Formulary, record: FullRecord) -> dict[str, frozenset[str]]:
    """Map each drug to itself plus every companion A5 requires with it, transitively."""
    required: dict[str, set[str]] = {drug_id: set() for drug_id in formulary.drug_ids}
    for coprescription in formulary.coprescriptions:
        if coprescription.risk_factor_id in record.present_risk_factors:
            required[coprescription.drug_id].add(coprescription.companion_drug_id)
    return {drug_id: _closure(drug_id, required) for drug_id in formulary.drug_ids}


def _closure(drug_id: str, required: Mapping[str, set[str]]) -> frozenset[str]:
    """Return ``drug_id`` and every drug reachable from it through ``required``."""
    bundle = {drug_id}
    pending = [drug_id]
    while pending:
        for companion in sorted(required[pending.pop()] - bundle):
            bundle.add(companion)
            pending.append(companion)
    return frozenset(bundle)
