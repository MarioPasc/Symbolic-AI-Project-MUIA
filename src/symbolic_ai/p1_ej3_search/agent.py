"""The cost-aware prescribing agent of EJ3: Agent 1 decides, a search picks the cheapest regimen.

Agent 1 (:class:`~symbolic_ai.p1_ej1_logic.PrescribingAgent`) is run unchanged and keeps the
decision PRESCRIBE / REQUEST_TEST / REFER. Only when it prescribes does this agent act, as a
problem-solving agent (AIMA 4th ed. §3.1): it formulates the search problem of the worst-case
record Γ⁺, searches it, and returns the cheapest regimen in place of the first model DPLL found.
It takes three things from Agent 1's decision: the proof that a regimen exists, the essential
and excluded drugs (to prune the search) and the regimen itself (an upper bound on the cost).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic import (
    Action,
    Decision,
    DPLLSolver,
    FullRecord,
    PrescribingAgent,
    gamma,
    gamma_plus,
    symbol_for_condition,
    symbol_for_drug,
    symbol_for_risk_factor,
    unknown_risk_factors,
)
from symbolic_ai.p1_ej3_search.errors import SearchInvariantError
from symbolic_ai.p1_ej3_search.problem import DomainPruning, RegimenProblem
from symbolic_ai.p1_ej3_search.search import Algorithm, SearchResult, search

__all__ = ["CostAwareAgent", "CostedDecision", "worst_case_record"]


@dataclass(frozen=True, slots=True)
class CostedDecision:
    """The agent's decision for one encounter.

    Parameters
    ----------
    decision : Decision
        Agent 1's decision, exactly as Agent 1 returned it (its action, its regimen, its tests,
        its classification and its trace).
    regimen : frozenset[str]
        The regimen the search returned; empty unless the action is PRESCRIBE.
    cost : int | None
        The cost of ``regimen`` in euro cents per month (PRESCRIBE only).
    baseline_cost : int | None
        The cost of Agent 1's regimen, ``decision.regimen`` (PRESCRIBE only).
    search : SearchResult | None
        The search that found ``regimen``, with its effort counts (PRESCRIBE only).
    """

    decision: Decision
    regimen: frozenset[str] = frozenset()
    cost: int | None = None
    baseline_cost: int | None = None
    search: SearchResult | None = None

    @property
    def action(self) -> Action:
        """The action taken, which is always Agent 1's."""
        return self.decision.action

    @property
    def saving(self) -> int | None:
        """How much cheaper ``regimen`` is than Agent 1's, in euro cents per month."""
        if self.cost is None or self.baseline_cost is None:
            return None
        return self.baseline_cost - self.cost


def worst_case_record(encounter: Encounter, formulary: Formulary) -> FullRecord:
    """Return the record of Γ⁺: the conditions present and every risk factor not known absent.

    Parameters
    ----------
    encounter : Encounter
        The clinical record of one appointment.
    formulary : Formulary
        The formulary whose conditions and risk factors bound the closed world.

    Returns
    -------
    FullRecord
        The present conditions, with the present and the unknown risk factors taken as present.
    """
    present = encounter.risk_factors_with(RiskStatus.PRESENT) & set(formulary.risk_factor_ids)
    return FullRecord(
        conditions=encounter.conditions & set(formulary.condition_ids),
        present_risk_factors=present | unknown_risk_factors(encounter, formulary),
    )


class CostAwareAgent:
    """Decides as Agent 1 does and, when it prescribes, prescribes the cheapest safe regimen.

    Parameters
    ----------
    formulary : Formulary
        The stable knowledge of the environment (axioms A1-A6), shared with Agent 1.
    costs : Mapping[str, int]
        The cost of each drug, in euro cents per month of treatment.
    solver : DPLLSolver
        The SAT solver of the embedded Agent 1.
    algorithm : Algorithm
        The search algorithm (default: A*).
    prune : bool
        Whether Agent 1 classifies every drug and the search is pruned with the result
        (``2 * len(formulary.drug_ids)`` extra solver calls per prescription).
    """

    def __init__(
        self,
        formulary: Formulary,
        costs: Mapping[str, int],
        solver: DPLLSolver,
        algorithm: Algorithm = Algorithm.ASTAR,
        prune: bool = True,
    ) -> None:
        self._formulary = formulary
        self._costs = costs
        self._algorithm = algorithm
        self._prescriber = PrescribingAgent(formulary, solver, classify=prune)

    def decide(self, encounter: Encounter) -> CostedDecision:
        """Run Agent 1 on ``encounter`` and, if it prescribes, search for the cheapest regimen.

        Parameters
        ----------
        encounter : Encounter
            The clinical record of one appointment.

        Returns
        -------
        CostedDecision
            Agent 1's decision, with the regimen found by the search when the action is PRESCRIBE.

        Raises
        ------
        LemmaPreconditionError
            Raised by Agent 1 if an unknown risk factor occurs positively in Γ.
        CostError
            If a drug of the formulary has no cost, or a negative one.
        SearchInvariantError
            If the search contradicts Agent 1 (see :class:`SearchInvariantError`).
        """
        decision = self._prescriber.decide(encounter)
        if decision.action is not Action.PRESCRIBE:
            return CostedDecision(decision)

        pruning = (
            DomainPruning.from_classification(decision.classification)
            if decision.classification is not None
            else None
        )
        problem = RegimenProblem(
            self._formulary, self._costs, worst_case_record(encounter, self._formulary), pruning
        )
        result = search(problem, self._algorithm)
        baseline_cost = problem.cost(decision.regimen)
        regimen, cost = self._certified(encounter, result, baseline_cost)
        return CostedDecision(decision, regimen, cost, baseline_cost, result)

    def _certified(
        self, encounter: Encounter, result: SearchResult, baseline_cost: int
    ) -> tuple[frozenset[str], int]:
        """Return the regimen and cost of ``result`` after checking them against Agent 1."""
        label = f"encounter {encounter.encounter_id}, {result.algorithm.value}"
        if result.regimen is None or result.cost is None:
            raise SearchInvariantError(f"{label}: Γ⁺ is satisfiable but the search found nothing")
        broken = _broken_clauses(self._formulary, encounter, result.regimen)
        if broken:
            raise SearchInvariantError(
                f"{label}: regimen {sorted(result.regimen)} breaks {len(broken)} clause(s) of Γ⁺:\n"
                + "\n".join(broken)
            )
        if result.algorithm.optimal and result.cost > baseline_cost:
            raise SearchInvariantError(
                f"{label}: regimen {sorted(result.regimen)} costs {result.cost}, more than "
                f"Agent 1's ({baseline_cost})"
            )
        return result.regimen, result.cost


def _broken_clauses(
    formulary: Formulary, encounter: Encounter, regimen: frozenset[str]
) -> tuple[str, ...]:
    """Return the clauses of Γ⁺ that ``regimen`` does not satisfy, as text; empty for a model.

    The assignment is the regimen for the decision symbols and the worst-case record for the
    rest, so this checks the search's set-based reading of A1-A6 against EJ1's clauses.
    """
    record = worst_case_record(encounter, formulary)
    true_symbols = (
        {symbol_for_drug(d) for d in regimen}
        | {symbol_for_condition(c) for c in record.conditions}
        | {symbol_for_risk_factor(r) for r in record.present_risk_factors}
    )
    clauses = gamma_plus(gamma(formulary, encounter), unknown_risk_factors(encounter, formulary))
    return tuple(
        f"{clause.axiom}:{clause.source} ({clause})"
        for clause in clauses
        if not any(
            (literal.symbol in true_symbols) == literal.positive for literal in clause.literals
        )
    )
