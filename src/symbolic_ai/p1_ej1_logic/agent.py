"""The prescribing agent: Algorithm 1 of the report (PRESCRIBE / REQUEST_TEST / REFER).

At each encounter the agent builds Γ (axioms A1-A6 plus the percepts of the record), then reasons
about the two brackets Γ⁺ (worst case: every unknown risk factor present) and Γ⁻ (best case: every
unknown risk factor absent). If Γ⁺ is satisfiable it prescribes a model of Γ⁺ and, optionally,
classifies every drug as essential, excluded or optional by two entailment checks per drug. If Γ⁺
is unsatisfiable but Γ⁻ is satisfiable, it asks which pending test would unlock a regimen. If
neither is satisfiable, it refers the patient. Safety of this rule rests on every unknown risk
factor occurring only negatively in Γ (``EJ1-sat/README.md`` §7.5), checked before solving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic.encoding import AxiomTag as _AxiomTag
from symbolic_ai.p1_ej1_logic.encoding import (
    Clause,
    Literal,
    check_lemma_precondition,
    gamma,
    gamma_minus,
    gamma_plus,
    gamma_u,
    symbol_for_drug,
    unknown_risk_factors,
)
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver, SolveResult

__all__ = ["Action", "Decision", "DrugClassification", "PrescribingAgent", "SatCall"]


class Action(StrEnum):
    """The three actions of Algorithm 1."""

    PRESCRIBE = "prescribe"
    REQUEST_TEST = "request_test"
    REFER = "refer"


class DrugClassification(StrEnum):
    """A drug's status in every model of Γ⁺: always in, always out, or a free choice."""

    ESSENTIAL = "essential"
    EXCLUDED = "excluded"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class SatCall:
    """One SAT call made while deciding, kept for the explanation trace.

    ``calls``, ``decisions`` and ``failures`` are the DPLL effort counts of
    :class:`~symbolic_ai.p1_ej1_logic.solver.SolveResult`.
    """

    name: str
    satisfiable: bool
    calls: int
    decisions: int = 0
    failures: int = 0

    @classmethod
    def of(cls, name: str, result: SolveResult) -> SatCall:
        """Record ``result`` under ``name``."""
        return cls(name, result.satisfiable, result.calls, result.decisions, result.failures)


@dataclass(frozen=True, slots=True)
class Decision:
    """The agent's decision for one encounter.

    Parameters
    ----------
    action : Action
        PRESCRIBE, REQUEST_TEST or REFER.
    regimen : frozenset[str]
        The prescribed drug identifiers; non-empty only for PRESCRIBE.
    tests : tuple[str, ...]
        The risk factor identifiers to request; non-empty only for REQUEST_TEST.
    classification : Mapping[str, DrugClassification] | None
        Every formulary drug's classification, when requested and Γ⁺ is satisfiable.
    trace : tuple[SatCall, ...]
        Every SAT call issued while deciding, in the order they were made.
    """

    action: Action
    regimen: frozenset[str] = frozenset()
    tests: tuple[str, ...] = ()
    classification: Mapping[str, DrugClassification] | None = None
    trace: tuple[SatCall, ...] = field(default_factory=tuple)


class PrescribingAgent:
    """Decides PRESCRIBE / REQUEST_TEST / REFER for one encounter, given one formulary.

    Parameters
    ----------
    formulary : Formulary
        The stable knowledge of the environment (axioms A1-A6).
    solver : DPLLSolver
        The complete SAT solver used to answer every query.
    classify : bool
        Whether to classify every drug as essential, excluded or optional after a PRESCRIBE
        (costs ``2 * len(formulary.drug_ids)`` extra solver calls).
    """

    def __init__(self, formulary: Formulary, solver: DPLLSolver, classify: bool = True) -> None:
        self._formulary = formulary
        self._solver = solver
        self._classify = classify

    def decide(self, encounter: Encounter) -> Decision:
        """Run Algorithm 1 for one encounter.

        Parameters
        ----------
        encounter : Encounter
            The clinical record of one appointment.

        Returns
        -------
        Decision
            The action taken, with its regimen, tests, classification and trace.

        Raises
        ------
        LemmaPreconditionError
            If some unknown risk factor occurs positively in Γ, checked before any solving.
        """
        base = gamma(self._formulary, encounter)
        unknowns = unknown_risk_factors(encounter, self._formulary)
        check_lemma_precondition(base, unknowns)

        trace: list[SatCall] = []

        plus_clauses = gamma_plus(base, unknowns)
        plus_result = self._solver.solve(plus_clauses)
        trace.append(SatCall.of("SAT(Gamma+)", plus_result))

        if plus_result.satisfiable:
            return self._prescribe(plus_clauses, plus_result, trace)

        minus_clauses = gamma_minus(base, unknowns)
        minus_result = self._solver.solve(minus_clauses)
        trace.append(SatCall.of("SAT(Gamma-)", minus_result))

        if minus_result.satisfiable:
            return self._request_test(base, unknowns, trace)

        return Decision(Action.REFER, trace=tuple(trace))

    def _prescribe(
        self, plus_clauses: tuple[Clause, ...], plus_result: SolveResult, trace: list[SatCall]
    ) -> Decision:
        """Build the PRESCRIBE decision: the regimen of a model of Γ⁺, and its classification."""
        assert plus_result.model is not None
        regimen = frozenset(
            drug_id
            for drug_id in self._formulary.drug_ids
            if plus_result.model[symbol_for_drug(drug_id)]
        )
        classification = None
        if self._classify:
            classification = self._classify_drugs(plus_clauses, trace)
        return Decision(
            Action.PRESCRIBE, regimen=regimen, classification=classification, trace=tuple(trace)
        )

    def _classify_drugs(
        self, plus_clauses: tuple[Clause, ...], trace: list[SatCall]
    ) -> Mapping[str, DrugClassification]:
        """Classify every drug: ESSENTIAL iff UNSAT(Γ⁺ ∧ ¬T_d); EXCLUDED iff UNSAT(Γ⁺ ∧ T_d)."""
        classification: dict[str, DrugClassification] = {}
        for drug_id in self._formulary.drug_ids:
            symbol = symbol_for_drug(drug_id)

            not_d = (
                *plus_clauses,
                Clause((Literal(symbol, positive=False),), _AxiomTag.ASSUMPTION, drug_id),
            )
            not_d_result = self._solver.solve(not_d)
            trace.append(SatCall.of(f"SAT(Gamma+ & ~T_{drug_id})", not_d_result))
            essential = not not_d_result.satisfiable

            d_true = (
                *plus_clauses,
                Clause((Literal(symbol, positive=True),), _AxiomTag.ASSUMPTION, drug_id),
            )
            d_true_result = self._solver.solve(d_true)
            trace.append(SatCall.of(f"SAT(Gamma+ & T_{drug_id})", d_true_result))
            excluded = not d_true_result.satisfiable

            if essential:
                classification[drug_id] = DrugClassification.ESSENTIAL
            elif excluded:
                classification[drug_id] = DrugClassification.EXCLUDED
            else:
                classification[drug_id] = DrugClassification.OPTIONAL
        return classification

    def _request_test(
        self, base: tuple[Clause, ...], unknowns: frozenset[str], trace: list[SatCall]
    ) -> Decision:
        """Ask, for each unknown u, whether Γ_u is satisfiable; request the first that qualifies."""
        qualifying: list[str] = []
        for u in sorted(unknowns):
            u_clauses = gamma_u(base, unknowns, u)
            u_result = self._solver.solve(u_clauses)
            trace.append(SatCall.of(f"SAT(Gamma_{u})", u_result))
            if u_result.satisfiable:
                qualifying.append(u)
        tests = (qualifying[0],) if qualifying else tuple(sorted(unknowns))
        return Decision(Action.REQUEST_TEST, tests=tests, trace=tuple(trace))
