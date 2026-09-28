"""Propositional encoding of the formulary and of one encounter: axioms A1-A6 and percepts.

Symbols follow ``EJ1-sat/README.md`` §3: ``T_<drug_id>`` (decisions), ``H_<condition_id>``
(conditions, always observed, closed world) and ``R_<risk_factor_id>`` (risk factors, observed
or unknown). The six rule schemas of §4 are grounded here into :class:`Clause` objects, which
:func:`clause_to_expr` converts into ``aima.logic`` expressions for the solver module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from aima.utils import Expr
from symbolic_ai.dataloader.models import Encounter, Formulary, RiskStatus
from symbolic_ai.p1_ej1_logic.errors import EncodingError, LemmaPreconditionError

__all__ = [
    "AxiomTag",
    "Clause",
    "Literal",
    "assumption_clauses",
    "axiom_clauses",
    "check_lemma_precondition",
    "clause_to_expr",
    "gamma",
    "gamma_minus",
    "gamma_plus",
    "gamma_u",
    "percept_clauses",
    "symbol_for_condition",
    "symbol_for_drug",
    "symbol_for_risk_factor",
    "unknown_risk_factors",
]

_logger = logging.getLogger(__name__)


class AxiomTag(StrEnum):
    """Provenance tag of a :class:`Clause`: which rule schema produced it."""

    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    PERCEPT = "PERCEPT"
    ASSUMPTION = "ASSUMPTION"


@dataclass(frozen=True, slots=True)
class Literal:
    """One literal of a clause: a symbol name and its polarity."""

    symbol: str
    positive: bool = True

    def negate(self) -> Literal:
        """Return the literal with the opposite polarity."""
        return Literal(self.symbol, not self.positive)

    def __str__(self) -> str:
        return self.symbol if self.positive else f"~{self.symbol}"


@dataclass(frozen=True, slots=True)
class Clause:
    """A disjunction of literals, tagged with the rule that produced it.

    Parameters
    ----------
    literals : tuple[Literal, ...]
        The literals of the disjunction; never empty for a well-formed clause.
    axiom : AxiomTag
        Which schema grounded this clause (``A1``-``A6``), or ``PERCEPT`` / ``ASSUMPTION``.
    source : str
        The formulary or encounter entry this clause was grounded from (a condition, drug,
        risk factor or a comma-joined tuple of identifiers), for explanations and error messages.
    """

    literals: tuple[Literal, ...]
    axiom: AxiomTag
    source: str

    def __str__(self) -> str:
        return " | ".join(str(literal) for literal in self.literals)


def symbol_for_condition(condition_id: str) -> str:
    """Return the symbol ``H_<condition_id>``."""
    return f"H_{condition_id}"


def symbol_for_drug(drug_id: str) -> str:
    """Return the symbol ``T_<drug_id>``."""
    return f"T_{drug_id}"


def symbol_for_risk_factor(risk_factor_id: str) -> str:
    """Return the symbol ``R_<risk_factor_id>``."""
    return f"R_{risk_factor_id}"


def _a1_clauses(formulary: Formulary) -> list[Clause]:
    """A1 - coverage: ``¬H_c | T_d1 | ... | T_dk`` for every condition c, one clause per c."""
    clauses = []
    for condition_id in formulary.condition_ids:
        candidates = sorted(formulary.candidates.get(condition_id, frozenset()))
        literals = (
            Literal(symbol_for_condition(condition_id), positive=False),
            *(Literal(symbol_for_drug(drug_id), positive=True) for drug_id in candidates),
        )
        clauses.append(Clause(literals, AxiomTag.A1, condition_id))
    return clauses


def _a2_clauses(formulary: Formulary) -> list[Clause]:
    """A2 - justification: ``¬T_d | H_c... | T_d'...`` for every drug d, one clause per d."""
    clauses = []
    for drug_id in formulary.drug_ids:
        indications = sorted(formulary.indications(drug_id))
        companions = sorted(
            {cp.drug_id for cp in formulary.coprescriptions if cp.companion_drug_id == drug_id}
        )
        literals = [Literal(symbol_for_drug(drug_id), positive=False)]
        literals.extend(Literal(symbol_for_condition(c), positive=True) for c in indications)
        literals.extend(Literal(symbol_for_drug(d), positive=True) for d in companions)
        clauses.append(Clause(tuple(literals), AxiomTag.A2, drug_id))
    return clauses


def _a3_clauses(formulary: Formulary) -> list[Clause]:
    """A3 - adverse interactions: ``¬T_a | ¬T_b`` for every interacting pair."""
    return [
        Clause(
            (
                Literal(symbol_for_drug(interaction.drug_a), positive=False),
                Literal(symbol_for_drug(interaction.drug_b), positive=False),
            ),
            AxiomTag.A3,
            f"{interaction.drug_a},{interaction.drug_b}",
        )
        for interaction in sorted(formulary.interactions, key=lambda i: (i.drug_a, i.drug_b))
    ]


def _a4_clauses(formulary: Formulary) -> list[Clause]:
    """A4 - contraindications: ``¬R_r | ¬T_d`` for every (risk factor, drug) pair."""
    return [
        Clause(
            (
                Literal(symbol_for_risk_factor(x.risk_factor_id), positive=False),
                Literal(symbol_for_drug(x.drug_id), positive=False),
            ),
            AxiomTag.A4,
            f"{x.risk_factor_id},{x.drug_id}",
        )
        for x in sorted(formulary.contraindications, key=lambda c: (c.risk_factor_id, c.drug_id))
    ]


def _a5_clauses(formulary: Formulary) -> list[Clause]:
    """A5 - co-prescription: ``¬T_d | ¬R_r | T_d'`` for every co-prescription rule."""
    return [
        Clause(
            (
                Literal(symbol_for_drug(p.drug_id), positive=False),
                Literal(symbol_for_risk_factor(p.risk_factor_id), positive=False),
                Literal(symbol_for_drug(p.companion_drug_id), positive=True),
            ),
            AxiomTag.A5,
            f"{p.drug_id},{p.risk_factor_id},{p.companion_drug_id}",
        )
        for p in sorted(
            formulary.coprescriptions,
            key=lambda p: (p.drug_id, p.risk_factor_id, p.companion_drug_id),
        )
    ]


def _a6_clauses(formulary: Formulary) -> list[Clause]:
    """A6 - no duplication inside a family: ``¬T_a | ¬T_b`` for every pair of an exclusive class."""
    exclusive_ids = sorted(c.class_id for c in formulary.drug_classes if c.exclusive)
    families = formulary.exclusive_families()
    clauses = []
    for class_id, members in zip(exclusive_ids, families, strict=True):
        for drug_a, drug_b in combinations(sorted(members), 2):
            clauses.append(
                Clause(
                    (
                        Literal(symbol_for_drug(drug_a), positive=False),
                        Literal(symbol_for_drug(drug_b), positive=False),
                    ),
                    AxiomTag.A6,
                    class_id,
                )
            )
    return clauses


def axiom_clauses(formulary: Formulary) -> tuple[Clause, ...]:
    """Ground axioms A1-A6 of ``formulary`` into clauses, in deterministic order.

    Parameters
    ----------
    formulary : Formulary
        The formulary to ground.

    Returns
    -------
    tuple[Clause, ...]
        A1 (one per condition), A2 (one per drug), A3 (one per interaction), A4 (one per
        contraindication), A5 (one per co-prescription), A6 (one per pair inside an exclusive
        class), concatenated in that order.
    """
    return tuple(
        _a1_clauses(formulary)
        + _a2_clauses(formulary)
        + _a3_clauses(formulary)
        + _a4_clauses(formulary)
        + _a5_clauses(formulary)
        + _a6_clauses(formulary)
    )


def percept_clauses(encounter: Encounter, formulary: Formulary) -> tuple[Clause, ...]:
    """Encode ``encounter`` as unit clauses (``Make-Percept-Sentence``).

    Closed world over ``formulary.condition_ids``: every condition present at the encounter
    becomes ``H_c``, every other formulary condition becomes ``¬H_c``. An observed risk factor
    becomes ``R_r`` (present) or ``¬R_r`` (absent); an unknown risk factor gets no clause.

    Parameters
    ----------
    encounter : Encounter
        The clinical record of one appointment.
    formulary : Formulary
        The formulary whose conditions and risk factors define the closed world.

    Returns
    -------
    tuple[Clause, ...]
        One unit clause per formulary condition, plus one per formulary risk factor whose
        status at this encounter is ``present`` or ``absent``.
    """
    unknown_conditions = encounter.conditions - set(formulary.condition_ids)
    for condition_id in sorted(unknown_conditions):
        _logger.warning(
            "encounter %s: condition %s is outside formulary %s and is ignored",
            encounter.encounter_id,
            condition_id,
            formulary.version,
        )

    clauses = [
        Clause(
            (
                Literal(
                    symbol_for_condition(condition_id),
                    positive=condition_id in encounter.conditions,
                ),
            ),
            AxiomTag.PERCEPT,
            condition_id,
        )
        for condition_id in formulary.condition_ids
    ]

    for risk_factor_id in formulary.risk_factor_ids:
        status = encounter.risk_factors.get(risk_factor_id)
        if status is None or status is RiskStatus.UNKNOWN:
            continue
        clauses.append(
            Clause(
                (
                    Literal(
                        symbol_for_risk_factor(risk_factor_id),
                        positive=status is RiskStatus.PRESENT,
                    ),
                ),
                AxiomTag.PERCEPT,
                risk_factor_id,
            )
        )
    return tuple(clauses)


def unknown_risk_factors(encounter: Encounter, formulary: Formulary) -> frozenset[str]:
    """Return U, the formulary risk factors whose status at this encounter is unknown.

    Parameters
    ----------
    encounter : Encounter
        The clinical record of one appointment.
    formulary : Formulary
        The formulary whose risk factors bound the closed world.

    Returns
    -------
    frozenset[str]
        Risk factor identifiers with status ``unknown``.
    """
    return frozenset(
        r for r in formulary.risk_factor_ids if encounter.risk_factors.get(r) is RiskStatus.UNKNOWN
    )


def gamma(formulary: Formulary, encounter: Encounter) -> tuple[Clause, ...]:
    """Γ: axioms A1-A6 grounded on ``formulary`` plus the percepts of ``encounter``."""
    return axiom_clauses(formulary) + percept_clauses(encounter, formulary)


def assumption_clauses(unknowns: frozenset[str], *, present: bool) -> tuple[Clause, ...]:
    """Return one unit clause per unknown risk factor, assuming it ``present`` or not.

    Parameters
    ----------
    unknowns : frozenset[str]
        Risk factor identifiers with unknown status.
    present : bool
        Whether to assume each of them present (``True``, for Γ⁺) or absent (``False``, for Γ⁻).

    Returns
    -------
    tuple[Clause, ...]
        One unit clause per element of ``unknowns``, sorted by identifier.
    """
    return tuple(
        Clause((Literal(symbol_for_risk_factor(u), positive=present),), AxiomTag.ASSUMPTION, u)
        for u in sorted(unknowns)
    )


def gamma_plus(base: tuple[Clause, ...], unknowns: frozenset[str]) -> tuple[Clause, ...]:
    """Γ⁺: ``base`` plus every unknown risk factor assumed present (the worst case)."""
    return base + assumption_clauses(unknowns, present=True)


def gamma_minus(base: tuple[Clause, ...], unknowns: frozenset[str]) -> tuple[Clause, ...]:
    """Γ⁻: ``base`` plus every unknown risk factor assumed absent (the best case)."""
    return base + assumption_clauses(unknowns, present=False)


def gamma_u(base: tuple[Clause, ...], unknowns: frozenset[str], u: str) -> tuple[Clause, ...]:
    """Γ_u: ``base`` plus ``¬R_u`` and every other unknown risk factor assumed present.

    Parameters
    ----------
    base : tuple[Clause, ...]
        Γ: axioms and percepts.
    unknowns : frozenset[str]
        U, the risk factors with unknown status at this encounter.
    u : str
        The risk factor being tested; must be a member of ``unknowns``.

    Returns
    -------
    tuple[Clause, ...]
        Γ_u as defined in ``EJ1-sat/README.md`` §7.3.

    Raises
    ------
    EncodingError
        If ``u`` is not a member of ``unknowns``.
    """
    if u not in unknowns:
        raise EncodingError(f"{u} is not an unknown risk factor of {sorted(unknowns)}")
    others = unknowns - {u}
    not_u = Clause((Literal(symbol_for_risk_factor(u), positive=False),), AxiomTag.ASSUMPTION, u)
    return (*base, not_u, *assumption_clauses(others, present=True))


def check_lemma_precondition(clauses: tuple[Clause, ...], unknowns: frozenset[str]) -> None:
    """Check that every unknown risk factor occurs only negatively in ``clauses``.

    Parameters
    ----------
    clauses : tuple[Clause, ...]
        Γ: axioms and percepts, before any Γ⁺/Γ⁻/Γ_u assumption is added.
    unknowns : frozenset[str]
        U, the risk factors with unknown status.

    Raises
    ------
    LemmaPreconditionError
        If some clause contains a positive literal ``R_u`` for a ``u`` in ``unknowns``.
    """
    unknown_symbols = {symbol_for_risk_factor(u) for u in unknowns}
    violations = tuple(
        f"{clause.axiom}:{clause.source} contains {literal.symbol} positively"
        for clause in clauses
        for literal in clause.literals
        if literal.positive and literal.symbol in unknown_symbols
    )
    if violations:
        raise LemmaPreconditionError(violations)


def clause_to_expr(clause: Clause) -> Expr:
    """Convert one clause into an ``aima`` ``Expr`` disjunction, for ``aima.logic.dpll``.

    Parameters
    ----------
    clause : Clause
        A non-empty disjunction of literals.

    Returns
    -------
    Expr
        The literals joined by ``|``, each negated with ``~`` when not positive.

    Raises
    ------
    EncodingError
        If the clause has no literals (the empty clause is always false and never appears
        in a well-formed encoding).
    """
    if not clause.literals:
        raise EncodingError(f"empty clause from {clause.axiom}:{clause.source}")
    exprs = [_literal_to_expr(literal) for literal in clause.literals]
    result = exprs[0]
    for expr in exprs[1:]:
        result = result | expr
    return result


def _literal_to_expr(literal: Literal) -> Expr:
    """Convert one literal into a (possibly negated) ``aima`` symbol ``Expr``."""
    symbol = Expr(literal.symbol)
    return symbol if literal.positive else ~symbol
