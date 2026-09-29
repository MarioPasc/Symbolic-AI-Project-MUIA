"""EJ1: the propositional SAT prescribing agent (Algorithm 1 of the report)."""

from __future__ import annotations

from symbolic_ai.p1_ej1_logic.agent import (
    Action,
    Decision,
    DrugClassification,
    PrescribingAgent,
    SatCall,
)
from symbolic_ai.p1_ej1_logic.encoding import (
    AxiomTag,
    Clause,
    Literal,
    axiom_clauses,
    check_lemma_precondition,
    clause_to_expr,
    gamma,
    gamma_minus,
    gamma_plus,
    gamma_u,
    percept_clauses,
    symbol_for_condition,
    symbol_for_drug,
    symbol_for_risk_factor,
    unknown_risk_factors,
)
from symbolic_ai.p1_ej1_logic.errors import (
    CertificateError,
    EJ1Error,
    EncodingError,
    ExplanationError,
    LemmaPreconditionError,
    OracleError,
    ResultsFormatError,
)
from symbolic_ai.p1_ej1_logic.explain import minimal_unsatisfiable_subset
from symbolic_ai.p1_ej1_logic.semantics import FullRecord, RegimenSpace
from symbolic_ai.p1_ej1_logic.solver import DPLLSolver, SolveResult

#: The database version EJ1 is pinned to (``01-database.md`` §2): the 50 rule clauses of the report.
EJ1_FORMULARY_VERSION = "1.0.0"

__all__ = [
    "EJ1_FORMULARY_VERSION",
    "Action",
    "AxiomTag",
    "CertificateError",
    "Clause",
    "DPLLSolver",
    "Decision",
    "DrugClassification",
    "EJ1Error",
    "EncodingError",
    "ExplanationError",
    "FullRecord",
    "LemmaPreconditionError",
    "Literal",
    "OracleError",
    "PrescribingAgent",
    "RegimenSpace",
    "ResultsFormatError",
    "SatCall",
    "SolveResult",
    "axiom_clauses",
    "check_lemma_precondition",
    "clause_to_expr",
    "gamma",
    "gamma_minus",
    "gamma_plus",
    "gamma_u",
    "minimal_unsatisfiable_subset",
    "percept_clauses",
    "symbol_for_condition",
    "symbol_for_drug",
    "symbol_for_risk_factor",
    "unknown_risk_factors",
]
