"""Exceptions raised by the EJ1 propositional prescribing agent."""

from __future__ import annotations


class EJ1Error(Exception):
    """Base class of every exception raised by ``symbolic_ai.p1_ej1_logic``."""


class EncodingError(EJ1Error):
    """A formulary or an encounter cannot be encoded into propositional clauses."""


class LemmaPreconditionError(EJ1Error):
    """An unknown risk factor occurs in a positive literal, breaking Lemma 1's precondition.

    Safety of the PRESCRIBE / REQUEST_TEST / REFER decision rule (``EJ1-sat/README.md`` §7.5)
    relies on every unknown risk factor occurring only negatively (``¬R_u``) in the knowledge
    base, so that assuming the worst case (all unknowns present) is the most restrictive one.
    """

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        message = f"{len(violations)} lemma precondition violation(s):\n" + "\n".join(violations)
        super().__init__(message)


class CertificateError(EJ1Error):
    """A model returned by the solver does not satisfy every clause of the input.

    Raised by the certificate check that every returned model passes before it is used
    (``02-code-architecture.md`` §5): a defensive, solver-independent verification.
    """

    def __init__(self, unsatisfied: tuple[str, ...]) -> None:
        self.unsatisfied = unsatisfied
        message = f"model fails {len(unsatisfied)} clause(s):\n" + "\n".join(unsatisfied)
        super().__init__(message)
