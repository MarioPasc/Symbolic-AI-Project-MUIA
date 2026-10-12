"""Exceptions raised by EJ4, the safe transition between two regimens."""

from __future__ import annotations


class EJ4Error(Exception):
    """Base class of every exception raised by ``symbolic_ai.p2_ej4_transition``."""


class ProblemError(EJ4Error):
    """A ``TransitionProblem`` breaks one of its invariants (see ``problem.TransitionProblem``)."""


class InstanceError(EJ4Error):
    """The data cannot produce the requested instances (an unknown identifier, a missing row)."""


class ModelError(EJ4Error):
    """The CSP builder was asked for something the model does not define (a negative horizon)."""


class SolverError(EJ4Error):
    """A solver configuration is invalid, or a solver run broke one of its own guarantees."""


class SolverTimeoutError(SolverError):
    """A solver run exceeded its time budget; the result is unknown, not infeasible."""


class VerificationError(EJ4Error):
    """The checker or the oracle got input it cannot judge (e.g. a schedule with missing days)."""
