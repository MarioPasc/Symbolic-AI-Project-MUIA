"""Exceptions raised by the EJ3 cost-aware search agent and its experiments."""

from __future__ import annotations


class EJ3Error(Exception):
    """Base class of every exception raised by ``symbolic_ai.p1_ej3_search``."""


class CostError(EJ3Error):
    """A drug of the formulary has no cost, or a negative one.

    The search needs a non-negative cost for every drug it may prescribe: a missing cost has no
    sensible default, and a negative step cost breaks the optimality of A*.
    """


class RecordError(EJ3Error):
    """A record or a pruning names an identifier that the formulary does not have."""


class SearchInvariantError(EJ3Error):
    """The search contradicts what Agent 1 proved about the same encounter.

    Raised when Agent 1 found a model of Γ⁺ and the search finds no regimen, returns one that
    breaks a clause of Γ⁺, or (for an optimal algorithm) returns one dearer than Agent 1's. None of
    these can happen if the search problem and the clauses of EJ1 describe the same axioms, so
    each is checked before a regimen is used, as EJ1 checks every model of its solver.
    """
