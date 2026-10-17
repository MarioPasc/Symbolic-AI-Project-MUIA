"""EJ4 (Práctica 2): the safe transition between two regimens, as a CSP indexed by day.

After an event (a new risk factor or a new condition) a patient's regimen must change, but not at
once: a stopped drug stays in the body for some days and keeps interacting, some drugs must be
tapered, and a treated condition may never be left untreated. EJ4 finds the day-by-day regimen
that reaches a safe one in the fewest days, with aima-python's CSP solver (AIMA 4th ed. ch. 5).

Reused code: the CSP algorithms are those of aima-python
(https://github.com/aimacode/aima-python, commit f104e03, MIT licence), vendored unmodified in
``src/aima/`` (see ``src/aima/VENDORED.md``) and called without changes. ``csp_builder.py`` builds
an ``aima.csp.CSP``; ``solve.py`` runs ``aima.csp.AC3``, ``backtracking_search``,
``first_unassigned_variable``, ``num_legal_values`` (inside a deterministic MRV), ``lcv``,
``no_inference``, ``forward_checking``, ``mac`` and ``min_conflicts``, and only adds deterministic
tie-breaks, budgets and a trace around them. The oracle (``oracle.py``) writes its own budgeted
loop after ``aima.search.breadth_first_graph_search``. Textbook: Russell and Norvig, *Artificial
Intelligence: A Modern Approach*, 4th ed., ch. 5 (§5.1 CSPs, §5.2 node and arc consistency, §5.3
backtracking, ordering and inference, §5.4 min-conflicts) and §3.4.1 (breadth-first search).

The public names exported here are the frozen contract only (``problem.py``); the other modules
are imported directly from their submodules.
"""

from __future__ import annotations

from symbolic_ai.p2_ej4_transition.errors import (
    EJ4Error,
    InstanceError,
    ModelError,
    ProblemError,
    SolverError,
    SolverTimeoutError,
    VerificationError,
)
from symbolic_ai.p2_ej4_transition.problem import (
    Day,
    Event,
    EventKind,
    Schedule,
    TransitionProblem,
    Value,
    drug_of,
    is_full_dose,
)

#: The database version EJ4 is pinned to: the formulary, the costs and the pharmacokinetics.
EJ4_DATA_VERSION = "1.3.0"

__all__ = [
    "EJ4_DATA_VERSION",
    "Day",
    "EJ4Error",
    "Event",
    "EventKind",
    "InstanceError",
    "ModelError",
    "ProblemError",
    "Schedule",
    "SolverError",
    "SolverTimeoutError",
    "TransitionProblem",
    "Value",
    "VerificationError",
    "drug_of",
    "is_full_dose",
]
