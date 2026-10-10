"""EJ3: the cost-aware prescribing agent (informed search for the cheapest safe regimen).

Agent 1 of EJ1 decides whether to prescribe; when it does, this exercise searches the regimens
of the worst-case record with A* (or IDA*, greedy best-first or uniform-cost search, for
comparison) and prescribes the cheapest, using the costs of data 1.2.0.
"""

from __future__ import annotations

from symbolic_ai.p1_ej3_search.agent import CostAwareAgent, CostedDecision, worst_case_record
from symbolic_ai.p1_ej3_search.errors import (
    CostError,
    EJ3Error,
    RecordError,
    SearchInvariantError,
)
from symbolic_ai.p1_ej3_search.problem import DomainPruning, RegimenProblem
from symbolic_ai.p1_ej3_search.search import Algorithm, SearchResult, search

#: The database version EJ3 is pinned to: the formulary of EJ1 plus the drug costs.
EJ3_DATA_VERSION = "1.2.0"

__all__ = [
    "EJ3_DATA_VERSION",
    "Algorithm",
    "CostAwareAgent",
    "CostError",
    "CostedDecision",
    "DomainPruning",
    "EJ3Error",
    "RecordError",
    "RegimenProblem",
    "SearchInvariantError",
    "SearchResult",
    "search",
    "worst_case_record",
]
