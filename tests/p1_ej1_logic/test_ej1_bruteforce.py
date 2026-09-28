"""Brute-force model counts against 01-database.md §6 (harness H-EJ1), independent of DPLLSolver.

Marked ``slow``: each case enumerates 2**18 assignments of the formulary's 18 drug symbols
(``docs/HARNESSES/README.md``, H-EJ1).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic.encoding import gamma, gamma_minus, gamma_plus, unknown_risk_factors

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import oracle  # noqa: E402

pytestmark = pytest.mark.slow


def test_e001_gamma_plus_115_models(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    e001 = encounters["E001"]
    base = gamma(formulary, e001)
    unknowns = unknown_risk_factors(e001, formulary)
    clauses = gamma_plus(base, unknowns)
    assert oracle.count_models_of_clauses(clauses) == 115


def test_e002_gamma_minus_94_models(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    e002 = encounters["E002"]
    base = gamma(formulary, e002)
    unknowns = unknown_risk_factors(e002, formulary)
    clauses = gamma_minus(base, unknowns)
    assert oracle.count_models_of_clauses(clauses) == 94


def test_e004_gamma_plus_one_model(
    formulary: Formulary, encounters: Mapping[str, Encounter]
) -> None:
    e004 = encounters["E004"]
    base = gamma(formulary, e004)
    unknowns = unknown_risk_factors(e004, formulary)
    clauses = gamma_plus(base, unknowns)
    assert oracle.count_models_of_clauses(clauses) == 1
