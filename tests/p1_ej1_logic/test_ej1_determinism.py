"""Determinism across PYTHONHASHSEED: the same encounter always yields the same prescription.

``aima.utils.Expr.__hash__`` depends on Python's string hash, which is randomised per process
unless ``PYTHONHASHSEED`` is fixed. Calling ``aima.logic.dpll`` with symbols sorted by name (never
``dpll_satisfiable``, which collects symbols from a ``set``) is what makes ``DPLLSolver``
independent of it (``EJ1-sat/solver-choice.md`` §7). This test proves it end to end, through
``main.py``, in two real subprocesses, with the dataloader stubbed by
``_determinism_subprocess.py`` (integration-free: no real ``data/`` is read).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "_determinism_subprocess.py"


def _run_with_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout


def test_same_output_across_hash_seeds() -> None:
    output_seed_0 = _run_with_seed("0")
    output_seed_1 = _run_with_seed("1")

    assert output_seed_0 == output_seed_1
    assert "=== E001" in output_seed_0
    assert "action: prescribe" in output_seed_0
