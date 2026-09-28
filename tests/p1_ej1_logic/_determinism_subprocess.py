"""Standalone script run in a subprocess by ``test_ej1_determinism.py``.

Not a test module (no ``test_`` prefix, never collected by pytest). Builds the full 1.0.0
formulary and E001-E005 from ``fixture_data.py`` (the dataloader is a stub in this worktree),
monkeypatches the loader names ``symbolic_ai.p1_ej1_logic.main`` imports, and runs the CLI on every
encounter, printing to stdout exactly as ``python -m symbolic_ai.p1_ej1_logic.main --all`` would.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fixture_data  # noqa: E402

from symbolic_ai.p1_ej1_logic import main as main_module  # noqa: E402


def run() -> None:
    """Monkeypatch the dataloader names and run the CLI on every fixture encounter."""
    formulary = fixture_data.full_formulary()
    all_encounters = fixture_data.encounters()

    main_module.load_formulary = lambda data_dir=None, version=None: formulary
    main_module.load_encounters = lambda data_dir=None: all_encounters
    main_module.load_encounter = lambda encounter_id, data_dir=None: all_encounters[encounter_id]

    main_module.main(["--all", "--first-value", "false"])


if __name__ == "__main__":
    run()
