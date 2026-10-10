"""Standalone script run in a subprocess by ``test_ej3_main.py``.

Not a test module (no ``test_`` prefix, never collected by pytest). Builds the full formulary,
the costs and E001-E005 from the fixture modules, replaces the loader names that
``symbolic_ai.p1_ej3_search.main`` imports, and runs the CLI on every encounter with the
algorithm given as argument, printing exactly what ``main --all`` would.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for directory in (_THIS_DIR, _THIS_DIR.parent / "p1_ej1_logic"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ej3_fixture_data  # noqa: E402
import fixture_data  # noqa: E402

from symbolic_ai.dataloader.models import DrugCost  # noqa: E402
from symbolic_ai.p1_ej3_search import main as main_module  # noqa: E402


def run(algorithm: str) -> None:
    """Replace the dataloader names and run the CLI on every fixture encounter."""
    formulary = fixture_data.full_formulary()
    all_encounters = fixture_data.encounters()
    priced = {d: DrugCost(d, cost) for d, cost in ej3_fixture_data.COSTS.items()}

    main_module.load_formulary = lambda data_dir=None, version=None: formulary
    main_module.load_drug_costs = lambda data_dir=None, version=None: priced
    main_module.load_encounters = lambda data_dir=None: all_encounters

    main_module.main(["--all", "--algorithm", algorithm])


if __name__ == "__main__":
    run(sys.argv[1])
