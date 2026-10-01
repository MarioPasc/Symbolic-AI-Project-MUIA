"""Fixtures for EJ2 tests: the real ontology (data 1.1.0), its closure, and the toy ontology."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from symbolic_ai.dataloader import load_ontology
from symbolic_ai.p1_ej2_ontology.forward_chaining import Closure, KnowledgeBase, fc_closure
from symbolic_ai.p1_ej2_ontology.ontology import Ontology, build_ontology, to_knowledge_base

# ``--import-mode=importlib`` does not add test directories to ``sys.path``; the plain helper
# module ``ej2_toy_data`` is imported from here, as EJ1's conftest does for ``fixture_data``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ej2_toy_data import toy_ontology_data  # noqa: E402


@pytest.fixture(scope="session")
def real_ontology() -> Ontology:
    """The ontology of the committed database."""
    return build_ontology(load_ontology())


@pytest.fixture(scope="session")
def real_kb(real_ontology: Ontology) -> KnowledgeBase:
    """The Datalog KB of the real ontology."""
    return to_knowledge_base(real_ontology)


@pytest.fixture(scope="session")
def real_closure(real_kb: KnowledgeBase) -> Closure:
    """Cl(KB) of the real ontology."""
    return fc_closure(real_kb)


@pytest.fixture
def toy_ontology() -> Ontology:
    """The toy ontology of ``ej2_toy_data.py``."""
    return build_ontology(toy_ontology_data())
