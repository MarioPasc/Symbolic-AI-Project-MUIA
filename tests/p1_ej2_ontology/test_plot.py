"""Fig. 2: the Graphviz source covers the ontology, is deterministic, and renders reproducibly."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from symbolic_ai.dataloader import load_formulary
from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError
from symbolic_ai.p1_ej2_ontology.experiments import (
    as_jsonable,
    run_formulary_derivation,
    told_knowledge,
)
from symbolic_ai.p1_ej2_ontology.ontology import Ontology
from symbolic_ai.p1_ej2_ontology.plot import build_ontology_graph, plot_ontology
from symbolic_ai.p1_ej2_ontology.reasoner import reason

pytestmark = pytest.mark.integration

requires_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not found")


@pytest.fixture(scope="module")
def results(real_ontology: Ontology) -> Mapping[str, object]:
    """The two sections of results.json the figure reads."""
    derivation = run_formulary_derivation(
        reason(real_ontology), load_formulary(version="1.0.0"), oracle_run=None
    )
    return {
        "ontology": told_knowledge(real_ontology),
        "formulary_derivation": as_jsonable(derivation),
    }


@requires_dot
def test_source_has_every_category_and_drug(
    results: Mapping[str, object], real_ontology: Ontology
) -> None:
    source = build_ontology_graph(results, inferred=False).source
    for category_id in real_ontology.category_ids:
        if category_id != "therapeutic_families":
            assert f"\t{category_id} [" in source
    for drug_id in real_ontology.drug_ids:
        assert f"\t{drug_id} [" in source
    assert "legend" not in source
    assert "therapeutic_families [" not in source


@requires_dot
def test_source_uses_spanish_names_and_abbreviations(results: Mapping[str, object]) -> None:
    source = build_ontology_graph(results, inferred=False).source
    assert "Ibuprofeno" in source and "Bloqueantes del SRAA" in source
    assert "trata HTA" in source and "CI: EMB" in source
    assert "cop: EDAD65 → omeprazol" in source
    assert "Times-Roman" in source


@requires_dot
def test_source_is_identical_across_two_builds(results: Mapping[str, object]) -> None:
    for inferred in (False, True):
        first = build_ontology_graph(results, inferred=inferred).source
        assert build_ontology_graph(results, inferred=inferred).source == first


@requires_dot
def test_inferred_variant_adds_inherited_rows(results: Mapping[str, object]) -> None:
    told = build_ontology_graph(results, inferred=False).source
    inferred = build_ontology_graph(results, inferred=True).source
    assert "trata FA; CI: EMB" in inferred and "trata FA; CI: EMB" not in told
    assert "clinical_objects" not in told and "clinical_objects" not in inferred


@requires_dot
def test_rendering_twice_gives_identical_files(
    results: Mapping[str, object], tmp_path: Path
) -> None:
    first = plot_ontology(results, tmp_path / "a" / "fig", inferred=False)
    second = plot_ontology(results, tmp_path / "b" / "fig", inferred=False)
    assert [p.suffix for p in first] == [".dot", ".pdf", ".png"]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()


def test_missing_section_raises() -> None:
    with pytest.raises(ResultsFormatError, match="ontology"):
        build_ontology_graph({}, inferred=False)


def test_malformed_record_raises(results: Mapping[str, object]) -> None:
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    broken = {**results, "ontology": {**ontology, "drugs": [{"id": 3}]}}
    with pytest.raises(ResultsFormatError):
        build_ontology_graph(broken, inferred=False)
