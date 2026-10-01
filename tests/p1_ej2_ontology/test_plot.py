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
    assert "trata DOL; CI: EPI" in inferred  # tramadol, through the defined category


@requires_dot
def test_there_is_no_upper_ontology(results: Mapping[str, object]) -> None:
    for inferred in (False, True):
        source = build_ontology_graph(results, inferred=inferred).source
        for upper in ("clinical_objects", "conditions", "risk_factors", "Objetos clínicos"):
            assert upper not in source


@requires_dot
def test_multiple_inheritance_draws_every_parent_arrow(results: Mapping[str, object]) -> None:
    source = build_ontology_graph(results, inferred=False).source
    for parent in ("antidepressants", "bleeding_risk_drugs", "serotonergic_drugs"):
        assert f"\t{parent} -> ssris [dir=back]" in source
    assert "\tbradycardic_drugs -> non_dihydropyridines [dir=back]" in source
    assert "\tbleeding_risk_drugs -> anticoagulants [dir=back]" in source


@requires_dot
def test_defined_category_is_dashed_with_its_definition_and_hollow_arrows(
    results: Mapping[str, object],
) -> None:
    source = build_ontology_graph(results, inferred=False).source
    node = next(line for line in source.splitlines() if "\tserotonergic_opioids [" in line)
    assert 'style="rounded,filled,dashed"' in node
    assert "≡ Opioides ∩ Serotoninérgicos" in node
    for conjunct in ("opioids", "serotonergic_drugs"):
        assert f"\t{conjunct} -> serotonergic_opioids [arrowtail=empty dir=back]" in source
    assert "cluster_serotonergic_opioids" in source


@requires_dot
def test_tramadol_has_two_membership_arrows_and_none_to_the_defined_category(
    results: Mapping[str, object],
) -> None:
    source = build_ontology_graph(results, inferred=False).source
    arrows = [line for line in source.splitlines() if "-> tramadol [" in line]
    assert [a.split("->")[0].strip() for a in arrows] == ["opioids", "serotonergic_drugs"]
    assert all("style=dashed" in a for a in arrows)


@requires_dot
def test_inferred_variant_draws_the_classification_of_tramadol(
    results: Mapping[str, object],
) -> None:
    told = build_ontology_graph(results, inferred=False).source
    inferred = build_ontology_graph(results, inferred=True).source
    classification = "\tserotonergic_opioids -> tramadol ["
    assert classification not in told
    assert classification in inferred


@requires_dot
def test_self_links_are_loops_with_their_effect(results: Mapping[str, object]) -> None:
    source = build_ontology_graph(results, inferred=False).source
    for category_id in ("bleeding_risk_drugs", "bradycardic_drugs", "serotonergic_drugs"):
        assert f"\t{category_id} -> {category_id} [" in source
    for effect in ("int.: hemorragia", "int.: bradicardia, BAV", "int.: sínd. serotoninérgico"):
        assert effect in source


@requires_dot
def test_disjointness_tags_survive_multiple_inheritance(results: Mapping[str, object]) -> None:
    source = build_ontology_graph(results, inferred=False).source
    for category_id in ("analgesics", "antidepressants", "antidiabetics"):
        node = next(line for line in source.splitlines() if f"\t{category_id} [" in line)
        assert "disjuntas" in node


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
