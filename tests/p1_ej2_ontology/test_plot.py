"""Fig. 2: each told fact is an edge of its relation's colour, nodes follow the frozen encoding,
the layout fits the page, and the files are reproducible. Fig. 3: each panel edge has the line
style of its provenance in the ``taxonomy`` section, and only the panel's nodes are drawn."""

from __future__ import annotations

import json
import re
import shutil
import struct
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from symbolic_ai.dataloader import load_formulary
from symbolic_ai.p1_ej2_ontology import plot
from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError
from symbolic_ai.p1_ej2_ontology.experiments import (
    as_jsonable,
    run_formulary_derivation,
    run_taxonomy,
    told_knowledge,
)
from symbolic_ai.p1_ej2_ontology.ontology import Ontology
from symbolic_ai.p1_ej2_ontology.plot import (
    CATEGORY_ORDER,
    INFERRED_MEMBER_COLOUR,
    RELATION_COLOURS,
    TAXONOMY_PANELS,
    Relation,
    TaxonomyPanel,
    build_ontology_graph,
    build_taxonomy_graph,
    condition_node,
    plot_ontology,
    plot_taxonomy_diff,
    risk_factor_node,
    taxonomy_figure_stem,
)
from symbolic_ai.p1_ej2_ontology.reasoner import reason
from symbolic_ai.viz import TOL

requires_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not found")

_EDGE = re.compile(r"^\t(\S+) -> (\S+) \[(.*)\]$")
_NODE = re.compile(r"^\t(\S+) \[(.*)\]$")
#: An attribute value: an HTML-like label (``<<...>>``, Fig. 3), a quoted string or a bare word.
_ATTRIBUTE = re.compile(r'(\w+)=(<<.*?>>|"(?:[^"\\]|\\.)*"|[^\s\]]+)')
#: Figure limits of the results design §12.2: full IEEE text width, at most 5 in tall.
_MAX_WIDTH_IN, _MAX_HEIGHT_IN, _PNG_DPI = 7.16, 5.0, 300

_Edge = tuple[str, str, dict[str, str]]
_Pair = tuple[str, str]


def _attributes(text: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in _ATTRIBUTE.findall(text)}


def _edges(source: str) -> list[_Edge]:
    """Return the edges of a ``.dot`` source as (tail, head, attributes)."""
    matches = (_EDGE.match(line) for line in source.splitlines())
    return [(m[1], m[2], _attributes(m[3])) for m in matches if m]


def _nodes(source: str) -> dict[str, dict[str, str]]:
    """Return the node statements of a ``.dot`` source, by node identifier."""
    lines = (line for line in source.splitlines() if " -> " not in line)
    matches = (_NODE.match(line) for line in lines)
    return {m[1]: _attributes(m[2]) for m in matches if m and m[1] not in {"node", "edge"}}


def _position(node: Mapping[str, str]) -> tuple[float, float]:
    x, y = node["pos"].rstrip("!").split(",")
    return float(x), float(y)


def _cop_node(nodes: Mapping[str, Mapping[str, str]]) -> str:
    return next(n for n, a in nodes.items() if a.get("label") == "cop")


def _expected_facts(ontology: Mapping[str, Any], cop: str) -> dict[Relation, set[_Pair]]:
    """Return the told facts of each relation, read from the ``ontology`` section directly."""
    copres = ontology["coprescriptions"]
    return {
        Relation.SUBSET: {(c["id"], p) for c in ontology["categories"] for p in c["parents"]},
        Relation.MEMBER: {(d["id"], c) for d in ontology["drugs"] for c in d["categories"]},
        Relation.TREATS: {
            (i["subject"], condition_node(i["condition"])) for i in ontology["indications"]
        },
        Relation.CONTRAINDICATED: {
            (i["subject"], risk_factor_node(i["risk_factor"]))
            for i in ontology["contraindications"]
        },
        Relation.COPRESCRIPTION: {(c["subject"], cop) for c in copres}
        | {(cop, risk_factor_node(c["risk_factor"])) for c in copres}
        | {(cop, c["companion"]) for c in copres},
        Relation.INTERACTION: {(i["subject_a"], i["subject_b"]) for i in ontology["interactions"]},
        Relation.DISJOINT: {
            pair
            for s in ontology["disjoint_sets"]
            for pair in combinations(sorted(s["categories"]), 2)
        },
        Relation.FAMILY: {(f, "therapeutic_families") for f in ontology["families"]},
        Relation.DEFINITION: {
            (d["category"], k) for d in ontology["definitions"] for k in d["conjuncts"]
        },
    }


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


@pytest.fixture(scope="module")
def told_source(results: Mapping[str, object]) -> str:
    """The ``.dot`` source of the told figure (the one the report shows)."""
    return build_ontology_graph(results, inferred=False).source


# --- the frozen encoding --------------------------------------------------------------------------


def test_relation_colours_are_the_frozen_values_of_the_design() -> None:
    assert dict(RELATION_COLOURS) == {
        "Subset": "#000000",
        "Member": "#4477AA",
        "TrataCat": "#228833",
        "CICat": "#EE6677",
        "CopCat": "#EE7733",
        "IntCat": "#AA3377",
        "Disj": "#66CCEE",
        "Familia": "#CCBB44",
        "Definition": "#888888",  # §12.1's allowed darkening of #BBBBBB
    }
    assert INFERRED_MEMBER_COLOUR == "#4477AA80"


def test_relation_colours_are_tol_colours_black_or_the_darkened_grey() -> None:
    allowed = {*TOL.values(), "#000000", "#888888"}
    assert set(RELATION_COLOURS.values()) <= allowed
    assert len(set(RELATION_COLOURS.values())) == len(RELATION_COLOURS)


@pytest.mark.integration
@requires_dot
@pytest.mark.parametrize(
    ("relation", "count"),
    [
        (Relation.SUBSET, 36),
        (Relation.MEMBER, 19),
        (Relation.TREATS, 6),
        (Relation.CONTRAINDICATED, 10),
        (Relation.COPRESCRIPTION, 3),
        (Relation.INTERACTION, 3),
        (Relation.DISJOINT, 13),
        (Relation.FAMILY, 5),
        (Relation.DEFINITION, 2),
    ],
)
def test_every_told_fact_is_one_edge_of_its_relation_colour(
    results: Mapping[str, object], told_source: str, relation: Relation, count: int
) -> None:
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    expected = _expected_facts(ontology, _cop_node(_nodes(told_source)))[relation]
    drawn = [
        (tail, head)
        for tail, head, attributes in _edges(told_source)
        if attributes["color"] == RELATION_COLOURS[relation]
    ]
    assert len(drawn) == count == len(expected)
    assert set(drawn) == expected


@pytest.mark.integration
@requires_dot
def test_the_told_figure_has_no_other_edge(told_source: str) -> None:
    assert len(_edges(told_source)) == 36 + 19 + 6 + 10 + 3 + 3 + 13 + 5 + 2


@pytest.mark.integration
@requires_dot
def test_symmetric_relations_have_no_arrowhead_and_interactions_are_loops(
    told_source: str,
) -> None:
    undirected = {RELATION_COLOURS[Relation.DISJOINT], RELATION_COLOURS[Relation.INTERACTION]}
    for tail, head, attributes in _edges(told_source):
        assert (attributes.get("dir") == "none") == (attributes["color"] in undirected)
        if attributes["color"] == RELATION_COLOURS[Relation.INTERACTION]:
            assert tail == head
            assert (attributes["tailport"], attributes["headport"]) == ("ne", "nw")


@pytest.mark.integration
@requires_dot
def test_only_subset_and_membership_edges_constrain_the_layout(told_source: str) -> None:
    ranking = {RELATION_COLOURS[Relation.SUBSET], RELATION_COLOURS[Relation.MEMBER]}
    for _, _, attributes in _edges(told_source):
        assert ("constraint" not in attributes) == (attributes["color"] in ranking)


@pytest.mark.integration
@requires_dot
def test_nodes_have_the_shapes_of_the_design(
    results: Mapping[str, object], told_source: str
) -> None:
    nodes = _nodes(told_source)
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    for record in ontology["categories"]:
        node = nodes[record["id"]]
        assert (node["shape"], node["style"]) == ("box", "rounded,filled")
    assert nodes["serotonergic_opioids"]["color"] == RELATION_COLOURS[Relation.DEFINITION]
    assert nodes["therapeutic_families"]["peripheries"] == "2"
    for record in ontology["drugs"]:
        assert nodes[record["id"]]["shape"] == "ellipse"
    for identifier in ("HTN", "AF", "T2D", "DEP", "PAIN", "GERD"):
        node = nodes[condition_node(identifier)]
        assert (node["shape"], node["style"]) == ("box", "filled")
        assert node["color"] == RELATION_COLOURS[Relation.TREATS]
    for identifier in ("PREG", "CKD", "LIVER", "EPI", "ASTHMA", "AGE65"):
        node = nodes[risk_factor_node(identifier)]
        assert (node["shape"], node["style"]) == ("box", "filled")
        assert node["color"] == RELATION_COLOURS[Relation.CONTRAINDICATED]
    cop = nodes[_cop_node(nodes)]
    assert (cop["shape"], cop["color"]) == ("diamond", RELATION_COLOURS[Relation.COPRESCRIPTION])


@pytest.mark.integration
@requires_dot
def test_labels_are_names_and_abbreviations_without_tags_or_atc_codes(
    results: Mapping[str, object], told_source: str
) -> None:
    nodes = _nodes(told_source)
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    for record in ontology["categories"]:
        name = str(record["name_es"])
        expected = name.replace(" ", r"\n", 1) if record["id"] == "therapeutic_families" else name
        assert nodes[record["id"]]["label"] == expected
        if record.get("atc_code"):
            assert str(record["atc_code"]) not in told_source
    assert nodes[condition_node("HTN")]["label"] == "HTA"
    assert nodes[risk_factor_node("AGE65")]["label"] == "EDAD65"
    for tag in ("trata", "CI:", "int.", "partición", "disjuntas", "≡", "legend"):
        assert tag not in told_source
    assert "Times-Roman" in told_source


# --- layout ---------------------------------------------------------------------------------------


@pytest.mark.integration
@requires_dot
def test_every_node_is_pinned_for_neato(told_source: str) -> None:
    assert "neato -n2" in told_source.splitlines()[0]
    assert all(node["pos"].endswith("!") for node in _nodes(told_source).values())


@pytest.mark.integration
@requires_dot
def test_columns_run_from_the_root_to_the_drugs_in_category_order(
    results: Mapping[str, object], told_source: str
) -> None:
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    nodes = {n: _position(a) for n, a in _nodes(told_source).items()}
    categories = [str(c["id"]) for c in ontology["categories"]]
    assert set(CATEGORY_ORDER) == set(categories)
    drugs = {str(d["id"]) for d in ontology["drugs"]}
    risk_x = {x for n, (x, _) in nodes.items() if n.startswith("R_")}
    assert len(risk_x) == 1
    assert max(nodes[c][0] for c in categories) < risk_x.pop() < min(nodes[d][0] for d in drugs)
    assert nodes["drugs"][0] == min(x for x, _ in nodes.values())
    for x in {nodes[c][0] for c in categories}:
        column = [c for c in CATEGORY_ORDER if nodes[c][0] == x]
        heights = [nodes[c][1] for c in column]
        assert heights == sorted(heights, reverse=True), column


@pytest.mark.integration
@requires_dot
def test_figure_fits_the_text_width_and_five_inches(
    results: Mapping[str, object], tmp_path: Path
) -> None:
    _, _, png = plot_ontology(results, tmp_path / "fig", inferred=False)
    width, height = struct.unpack(">II", png.read_bytes()[16:24])
    assert width / _PNG_DPI <= _MAX_WIDTH_IN
    assert height / _PNG_DPI <= _MAX_HEIGHT_IN


@pytest.mark.integration
@requires_dot
def test_pdf_embeds_times_new_roman_only(results: Mapping[str, object], tmp_path: Path) -> None:
    _, pdf, _ = plot_ontology(results, tmp_path / "fig", inferred=False)
    fonts = set(re.findall(rb"/BaseFont /(?:[A-Z]{6}\+)?([A-Za-z0-9-]+)", pdf.read_bytes()))
    assert fonts == {b"TimesNewRomanPSMT"}


# --- the inferred variant and reproducibility -----------------------------------------------------


@pytest.mark.integration
@requires_dot
def test_inferred_variant_adds_only_the_derived_membership_of_tramadol(
    results: Mapping[str, object], told_source: str
) -> None:
    inferred = build_ontology_graph(results, inferred=True).source
    told_lines, inferred_lines = set(told_source.splitlines()), set(inferred.splitlines())
    added = [line for line in inferred.splitlines() if line not in told_lines]
    removed = [line for line in told_source.splitlines() if line not in inferred_lines]
    assert removed == ["digraph fig_ej2_ontology {"]
    assert added == [
        "digraph fig_ej2_ontology_inferred {",
        f'\ttramadol -> serotonergic_opioids [color="{INFERRED_MEMBER_COLOUR}"]',
    ]


@pytest.mark.integration
@requires_dot
def test_source_is_identical_across_two_builds(results: Mapping[str, object]) -> None:
    for inferred in (False, True):
        first = build_ontology_graph(results, inferred=inferred).source
        assert build_ontology_graph(results, inferred=inferred).source == first


@pytest.mark.integration
@requires_dot
def test_rendering_twice_gives_identical_files(
    results: Mapping[str, object], tmp_path: Path
) -> None:
    first = plot_ontology(results, tmp_path / "a" / "fig", inferred=False)
    second = plot_ontology(results, tmp_path / "b" / "fig", inferred=False)
    assert [p.suffix for p in first] == [".dot", ".pdf", ".png"]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()


# --- other ontologies and failure paths -------------------------------------------------------


@requires_dot
def test_a_toy_ontology_without_families_draws_an_interaction_pair_as_an_edge(
    toy_ontology: Ontology,
) -> None:
    section = {**told_knowledge(toy_ontology), "families": []}
    edges = _edges(build_ontology_graph({"ontology": section}, inferred=False).source)
    interactions = [
        (tail, head)
        for tail, head, attributes in edges
        if attributes["color"] == RELATION_COLOURS[Relation.INTERACTION]
    ]
    assert interactions == [("a", "b")]
    assert len(edges) == 4 + 3 + 1 + 1 + 3 + 1 + 1


def test_families_without_their_category_of_categories_raise(toy_ontology: Ontology) -> None:
    with pytest.raises(ResultsFormatError, match="therapeutic_families"):
        build_ontology_graph({"ontology": told_knowledge(toy_ontology)}, inferred=False)


def test_missing_section_raises() -> None:
    with pytest.raises(ResultsFormatError, match="ontology"):
        build_ontology_graph({}, inferred=False)


@pytest.mark.integration
def test_malformed_record_raises(results: Mapping[str, object]) -> None:
    ontology = results["ontology"]
    assert isinstance(ontology, Mapping)
    broken = {**results, "ontology": {**ontology, "drugs": [{"id": 3}]}}
    with pytest.raises(ResultsFormatError):
        build_ontology_graph(broken, inferred=False)


# --- Fig. 3: the deduced taxonomy -------------------------------------------------------------

#: One IEEE column, and both panels together at most 2.6 in tall (panel (b) about 1 in).
_COLUMN_WIDTH_IN, _PANELS_HEIGHT_IN, _PANEL_B_MAX_HEIGHT_IN = 3.45, 2.6, 1.1
_SUBSET, _MEMBER = RELATION_COLOURS[Relation.SUBSET], RELATION_COLOURS[Relation.MEMBER]


@pytest.fixture(scope="module")
def full_results(real_ontology: Ontology, results: Mapping[str, object]) -> Mapping[str, Any]:
    """The sections Fig. 3 reads, through a JSON round trip as in ``results.json``."""
    tax = run_taxonomy(reason(real_ontology), oracle_run=None, oracle_prototypes=None)
    loaded: Mapping[str, Any] = json.loads(json.dumps({**results, "taxonomy": as_jsonable(tax)}))
    return loaded


def _expected_taxonomy_edges(
    results: Mapping[str, Any], panel: TaxonomyPanel
) -> dict[_Pair, tuple[str, str]]:
    """Return (colour, style) of every edge a panel must draw, read from the results directly."""
    nodes = set(TAXONOMY_PANELS[panel])
    tax, ontology = results["taxonomy"], results["ontology"]
    expected: dict[_Pair, tuple[str, str]] = {}
    for e in tax["direct_edges"]:
        style = "solid" if e["status"] == "told" else "dashed"
        expected[e["child"], e["parent"]] = (_SUBSET, style)
    for e in tax["told_edges_made_indirect"]:
        expected[e["child"], e["parent"]] = (_SUBSET + "73", "dotted")
    for d in ontology["drugs"]:
        for c in d["categories"]:
            if c in tax["most_specific_categories"][d["id"]]:
                expected[d["id"], c] = (_MEMBER, "solid")
    for m in tax["new_direct_memberships"]:
        expected[m["drug"], m["category"]] = (_MEMBER, "dashed")
    for m in tax["memberships_made_indirect"]:
        expected[m["child"], m["parent"]] = (_MEMBER + "73", "dotted")
    return {pair: v for pair, v in expected.items() if set(pair) <= nodes}


@pytest.mark.integration
@requires_dot
@pytest.mark.parametrize("panel", list(TaxonomyPanel))
def test_every_panel_edge_has_the_style_of_its_status(
    full_results: Mapping[str, Any], panel: TaxonomyPanel
) -> None:
    source = build_taxonomy_graph(full_results, panel).source
    drawn = {(t, h): (a["color"], a["style"]) for t, h, a in _edges(source)}
    assert len(drawn) == len(_edges(source))
    assert drawn == _expected_taxonomy_edges(full_results, panel)


@pytest.mark.integration
@requires_dot
@pytest.mark.parametrize(
    ("panel", "counts"),
    [
        # (solid, dashed, dotted) per panel
        (TaxonomyPanel.A, (5, 6, 1)),
        (TaxonomyPanel.B, (0, 4, 2)),
    ],
)
def test_panel_edge_counts(
    full_results: Mapping[str, Any], panel: TaxonomyPanel, counts: tuple[int, int, int]
) -> None:
    styles = [a["style"] for _, _, a in _edges(build_taxonomy_graph(full_results, panel).source)]
    assert (styles.count("solid"), styles.count("dashed"), styles.count("dotted")) == counts


@pytest.mark.integration
@requires_dot
@pytest.mark.parametrize("panel", list(TaxonomyPanel))
def test_panels_draw_only_their_own_nodes(
    full_results: Mapping[str, Any], panel: TaxonomyPanel
) -> None:
    source = build_taxonomy_graph(full_results, panel).source
    allowed = set(TAXONOMY_PANELS[panel])
    assert set(_nodes(source)) == allowed
    for tail, head, _ in _edges(source):
        assert {tail, head} <= allowed


@pytest.mark.integration
@requires_dot
def test_hbpm_has_no_edge_to_the_pregnancy_contraindication(
    full_results: Mapping[str, Any],
) -> None:
    pairs = {tuple(p) for p in full_results["taxonomy"]["pairs"]}
    assert ("low_molecular_weight_heparins", "contraindicated_PREG") not in pairs
    edges = _edges(build_taxonomy_graph(full_results, TaxonomyPanel.A).source)
    assert ("vitamin_k_antagonists", "contraindicated_PREG") in {(t, h) for t, h, _ in edges}
    assert [(t, h) for t, h, _ in edges if t == "low_molecular_weight_heparins"] == [
        ("low_molecular_weight_heparins", "anticoagulants")
    ]


@pytest.mark.integration
@requires_dot
def test_panel_nodes_follow_the_encoding_of_fig_2(full_results: Mapping[str, Any]) -> None:
    nodes = {
        **_nodes(build_taxonomy_graph(full_results, TaxonomyPanel.A).source),
        **_nodes(build_taxonomy_graph(full_results, TaxonomyPanel.B).source),
    }
    grey = RELATION_COLOURS[Relation.DEFINITION]
    assert nodes["contraindicated_PREG"]["label"] == (
        '<<i>Contraind</i><font point-size="9"><sub>EMB</sub></font>>'
    )
    assert nodes["candidate_AF"]["label"] == (
        '<<i>Candidato</i><font point-size="9"><sub>FA</sub></font>>'
    )
    assert nodes["serotonergic_opioids"]["label"] == "Opioides serotoninérgicos"
    assert nodes["low_molecular_weight_heparins"]["label"] == "HBPM"
    for defined in ("candidate_AF", "contraindicated_PREG", "contraindicated_EPI"):
        assert nodes[defined]["color"] == grey
    assert nodes["serotonergic_opioids"]["color"] == grey
    for primitive in ("drugs", "anticoagulants", "opioids"):
        assert "color" not in nodes[primitive]
        assert (nodes[primitive]["shape"], nodes[primitive]["style"]) == ("box", "rounded,filled")
    assert (nodes["tramadol"]["shape"], nodes["tramadol"]["label"]) == ("ellipse", "Tramadol")


@pytest.mark.integration
@requires_dot
def test_panels_fit_one_column_and_2_6_inches(
    full_results: Mapping[str, Any], tmp_path: Path
) -> None:
    sizes = {}
    for panel in TaxonomyPanel:
        _, _, png = plot_taxonomy_diff(full_results, tmp_path / panel.value, panel=panel)
        width, height = struct.unpack(">II", png.read_bytes()[16:24])
        sizes[panel] = (width / _PNG_DPI, height / _PNG_DPI)
    assert all(width <= _COLUMN_WIDTH_IN for width, _ in sizes.values())
    assert sum(height for _, height in sizes.values()) <= _PANELS_HEIGHT_IN
    assert sizes[TaxonomyPanel.B][1] <= _PANEL_B_MAX_HEIGHT_IN


@pytest.mark.integration
@requires_dot
def test_panel_pdf_embeds_times_new_roman_only(
    full_results: Mapping[str, Any], tmp_path: Path
) -> None:
    _, pdf, _ = plot_taxonomy_diff(full_results, tmp_path / "a", panel=TaxonomyPanel.A)
    fonts = set(re.findall(rb"/BaseFont /(?:[A-Z]{6}\+)?([A-Za-z0-9-]+)", pdf.read_bytes()))
    assert fonts == {b"TimesNewRomanPSMT", b"TimesNewRomanPS-ItalicMT"}


@pytest.mark.integration
@requires_dot
def test_panels_are_identical_across_two_renderings(
    full_results: Mapping[str, Any], tmp_path: Path
) -> None:
    for panel in TaxonomyPanel:
        stem = taxonomy_figure_stem(panel)
        first = plot_taxonomy_diff(full_results, tmp_path / "1" / stem, panel=panel)
        second = plot_taxonomy_diff(full_results, tmp_path / "2" / stem, panel=panel)
        assert [p.name for p in first] == [f"{stem}.dot", f"{stem}.pdf", f"{stem}.png"]
        for a, b in zip(first, second, strict=True):
            assert a.read_bytes() == b.read_bytes()


def test_panel_without_the_taxonomy_section_raises(toy_ontology: Ontology) -> None:
    with pytest.raises(ResultsFormatError, match="taxonomy"):
        build_taxonomy_graph({"ontology": told_knowledge(toy_ontology)}, TaxonomyPanel.B)


@pytest.mark.integration
def test_panel_with_an_unknown_node_raises(
    full_results: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plot, "TAXONOMY_PANELS", {TaxonomyPanel.B: ("opioids", "no_such_id")})
    with pytest.raises(ResultsFormatError, match="no_such_id"):
        build_taxonomy_graph(full_results, TaxonomyPanel.B)
