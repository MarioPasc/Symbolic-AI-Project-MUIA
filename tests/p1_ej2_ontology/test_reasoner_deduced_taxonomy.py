"""The deduced taxonomy (direct edges, told links made indirect, most specific categories).

Hand-checked on a synthetic order and on the toy ontology, then against the values the results
redesign proposal §2.2 computed from ``results.json`` (data 1.1.0).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx
import pytest
from ej2_toy_data import toy_ontology_data

from symbolic_ai.dataloader.models import SubcategoryEdge
from symbolic_ai.p1_ej2_ontology.errors import CyclicTaxonomyError
from symbolic_ai.p1_ej2_ontology.ontology import Ontology, build_ontology
from symbolic_ai.p1_ej2_ontology.reasoner import (
    DeducedTaxonomy,
    DirectEdge,
    DrugMembership,
    EdgeStatus,
    IndirectLink,
    SameMembers,
    Taxonomy,
    deduced_taxonomy,
    direct_edges,
    equivalences,
    membership_changes,
    most_specific_categories,
    reason,
    redundant_told_edges,
    same_members_not_equivalent,
    taxonomy,
    told_edges_made_indirect,
    transitive_reduction,
)

TOLD, DEDUCED = EdgeStatus.TOLD, EdgeStatus.DEDUCED


def _taxonomy(
    direct: Iterable[tuple[str, str]], told: Iterable[tuple[str, str]], extra: Iterable[str] = ()
) -> Taxonomy:
    """Return the Taxonomy whose ⊑ is the transitive closure of ``direct`` (child -> parent)."""
    graph: nx.DiGraph[str] = nx.DiGraph(list(direct))
    graph.add_nodes_from(extra)
    categories = tuple(sorted(graph.nodes))
    return Taxonomy(
        categories=categories,
        subsumers={c: tuple(sorted(nx.descendants(graph, c))) for c in categories},
        told=frozenset(told),
    )


#: A chain d ⊑ c ⊑ x ⊑ b ⊑ a where x was never told: c ⊂ b was told instead, and d ⊂ a is told
#: although d ⊂ c ⊂ b ⊂ a already implies it.
CHAIN = _taxonomy(
    direct=[("d", "c"), ("c", "x"), ("x", "b"), ("b", "a")],
    told=[("d", "c"), ("c", "b"), ("b", "a"), ("d", "a")],
)
#: Drugs of each category of CHAIN: m1 told in c, m2 told in b, m3 told in a.
CHAIN_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "a": ("m1", "m2", "m3"),
    "b": ("m1", "m2"),
    "x": ("m1", "m2"),
    "c": ("m1",),
    "d": ("m1",),
}
CHAIN_TOLD_MEMBERSHIPS: Mapping[str, tuple[str, ...]] = {"m1": ("c",), "m2": ("b",), "m3": ("a",)}
#: p and q subsume each other: ⊑ has a cycle.
EQUIVALENT = _taxonomy(direct=[("p", "q"), ("q", "p"), ("q", "top")], told=[("p", "q")])


# --- a synthetic order --------------------------------------------------------------------------


def test_direct_edges_of_a_chain_are_its_links_with_their_provenance() -> None:
    assert direct_edges(CHAIN) == (
        DirectEdge("b", "a", TOLD),
        DirectEdge("c", "x", DEDUCED),
        DirectEdge("d", "c", TOLD),
        DirectEdge("x", "b", DEDUCED),
    )


def test_told_links_skipping_a_category_are_made_indirect_with_every_intermediate() -> None:
    assert told_edges_made_indirect(CHAIN) == (
        IndirectLink("c", "b", ("x",)),
        IndirectLink("d", "a", ("b", "c", "x")),
    )


def test_only_a_told_link_implied_by_other_told_links_is_redundant() -> None:
    # c ⊂ b is made indirect by the deduced x, but no other told link implies it.
    assert redundant_told_edges(CHAIN) == (IndirectLink("d", "a", ("b", "c")),)


def test_a_chain_has_no_equivalences() -> None:
    assert equivalences(CHAIN) == ()


def test_most_specific_categories_are_the_minimal_ones() -> None:
    assert most_specific_categories(CHAIN, CHAIN_MEMBERS) == {
        "m1": ("d",),
        "m2": ("x",),
        "m3": ("a",),
    }


def test_membership_changes_against_the_told_memberships() -> None:
    most_specific = most_specific_categories(CHAIN, CHAIN_MEMBERS)
    made_indirect, new_direct = membership_changes(CHAIN, most_specific, CHAIN_TOLD_MEMBERSHIPS)
    assert made_indirect == (IndirectLink("m1", "c", ("d",)), IndirectLink("m2", "b", ("x",)))
    assert new_direct == (DrugMembership("m1", "d"), DrugMembership("m2", "x"))


def test_same_members_without_equivalence_records_the_one_way_subsumption() -> None:
    assert same_members_not_equivalent(CHAIN, CHAIN_MEMBERS) == (
        SameMembers("b", "x", ("m1", "m2"), False, True),
        SameMembers("c", "d", ("m1",), False, True),
    )


def test_empty_categories_are_never_paired() -> None:
    members = {"a": (), "b": (), "c": ("m1",), "d": ("m1",), "x": ()}
    assert [(r.a, r.b) for r in same_members_not_equivalent(CHAIN, members)] == [("c", "d")]


def test_a_single_category_has_no_direct_edge() -> None:
    lone = _taxonomy(direct=[], told=[], extra=["only"])
    assert direct_edges(lone) == ()
    assert told_edges_made_indirect(lone) == redundant_told_edges(lone) == ()


def test_equivalent_categories_are_reported_and_paired_only_when_not_equivalent() -> None:
    assert equivalences(EQUIVALENT) == (("p", "q"),)
    members = {"p": ("m",), "q": ("m",), "top": ("m",)}
    pairs = [(r.a, r.b) for r in same_members_not_equivalent(EQUIVALENT, members)]
    assert pairs == [("p", "top"), ("q", "top")]


def test_equivalent_categories_have_no_unique_direct_edges() -> None:
    with pytest.raises(CyclicTaxonomyError, match="cycle"):
        direct_edges(EQUIVALENT)
    with pytest.raises(CyclicTaxonomyError):
        transitive_reduction(["p", "q"], [("p", "q"), ("q", "p")])


# --- the toy ontology ---------------------------------------------------------------------------


@pytest.fixture
def toy_deduced() -> DeducedTaxonomy:
    """The toy ontology with the redundant told link a1 ⊂ drugs added."""
    edges = (
        SubcategoryEdge("a", "drugs"),
        SubcategoryEdge("a1", "a"),
        SubcategoryEdge("a1", "drugs"),
        SubcategoryEdge("a2", "a"),
        SubcategoryEdge("b", "drugs"),
    )
    reasoned = reason(build_ontology(toy_ontology_data(subcategories=edges)))
    return deduced_taxonomy(reasoned, taxonomy(reasoned))


def test_toy_direct_edges_place_the_defined_categories(toy_deduced: DeducedTaxonomy) -> None:
    # a treats H1, so a ⊑ candidate_H1 ⊑ drugs; a1 is contraindicated by R1.
    assert toy_deduced.direct_edges == (
        DirectEdge("a", "candidate_H1", DEDUCED),
        DirectEdge("a1", "a", TOLD),
        DirectEdge("a1", "contraindicated_R1", DEDUCED),
        DirectEdge("a2", "a", TOLD),
        DirectEdge("b", "drugs", TOLD),
        DirectEdge("candidate_H1", "drugs", DEDUCED),
        DirectEdge("contraindicated_R1", "drugs", DEDUCED),
    )


def test_toy_told_links_made_indirect_and_redundant(toy_deduced: DeducedTaxonomy) -> None:
    assert toy_deduced.told_edges_made_indirect == (
        IndirectLink("a", "drugs", ("candidate_H1",)),
        IndirectLink("a1", "drugs", ("a", "candidate_H1", "contraindicated_R1")),
    )
    assert toy_deduced.redundant_told_edges == (IndirectLink("a1", "drugs", ("a",)),)
    assert toy_deduced.equivalences == ()


def test_toy_members_and_most_specific_categories(toy_deduced: DeducedTaxonomy) -> None:
    assert toy_deduced.members["candidate_H1"] == ("x1", "x2")
    assert toy_deduced.members["contraindicated_R1"] == ("x1",)
    assert toy_deduced.most_specific_categories == {"x1": ("a1",), "x2": ("a2",), "y": ("b",)}
    assert toy_deduced.memberships_made_indirect == toy_deduced.new_direct_memberships == ()
    assert toy_deduced.same_members_not_equivalent == (
        SameMembers("a", "candidate_H1", ("x1", "x2"), True, False),
        SameMembers("a1", "contraindicated_R1", ("x1",), True, False),
    )


# --- the real ontology (data 1.1.0) -------------------------------------------------------------


@pytest.fixture(scope="module")
def real_deduced(real_ontology: Ontology) -> DeducedTaxonomy:
    """The deduced taxonomy of the committed database."""
    reasoned = reason(real_ontology)
    return deduced_taxonomy(reasoned, taxonomy(reasoned))


@pytest.mark.integration
def test_real_direct_edges_are_60_half_told(real_deduced: DeducedTaxonomy) -> None:
    statuses = [e.status for e in real_deduced.direct_edges]
    assert (len(statuses), statuses.count(TOLD), statuses.count(DEDUCED)) == (60, 30, 30)
    assert real_deduced.equivalences == ()


@pytest.mark.integration
def test_real_therapeutic_groups_move_below_their_candidate_category(
    real_deduced: DeducedTaxonomy,
) -> None:
    assert real_deduced.told_edges_made_indirect == (
        IndirectLink("analgesics", "drugs", ("candidate_PAIN",)),
        IndirectLink("anticoagulants", "drugs", ("bleeding_risk_drugs", "candidate_AF")),
        IndirectLink("antidepressants", "drugs", ("candidate_DEP",)),
        IndirectLink("antidiabetics", "drugs", ("candidate_T2D",)),
        IndirectLink("antihypertensives", "drugs", ("candidate_HTN",)),
        IndirectLink("proton_pump_inhibitors", "drugs", ("candidate_GERD",)),
    )


@pytest.mark.integration
def test_real_redundant_told_edge_is_anticoagulants_to_drugs(
    real_deduced: DeducedTaxonomy,
) -> None:
    assert real_deduced.redundant_told_edges == (
        IndirectLink("anticoagulants", "drugs", ("bleeding_risk_drugs",)),
    )


@pytest.mark.integration
def test_real_members_cover_44_categories_two_of_them_empty(
    real_deduced: DeducedTaxonomy,
) -> None:
    assert len(real_deduced.members) == 44
    empty = [c for c, drugs in real_deduced.members.items() if not drugs]
    assert empty == ["contraindicated_AGE65", "low_molecular_weight_heparins"]


@pytest.mark.integration
def test_real_same_members_not_equivalent_are_eleven_pairs(
    real_deduced: DeducedTaxonomy,
) -> None:
    rows = {(r.a, r.b): r for r in real_deduced.same_members_not_equivalent}
    assert len(rows) == 11
    htn = rows["antihypertensives", "candidate_HTN"]
    assert (htn.a_subsumed_by_b, htn.b_subsumed_by_a) == (True, False)
    assert len(htn.members) == 7
    epi = rows["contraindicated_EPI", "opioids"]
    assert (epi.members, epi.a_subsumed_by_b, epi.b_subsumed_by_a) == (("tramadol",), False, False)


@pytest.mark.integration
def test_real_most_specific_categories_are_the_told_ones_except_tramadol(
    real_ontology: Ontology, real_deduced: DeducedTaxonomy
) -> None:
    most_specific = real_deduced.most_specific_categories
    assert set(most_specific) == set(real_ontology.drug_ids)
    changed = {d: m for d, m in most_specific.items() if m != real_ontology.told_categories_of(d)}
    assert changed == {"tramadol": ("serotonergic_opioids",)}


@pytest.mark.integration
def test_real_membership_changes_are_tramadols(real_deduced: DeducedTaxonomy) -> None:
    assert real_deduced.memberships_made_indirect == (
        IndirectLink("tramadol", "opioids", ("serotonergic_opioids",)),
        IndirectLink("tramadol", "serotonergic_drugs", ("serotonergic_opioids",)),
    )
    assert real_deduced.new_direct_memberships == (
        DrugMembership("tramadol", "serotonergic_opioids"),
    )
