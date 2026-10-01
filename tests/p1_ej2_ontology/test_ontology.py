"""The ontology and its knowledge base: the counts of results design §11 and the build errors."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from ej2_toy_data import toy_ontology_data

from aima.logic import is_variable
from symbolic_ai.dataloader.models import (
    Category,
    CategoryDefinition,
    InteractionLink,
    Membership,
    SubcategoryEdge,
)
from symbolic_ai.p1_ej2_ontology.errors import (
    CyclicTaxonomyError,
    OntologyError,
    UnknownIdentifierError,
)
from symbolic_ai.p1_ej2_ontology.forward_chaining import KnowledgeBase, fc_closure
from symbolic_ai.p1_ej2_ontology.ontology import (
    MEMBER,
    DefinedKind,
    Ontology,
    atom,
    build_ontology,
    category_symbol,
    condition_symbol,
    drug_symbol,
    identifier,
    prototype_symbol,
    risk_factor_symbol,
    tagged_rules,
    to_knowledge_base,
)

# --- the real ontology (data 1.1.0, amended) ------------------------------------------------


@pytest.mark.integration
def test_real_ontology_has_32_drug_and_12_plus_1_defined_categories(
    real_ontology: Ontology,
) -> None:
    assert len(real_ontology.category_ids) == 33
    assert len(real_ontology.drug_categories) == 32
    assert len(real_ontology.primitive_categories) == 31
    assert real_ontology.conjunctive_ids == ("serotonergic_opioids",)
    assert len(real_ontology.defined_categories) == 12
    assert len(real_ontology.named_categories) == 44
    kinds = Counter(d.kind for d in real_ontology.defined_categories)
    assert kinds == {DefinedKind.CANDIDATE: 6, DefinedKind.CONTRAINDICATED: 6}


@pytest.mark.integration
def test_defined_category_is_a_drug_category_through_its_conjuncts(
    real_ontology: Ontology,
) -> None:
    assert "serotonergic_opioids" in real_ontology.drug_categories
    assert "serotonergic_opioids" not in real_ontology.parents()
    assert real_ontology.conjuncts() == {"serotonergic_opioids": ("opioids", "serotonergic_drugs")}
    assert "therapeutic_families" not in real_ontology.drug_categories
    assert "drugs" in real_ontology.drug_categories


@pytest.mark.integration
def test_told_facts_by_predicate(real_kb: KnowledgeBase) -> None:
    counts = Counter(fact.op for fact in real_kb.facts)
    assert counts == {
        "Member": 19,  # 17 drugs in one category, tramadol in two; no upper ontology
        "Subset": 36,
        "TrataCat": 6,
        "CICat": 10,  # every contraindication is on a category
        "IntCat": 3,  # three self-links
        "CopCat": 1,
        "Familia": 5,
        "Disj": 13,  # 3 + 1 + 1 + 1 + 3 + 1 + 3 pairs from the 7 disjoint sets
    }
    assert len(real_kb.facts) == 93


@pytest.mark.integration
def test_interactions_are_told_as_self_links(real_kb: KnowledgeBase) -> None:
    told = sorted(str(fact) for fact in real_kb.facts if fact.op == "IntCat")
    assert told == [
        "IntCat(C_bleeding_risk_drugs, C_bleeding_risk_drugs)",
        "IntCat(C_bradycardic_drugs, C_bradycardic_drugs)",
        "IntCat(C_serotonergic_drugs, C_serotonergic_drugs)",
    ]


@pytest.mark.integration
def test_rules_are_o1_to_o6_and_the_clauses_of_every_definition(
    real_ontology: Ontology,
) -> None:
    tags = Counter(rule.tag for rule in tagged_rules(real_ontology))
    # O6: 3 clauses for each of the 12 defined categories of §4.5, 2 + 1 for serotonergic_opioids.
    assert tags == {"O1": 1, "O2": 1, "O3": 3, "O4": 2, "O5": 1, "O6": 36 + 3}


@pytest.mark.integration
def test_definition_clauses_cover_both_directions(real_ontology: Ontology) -> None:
    clauses = {str(rule.clause) for rule in tagged_rules(real_ontology)}
    assert {
        "((Member(x, C_opioids) & Member(x, C_serotonergic_drugs)) ==> "
        "Member(x, C_serotonergic_opioids))",
        "(Member(x, C_serotonergic_opioids) ==> Member(x, C_opioids))",
        "(Member(x, C_serotonergic_opioids) ==> Member(x, C_serotonergic_drugs))",
    } <= clauses


@pytest.mark.integration
def test_constants_are_never_aima_variables(real_kb: KnowledgeBase) -> None:
    arguments = {argument for fact in real_kb.facts for argument in fact.args}
    assert arguments
    assert not any(is_variable(argument) for argument in arguments)


def test_symbols_carry_their_prefix_and_round_trip() -> None:
    symbols = {
        drug_symbol("warfarin"): "D_warfarin",
        category_symbol("nsaids"): "C_nsaids",
        condition_symbol("HTN"): "H_HTN",
        risk_factor_symbol("CKD"): "R_CKD",
        prototype_symbol("nsaids"): "P_nsaids",
    }
    for symbol, text in symbols.items():
        assert str(symbol) == text
        assert identifier(symbol) == text[2:]


# --- the toy ontology and the failure paths -------------------------------------------------


def test_toy_ontology_drug_categories_and_defined_ones(toy_ontology: Ontology) -> None:
    assert toy_ontology.drug_categories == ("a", "a1", "a2", "b", "drugs")
    assert toy_ontology.defined_ids == ("candidate_H1", "contraindicated_R1")
    assert toy_ontology.conjunctive_ids == ()
    assert toy_ontology.told_categories_of("x1") == ("a1",)


def _toy_with_ab(*memberships: Membership) -> Ontology:
    """The toy ontology plus ab ≡ a ⊓ b (no told parent, no told member) and extra memberships."""
    data = toy_ontology_data()
    return build_ontology(
        replace(
            data,
            categories=(*data.categories, Category("ab", "AB", "AB")),
            memberships=(*data.memberships, *memberships),
            definitions=(CategoryDefinition("ab", ("a", "b")),),
        )
    )


def test_object_told_in_both_conjuncts_is_classified_into_the_defined_category() -> None:
    ontology = _toy_with_ab(Membership("y", "a"))  # y is told in b and in a
    closure = fc_closure(to_knowledge_base(ontology))
    assert atom(MEMBER, drug_symbol("y"), category_symbol("ab")) in closure.derived
    assert atom(MEMBER, drug_symbol("x1"), category_symbol("ab")) not in closure.facts
    assert ontology.told_categories_of("y") == ("a", "b")


def test_defined_category_is_below_drugs_through_its_conjuncts() -> None:
    ontology = _toy_with_ab()
    assert "ab" in ontology.drug_categories
    assert ontology.primitive_categories == ("a", "a1", "a2", "b", "drugs")


def test_cycle_through_a_definition_raises() -> None:
    # a ≡ a1 ⊓ b puts a below a1, while a1 ⊂ a is told.
    data = replace(toy_ontology_data(), definitions=(CategoryDefinition("a", ("a1", "b")),))
    with pytest.raises(CyclicTaxonomyError):
        build_ontology(data)


def test_unknown_conjunct_raises() -> None:
    data = replace(toy_ontology_data(), definitions=(CategoryDefinition("a", ("b", "ghost")),))
    with pytest.raises(UnknownIdentifierError, match="ghost"):
        build_ontology(data)


def test_unknown_membership_category_raises() -> None:
    data = replace(toy_ontology_data(), memberships=(Membership("x1", "nowhere"),))
    with pytest.raises(UnknownIdentifierError, match="nowhere"):
        build_ontology(data)


def test_unknown_link_subject_raises() -> None:
    interactions = (InteractionLink("a", "ghost", "bleeding", "major"),)
    with pytest.raises(UnknownIdentifierError, match="ghost"):
        build_ontology(toy_ontology_data(interactions=interactions))


def test_missing_drugs_category_raises() -> None:
    data = toy_ontology_data()
    categories = tuple(c for c in data.categories if c.category_id != "drugs")
    edges = tuple(e for e in data.subcategories if e.parent_id != "drugs")
    with pytest.raises(UnknownIdentifierError, match="drugs"):
        build_ontology(replace(data, categories=categories, subcategories=edges))


def test_cyclic_taxonomy_raises() -> None:
    edges = (
        SubcategoryEdge("a", "drugs"),
        SubcategoryEdge("a1", "a"),
        SubcategoryEdge("a", "a1"),
        SubcategoryEdge("a2", "a"),
        SubcategoryEdge("b", "drugs"),
    )
    with pytest.raises(CyclicTaxonomyError):
        build_ontology(toy_ontology_data(subcategories=edges))


def test_interaction_between_a_drug_and_a_category_raises() -> None:
    interactions = (InteractionLink("a", "x1", "bleeding", "major"),)
    with pytest.raises(OntologyError, match="two categories or two drugs"):
        build_ontology(toy_ontology_data(interactions=interactions))


def test_drug_interacting_with_itself_raises() -> None:
    interactions = (InteractionLink("x1", "x1", "bleeding", "major"),)
    with pytest.raises(OntologyError, match="cannot interact with itself"):
        build_ontology(toy_ontology_data(interactions=interactions))


def test_self_link_on_a_category_is_told_once() -> None:
    interactions = (InteractionLink("a", "a", "bleeding", "major"),)
    kb = to_knowledge_base(build_ontology(toy_ontology_data(interactions=interactions)))
    assert [str(fact) for fact in kb.facts if fact.op == "IntCat"] == ["IntCat(C_a, C_a)"]


def test_unknown_category_lookup_raises(toy_ontology: Ontology) -> None:
    with pytest.raises(UnknownIdentifierError):
        toy_ontology.category("ghost")
    with pytest.raises(UnknownIdentifierError):
        toy_ontology.told_categories_of("ghost")


def test_drug_interaction_is_told_in_both_directions() -> None:
    interactions = (InteractionLink("x1", "y", "bleeding", "major"),)
    kb = to_knowledge_base(build_ontology(toy_ontology_data(interactions=interactions)))
    told = {str(fact) for fact in kb.facts if fact.op == "Int"}
    assert told == {"Int(D_x1, D_y)", "Int(D_y, D_x1)"}
