"""The ontology and its knowledge base: the counts of results design §2 and the build errors."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from ej2_toy_data import toy_ontology_data

from aima.logic import is_variable
from symbolic_ai.dataloader.models import InteractionLink, Membership, SubcategoryEdge
from symbolic_ai.p1_ej2_ontology.errors import (
    CyclicTaxonomyError,
    OntologyError,
    UnknownIdentifierError,
)
from symbolic_ai.p1_ej2_ontology.forward_chaining import KnowledgeBase
from symbolic_ai.p1_ej2_ontology.ontology import (
    DefinedKind,
    Ontology,
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

# --- the real ontology (data 1.1.0) ---------------------------------------------------------


@pytest.mark.integration
def test_real_ontology_has_29_drug_and_12_defined_categories(real_ontology: Ontology) -> None:
    assert len(real_ontology.category_ids) == 33
    assert len(real_ontology.drug_categories) == 29
    assert len(real_ontology.defined_categories) == 12
    assert len(real_ontology.named_categories) == 41
    kinds = Counter(d.kind for d in real_ontology.defined_categories)
    assert kinds == {DefinedKind.CANDIDATE: 6, DefinedKind.CONTRAINDICATED: 6}


@pytest.mark.integration
def test_upper_categories_are_not_drug_categories(real_ontology: Ontology) -> None:
    upper = {"clinical_objects", "conditions", "risk_factors", "therapeutic_families"}
    assert upper.isdisjoint(real_ontology.drug_categories)
    assert "drugs" in real_ontology.drug_categories


@pytest.mark.integration
def test_told_facts_by_predicate(real_kb: KnowledgeBase) -> None:
    counts = Counter(fact.op for fact in real_kb.facts)
    assert counts == {
        "Member": 18 + 6 + 6,  # drugs in their leaf; conditions and risk factors in their upper
        "Subset": 31,
        "TrataCat": 6,
        "CICat": 9,
        "CI": 1,  # tramadol, EPI (object level)
        "IntCat": 5,
        "CopCat": 1,
        "Familia": 5,
        "Disj": 16,  # 3 + 1 + 1 + 1 + 3 + 3 + 1 + 3 pairs from the 8 disjoint sets
    }
    assert len(real_kb.facts) == 104


@pytest.mark.integration
def test_rules_are_o1_to_o6_and_three_clauses_per_definition(real_ontology: Ontology) -> None:
    tags = Counter(rule.tag for rule in tagged_rules(real_ontology))
    assert tags == {"O1": 1, "O2": 1, "O3": 3, "O4": 1, "O5": 1, "O6": 1, "O7": 36}


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
    assert toy_ontology.leaf_of("x1") == "a1"


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


def test_unknown_category_lookup_raises(toy_ontology: Ontology) -> None:
    with pytest.raises(UnknownIdentifierError):
        toy_ontology.category("ghost")
    with pytest.raises(UnknownIdentifierError):
        toy_ontology.leaf_of("ghost")


def test_drug_interaction_is_told_in_both_directions() -> None:
    interactions = (InteractionLink("x1", "y", "bleeding", "major"),)
    kb = to_knowledge_base(build_ontology(toy_ontology_data(interactions=interactions)))
    told = {str(fact) for fact in kb.facts if fact.op == "Int"}
    assert told == {"Int(D_x1, D_y)", "Int(D_y, D_x1)"}
