"""The DL queries on the closure and the derived formulary, against results design §11.3."""

from __future__ import annotations

import pytest

from symbolic_ai.dataloader import load_formulary
from symbolic_ai.p1_ej2_ontology.errors import UnknownIdentifierError
from symbolic_ai.p1_ej2_ontology.ontology import (
    MEMBER,
    Ontology,
    atom,
    category_symbol,
    drug_symbol,
)
from symbolic_ai.p1_ej2_ontology.reasoner import (
    Clash,
    LinkLevel,
    ReasonedOntology,
    add_membership,
    candidate_sources,
    clashes,
    classify,
    contraindication_sources,
    derive_formulary,
    drug_members,
    family_pairs,
    inherited_by_new_member,
    interaction_sources,
    partition_coverage,
    reason,
    subsumes,
    taxonomy,
)


@pytest.fixture(scope="module")
def reasoned(real_ontology: Ontology) -> ReasonedOntology:
    """The real ontology with its closure."""
    return reason(real_ontology)


pytestmark = pytest.mark.integration


# --- classification -------------------------------------------------------------------------


def test_classification_of_ibuprofen(reasoned: ReasonedOntology) -> None:
    assert classify(reasoned, "ibuprofen") == (
        "analgesics",
        "bleeding_risk_drugs",
        "candidate_PAIN",
        "contraindicated_CKD",
        "drugs",
        "nsaids",
    )


def test_ibuprofen_is_not_contraindicated_by_age65(reasoned: ReasonedOntology) -> None:
    # AGE65 triggers a coprescription for NSAIDs, not a contraindication.
    assert "contraindicated_AGE65" not in classify(reasoned, "ibuprofen")


def test_tramadol_is_classified_into_serotonergic_opioids_from_its_two_told_memberships(
    reasoned: ReasonedOntology,
) -> None:
    assert reasoned.ontology.told_categories_of("tramadol") == ("opioids", "serotonergic_drugs")
    assert classify(reasoned, "tramadol") == (
        "analgesics",
        "candidate_PAIN",
        "contraindicated_EPI",
        "drugs",
        "opioids",
        "serotonergic_drugs",
        "serotonergic_opioids",
    )
    classified = atom(MEMBER, drug_symbol("tramadol"), category_symbol("serotonergic_opioids"))
    assert classified in reasoned.closure.derived and classified not in reasoned.closure.told


def test_tramadol_inherits_its_epilepsy_contraindication_from_the_defined_category(
    reasoned: ReasonedOntology,
) -> None:
    sources = contraindication_sources(reasoned, "EPI", "tramadol")
    assert [(s.subject_id, s.level) for s in sources] == [
        ("serotonergic_opioids", LinkLevel.CATEGORY)
    ]


def test_only_tramadol_is_a_serotonergic_opioid(reasoned: ReasonedOntology) -> None:
    # sertraline is serotonergic but not an opioid: one conjunct is not enough.
    assert drug_members(reasoned.closure, "serotonergic_opioids") == ("tramadol",)
    assert drug_members(reasoned.closure, "serotonergic_drugs") == ("sertraline", "tramadol")


def test_classify_unknown_drug_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        classify(reasoned, "aspirin")


# --- consistency ----------------------------------------------------------------------------


def test_the_ontology_is_consistent(reasoned: ReasonedOntology) -> None:
    assert clashes(reasoned.closure) == ()


def test_partitions_are_exhaustive_on_the_told_objects(reasoned: ReasonedOntology) -> None:
    coverage = {p.set_id: p for p in partition_coverage(reasoned)}
    assert sorted(coverage) == ["anticoag", "beta", "ccb", "raas"]
    assert all(p.uncovered == () for p in coverage.values())
    assert all(len(p.members) == 2 for p in coverage.values())


@pytest.mark.parametrize(
    ("drug_id", "category_id", "expected"),
    [
        ("ibuprofen", "opioids", (Clash("ibuprofen", "nsaids", "opioids"),)),
        ("ibuprofen", "antidepressants", ()),
        ("ibuprofen", "serotonergic_opioids", (Clash("ibuprofen", "nsaids", "opioids"),)),
        ("tramadol", "nsaids", (Clash("tramadol", "nsaids", "opioids"),)),
        ("sertraline", "opioids", ()),
    ],
    ids=[
        "disjoint_sibling",
        "other_group",
        "through_the_definition",
        "second_told_membership",
        "classified_without_clash",
    ],
)
def test_mutation_clashes(
    reasoned: ReasonedOntology, drug_id: str, category_id: str, expected: tuple[Clash, ...]
) -> None:
    assert clashes(add_membership(reasoned, drug_id, category_id)) == expected


def test_a_mutation_can_classify_a_drug_into_the_defined_category(
    reasoned: ReasonedOntology,
) -> None:
    # sertraline ∈ opioids, added to sertraline ∈ ssris ⊂ serotonergic_drugs, meets the definition.
    closure = add_membership(reasoned, "sertraline", "opioids")
    assert atom(MEMBER, drug_symbol("sertraline"), category_symbol("serotonergic_opioids")) in (
        closure.facts
    )


def test_mutation_with_unknown_category_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        add_membership(reasoned, "ibuprofen", "ghosts")


# --- subsumption (P4.1) ---------------------------------------------------------------------


def test_taxonomy_counts(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    primitive = set(reasoned.ontology.primitive_categories)
    defined = set(reasoned.ontology.defined_ids) | {"serotonergic_opioids"}
    pairs = result.pairs()
    assert (len(primitive), len(defined)) == (31, 13)
    assert len(result.told) == 36
    assert sum(c in primitive and d in primitive for c, d in pairs) == 68
    assert sum(c in primitive and d in defined for c, d in pairs) == 38
    assert sum(c in defined and d in primitive for c, d in pairs) == 16
    assert [(c, d) for c, d in pairs if c in defined and d in defined] == [
        ("serotonergic_opioids", "candidate_PAIN"),
        ("serotonergic_opioids", "contraindicated_EPI"),
    ]
    assert len(pairs) == 124


def test_told_links_are_among_the_inferred_pairs(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    assert result.told <= set(result.pairs())


def test_multiple_inheritance_gives_ssris_four_proper_ancestors(
    reasoned: ReasonedOntology,
) -> None:
    result = taxonomy(reasoned)
    primitive = set(reasoned.ontology.primitive_categories)
    assert [c for c in result.subsumers["ssris"] if c in primitive] == [
        "antidepressants",
        "bleeding_risk_drugs",
        "drugs",
        "serotonergic_drugs",
    ]


def test_pregnancy_contrast_hbpm(reasoned: ReasonedOntology) -> None:
    for category_id in ("vitamin_k_antagonists", "direct_oral_anticoagulants", "raas_blockers"):
        assert subsumes(reasoned, category_id, "contraindicated_PREG")
    assert not subsumes(reasoned, "anticoagulants", "contraindicated_PREG")
    assert not subsumes(reasoned, "low_molecular_weight_heparins", "contraindicated_PREG")
    assert subsumes(reasoned, "low_molecular_weight_heparins", "candidate_AF")


def test_serotonergic_opioids_is_subsumed_by_ci_epi_by_definition(
    reasoned: ReasonedOntology,
) -> None:
    result = taxonomy(reasoned)
    assert [c for c, d in result.pairs() if d == "contraindicated_EPI"] == ["serotonergic_opioids"]
    assert result.subsumers["serotonergic_opioids"] == (
        "analgesics",
        "candidate_PAIN",
        "contraindicated_EPI",
        "drugs",
        "opioids",
        "serotonergic_drugs",
    )
    # Neither conjunct alone is subsumed by the definition or by the contraindication.
    for conjunct in ("opioids", "serotonergic_drugs"):
        assert not subsumes(reasoned, conjunct, "serotonergic_opioids")
        assert not subsumes(reasoned, conjunct, "contraindicated_EPI")


def test_subsumes_unknown_category_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        subsumes(reasoned, "nsaids", "ghosts")


# --- inheritance by a new member (P4.2) -----------------------------------------------------


def test_new_nsaid_inherits_ten_rows(reasoned: ReasonedOntology) -> None:
    inherited = inherited_by_new_member(reasoned, "nsaids")
    assert inherited.candidates == ("PAIN",)
    assert inherited.contraindications == ("CKD",)
    assert inherited.interactions == ("apixaban", "ibuprofen", "sertraline", "warfarin")
    assert inherited.coprescriptions == (("AGE65", "omeprazole"),)
    assert inherited.family_pairs == (
        ("analgesics", "ibuprofen"),
        ("analgesics", "paracetamol"),
        ("analgesics", "tramadol"),
    )
    assert inherited.rows == 10
    assert inherited.clashes == ()


def test_self_link_makes_a_new_nsaid_interact_with_ibuprofen(reasoned: ReasonedOntology) -> None:
    # Same-category pair: both are members of bleeding_risk_drugs, whose self-link relates any
    # two different members.
    inherited = inherited_by_new_member(reasoned, "nsaids")
    assert "ibuprofen" in inherited.interactions
    assert "bleeding_risk_drugs" in inherited.categories
    sources = interaction_sources(reasoned, "ibuprofen", "warfarin")
    assert [s.subject_id for s in sources] == ["bleeding_risk_drugs+bleeding_risk_drugs"]


def test_self_link_never_pairs_a_drug_with_itself(reasoned: ReasonedOntology) -> None:
    derived = derive_formulary(reasoned)
    assert all(i.drug_a < i.drug_b for i in derived.interactions)
    pairs = [str(f) for f in reasoned.closure.with_predicate("Int") if f.args[0] == f.args[1]]
    assert pairs == []


@pytest.mark.parametrize(
    ("category_id", "rows"),
    [
        ("ssris", 8),
        ("low_molecular_weight_heparins", 7),
        ("serotonergic_opioids", 7),
        ("bleeding_risk_drugs", 4),
        ("bradycardic_drugs", 3),
        ("serotonergic_drugs", 2),
        ("drugs", 0),
    ],
)
def test_rows_a_new_member_inherits(
    reasoned: ReasonedOntology, category_id: str, rows: int
) -> None:
    assert inherited_by_new_member(reasoned, category_id).rows == rows


def test_new_hbpm_inherits_seven_rows(reasoned: ReasonedOntology) -> None:
    inherited = inherited_by_new_member(reasoned, "low_molecular_weight_heparins")
    assert inherited.candidates == ("AF",)
    assert inherited.contraindications == ()
    assert inherited.interactions == ("apixaban", "ibuprofen", "sertraline", "warfarin")
    assert inherited.family_pairs == (
        ("anticoagulants", "apixaban"),
        ("anticoagulants", "warfarin"),
    )
    assert inherited.rows == 7


def test_every_named_category_is_consistent(reasoned: ReasonedOntology) -> None:
    for category_id in reasoned.ontology.named_categories:
        assert inherited_by_new_member(reasoned, category_id).clashes == ()


# --- derived formulary (P3) -----------------------------------------------------------------


def test_derived_formulary_reproduces_1_0_0_plus_three_pairs(
    reasoned: ReasonedOntology,
) -> None:
    derived = derive_formulary(reasoned)
    reference = load_formulary(version="1.0.0")
    assert derived.version == "ontology@1.2.0"
    assert derived.candidates == reference.candidates
    contraindications = {(c.risk_factor_id, c.drug_id) for c in derived.contraindications}
    assert contraindications == {(c.risk_factor_id, c.drug_id) for c in reference.contraindications}
    assert {
        (c.drug_id, c.risk_factor_id, c.companion_drug_id) for c in derived.coprescriptions
    } == {(c.drug_id, c.risk_factor_id, c.companion_drug_id) for c in reference.coprescriptions}
    assert family_pairs(derived) == family_pairs(reference)
    assert len(family_pairs(derived)) == 7
    derived_pairs = {(i.drug_a, i.drug_b) for i in derived.interactions}
    reference_pairs = {(i.drug_a, i.drug_b) for i in reference.interactions}
    assert reference_pairs <= derived_pairs
    assert derived_pairs - reference_pairs == {
        ("apixaban", "sertraline"),
        ("apixaban", "warfarin"),
        ("bisoprolol", "propranolol"),
    }
    # Two of the three extra pairs are already A6 family pairs of 1.0.0.
    assert (derived_pairs - reference_pairs) & set(family_pairs(reference)) == {
        ("apixaban", "warfarin"),
        ("bisoprolol", "propranolol"),
    }


def test_extra_pair_comes_from_the_same_link_as_sertraline_warfarin(
    reasoned: ReasonedOntology,
) -> None:
    extra = interaction_sources(reasoned, "apixaban", "sertraline")
    listed = interaction_sources(reasoned, "sertraline", "warfarin")
    assert [s.subject_id for s in extra] == ["bleeding_risk_drugs+bleeding_risk_drugs"]
    assert extra == listed


def test_sertraline_tramadol_comes_from_the_serotonergic_self_link(
    reasoned: ReasonedOntology,
) -> None:
    sources = interaction_sources(reasoned, "sertraline", "tramadol")
    assert [(s.subject_id, s.text) for s in sources] == [
        ("serotonergic_drugs+serotonergic_drugs", "serotonin syndrome|major")
    ]


def test_derived_classes_keep_the_ej1_class_ids(reasoned: ReasonedOntology) -> None:
    derived = derive_formulary(reasoned)
    reference = load_formulary(version="1.0.0")
    assert derived.class_members == reference.class_members
    assert all(c.exclusive for c in derived.drug_classes)


def test_candidate_sources_are_category_level(reasoned: ReasonedOntology) -> None:
    sources = candidate_sources(reasoned, "HTN", "verapamil")
    assert [(s.subject_id, s.level) for s in sources] == [("antihypertensives", LinkLevel.CATEGORY)]


def test_derived_formulary_is_deterministic(real_ontology: Ontology) -> None:
    assert derive_formulary(reason(real_ontology)) == derive_formulary(reason(real_ontology))
