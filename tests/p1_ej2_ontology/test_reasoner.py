"""The DL queries on the closure and the derived formulary, against results design §3."""

from __future__ import annotations

import pytest

from symbolic_ai.dataloader import load_formulary
from symbolic_ai.p1_ej2_ontology.errors import UnknownIdentifierError
from symbolic_ai.p1_ej2_ontology.ontology import Ontology
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
        "candidate_PAIN",
        "clinical_objects",
        "contraindicated_CKD",
        "drugs",
        "nsaids",
    )


def test_ibuprofen_is_not_contraindicated_by_age65(reasoned: ReasonedOntology) -> None:
    # AGE65 triggers a coprescription for NSAIDs, not a contraindication.
    assert "contraindicated_AGE65" not in classify(reasoned, "ibuprofen")


def test_tramadol_inherits_no_epilepsy_contraindication_but_has_its_own(
    reasoned: ReasonedOntology,
) -> None:
    assert "contraindicated_EPI" in classify(reasoned, "tramadol")
    sources = contraindication_sources(reasoned, "EPI", "tramadol")
    assert [(s.subject_id, s.level) for s in sources] == [("tramadol", LinkLevel.OBJECT)]


def test_classify_unknown_drug_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        classify(reasoned, "aspirin")


# --- consistency ----------------------------------------------------------------------------


def test_the_ontology_is_consistent(reasoned: ReasonedOntology) -> None:
    assert clashes(reasoned.closure) == ()


def test_partitions_are_exhaustive_on_the_told_objects(reasoned: ReasonedOntology) -> None:
    coverage = {p.set_id: p for p in partition_coverage(reasoned)}
    assert sorted(coverage) == ["anticoag", "beta", "ccb", "raas", "upper"]
    assert all(p.uncovered == () for p in coverage.values())
    assert len(coverage["upper"].members) == 30  # 18 drugs, 6 conditions, 6 risk factors


@pytest.mark.parametrize(
    ("drug_id", "category_id", "expected"),
    [
        ("ibuprofen", "opioids", (Clash("ibuprofen", "nsaids", "opioids"),)),
        ("ibuprofen", "antidepressants", ()),
        ("ibuprofen", "serotonergic_opioids", (Clash("ibuprofen", "nsaids", "opioids"),)),
        ("warfarin", "conditions", (Clash("warfarin", "conditions", "drugs"),)),
    ],
    ids=["disjoint_sibling", "other_group", "through_inheritance", "upper_partition"],
)
def test_mutation_clashes(
    reasoned: ReasonedOntology, drug_id: str, category_id: str, expected: tuple[Clash, ...]
) -> None:
    assert clashes(add_membership(reasoned, drug_id, category_id)) == expected


def test_mutation_with_unknown_category_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        add_membership(reasoned, "ibuprofen", "ghosts")


# --- subsumption (P4.1) ---------------------------------------------------------------------


def test_taxonomy_counts(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    primitive = set(reasoned.ontology.drug_categories)
    defined = set(reasoned.ontology.defined_ids)
    pairs = result.pairs()
    assert len(result.told) == 28
    assert sum(c in primitive and d in primitive for c, d in pairs) == 57
    assert sum(c in primitive and d in defined for c, d in pairs) == 39
    assert [(c, d) for c, d in pairs if c in defined] == [(c, "drugs") for c in sorted(defined)]
    assert len(pairs) == 108


def test_told_links_are_among_the_inferred_pairs(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    assert result.told <= set(result.pairs())


def test_pregnancy_contrast_hbpm(reasoned: ReasonedOntology) -> None:
    for category_id in ("vitamin_k_antagonists", "direct_oral_anticoagulants", "raas_blockers"):
        assert subsumes(reasoned, category_id, "contraindicated_PREG")
    assert not subsumes(reasoned, "anticoagulants", "contraindicated_PREG")
    assert not subsumes(reasoned, "low_molecular_weight_heparins", "contraindicated_PREG")
    assert subsumes(reasoned, "low_molecular_weight_heparins", "candidate_AF")


def test_no_category_is_subsumed_by_ci_epi(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    assert [c for c, d in result.pairs() if d == "contraindicated_EPI"] == []
    assert not subsumes(reasoned, "serotonergic_opioids", "contraindicated_EPI")


def test_subsumes_unknown_category_raises(reasoned: ReasonedOntology) -> None:
    with pytest.raises(UnknownIdentifierError):
        subsumes(reasoned, "nsaids", "ghosts")


# --- inheritance by a new member (P4.2) -----------------------------------------------------


def test_new_nsaid_inherits_nine_rows(reasoned: ReasonedOntology) -> None:
    inherited = inherited_by_new_member(reasoned, "nsaids")
    assert inherited.candidates == ("PAIN",)
    assert inherited.contraindications == ("CKD",)
    assert inherited.interactions == ("apixaban", "sertraline", "warfarin")
    assert inherited.coprescriptions == (("AGE65", "omeprazole"),)
    assert inherited.family_pairs == (
        ("analgesics", "ibuprofen"),
        ("analgesics", "paracetamol"),
        ("analgesics", "tramadol"),
    )
    assert inherited.rows == 9
    assert inherited.clashes == ()


def test_new_hbpm_inherits_five_rows(reasoned: ReasonedOntology) -> None:
    inherited = inherited_by_new_member(reasoned, "low_molecular_weight_heparins")
    assert inherited.candidates == ("AF",)
    assert inherited.contraindications == ()
    assert inherited.interactions == ("ibuprofen", "sertraline")
    assert inherited.family_pairs == (
        ("anticoagulants", "apixaban"),
        ("anticoagulants", "warfarin"),
    )
    assert inherited.rows == 5


def test_every_named_category_is_consistent(reasoned: ReasonedOntology) -> None:
    for category_id in reasoned.ontology.named_categories:
        assert inherited_by_new_member(reasoned, category_id).clashes == ()


# --- derived formulary (P3) -----------------------------------------------------------------


def test_derived_formulary_reproduces_1_0_0_plus_one_pair(reasoned: ReasonedOntology) -> None:
    derived = derive_formulary(reasoned)
    reference = load_formulary(version="1.0.0")
    assert derived.version == "ontology@1.1.0"
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
    assert derived_pairs - reference_pairs == {("apixaban", "sertraline")}


def test_extra_pair_comes_from_the_same_link_as_sertraline_warfarin(
    reasoned: ReasonedOntology,
) -> None:
    extra = interaction_sources(reasoned, "apixaban", "sertraline")
    listed = interaction_sources(reasoned, "sertraline", "warfarin")
    assert [s.subject_id for s in extra] == ["anticoagulants+ssris"]
    assert extra == listed


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
