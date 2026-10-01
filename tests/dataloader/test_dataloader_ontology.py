"""``load_ontology`` on the real data 1.1.0 and one broken fixture per ontology validation rule.

The content checked here is the frozen table of the EJ2 results design §6.1, as amended by §11.1
(R1 multiple inheritance, R2 a category defined by its conjuncts, R3a no upper ontology).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import (
    CategoryDefinition,
    DatabaseValidationError,
    load_formulary,
    load_ontology,
    validate_database,
)

pytestmark = pytest.mark.integration


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture text not found in {path}: {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _append(path: Path, line: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")


def _problems(data_dir: Path) -> tuple[str, ...]:
    with pytest.raises(DatabaseValidationError) as excinfo:
        validate_database(data_dir=data_dir)
    return excinfo.value.problems


# --- the real data --------------------------------------------------------------------------


def test_ontology_table_counts(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    assert ontology.version == "1.1.0"
    assert len(ontology.categories) == 33
    assert len(ontology.subcategories) == 36
    assert len(ontology.definitions) == 1
    assert len(ontology.memberships) == 19
    assert len(ontology.indications) == 6
    assert len(ontology.contraindications) == 10
    assert len(ontology.interactions) == 3
    assert len(ontology.coprescriptions) == 1
    assert len(ontology.families) == 5
    assert len(ontology.disjoint_sets) == 7
    assert sum(s.partition_of is not None for s in ontology.disjoint_sets) == 4


def test_the_upper_ontology_is_gone(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    category_ids = {c.category_id for c in ontology.categories}
    assert category_ids.isdisjoint({"clinical_objects", "conditions", "risk_factors"})
    assert "upper" not in {s.set_id for s in ontology.disjoint_sets}


def test_ontology_carries_the_formulary_identifiers(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    formulary = load_formulary(data_dir=real_data_dir, version="1.0.0")
    assert ontology.drugs == formulary.drugs
    assert ontology.conditions == formulary.conditions
    assert ontology.risk_factors == formulary.risk_factors


def test_every_drug_has_a_told_membership_and_tramadol_has_two(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    assert {m.object_id for m in ontology.memberships} == {d.drug_id for d in ontology.drugs}
    tramadol = [m.category_id for m in ontology.memberships if m.object_id == "tramadol"]
    assert tramadol == ["opioids", "serotonergic_drugs"]


def test_ssris_have_three_told_parents(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    parents = [e.parent_id for e in ontology.subcategories if e.category_id == "ssris"]
    assert parents == ["antidepressants", "bleeding_risk_drugs", "serotonergic_drugs"]


def test_serotonergic_opioids_is_defined_by_two_conjuncts(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    assert ontology.definitions == (
        CategoryDefinition("serotonergic_opioids", ("opioids", "serotonergic_drugs")),
    )
    assert "serotonergic_opioids" not in {e.category_id for e in ontology.subcategories}


def test_every_contraindication_link_is_on_a_category(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    category_ids = {c.category_id for c in ontology.categories}
    assert all(c.subject_id in category_ids for c in ontology.contraindications)
    epilepsy = [c.subject_id for c in ontology.contraindications if c.risk_factor_id == "EPI"]
    assert epilepsy == ["serotonergic_opioids"]


def test_interactions_are_three_self_links(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    assert [(i.subject_a, i.subject_b) for i in ontology.interactions] == [
        ("bleeding_risk_drugs", "bleeding_risk_drugs"),
        ("bradycardic_drugs", "bradycardic_drugs"),
        ("serotonergic_drugs", "serotonergic_drugs"),
    ]


def test_disjoint_sets_are_sorted_with_their_partitions(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    by_id = {s.set_id: s for s in ontology.disjoint_sets}
    assert by_id["anticoag"].category_ids == (
        "direct_oral_anticoagulants",
        "low_molecular_weight_heparins",
        "vitamin_k_antagonists",
    )
    assert by_id["anticoag"].partition_of == "anticoagulants"
    assert by_id["analg"].partition_of is None


def test_no_ontology_exists_at_version_1_0_0(real_data_dir: Path) -> None:
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_ontology(data_dir=real_data_dir, version="1.0.0")
    assert len(excinfo.value.problems) == 18  # every drug lacks a told membership


def test_formulary_1_0_0_is_unchanged_by_data_1_1_0(real_data_dir: Path) -> None:
    formulary = load_formulary(data_dir=real_data_dir, version="1.0.0")
    assert formulary.version == "1.0.0"
    assert sum(len(drugs) for drugs in formulary.candidates.values()) == 18
    assert len(formulary.interactions) == 7
    assert len(formulary.contraindications) == 11
    assert len(formulary.coprescriptions) == 1
    assert len(formulary.drug_classes) == 5


# --- one broken fixture per validation rule --------------------------------------------------


def test_unknown_link_subject_is_reported(sandbox_data_dir: Path) -> None:
    _replace(sandbox_data_dir / "ontology" / "indications.csv", "analgesics,PAIN", "pills,PAIN")
    problems = _problems(sandbox_data_dir)
    assert any("'pills' is neither a category nor a drug" in p for p in problems)


def test_unknown_membership_category_is_reported(sandbox_data_dir: Path) -> None:
    _replace(
        sandbox_data_dir / "ontology" / "memberships.csv", "ibuprofen,nsaids", "ibuprofen,pills"
    )
    problems = _problems(sandbox_data_dir)
    assert any("'pills' does not match any ontology_categories" in p for p in problems)


def test_overlapping_subject_namespaces_are_reported(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "categories.csv", "warfarin,Warfarina,Warfarin,,1.1.0")
    problems = _problems(sandbox_data_dir)
    assert any("category_id='warfarin'" in p and "also a drug_id" in p for p in problems)


def test_cyclic_subcategory_graph_is_reported(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "subcategories.csv", "drugs,nsaids,1.1.0")
    problems = _problems(sandbox_data_dir)
    assert any("definition edges has a cycle" in p for p in problems)


def test_cycle_through_a_definition_edge_is_reported(sandbox_data_dir: Path) -> None:
    # serotonergic_drugs ⊂ serotonergic_opioids, while the definition puts serotonergic_opioids
    # below serotonergic_drugs: acyclic as a subcategory graph, cyclic with the definition edges.
    _append(
        sandbox_data_dir / "ontology" / "subcategories.csv",
        "serotonergic_drugs,serotonergic_opioids,1.1.0",
    )
    problems = _problems(sandbox_data_dir)
    cycles = [p for p in problems if "definition edges has a cycle" in p]
    assert len(cycles) == 1
    assert "serotonergic_drugs" in cycles[0] and "serotonergic_opioids" in cycles[0]


def test_unordered_interaction_subjects_are_reported(sandbox_data_dir: Path) -> None:
    _append(
        sandbox_data_dir / "ontology" / "interactions.csv",
        "nsaids,anticoagulants,bleeding,major,1.1.0",
    )
    problems = _problems(sandbox_data_dir)
    assert any("subject_a='nsaids'" in p and "lexicographically after" in p for p in problems)


def test_self_link_interactions_are_valid(sandbox_data_dir: Path) -> None:
    _append(
        sandbox_data_dir / "ontology" / "interactions.csv", "nsaids,nsaids,bleeding,major,1.1.0"
    )
    validate_database(data_dir=sandbox_data_dir)


def test_second_told_membership_is_valid(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "memberships.csv", "sertraline,opioids,1.1.0")
    validate_database(data_dir=sandbox_data_dir)
    told = [m for m in load_ontology(data_dir=sandbox_data_dir).memberships]
    assert [m.category_id for m in told if m.object_id == "sertraline"] == ["opioids", "ssris"]


def test_missing_membership_fails_load_ontology_but_not_the_database(
    sandbox_data_dir: Path,
) -> None:
    _replace(sandbox_data_dir / "ontology" / "memberships.csv", "ibuprofen,nsaids,1.1.0\n", "")
    validate_database(data_dir=sandbox_data_dir)  # EJ1 keeps working
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_ontology(data_dir=sandbox_data_dir)
    assert excinfo.value.problems == (
        "ontology_memberships: drug 'ibuprofen' has no told membership, expected at least one",
    )


def test_unknown_conjunct_is_reported(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "definitions.csv", "serotonergic_opioids,ghosts,1.1.0")
    problems = _problems(sandbox_data_dir)
    assert any("'ghosts' does not match any ontology_categories" in p for p in problems)


def test_definition_with_one_conjunct_is_reported(sandbox_data_dir: Path) -> None:
    _replace(
        sandbox_data_dir / "ontology" / "definitions.csv",
        "serotonergic_opioids,serotonergic_drugs,1.1.0\n",
        "",
    )
    problems = _problems(sandbox_data_dir)
    assert problems == (
        "ontology_definitions[category_id='serotonergic_opioids']: fewer than two conjuncts",
    )


def test_defined_category_with_a_told_member_is_reported(sandbox_data_dir: Path) -> None:
    _append(
        sandbox_data_dir / "ontology" / "memberships.csv", "tramadol,serotonergic_opioids,1.1.0"
    )
    problems = _problems(sandbox_data_dir)
    assert any(
        "category_id='serotonergic_opioids'" in p and "no told member" in p for p in problems
    )


def test_defined_category_with_a_told_parent_is_reported(sandbox_data_dir: Path) -> None:
    _append(
        sandbox_data_dir / "ontology" / "subcategories.csv", "serotonergic_opioids,opioids,1.1.0"
    )
    problems = _problems(sandbox_data_dir)
    assert any(
        "category_id='serotonergic_opioids'" in p and "no told parent" in p for p in problems
    )


def test_disjoint_set_with_one_member_is_reported(sandbox_data_dir: Path) -> None:
    _replace(
        sandbox_data_dir / "ontology" / "disjoint_sets.csv",
        "antidep,other_antidepressants,,1.1.0\n",
        "",
    )
    problems = _problems(sandbox_data_dir)
    assert any("set_id='antidep'" in p and "fewer than two members" in p for p in problems)


def test_inconsistent_partition_of_is_reported(sandbox_data_dir: Path) -> None:
    _replace(
        sandbox_data_dir / "ontology" / "disjoint_sets.csv",
        "raas,arbs,raas_blockers",
        "raas,arbs,antihypertensives",
    )
    problems = _problems(sandbox_data_dir)
    assert any("set_id='raas'" in p and "inconsistent partition_of" in p for p in problems)


def test_all_ontology_problems_are_reported_at_once(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "subcategories.csv", "drugs,nsaids,1.1.0")
    _append(
        sandbox_data_dir / "ontology" / "memberships.csv", "tramadol,serotonergic_opioids,1.1.0"
    )
    problems = _problems(sandbox_data_dir)
    assert any("cycle" in p for p in problems)
    assert any("no told member" in p for p in problems)
