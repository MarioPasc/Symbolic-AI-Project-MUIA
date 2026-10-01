"""``load_ontology`` on the real data 1.1.0 and one broken fixture per ontology validation rule.

The content checked here is the frozen table of the EJ2 results design §6.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from symbolic_ai.dataloader import (
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
    assert len(ontology.subcategories) == 31
    assert len(ontology.memberships) == 18
    assert len(ontology.indications) == 6
    assert len(ontology.contraindications) == 10
    assert len(ontology.interactions) == 5
    assert len(ontology.coprescriptions) == 1
    assert len(ontology.families) == 5
    assert len(ontology.disjoint_sets) == 8
    assert sum(s.partition_of is not None for s in ontology.disjoint_sets) == 5


def test_ontology_carries_the_formulary_identifiers(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    formulary = load_formulary(data_dir=real_data_dir, version="1.0.0")
    assert ontology.drugs == formulary.drugs
    assert ontology.conditions == formulary.conditions
    assert ontology.risk_factors == formulary.risk_factors


def test_every_drug_has_exactly_one_told_membership(real_data_dir: Path) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    members = sorted(m.object_id for m in ontology.memberships)
    assert members == sorted(d.drug_id for d in ontology.drugs)


def test_contraindication_links_split_into_nine_categories_and_one_drug(
    real_data_dir: Path,
) -> None:
    ontology = load_ontology(data_dir=real_data_dir)
    category_ids = {c.category_id for c in ontology.categories}
    on_categories = [c for c in ontology.contraindications if c.subject_id in category_ids]
    assert len(on_categories) == 9
    assert [c.subject_id for c in ontology.contraindications if c not in on_categories] == [
        "tramadol"
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
    assert len(excinfo.value.problems) == 18  # every drug lacks its told membership


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
    assert any("subcategory graph has a cycle" in p for p in problems)


def test_unordered_interaction_subjects_are_reported(sandbox_data_dir: Path) -> None:
    _replace(
        sandbox_data_dir / "ontology" / "interactions.csv",
        "anticoagulants,nsaids,",
        "nsaids,anticoagulants,",
    )
    problems = _problems(sandbox_data_dir)
    assert any("subject_a='nsaids'" in p and "lexicographically" in p for p in problems)


def test_second_told_membership_is_reported(sandbox_data_dir: Path) -> None:
    _append(sandbox_data_dir / "ontology" / "memberships.csv", "ibuprofen,opioids,1.1.0")
    problems = _problems(sandbox_data_dir)
    assert any("object_id='ibuprofen'" in p and "2 told memberships" in p for p in problems)


def test_missing_membership_fails_load_ontology_but_not_the_database(
    sandbox_data_dir: Path,
) -> None:
    _replace(sandbox_data_dir / "ontology" / "memberships.csv", "ibuprofen,nsaids,1.1.0\n", "")
    validate_database(data_dir=sandbox_data_dir)  # EJ1 keeps working
    with pytest.raises(DatabaseValidationError) as excinfo:
        load_ontology(data_dir=sandbox_data_dir)
    assert excinfo.value.problems == (
        "ontology_memberships: drug 'ibuprofen' has no told membership, expected exactly one",
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
    _append(sandbox_data_dir / "ontology" / "memberships.csv", "ibuprofen,opioids,1.1.0")
    problems = _problems(sandbox_data_dir)
    assert any("cycle" in p for p in problems)
    assert any("2 told memberships" in p for p in problems)
