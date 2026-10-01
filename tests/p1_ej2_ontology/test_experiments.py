"""Experiments P3 and P4 on the real data, against the PREDICTED values of results design §11.3."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import pytest

from symbolic_ai.dataloader import load_formulary
from symbolic_ai.p1_ej2_ontology.experiments import (
    ConsistencyResult,
    as_jsonable,
    expected_clash_sets,
    mutation_candidates,
    run_consistency,
    run_fixed_point_cost,
    run_formulary_derivation,
    run_inheritance,
    run_taxonomy,
    told_knowledge,
)
from symbolic_ai.p1_ej2_ontology.ontology import Ontology
from symbolic_ai.p1_ej2_ontology.reasoner import ReasonedOntology, reason

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def reasoned(real_ontology: Ontology) -> ReasonedOntology:
    """The real ontology with its closure."""
    return reason(real_ontology)


@pytest.fixture(scope="module")
def consistency(reasoned: ReasonedOntology) -> ConsistencyResult:
    """P4.3 without the oracle (forward chaining against the networkx labels)."""
    return run_consistency(reasoned, oracle_run=None)


# --- P3 ---------------------------------------------------------------------------------------


def test_p3_table_matches_the_prediction(reasoned: ReasonedOntology) -> None:
    result = run_formulary_derivation(reasoned, load_formulary(version="1.0.0"), oracle_run=None)
    table = {
        t.table: (
            t.rows_reference,
            t.statements,
            t.statements_on_categories,
            t.derived,
            t.in_both,
            t.only_derived,
            t.only_reference,
        )
        for t in result.tables
    }
    only_derived = (
        ("apixaban", "sertraline"),
        ("apixaban", "warfarin"),
        ("bisoprolol", "propranolol"),
    )
    assert table == {
        "candidates": (18, 6, 6, 18, 18, (), ()),
        "contraindications": (11, 10, 10, 11, 11, (), ()),
        "interactions": (7, 3, 3, 10, 7, only_derived, ()),
        "coprescriptions": (1, 1, 1, 1, 1, (), ()),
        "families": (7, 5, 5, 7, 7, (), ()),
    }
    families = next(t for t in result.tables if t.table == "families")
    assert (families.classes_reference, families.classes_derived) == (5, 5)
    assert result.oracle_memberships is None and result.oracle_rows is None


def test_p3_rows_record_their_provenance(reasoned: ReasonedOntology) -> None:
    result = run_formulary_derivation(reasoned, load_formulary(version="1.0.0"), oracle_run=None)
    rows = {(r.table, r.key): r for r in result.rows}
    extra = rows[("interactions", ("apixaban", "sertraline"))]
    assert (extra.in_reference, extra.sources, extra.level) == (
        False,
        ("bleeding_risk_drugs+bleeding_risk_drugs",),
        "category",
    )
    tramadol = rows[("contraindications", ("EPI", "tramadol"))]
    assert (tramadol.sources, tramadol.level) == (("serotonergic_opioids",), "category")
    assert len(result.rows) == 18 + 11 + 10 + 1 + 7
    assert all(r.oracle_agrees is None for r in result.rows)


def test_p3_records_the_drugs_classified_into_the_defined_category(
    reasoned: ReasonedOntology,
) -> None:
    result = run_formulary_derivation(reasoned, load_formulary(version="1.0.0"), oracle_run=None)
    assert result.defined_category_members == {"serotonergic_opioids": ("tramadol",)}


def test_p3_flags_the_extra_pairs_that_are_also_family_pairs(reasoned: ReasonedOntology) -> None:
    result = run_formulary_derivation(reasoned, load_formulary(version="1.0.0"), oracle_run=None)
    interactions = [r for r in result.rows if r.table == "interactions"]
    flagged = [r.key for r in interactions if r.also_family_pair]
    assert flagged == [("apixaban", "warfarin"), ("bisoprolol", "propranolol")]
    assert all(r.also_family_pair is False for r in interactions if r.key not in flagged)
    assert all(r.also_family_pair is None for r in result.rows if r.table != "interactions")


# --- P4.1, P4.2 -------------------------------------------------------------------------------


def test_p4_taxonomy_counts(reasoned: ReasonedOntology) -> None:
    result = run_taxonomy(reasoned, oracle_run=None, oracle_prototypes=None)
    counts = (
        result.n_primitive,
        result.n_defined,
        result.defined_by_conjuncts,
        result.told_pairs,
        result.primitive_pairs,
        result.primitive_to_defined,
        result.defined_to_primitive,
        result.defined_to_defined,
        result.total_pairs,
    )
    assert counts == (31, 13, ("serotonergic_opioids",), 36, 68, 38, 16, 2, 124)
    assert result.subsumed_by_defined["contraindicated_PREG"] == (
        "ace_inhibitors",
        "arbs",
        "direct_oral_anticoagulants",
        "raas_blockers",
        "vitamin_k_antagonists",
    )
    assert result.subsumed_by_defined["contraindicated_EPI"] == ("serotonergic_opioids",)
    assert result.subsumed_by_defined["contraindicated_AGE65"] == ()
    assert result.subsumed_by_defined["serotonergic_opioids"] == ()


def test_p4_primitive_to_defined_pairs_per_defined_category(reasoned: ReasonedOntology) -> None:
    result = run_taxonomy(reasoned, oracle_run=None, oracle_prototypes=None)
    primitive = set(reasoned.ontology.primitive_categories)
    per_category = {
        d: sum(c in primitive for c in subs) for d, subs in result.subsumed_by_defined.items()
    }
    assert per_category == {
        "candidate_AF": 4,
        "candidate_DEP": 3,
        "candidate_GERD": 1,
        "candidate_HTN": 11,
        "candidate_PAIN": 4,
        "candidate_T2D": 4,
        "contraindicated_AGE65": 0,
        "contraindicated_ASTHMA": 1,
        "contraindicated_CKD": 4,
        "contraindicated_EPI": 0,
        "contraindicated_LIVER": 1,
        "contraindicated_PREG": 5,
        "serotonergic_opioids": 0,
    }


def test_p4_inheritance_range_and_examples(reasoned: ReasonedOntology) -> None:
    result = run_inheritance(reasoned, oracle_prototypes=None)
    rows = {r.category_id: r for r in result.rows}
    assert len(rows) == 32
    assert rows["nsaids"].rows == 10
    assert rows["nsaids"].coprescriptions == ("AGE65->omeprazole",)
    assert rows["nsaids"].interactions == ("apixaban", "ibuprofen", "sertraline", "warfarin")
    assert rows["ssris"].rows == 8
    assert rows["low_molecular_weight_heparins"].rows == 7
    assert rows["serotonergic_opioids"].rows == 7
    assert (result.min_rows, result.max_rows) == (0, 10)
    assert rows["drugs"].rows == 0
    below_drugs = [r.rows for r in result.rows if r.category_id != "drugs"]
    assert (len(below_drugs), min(below_drugs), max(below_drugs)) == (31, 1, 10)


# --- P4.3 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("drug_id", "category_id", "expected"),
    [
        ("ibuprofen", "opioids", ("analg",)),
        ("ibuprofen", "antidepressants", ()),
        ("ibuprofen", "serotonergic_opioids", ("analg",)),  # through the definition edge
        ("tramadol", "nsaids", ("analg",)),  # through tramadol's told opioids
        ("sertraline", "opioids", ()),
        ("warfarin", "low_molecular_weight_heparins", ("anticoag",)),
        ("warfarin", "nsaids", ()),
    ],
)
def test_expected_clash_labels_on_hand_checked_cases(
    real_ontology: Ontology, drug_id: str, category_id: str, expected: tuple[str, ...]
) -> None:
    assert expected_clash_sets(real_ontology.data, drug_id, category_id) == expected


def test_mutation_candidates_leave_out_own_categories_and_families(
    reasoned: ReasonedOntology,
) -> None:
    mutations = mutation_candidates(reasoned)
    assert len(mutations) == 18 * 32 - 69 == 507
    assert ("ibuprofen", "nsaids") not in mutations
    assert ("ibuprofen", "bleeding_risk_drugs") not in mutations
    assert ("tramadol", "serotonergic_opioids") not in mutations  # classified, not told
    assert all(k != "therapeutic_families" for _, k in mutations)
    assert ("ibuprofen", "opioids") in mutations


@pytest.mark.slow
def test_p4_consistency_detects_every_clash_and_flags_nothing_else(
    consistency: ConsistencyResult,
) -> None:
    assert consistency.kb_clashes == ()
    assert consistency.inconsistent_categories == ()
    assert consistency.categories_checked == 44
    assert (consistency.n_mutations, consistency.n_clash_expected) == (507, 26)
    assert consistency.n_clash_expected_from_upper == 0
    assert consistency.n_clash_expected_by_set == {
        "analg": 8,
        "anticoag": 4,
        "antidep": 2,
        "antidiab": 6,
        "beta": 2,
        "ccb": 2,
        "raas": 2,
    }
    assert consistency.n_clash_detected == 26
    assert (consistency.n_no_clash_expected, consistency.n_false_alarms) == (481, 0)
    assert all(p.uncovered == () for p in consistency.partitions)
    assert consistency.oracle_single_run is None and consistency.oracle_rechecked is None


@pytest.mark.slow
def test_p4_expected_clashes_per_drug(consistency: ConsistencyResult) -> None:
    per_drug = Counter(m.drug_id for m in consistency.mutations if m.expected_clash)
    assert per_drug == {
        "warfarin": 2,
        "apixaban": 2,
        "metformin": 2,
        "gliclazide": 2,
        "linagliptin": 2,
        "ibuprofen": 3,
        "paracetamol": 3,
        "tramadol": 2,
        "enalapril": 1,
        "losartan": 1,
        "bisoprolol": 1,
        "propranolol": 1,
        "amlodipine": 1,
        "verapamil": 1,
        "sertraline": 1,
        "mirtazapine": 1,
    }
    ibuprofen = [m.category_id for m in consistency.mutations if m.drug_id == "ibuprofen"]
    clashing = [k for k in ibuprofen if k in {"anilides", "opioids", "serotonergic_opioids"}]
    assert clashing == ["anilides", "opioids", "serotonergic_opioids"]


# --- P4.4 -------------------------------------------------------------------------------------


def test_p4_fixed_point_cost(reasoned: ReasonedOntology) -> None:
    result = run_fixed_point_cost(reasoned)
    assert (result.told_facts, result.rules, result.closure_facts) == (93, 47, 254)
    assert result.iterations == 4
    assert result.new_per_iteration == (61, 50, 44, 6)
    assert (result.predicates, result.constants, result.max_arity) == (13, 74, 3)
    assert result.bound == 13 * 74**3
    assert result.naive_equals_incremental
    assert result.subset_closure_matches_networkx
    assert result.facts_per_predicate["Int"] == 20
    assert result.facts_per_predicate["Incons"] == 0


# --- serialisation ----------------------------------------------------------------------------


def test_told_knowledge_is_json_ready_and_complete(real_ontology: Ontology) -> None:
    knowledge = told_knowledge(real_ontology)
    text = json.dumps(knowledge, sort_keys=True, ensure_ascii=False)
    assert json.loads(text) == knowledge
    categories, clauses = knowledge["categories"], knowledge["clauses"]
    assert isinstance(categories, list) and len(categories) == 33
    assert isinstance(clauses, list) and len(clauses) == 47
    assert knowledge["definitions"] == [
        {"category": "serotonergic_opioids", "conjuncts": ["opioids", "serotonergic_drugs"]}
    ]
    drugs = knowledge["drugs"]
    assert isinstance(drugs, list)
    tramadol = next(d for d in drugs if d["id"] == "tramadol")
    assert tramadol["categories"] == ["opioids", "serotonergic_drugs"]


@dataclass(frozen=True)
class _Record:
    values: tuple[tuple[str, str], ...]
    table: dict[str, int]


def test_as_jsonable_turns_tuples_and_mappings_into_lists_and_dicts() -> None:
    record = _Record(values=(("a", "b"),), table={"x": 1})
    assert as_jsonable(record) == {"values": [["a", "b"]], "table": {"x": 1}}
    assert as_jsonable((record,)) == [{"values": [["a", "b"]], "table": {"x": 1}}]


def test_as_jsonable_rejects_other_objects() -> None:
    with pytest.raises(TypeError):
        as_jsonable(object())
