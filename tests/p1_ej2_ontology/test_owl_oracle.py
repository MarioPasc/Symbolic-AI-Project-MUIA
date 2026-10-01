"""The HermiT oracle: agreement with forward chaining, mutations, and its failure paths.

Tests that run HermiT are ``integration`` and are skipped, with the reason, when owlready2 or a
Java runtime is missing.
"""

from __future__ import annotations

import importlib

import pytest
from ej2_toy_data import toy_ontology_data

from symbolic_ai.dataloader.models import InteractionLink
from symbolic_ai.p1_ej2_ontology import owl_oracle
from symbolic_ai.p1_ej2_ontology.errors import OracleUnavailableError
from symbolic_ai.p1_ej2_ontology.ontology import Ontology
from symbolic_ai.p1_ej2_ontology.reasoner import ReasonedOntology, classify, reason, taxonomy


def _unavailable_reason() -> str | None:
    try:
        owl_oracle.check_available()
    except OracleUnavailableError as exc:
        return str(exc)
    return None


_REASON = _unavailable_reason()
requires_oracle = pytest.mark.skipif(_REASON is not None, reason=f"oracle unavailable: {_REASON}")


@pytest.fixture(scope="module")
def reasoned(real_ontology: Ontology) -> ReasonedOntology:
    """The real ontology with its closure."""
    return reason(real_ontology)


@pytest.fixture(scope="module")
def base_run(real_ontology: Ontology) -> owl_oracle.OracleRun:
    """HermiT's run on the export of the real data."""
    return owl_oracle.reason_base(real_ontology.data)


# --- agreement on the real data ---------------------------------------------------------------


@pytest.mark.integration
@requires_oracle
def test_hermit_finds_the_ontology_consistent(base_run: owl_oracle.OracleRun) -> None:
    assert base_run.consistent
    assert base_run.unsatisfiable == ()


@pytest.mark.integration
@requires_oracle
def test_hermit_classifies_every_drug_as_forward_chaining(
    reasoned: ReasonedOntology, base_run: owl_oracle.OracleRun
) -> None:
    for drug_id in reasoned.ontology.drug_ids:
        assert base_run.individuals["D_" + drug_id].categories == set(classify(reasoned, drug_id))


@pytest.mark.integration
@requires_oracle
def test_hermit_derives_the_extra_interaction(base_run: owl_oracle.OracleRun) -> None:
    assert "D_apixaban" in base_run.individuals["D_sertraline"].interacts
    assert "D_sertraline" not in base_run.individuals["D_sertraline"].interacts


@pytest.mark.integration
@requires_oracle
def test_hermit_hierarchy_equals_the_prototype_taxonomy(
    reasoned: ReasonedOntology, base_run: owl_oracle.OracleRun
) -> None:
    result = taxonomy(reasoned)
    named = set(result.categories)
    for category_id in result.categories:
        assert base_run.subsumers[category_id] & named == set(result.subsumers[category_id])


@pytest.mark.integration
@requires_oracle
@pytest.mark.parametrize(
    ("drug_id", "category_id", "consistent"),
    [
        ("ibuprofen", "opioids", False),
        ("ibuprofen", "antidepressants", True),
        ("ibuprofen", "serotonergic_opioids", False),
        ("warfarin", "conditions", False),
    ],
)
def test_per_mutation_runs_on_hand_checked_cases(
    real_ontology: Ontology, drug_id: str, category_id: str, consistent: bool
) -> None:
    assert owl_oracle.mutation_is_consistent(real_ontology.data, drug_id, category_id) is consistent


@pytest.mark.integration
@requires_oracle
def test_single_run_mutation_classes_agree_with_per_mutation_runs(
    real_ontology: Ontology,
) -> None:
    mutations = [
        ("ibuprofen", "opioids"),
        ("ibuprofen", "antidepressants"),
        ("ibuprofen", "serotonergic_opioids"),
        ("warfarin", "conditions"),
    ]
    unsatisfiable = owl_oracle.unsatisfiable_mutations(real_ontology.data, mutations)
    assert unsatisfiable == {mutations[0], mutations[2], mutations[3]}


@pytest.mark.integration
@requires_oracle
def test_prototype_individuals_inherit_like_forward_chaining(real_ontology: Ontology) -> None:
    run = owl_oracle.reason_with_prototypes(real_ontology.data, ["nsaids"])
    prototype = run.individuals["P_nsaids"]
    assert prototype.treats == {"PAIN"}
    assert prototype.contraindicated_by == {"CKD"}
    assert prototype.requires == {("AGE65", "omeprazole")}
    drugs = {name for name in prototype.interacts if name.startswith("D_")}
    assert drugs == {"D_apixaban", "D_sertraline", "D_warfarin"}


@requires_oracle
def test_swrl_rule_excludes_self_pairs() -> None:
    data = toy_ontology_data(interactions=(InteractionLink("a", "a", "bleeding", "major"),))
    run = owl_oracle.reason_base(data)
    assert run.individuals["D_x1"].interacts == {"D_x2"}
    assert run.individuals["D_y"].interacts == set()


# --- failure paths ----------------------------------------------------------------------------


def test_missing_owlready2_raises_oracle_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(owl_oracle.importlib, "import_module", fail)
    with pytest.raises(OracleUnavailableError, match="owlready2"):
        owl_oracle.check_available()
    assert owl_oracle.owlready2_version() is None


def test_missing_java_raises_oracle_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("owlready2")
    monkeypatch.setattr(owl_oracle.shutil, "which", lambda _: None)
    with pytest.raises(OracleUnavailableError, match="Java"):
        owl_oracle.check_available()


def test_oracle_module_does_not_import_forward_chaining() -> None:
    source = importlib.util.find_spec(owl_oracle.__name__)
    assert source is not None and source.origin is not None
    with open(source.origin, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in ("aima", "forward_chaining", "reasoner", "p1_ej2_ontology.ontology"):
        assert f"import {forbidden}" not in text and f"{forbidden} import" not in text
