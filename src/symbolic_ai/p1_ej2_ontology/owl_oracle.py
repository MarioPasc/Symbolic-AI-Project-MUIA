"""Independent oracle: an OWL 2 export of the ontology tables, reasoned by HermiT through owlready2.

The oracle shares only the dataloader's :class:`~symbolic_ai.dataloader.models.OntologyData` with
the forward-chaining reasoner: no clause, no aima ``Expr`` and no forward-chaining code. Mapping
(``README.md`` §5.4, results design §11.2): a category is a class; c ⊂ c' is ``SubClassOf``; a
told membership is a class assertion (a drug may have several); a member-property link written on
a category is ``SubClassOf(c, ObjectHasValue(p, v))`` (CLASSIC's *Fills*) and one written on a drug
is a property assertion; a defined category of §4.5 is ``EquivalentClasses(D, drugs ⊓ ∃p.{v})``
and a category defined by conjuncts is ``EquivalentClasses(D, k₁ ⊓ … ⊓ kₙ)``; a disjoint set is
``DisjointClasses``; every individual is in one ``DifferentIndividuals`` axiom (unique names); a
category-pair interaction is the DL-safe SWRL rule
``c(?a), c'(?b), DifferentFrom(?a, ?b) -> interacts(?a, ?b)``, one per direction, and a self-link
on c is the single rule ``c(?a), c(?b), DifferentFrom(?a, ?b) -> interacts(?a, ?b)``. Conditions
and risk factors are plain individuals (no upper ontology). The ternary coprescription link
``CopCat(c, r, d')`` becomes one property ``requires_<r>`` per risk factor with value d'. Families
are not exported: a category of categories needs OWL punning, and the family pairs follow from
class memberships.

HermiT (Glimm et al., J. Autom. Reasoning 53(3), 2014) is sound and complete for OWL 2 DL, so a
mistake in the translation to clauses, in the forward-chaining loop or in the queries shows up as a
disagreement. owlready2 is an optional dependency, imported only when the oracle runs; HermiT needs
a Java runtime.
"""

from __future__ import annotations

import graphlib
import importlib
import logging
import shutil
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from symbolic_ai.dataloader.models import OntologyData
from symbolic_ai.p1_ej2_ontology.errors import OracleUnavailableError

__all__ = [
    "IndividualView",
    "OracleRun",
    "check_available",
    "mutation_is_consistent",
    "owlready2_version",
    "reason_base",
    "reason_with_prototypes",
    "unsatisfiable_mutations",
]

logger = logging.getLogger(__name__)

_ONTOLOGY_IRI = "http://symai.local/ej2-ontology.owl#"
# Data identifiers the mapping refers to (kept here, not imported, so the oracle stays independent).
_DRUGS = "drugs"
_TREATS = "treats"
_CONTRAINDICATED_BY = "contraindicated_by"
_INTERACTS = "interacts"
_REQUIRES = "requires_"
_CLASS = "C_"
_DRUG = "D_"
_CONDITION = "H_"
_RISK_FACTOR = "R_"
_PROTOTYPE = "P_"
_MUTATION = "M_"


@dataclass(frozen=True, slots=True)
class IndividualView:
    """What HermiT infers about one individual.

    ``categories`` holds category identifiers (told and defined); ``interacts`` individual names
    (``D_<drug>``, ``P_<category>``); ``requires`` (risk factor, companion drug) pairs.
    """

    name: str
    categories: frozenset[str]
    treats: frozenset[str]
    contraindicated_by: frozenset[str]
    interacts: frozenset[str]
    requires: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class OracleRun:
    """The result of one HermiT run on an export.

    Parameters
    ----------
    consistent : bool
        Whether HermiT found the export consistent; if not, the other fields are empty.
    individuals : Mapping[str, IndividualView]
        Every named individual, by name.
    subsumers : Mapping[str, frozenset[str]]
        For every category class (told and defined), its proper named superclasses.
    unsatisfiable : tuple[str, ...]
        Classes equivalent to ``owl:Nothing`` (the class ``Nothing`` itself excluded), sorted.
    """

    consistent: bool
    individuals: Mapping[str, IndividualView]
    subsumers: Mapping[str, frozenset[str]]
    unsatisfiable: tuple[str, ...]


# --- availability -------------------------------------------------------------------------------


def _owlready() -> Any:
    """Import owlready2 lazily; raise :class:`OracleUnavailableError` if it or Java is missing."""
    try:
        module = importlib.import_module("owlready2")
    except ImportError as exc:
        raise OracleUnavailableError(
            "owlready2 is not installed (pip install 'symbolic-ai-practicas[owl]')"
        ) from exc
    java = str(module.JAVA_EXE)
    if shutil.which(java) is None:
        raise OracleUnavailableError(f"no Java runtime found ({java!r}); HermiT needs Java >= 11")
    return module


def check_available() -> None:
    """Raise :class:`OracleUnavailableError` unless owlready2 and a Java runtime are available."""
    _owlready()


def owlready2_version() -> str | None:
    """Return the installed owlready2 version, or ``None`` if it is not installed."""
    try:
        module = importlib.import_module("owlready2")
    except ImportError:
        return None
    return str(module.VERSION)


# --- export -------------------------------------------------------------------------------------


def _defined_categories(data: OntologyData) -> list[tuple[str, str, str]]:
    """Return (class id, property, value individual) for each defined category of §4.5."""
    defined = [
        (f"candidate_{c.condition_id}", _TREATS, _CONDITION + c.condition_id)
        for c in data.conditions
    ]
    defined += [
        (
            f"contraindicated_{r.risk_factor_id}",
            _CONTRAINDICATED_BY,
            _RISK_FACTOR + r.risk_factor_id,
        )
        for r in data.risk_factors
    ]
    return sorted(defined)


def _new_class(name: str, bases: tuple[Any, ...]) -> Any:
    """Create an owlready2 class; typed ``Any`` because owlready2 ships no type information."""
    created: Any = types.new_class(name, bases)
    return created


def _parents(data: OntologyData) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {c.category_id: [] for c in data.categories}
    for edge in data.subcategories:
        parents[edge.category_id].append(edge.parent_id)
    return {child: sorted(parents[child]) for child in sorted(parents)}


def _told_types(data: OntologyData) -> dict[str, list[str]]:
    """Return the told categories of every drug that has one, sorted."""
    told: dict[str, list[str]] = {}
    for membership in data.memberships:
        told.setdefault(membership.object_id, []).append(membership.category_id)
    return {drug_id: sorted(told[drug_id]) for drug_id in sorted(told)}


class _Export:
    """Builds the OWL ontology of :class:`OntologyData` in a fresh owlready2 world."""

    def __init__(self, owl: Any, data: OntologyData) -> None:
        self.owl = owl
        self.data = data
        self.world = owl.World()
        self.onto = self.world.get_ontology(_ONTOLOGY_IRI)
        self.classes: dict[str, Any] = {}
        self.individuals: dict[str, Any] = {}
        self.properties: dict[str, Any] = {}
        with self.onto:
            self._declare_classes()
            self._declare_definitions()
            self._declare_properties()
            self._declare_individuals()
            self._declare_links()
            self._declare_defined()
            self._declare_disjointness()
            self._declare_interactions()

    def _declare_classes(self) -> None:
        parents = _parents(self.data)
        for category_id in graphlib.TopologicalSorter(parents).static_order():
            bases = tuple(self.classes[p] for p in parents[category_id]) or (self.owl.Thing,)
            self.classes[category_id] = _new_class(_CLASS + category_id, bases)

    def _declare_definitions(self) -> None:
        """State D ≡ k₁ ⊓ … ⊓ kₙ for every category defined by conjuncts."""
        for definition in self.data.definitions:
            conjuncts = [self.classes[k] for k in definition.conjunct_ids]
            self.classes[definition.category_id].equivalent_to.append(self.owl.And(conjuncts))

    def _declare_properties(self) -> None:
        names = [_TREATS, _CONTRAINDICATED_BY, _INTERACTS]
        names += [_REQUIRES + r.risk_factor_id for r in self.data.risk_factors]
        for name in names:
            self.properties[name] = _new_class(name, (self.owl.ObjectProperty,))

    def _new_individual(self, name: str, category_ids: Sequence[str]) -> None:
        """Create an individual asserted in every class of ``category_ids`` (``Thing`` if none)."""
        classes = [self.classes[c] for c in category_ids]
        individual = (classes[0] if classes else self.owl.Thing)(name)
        individual.is_a.extend(classes[1:])
        self.individuals[name] = individual

    def _declare_individuals(self) -> None:
        told = _told_types(self.data)
        for drug in self.data.drugs:
            self._new_individual(_DRUG + drug.drug_id, told.get(drug.drug_id, []))
        for condition in self.data.conditions:
            self._new_individual(_CONDITION + condition.condition_id, [])
        for risk_factor in self.data.risk_factors:
            self._new_individual(_RISK_FACTOR + risk_factor.risk_factor_id, [])

    def _fill(self, subject_id: str, prop: str, value: str) -> None:
        """State that every member of the subject (a category or one drug) has ``prop`` = value."""
        target = self.individuals[value]
        if subject_id in self.classes:
            self.classes[subject_id].is_a.append(self.properties[prop].value(target))
        else:
            getattr(self.individuals[_DRUG + subject_id], prop).append(target)

    def _declare_links(self) -> None:
        for indication in self.data.indications:
            self._fill(indication.subject_id, _TREATS, _CONDITION + indication.condition_id)
        for contraindication in self.data.contraindications:
            risk_factor = _RISK_FACTOR + contraindication.risk_factor_id
            self._fill(contraindication.subject_id, _CONTRAINDICATED_BY, risk_factor)
        for coprescription in self.data.coprescriptions:
            prop = _REQUIRES + coprescription.risk_factor_id
            self._fill(coprescription.subject_id, prop, _DRUG + coprescription.companion_drug_id)

    def _declare_defined(self) -> None:
        drugs = self.classes[_DRUGS]
        for class_id, prop, value in _defined_categories(self.data):
            defined = _new_class(_CLASS + class_id, (self.owl.Thing,))
            restriction = self.properties[prop].value(self.individuals[value])
            defined.equivalent_to.append(drugs & restriction)
            self.classes[class_id] = defined

    def _declare_disjointness(self) -> None:
        for disjoint_set in self.data.disjoint_sets:
            self.owl.AllDisjoint([self.classes[c] for c in disjoint_set.category_ids])

    def _class_pair_rule(self, a: str, b: str) -> None:
        rule = self.owl.Imp()
        rule.set_as_rule(
            f"{_CLASS}{a}(?a), {_CLASS}{b}(?b), DifferentFrom(?a, ?b) -> {_INTERACTS}(?a, ?b)"
        )

    def _declare_interactions(self) -> None:
        for link in self.data.interactions:
            a, b = link.subject_a, link.subject_b
            if a in self.classes and b in self.classes:
                # A self-link's rule is symmetric already: one rule, not two identical ones.
                pairs = [(a, b)] if a == b else [(a, b), (b, a)]
                for first, second in pairs:
                    self._class_pair_rule(first, second)
            else:
                a_ind, b_ind = self.individuals[_DRUG + a], self.individuals[_DRUG + b]
                a_ind.interacts.append(b_ind)
                b_ind.interacts.append(a_ind)

    def add_prototypes(self, category_ids: Iterable[str]) -> None:
        """Add one individual ``P_<c>`` of class c per category."""
        with self.onto:
            for category_id in category_ids:
                self._new_individual(_PROTOTYPE + category_id, [category_id])

    def add_membership(self, drug_id: str, category_id: str) -> None:
        """Tell one extra class assertion (a mutation)."""
        self.individuals[_DRUG + drug_id].is_a.append(self.classes[category_id])

    def add_mutation_classes(
        self, mutations: Sequence[tuple[str, str]], told: Mapping[str, Sequence[str]]
    ) -> None:
        """Declare M_i ≡ told(d) ⊓ K for every mutation (d, K), told(d) = ⊓ of d's told classes."""
        with self.onto:
            for index, (drug_id, category_id) in enumerate(mutations):
                mutation = _new_class(f"{_MUTATION}{index}", (self.owl.Thing,))
                operands = [self.classes[c] for c in (*told[drug_id], category_id)]
                mutation.equivalent_to.append(self.owl.And(operands))

    def reason(self) -> None:
        """Declare every individual different from every other one, then run HermiT."""
        with self.onto:
            self.owl.AllDifferent(list(self.individuals.values()))
        self.owl.sync_reasoner_hermit(self.world, infer_property_values=True, debug=0)


# --- reading the results ------------------------------------------------------------------------


def _names(values: Iterable[Any], prefix: str) -> frozenset[str]:
    return frozenset(str(v.name)[len(prefix) :] for v in values if str(v.name).startswith(prefix))


def _view(export: _Export, name: str) -> IndividualView:
    individual = export.individuals[name]
    owl = export.owl
    categories = _names(
        (c for c in individual.INDIRECT_is_a if isinstance(c, owl.ThingClass)), _CLASS
    )
    requires = frozenset(
        (r.risk_factor_id, companion)
        for r in export.data.risk_factors
        for companion in _names(getattr(individual, _REQUIRES + r.risk_factor_id), _DRUG)
    )
    return IndividualView(
        name=name,
        categories=categories,
        treats=_names(individual.treats, _CONDITION),
        contraindicated_by=_names(individual.contraindicated_by, _RISK_FACTOR),
        interacts=frozenset(str(v.name) for v in individual.interacts),
        requires=requires,
    )


def _subsumers(export: _Export) -> dict[str, frozenset[str]]:
    owl = export.owl
    result = {}
    for category_id in sorted(export.classes):
        ancestors = export.classes[category_id].ancestors()
        named = _names((a for a in ancestors if isinstance(a, owl.ThingClass)), _CLASS)
        result[category_id] = named - {category_id}
    return result


def _run(export: _Export) -> OracleRun:
    try:
        export.reason()
    except export.owl.OwlReadyInconsistentOntologyError:
        return OracleRun(consistent=False, individuals={}, subsumers={}, unsatisfiable=())
    unsatisfiable = sorted(
        str(c.name) for c in export.world.inconsistent_classes() if c is not export.owl.Nothing
    )
    individuals = {name: _view(export, name) for name in sorted(export.individuals)}
    return OracleRun(
        consistent=True,
        individuals=individuals,
        subsumers=_subsumers(export),
        unsatisfiable=tuple(unsatisfiable),
    )


# --- public API ---------------------------------------------------------------------------------


def reason_base(data: OntologyData) -> OracleRun:
    """Export the ontology and let HermiT classify it and infer every property value.

    Raises
    ------
    OracleUnavailableError
        If owlready2 or Java is missing.
    """
    owl = _owlready()
    run = _run(_Export(owl, data))
    logger.info("HermiT base run: %d unsatisfiable classes", len(run.unsatisfiable))
    return run


def reason_with_prototypes(data: OntologyData, category_ids: Sequence[str]) -> OracleRun:
    """Export the ontology with one prototype individual ``P_<c>`` per category, and reason.

    Raises
    ------
    OracleUnavailableError
        If owlready2 or Java is missing.
    """
    owl = _owlready()
    export = _Export(owl, data)
    export.add_prototypes(category_ids)
    return _run(export)


def unsatisfiable_mutations(
    data: OntologyData, mutations: Sequence[tuple[str, str]]
) -> frozenset[tuple[str, str]]:
    """Return the mutations (d, K) whose class told(d) ⊓ K is unsatisfiable, in one HermiT run.

    told(d) is the intersection of d's told classes. A mutation d ∈ K makes the ontology
    inconsistent iff told(d) ⊓ K is unsatisfiable, because d's only assertions are its told
    classes and the SWRL rules only add role assertions, which cannot clash here.
    :func:`mutation_is_consistent` checks this equivalence on real per-mutation runs.

    Raises
    ------
    OracleUnavailableError
        If owlready2 or Java is missing.
    """
    owl = _owlready()
    export = _Export(owl, data)
    export.add_mutation_classes(mutations, _told_types(data))
    run = _run(export)
    unsatisfiable = set(run.unsatisfiable)
    return frozenset(
        mutation
        for index, mutation in enumerate(mutations)
        if f"{_MUTATION}{index}" in unsatisfiable
    )


def mutation_is_consistent(data: OntologyData, drug_id: str, category_id: str) -> bool:
    """Tell d ∈ K in a fresh export and report whether HermiT finds the ontology consistent.

    Raises
    ------
    OracleUnavailableError
        If owlready2 or Java is missing.
    """
    owl = _owlready()
    export = _Export(owl, data)
    export.add_membership(drug_id, category_id)
    try:
        export.reason()
    except owl.OwlReadyInconsistentOntologyError:
        return False
    return True
