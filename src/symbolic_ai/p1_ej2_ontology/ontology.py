"""The EJ2 ontology built from the data, and its translation into the definite clauses O1-O7.

Categories are reified as constants (AIMA §10.2). Membership x ∈ c is the atom ``Member(x, c)``
and the subcategory relation c ⊂ c' is ``Subset(c, c')``; the other predicates keep the names of
the report (``TrataCat``/``Trata``, ``CICat``/``CI``, ``IntCat``/``Int``, ``CopCat``/``Cop``,
``Familia``, ``Disj``, ``Incons``). Constants carry an upper-case prefix, because aima-python reads
a symbol that starts in lower case as a variable: ``D_<drug>``, ``C_<category>``,
``H_<condition>``, ``R_<risk factor>`` and ``P_<category>`` for the prototype of a category.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import networkx as nx

from aima.logic import expr
from aima.utils import Expr
from symbolic_ai.dataloader.models import Category, OntologyData
from symbolic_ai.p1_ej2_ontology.errors import (
    CyclicTaxonomyError,
    OntologyError,
    UnknownIdentifierError,
)
from symbolic_ai.p1_ej2_ontology.forward_chaining import KnowledgeBase

__all__ = [
    "AXIOMS",
    "CI",
    "CI_CAT",
    "CONDITIONS_CATEGORY",
    "COP",
    "COP_CAT",
    "DISJ",
    "DRUGS_CATEGORY",
    "FAMILIA",
    "FAMILIES_CATEGORY",
    "INCONS",
    "INT",
    "INT_CAT",
    "MEMBER",
    "RISK_FACTORS_CATEGORY",
    "SUBSET",
    "TRATA",
    "TRATA_CAT",
    "DefinedCategory",
    "DefinedKind",
    "Ontology",
    "TaggedRule",
    "atom",
    "build_ontology",
    "category_symbol",
    "condition_symbol",
    "drug_symbol",
    "identifier",
    "prototype_symbol",
    "risk_factor_symbol",
    "tagged_rules",
    "to_knowledge_base",
    "told_facts",
]

# --- vocabulary ---------------------------------------------------------------------------------

MEMBER = "Member"
SUBSET = "Subset"
TRATA_CAT = "TrataCat"
TRATA = "Trata"
CI_CAT = "CICat"
CI = "CI"
INT_CAT = "IntCat"
INT = "Int"
COP_CAT = "CopCat"
COP = "Cop"
FAMILIA = "Familia"
DISJ = "Disj"
INCONS = "Incons"

#: The category whose members are the drugs; the defined categories of O7 are subsets of it.
DRUGS_CATEGORY = "drugs"
#: Conditions and risk factors are members of these categories by construction.
CONDITIONS_CATEGORY = "conditions"
RISK_FACTORS_CATEGORY = "risk_factors"
#: The category of categories: its members are the therapeutic families (``Familia``).
FAMILIES_CATEGORY = "therapeutic_families"

_DRUG_PREFIX = "D_"
_CATEGORY_PREFIX = "C_"
_CONDITION_PREFIX = "H_"
_RISK_FACTOR_PREFIX = "R_"
_PROTOTYPE_PREFIX = "P_"

#: Axioms O1-O6 of the report (``README.md`` §4.6), as aima ``expr`` strings. O3 has one clause
#: per member property; O7 is generated per defined category by :func:`tagged_rules`.
AXIOMS: tuple[tuple[str, str], ...] = (
    ("O1", "(Member(x, c) & Subset(c, d)) ==> Member(x, d)"),
    ("O2", "(Subset(c, d) & Subset(d, e)) ==> Subset(c, e)"),
    ("O3", "(Member(x, c) & TrataCat(c, h)) ==> Trata(x, h)"),
    ("O3", "(Member(x, c) & CICat(c, r)) ==> CI(x, r)"),
    ("O3", "(Member(x, c) & CopCat(c, r, y)) ==> Cop(x, r, y)"),
    ("O4", "IntCat(c, d) ==> IntCat(d, c)"),
    ("O5", "(Member(a, c) & Member(b, d) & IntCat(c, d) & Distinct(a, b)) ==> Int(a, b)"),
    ("O6", "(Member(x, c) & Member(x, d) & Disj(c, d)) ==> Incons(x, c, d)"),
)


class DefinedKind(StrEnum):
    """The two families of defined categories (``README.md`` §4.5)."""

    CANDIDATE = "candidate"
    CONTRAINDICATED = "contraindicated"


@dataclass(frozen=True, slots=True)
class DefinedCategory:
    """A category defined by necessary and sufficient conditions (O7).

    ``candidate_<h>``: x ∈ it ⇔ x ∈ drugs ∧ Trata(x, h). ``contraindicated_<r>``: x ∈ it ⇔
    x ∈ drugs ∧ CI(x, r).
    """

    category_id: str
    kind: DefinedKind
    target_id: str


@dataclass(frozen=True, slots=True)
class TaggedRule:
    """One definite clause with the axiom it belongs to (``O1`` ... ``O7``)."""

    tag: str
    clause: Expr


@dataclass(frozen=True, slots=True)
class Ontology:
    """The ontology: the data tables plus what follows from their structure alone.

    Parameters
    ----------
    data : OntologyData
        The tables loaded by :func:`symbolic_ai.dataloader.load_ontology`.
    drug_categories : tuple[str, ...]
        ``drugs`` and every category below it in the told taxonomy, sorted.
    defined_categories : tuple[DefinedCategory, ...]
        One per condition and one per risk factor, sorted by identifier.
    """

    data: OntologyData
    drug_categories: tuple[str, ...]
    defined_categories: tuple[DefinedCategory, ...]

    @property
    def category_ids(self) -> tuple[str, ...]:
        """The told (primitive) categories, sorted."""
        return tuple(c.category_id for c in self.data.categories)

    @property
    def drug_ids(self) -> tuple[str, ...]:
        """The drugs, sorted."""
        return tuple(d.drug_id for d in self.data.drugs)

    @property
    def defined_ids(self) -> tuple[str, ...]:
        """The defined categories, sorted."""
        return tuple(d.category_id for d in self.defined_categories)

    @property
    def named_categories(self) -> tuple[str, ...]:
        """The categories of the taxonomy experiment: the drug categories, then the defined ones."""
        return self.drug_categories + self.defined_ids

    def category(self, category_id: str) -> Category:
        """Return a told category by identifier.

        Raises
        ------
        UnknownIdentifierError
            If ``category_id`` is not a told category.
        """
        for category in self.data.categories:
            if category.category_id == category_id:
                return category
        raise UnknownIdentifierError(f"unknown category {category_id!r}")

    def defined(self, category_id: str) -> DefinedCategory | None:
        """Return the defined category ``category_id``, or ``None`` if it is not defined."""
        return next((d for d in self.defined_categories if d.category_id == category_id), None)

    def is_category(self, identifier: str) -> bool:
        """Whether ``identifier`` is a told category."""
        return identifier in self.category_ids

    def is_drug(self, identifier: str) -> bool:
        """Whether ``identifier`` is a drug."""
        return identifier in self.drug_ids

    def leaf_of(self, drug_id: str) -> str:
        """Return the category of the told membership of ``drug_id``.

        Raises
        ------
        UnknownIdentifierError
            If ``drug_id`` has no told membership.
        """
        for membership in self.data.memberships:
            if membership.object_id == drug_id:
                return membership.category_id
        raise UnknownIdentifierError(f"drug {drug_id!r} has no told membership")

    def parents(self) -> Mapping[str, tuple[str, ...]]:
        """Return the told parents of every category that has one, sorted."""
        grouped: dict[str, list[str]] = {}
        for edge in self.data.subcategories:
            grouped.setdefault(edge.category_id, []).append(edge.parent_id)
        return {child: tuple(sorted(grouped[child])) for child in sorted(grouped)}


# --- symbols ------------------------------------------------------------------------------------


def drug_symbol(drug_id: str) -> Expr:
    """Return the constant of a drug, ``D_<drug_id>``."""
    return Expr(_DRUG_PREFIX + drug_id)


def category_symbol(category_id: str) -> Expr:
    """Return the constant of a category (told or defined), ``C_<category_id>``."""
    return Expr(_CATEGORY_PREFIX + category_id)


def condition_symbol(condition_id: str) -> Expr:
    """Return the constant of a condition, ``H_<condition_id>``."""
    return Expr(_CONDITION_PREFIX + condition_id)


def risk_factor_symbol(risk_factor_id: str) -> Expr:
    """Return the constant of a risk factor, ``R_<risk_factor_id>``."""
    return Expr(_RISK_FACTOR_PREFIX + risk_factor_id)


def prototype_symbol(category_id: str) -> Expr:
    """Return the fresh constant p_c of a category's prototype, ``P_<category_id>``."""
    return Expr(_PROTOTYPE_PREFIX + category_id)


def identifier(symbol: Expr) -> str:
    """Return the data identifier of a constant (its name without the two-character prefix)."""
    return str(symbol.op)[2:]


def atom(predicate: str, *arguments: Expr) -> Expr:
    """Return the atom ``predicate(arguments...)``."""
    return Expr(predicate, *arguments)


# --- construction -------------------------------------------------------------------------------


def _taxonomy_graph(data: OntologyData) -> nx.DiGraph[str]:
    """Return the told taxonomy as a directed graph, child -> parent."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(c.category_id for c in data.categories)
    graph.add_edges_from((e.category_id, e.parent_id) for e in data.subcategories)
    return graph


def _require(identifiers: Iterable[str], known: set[str], what: str) -> None:
    unknown = sorted(set(identifiers) - known)
    if unknown:
        raise UnknownIdentifierError(f"unknown {what}: {', '.join(unknown)}")


def _check_references(data: OntologyData) -> None:
    """Check that every identifier used by a table exists (the dataloader also checks this)."""
    categories = {c.category_id for c in data.categories}
    drugs = {d.drug_id for d in data.drugs}
    subjects = categories | drugs
    _require((e.category_id for e in data.subcategories), categories, "category")
    _require((e.parent_id for e in data.subcategories), categories, "category")
    _require((m.object_id for m in data.memberships), drugs, "drug")
    _require((m.category_id for m in data.memberships), categories, "category")
    _require((link.subject_id for link in data.indications), subjects, "subject")
    _require((link.subject_id for link in data.contraindications), subjects, "subject")
    _require((link.subject_id for link in data.coprescriptions), subjects, "subject")
    _require((link.companion_drug_id for link in data.coprescriptions), drugs, "drug")
    _require((link.subject_a for link in data.interactions), subjects, "subject")
    _require((link.subject_b for link in data.interactions), subjects, "subject")
    _require(data.families, categories, "category")
    _require((c for s in data.disjoint_sets for c in s.category_ids), categories, "category")
    _require({DRUGS_CATEGORY}, categories, "category")
    _require((link.condition_id for link in data.indications), _condition_ids(data), "condition")
    risk_factors = _risk_factor_ids(data)
    _require((link.risk_factor_id for link in data.contraindications), risk_factors, "risk factor")
    _require((link.risk_factor_id for link in data.coprescriptions), risk_factors, "risk factor")


def _condition_ids(data: OntologyData) -> set[str]:
    return {c.condition_id for c in data.conditions}


def _risk_factor_ids(data: OntologyData) -> set[str]:
    return {r.risk_factor_id for r in data.risk_factors}


def _defined_categories(data: OntologyData) -> tuple[DefinedCategory, ...]:
    defined = [
        DefinedCategory(f"candidate_{c.condition_id}", DefinedKind.CANDIDATE, c.condition_id)
        for c in data.conditions
    ] + [
        DefinedCategory(
            f"contraindicated_{r.risk_factor_id}", DefinedKind.CONTRAINDICATED, r.risk_factor_id
        )
        for r in data.risk_factors
    ]
    told = {c.category_id for c in data.categories}
    clashing = sorted(d.category_id for d in defined if d.category_id in told)
    if clashing:
        raise OntologyError(f"defined category names already used by told categories: {clashing}")
    return tuple(sorted(defined, key=lambda d: d.category_id))


def build_ontology(data: OntologyData) -> Ontology:
    """Build the :class:`Ontology` of the data: drug categories and defined categories.

    Parameters
    ----------
    data : OntologyData
        The tables loaded by :func:`symbolic_ai.dataloader.load_ontology`.

    Returns
    -------
    Ontology
        The ontology, ready to be translated by :func:`to_knowledge_base`.

    Raises
    ------
    UnknownIdentifierError
        If a table refers to an identifier that does not exist, or there is no ``drugs`` category.
    CyclicTaxonomyError
        If the told subcategory links contain a cycle.
    OntologyError
        If an interaction links a drug with a category, or a generated defined-category name is
        already a told category.
    """
    _check_references(data)
    graph = _taxonomy_graph(data)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = " -> ".join(child for child, _ in nx.find_cycle(graph))
        raise CyclicTaxonomyError(f"the told taxonomy has a cycle: {cycle}")
    mixed = [
        f"{link.subject_a}-{link.subject_b}"
        for link in data.interactions
        if _is_drug(data, link.subject_a) != _is_drug(data, link.subject_b)
    ]
    if mixed:
        raise OntologyError(f"an interaction must join two categories or two drugs: {mixed}")
    below_drugs = nx.ancestors(graph, DRUGS_CATEGORY)
    return Ontology(
        data=data,
        drug_categories=tuple(sorted({DRUGS_CATEGORY, *below_drugs})),
        defined_categories=_defined_categories(data),
    )


def _is_drug(data: OntologyData, identifier_: str) -> bool:
    return any(d.drug_id == identifier_ for d in data.drugs)


# --- translation into definite clauses ----------------------------------------------------------


def _subject_symbol(ontology: Ontology, subject_id: str) -> tuple[Expr, bool]:
    """Return the constant of a link subject and whether it is a category."""
    if ontology.is_category(subject_id):
        return category_symbol(subject_id), True
    return drug_symbol(subject_id), False


def _membership_facts(ontology: Ontology) -> list[Expr]:
    data = ontology.data
    facts = [
        atom(MEMBER, drug_symbol(m.object_id), category_symbol(m.category_id))
        for m in data.memberships
    ]
    if ontology.is_category(CONDITIONS_CATEGORY):
        facts += [
            atom(MEMBER, condition_symbol(c.condition_id), category_symbol(CONDITIONS_CATEGORY))
            for c in data.conditions
        ]
    if ontology.is_category(RISK_FACTORS_CATEGORY):
        facts += [
            atom(
                MEMBER, risk_factor_symbol(r.risk_factor_id), category_symbol(RISK_FACTORS_CATEGORY)
            )
            for r in data.risk_factors
        ]
    return facts


def _link_facts(ontology: Ontology) -> list[Expr]:
    data = ontology.data
    facts: list[Expr] = []
    for indication in data.indications:
        subject, on_category = _subject_symbol(ontology, indication.subject_id)
        predicate = TRATA_CAT if on_category else TRATA
        facts.append(atom(predicate, subject, condition_symbol(indication.condition_id)))
    for contraindication in data.contraindications:
        subject, on_category = _subject_symbol(ontology, contraindication.subject_id)
        predicate = CI_CAT if on_category else CI
        facts.append(atom(predicate, subject, risk_factor_symbol(contraindication.risk_factor_id)))
    for coprescription in data.coprescriptions:
        subject, on_category = _subject_symbol(ontology, coprescription.subject_id)
        predicate = COP_CAT if on_category else COP
        risk_factor = risk_factor_symbol(coprescription.risk_factor_id)
        companion = drug_symbol(coprescription.companion_drug_id)
        facts.append(atom(predicate, subject, risk_factor, companion))
    for interaction in data.interactions:
        a, on_category = _subject_symbol(ontology, interaction.subject_a)
        b, _ = _subject_symbol(ontology, interaction.subject_b)
        if on_category:
            facts.append(atom(INT_CAT, a, b))
        else:  # an object-level pair is told in both directions, as O4 + O5 would derive it
            facts += [atom(INT, a, b), atom(INT, b, a)]
    return facts


def _structure_facts(ontology: Ontology) -> list[Expr]:
    data = ontology.data
    facts = [
        atom(SUBSET, category_symbol(e.category_id), category_symbol(e.parent_id))
        for e in data.subcategories
    ]
    facts += [atom(FAMILIA, category_symbol(c)) for c in data.families]
    for disjoint_set in data.disjoint_sets:
        facts += [
            atom(DISJ, category_symbol(a), category_symbol(b))
            for a, b in itertools.combinations(disjoint_set.category_ids, 2)
        ]
    return facts


def told_facts(ontology: Ontology) -> tuple[Expr, ...]:
    """Return the told facts of the KB (``README.md`` §4.2-4.4), sorted by their text.

    Memberships (drugs in their leaf, conditions and risk factors in their upper category),
    subcategory links, the member-property links (category-level ``TrataCat``/``CICat``/
    ``CopCat``/``IntCat``, object-level ``Trata``/``CI``/``Cop``/``Int``), the families and one
    ``Disj(c, c')`` per pair ``c < c'`` of every disjoint set.

    Parameters
    ----------
    ontology : Ontology
        The ontology.

    Returns
    -------
    tuple[Expr, ...]
        Ground atoms.
    """
    facts = _membership_facts(ontology) + _structure_facts(ontology) + _link_facts(ontology)
    return tuple(sorted(set(facts), key=str))


def _definition_rules(defined: DefinedCategory) -> tuple[str, ...]:
    """Return the three definite clauses of one definition of O7 (both directions)."""
    category = f"{_CATEGORY_PREFIX}{defined.category_id}"
    drugs = f"{_CATEGORY_PREFIX}{DRUGS_CATEGORY}"
    if defined.kind is DefinedKind.CANDIDATE:
        property_atom = f"{TRATA}(x, {_CONDITION_PREFIX}{defined.target_id})"
    else:
        property_atom = f"{CI}(x, {_RISK_FACTOR_PREFIX}{defined.target_id})"
    return (
        f"({MEMBER}(x, {drugs}) & {property_atom}) ==> {MEMBER}(x, {category})",
        f"{MEMBER}(x, {category}) ==> {MEMBER}(x, {drugs})",
        f"{MEMBER}(x, {category}) ==> {property_atom}",
    )


def tagged_rules(ontology: Ontology) -> tuple[TaggedRule, ...]:
    """Return the rules O1-O7 of the KB, each with its axiom tag, in the order of the report.

    Parameters
    ----------
    ontology : Ontology
        The ontology (it fixes the defined categories of O7).

    Returns
    -------
    tuple[TaggedRule, ...]
        8 clauses for O1-O6 and 3 per defined category for O7.
    """
    rules = [TaggedRule(tag, expr(text)) for tag, text in AXIOMS]
    for defined in ontology.defined_categories:
        rules += [TaggedRule("O7", expr(text)) for text in _definition_rules(defined)]
    return tuple(rules)


def to_knowledge_base(ontology: Ontology) -> KnowledgeBase:
    """Translate the ontology into its Datalog knowledge base: told facts and the rules O1-O7.

    Parameters
    ----------
    ontology : Ontology
        The ontology.

    Returns
    -------
    KnowledgeBase
        The KB whose fixed point Cl(KB) answers the three queries of ``README.md`` §5.2.
    """
    rules = tuple(rule.clause for rule in tagged_rules(ontology))
    return KnowledgeBase(facts=told_facts(ontology), rules=rules)
