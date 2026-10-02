"""The description-logic tasks as queries on the fixed point, and the formulary derived from it.

Classification of an object x is {c : (x ∈ c) ∈ Cl(KB)}; the KB is consistent iff Cl(KB) has no
``Incons`` atom; c ⊑ d iff the prototype p_c, a fresh constant told only p_c ∈ c, satisfies
(p_c ∈ d) ∈ Cl(KB + {p_c ∈ c}) (``README.md`` §5.2). The prototype also tells what a new member of
c would inherit. :func:`derive_formulary` reads EJ1's drug-level tables off Cl(KB) (§5.3).
:func:`deduced_taxonomy` reads the direct-parent graph of ⊑ (its transitive reduction) and each
drug's most specific categories, and compares both with what was told.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import networkx as nx

from aima.utils import Expr
from symbolic_ai.dataloader.models import (
    AdverseInteraction,
    Contraindication,
    Coprescription,
    DrugClass,
    Formulary,
)
from symbolic_ai.p1_ej2_ontology.errors import CyclicTaxonomyError, UnknownIdentifierError
from symbolic_ai.p1_ej2_ontology.forward_chaining import (
    Closure,
    KnowledgeBase,
    ask,
    fc_closure,
    fc_extend,
)
from symbolic_ai.p1_ej2_ontology.ontology import (
    COP,
    FAMILIA,
    INCONS,
    INT,
    MEMBER,
    DefinedKind,
    Ontology,
    atom,
    category_symbol,
    drug_symbol,
    identifier,
    prototype_symbol,
    to_knowledge_base,
)

__all__ = [
    "Clash",
    "DeducedTaxonomy",
    "DirectEdge",
    "DrugMembership",
    "EdgeStatus",
    "IndirectLink",
    "Inheritance",
    "LinkLevel",
    "PartitionCoverage",
    "ReasonedOntology",
    "SameMembers",
    "Source",
    "Taxonomy",
    "add_membership",
    "candidate_sources",
    "category_members",
    "clashes",
    "classify",
    "contraindication_sources",
    "coprescription_sources",
    "deduced_taxonomy",
    "derive_formulary",
    "direct_edges",
    "drug_members",
    "equivalences",
    "family_pairs",
    "family_sources",
    "inherited_by_new_member",
    "interaction_sources",
    "membership_changes",
    "most_specific_categories",
    "partition_coverage",
    "prototype_closure",
    "reason",
    "redundant_told_edges",
    "same_members_not_equivalent",
    "subsumes",
    "taxonomy",
    "told_edges_made_indirect",
    "transitive_reduction",
]

logger = logging.getLogger(__name__)

_DRUG_CONSTANT_PREFIX = "D_"
_VARIABLE_X = Expr("x")
_VARIABLE_Y = Expr("y")


@dataclass(frozen=True, slots=True)
class ReasonedOntology:
    """An ontology with its knowledge base and the fixed point Cl(KB)."""

    ontology: Ontology
    kb: KnowledgeBase
    closure: Closure


@dataclass(frozen=True, slots=True)
class Clash:
    """An ``Incons(x, c, c')`` atom: object x belongs to two disjoint categories."""

    object_id: str
    category_a: str
    category_b: str


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Subsumption among the named categories, decided by prototypes.

    Parameters
    ----------
    categories : tuple[str, ...]
        The named categories: drug categories, then defined categories.
    subsumers : Mapping[str, tuple[str, ...]]
        For each category c, the named categories d ≠ c with c ⊑ d, sorted.
    told : frozenset[tuple[str, str]]
        The told subcategory links between named categories.
    """

    categories: tuple[str, ...]
    subsumers: Mapping[str, tuple[str, ...]]
    told: frozenset[tuple[str, str]]

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Return every proper pair (c, d) with c ⊑ d, sorted."""
        return tuple(sorted((c, d) for c in self.categories for d in self.subsumers[c]))


@dataclass(frozen=True, slots=True)
class Inheritance:
    """What a new member of a category inherits: the rows its prototype derives.

    ``interactions`` lists the existing drugs the prototype interacts with; ``family_pairs`` the
    (family, existing drug) pairs it would form; ``coprescriptions`` the (risk factor, companion)
    pairs it requires.
    """

    category_id: str
    categories: tuple[str, ...]
    candidates: tuple[str, ...]
    contraindications: tuple[str, ...]
    interactions: tuple[str, ...]
    coprescriptions: tuple[tuple[str, str], ...]
    family_pairs: tuple[tuple[str, str], ...]
    clashes: tuple[Clash, ...]

    @property
    def rows(self) -> int:
        """Number of drug-level formulary rows the new member receives."""
        return (
            len(self.candidates)
            + len(self.contraindications)
            + len(self.interactions)
            + len(self.coprescriptions)
            + len(self.family_pairs)
        )


@dataclass(frozen=True, slots=True)
class PartitionCoverage:
    """The closed-world exhaustiveness check of one partition on the told objects."""

    set_id: str
    parent_id: str
    parts: tuple[str, ...]
    members: tuple[str, ...]
    uncovered: tuple[str, ...]


class LinkLevel(StrEnum):
    """Whether a told link is written on a category or on one object."""

    CATEGORY = "category"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class Source:
    """The told statement a derived formulary row comes from."""

    subject_id: str
    level: LinkLevel
    text: str


class EdgeStatus(StrEnum):
    """Whether a direct edge of the deduced taxonomy was told or only deduced."""

    TOLD = "told"
    DEDUCED = "deduced"


@dataclass(frozen=True, slots=True)
class DirectEdge:
    """A direct subsumption c ⊑ d: no other named category m has c ⊑ m ⊑ d."""

    child: str
    parent: str
    status: EdgeStatus


@dataclass(frozen=True, slots=True)
class IndirectLink:
    """A told link from ``child`` to ``parent`` that passes through the categories ``via``.

    ``child`` is a category (a told c ⊂ d) or a drug (a told x ∈ d).
    """

    child: str
    parent: str
    via: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DrugMembership:
    """A membership x ∈ c of a drug in a named category."""

    drug: str
    category: str


@dataclass(frozen=True, slots=True)
class SameMembers:
    """Two categories with the same non-empty members in Cl(KB) that are not equivalent."""

    a: str
    b: str
    members: tuple[str, ...]
    a_subsumed_by_b: bool
    b_subsumed_by_a: bool


@dataclass(frozen=True, slots=True)
class DeducedTaxonomy:
    """The direct-parent graph of ⊑ and the most specific categories, against the told links.

    Parameters
    ----------
    direct_edges : tuple[DirectEdge, ...]
        The transitive reduction of ⊑ among the named categories, each edge told or deduced.
    told_edges_made_indirect : tuple[IndirectLink, ...]
        Told c ⊂ d that are not direct, with every named m such that c ⊑ m ⊑ d.
    redundant_told_edges : tuple[IndirectLink, ...]
        Told c ⊂ d implied by the other told ⊂ links alone, with the intermediate categories.
    equivalences : tuple[tuple[str, str], ...]
        Pairs c < d (identifier order) with c ⊑ d and d ⊑ c.
    members : Mapping[str, tuple[str, ...]]
        The drugs of each named category in Cl(KB).
    same_members_not_equivalent : tuple[SameMembers, ...]
        Pairs of categories with the same non-empty members that are not equivalent.
    most_specific_categories : Mapping[str, tuple[str, ...]]
        For each drug, the ⊑-minimal named categories it belongs to.
    memberships_made_indirect : tuple[IndirectLink, ...]
        Told x ∈ c with c not most specific for x, with the most specific m ⊑ c of x.
    new_direct_memberships : tuple[DrugMembership, ...]
        Most specific memberships that were not told.
    """

    direct_edges: tuple[DirectEdge, ...]
    told_edges_made_indirect: tuple[IndirectLink, ...]
    redundant_told_edges: tuple[IndirectLink, ...]
    equivalences: tuple[tuple[str, str], ...]
    members: Mapping[str, tuple[str, ...]]
    same_members_not_equivalent: tuple[SameMembers, ...]
    most_specific_categories: Mapping[str, tuple[str, ...]]
    memberships_made_indirect: tuple[IndirectLink, ...]
    new_direct_memberships: tuple[DrugMembership, ...]


# --- reasoning ----------------------------------------------------------------------------------


def reason(ontology: Ontology, *, incremental: bool = True) -> ReasonedOntology:
    """Translate the ontology into its KB and compute Cl(KB) by forward chaining.

    Parameters
    ----------
    ontology : Ontology
        The ontology.
    incremental : bool
        Passed to :func:`~symbolic_ai.p1_ej2_ontology.forward_chaining.fc_closure`.

    Returns
    -------
    ReasonedOntology
        The ontology, its KB and Cl(KB).
    """
    kb = to_knowledge_base(ontology)
    closure = fc_closure(kb, incremental=incremental)
    logger.info("Cl(KB): %d facts in %d iterations", len(closure.facts), closure.iterations)
    return ReasonedOntology(ontology=ontology, kb=kb, closure=closure)


def _categories_of(closure: Closure, symbol: Expr) -> tuple[str, ...]:
    answers = ask(closure, atom(MEMBER, symbol, _VARIABLE_Y))
    return tuple(sorted(identifier(theta[_VARIABLE_Y]) for theta in answers))


def _check_drug(ontology: Ontology, drug_id: str) -> None:
    if not ontology.is_drug(drug_id):
        raise UnknownIdentifierError(f"unknown drug {drug_id!r}")


def _check_category(ontology: Ontology, category_id: str) -> None:
    if not ontology.is_category(category_id) and ontology.defined(category_id) is None:
        raise UnknownIdentifierError(f"unknown category {category_id!r}")


def classify(reasoned: ReasonedOntology, drug_id: str) -> tuple[str, ...]:
    """Return every category, told or defined, that ``drug_id`` belongs to in Cl(KB).

    Raises
    ------
    UnknownIdentifierError
        If ``drug_id`` is not a drug of the ontology.
    """
    _check_drug(reasoned.ontology, drug_id)
    return _categories_of(reasoned.closure, drug_symbol(drug_id))


def drug_members(closure: Closure, category_id: str) -> tuple[str, ...]:
    """Return the drugs that belong to ``category_id`` in ``closure``, sorted."""
    answers = ask(closure, atom(MEMBER, _VARIABLE_X, category_symbol(category_id)))
    members = (theta[_VARIABLE_X] for theta in answers)
    return tuple(
        sorted(identifier(m) for m in members if str(m.op).startswith(_DRUG_CONSTANT_PREFIX))
    )


def clashes(closure: Closure) -> tuple[Clash, ...]:
    """Return every ``Incons`` atom of a closure; the KB is consistent iff there is none."""
    return tuple(
        Clash(identifier(fact.args[0]), identifier(fact.args[1]), identifier(fact.args[2]))
        for fact in closure.with_predicate(INCONS)
    )


def prototype_closure(reasoned: ReasonedOntology, category_id: str) -> Closure:
    """Return Cl(KB + {p_c ∈ c}) for the prototype p_c of a told or defined category.

    Raises
    ------
    UnknownIdentifierError
        If ``category_id`` is neither a told nor a defined category.
    """
    _check_category(reasoned.ontology, category_id)
    fact = atom(MEMBER, prototype_symbol(category_id), category_symbol(category_id))
    return fc_extend(reasoned.closure, reasoned.kb.rules, [fact])


def subsumes(reasoned: ReasonedOntology, sub: str, sup: str) -> bool:
    """Decide ``sub`` ⊑ ``sup`` (told or defined categories) by the prototype of ``sub``.

    Raises
    ------
    UnknownIdentifierError
        If either category is unknown.
    """
    _check_category(reasoned.ontology, sup)
    closure = prototype_closure(reasoned, sub)
    return atom(MEMBER, prototype_symbol(sub), category_symbol(sup)) in closure.facts


def taxonomy(reasoned: ReasonedOntology) -> Taxonomy:
    """Compute subsumption among the named categories, one prototype per category.

    Returns
    -------
    Taxonomy
        The proper subsumers of every named category and the told links among them.
    """
    ontology = reasoned.ontology
    named = ontology.named_categories
    named_set = set(named)
    subsumers: dict[str, tuple[str, ...]] = {}
    for category_id in named:
        closure = prototype_closure(reasoned, category_id)
        above = _categories_of(closure, prototype_symbol(category_id))
        subsumers[category_id] = tuple(c for c in above if c in named_set and c != category_id)
    told = frozenset(
        (e.category_id, e.parent_id)
        for e in ontology.data.subcategories
        if e.category_id in named_set and e.parent_id in named_set
    )
    return Taxonomy(categories=named, subsumers=subsumers, told=told)


# --- the deduced taxonomy -----------------------------------------------------------------------


def equivalences(result: Taxonomy) -> tuple[tuple[str, str], ...]:
    """Return every pair c < d (identifier order) with c ⊑ d and d ⊑ c, sorted."""
    return tuple((c, d) for c, d in result.pairs() if c < d and c in result.subsumers.get(d, ()))


def transitive_reduction(
    categories: Iterable[str], pairs: Iterable[tuple[str, str]]
) -> frozenset[tuple[str, str]]:
    """Return the direct pairs of a strict order: (c, d) with no m such that c ⊑ m ⊑ d.

    The reduction is unique only for a directed acyclic graph; ⊑ restricted to proper pairs is
    one exactly when no two categories are equivalent (Aho, Garey & Ullman, SIAM J. Comput. 1972).

    Parameters
    ----------
    categories : Iterable[str]
        The nodes of the order.
    pairs : Iterable[tuple[str, str]]
        The proper pairs (c, d), c ⊑ d and c ≠ d.

    Returns
    -------
    frozenset[tuple[str, str]]
        The edges of the transitive reduction, child to parent.

    Raises
    ------
    CyclicTaxonomyError
        If the pairs contain a cycle, i.e. two equivalent categories.
    """
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(categories)
    graph.add_edges_from(pairs)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        raise CyclicTaxonomyError(f"⊑ has a cycle (equivalent categories): {cycle}")
    reduced: nx.DiGraph[str] = nx.transitive_reduction(graph)
    return frozenset((str(c), str(d)) for c, d in reduced.edges)


def direct_edges(result: Taxonomy) -> tuple[DirectEdge, ...]:
    """Return the transitive reduction of ⊑ among the named categories, each told or deduced.

    Raises
    ------
    CyclicTaxonomyError
        If two named categories are equivalent.
    """
    reduced = transitive_reduction(result.categories, result.pairs())
    return tuple(
        DirectEdge(c, d, EdgeStatus.TOLD if (c, d) in result.told else EdgeStatus.DEDUCED)
        for c, d in sorted(reduced)
    )


def told_edges_made_indirect(result: Taxonomy) -> tuple[IndirectLink, ...]:
    """Return the told c ⊂ d that are not direct in ⊑, with each m such that c ⊑ m ⊑ d.

    Raises
    ------
    CyclicTaxonomyError
        If two named categories are equivalent.
    """
    reduced = transitive_reduction(result.categories, result.pairs())
    links = []
    for child, parent in sorted(result.told - reduced):
        via = sorted(
            m for m in result.subsumers[child] if m != parent and parent in result.subsumers[m]
        )
        links.append(IndirectLink(child, parent, tuple(via)))
    return tuple(links)


def redundant_told_edges(result: Taxonomy) -> tuple[IndirectLink, ...]:
    """Return the told c ⊂ d implied by the other told ⊂ links alone (no reasoning involved).

    ``via`` lists the categories m on a told path c ⊂ … ⊂ m ⊂ … ⊂ d that avoids the link itself.
    """
    told: nx.DiGraph[str] = nx.DiGraph(sorted(result.told))
    links = []
    for child, parent in sorted(result.told):
        told.remove_edge(child, parent)
        above = nx.descendants(told, child)
        if parent in above:
            via = sorted(m for m in above if parent in nx.descendants(told, m))
            links.append(IndirectLink(child, parent, tuple(via)))
        told.add_edge(child, parent)
    return tuple(links)


def category_members(reasoned: ReasonedOntology) -> dict[str, tuple[str, ...]]:
    """Return the drugs of every named category in Cl(KB), defined categories included."""
    return {
        c: drug_members(reasoned.closure, c) for c in sorted(reasoned.ontology.named_categories)
    }


def same_members_not_equivalent(
    result: Taxonomy, members: Mapping[str, tuple[str, ...]]
) -> tuple[SameMembers, ...]:
    """Return the pairs a < b with the same non-empty members that are not equivalent.

    Classification looks at the current members only; subsumption holds for every possible member,
    so two categories can agree on the first and still differ on the second.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for category_id in sorted(members):
        if members[category_id]:
            groups.setdefault(members[category_id], []).append(category_id)
    rows = []
    for shared, categories in groups.items():
        for a, b in itertools.combinations(categories, 2):
            a_below_b = b in result.subsumers[a]
            b_below_a = a in result.subsumers[b]
            if not (a_below_b and b_below_a):
                rows.append(SameMembers(a, b, shared, a_below_b, b_below_a))
    return tuple(sorted(rows, key=lambda r: (r.a, r.b)))


def most_specific_categories(
    result: Taxonomy, members: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Return, for each drug, the ⊑-minimal named categories among those it belongs to.

    Parameters
    ----------
    result : Taxonomy
        Subsumption among the named categories.
    members : Mapping[str, tuple[str, ...]]
        The drugs of each named category (:func:`category_members`).

    Returns
    -------
    dict[str, tuple[str, ...]]
        Drug -> its categories c with no other category m of the drug such that m ⊑ c, sorted.
    """
    categories_of: dict[str, set[str]] = {}
    for category_id, drugs in members.items():
        for drug_id in drugs:
            categories_of.setdefault(drug_id, set()).add(category_id)
    return {
        drug_id: tuple(
            sorted(
                c
                for c in categories_of[drug_id]
                if not any(c in result.subsumers[m] for m in categories_of[drug_id] - {c})
            )
        )
        for drug_id in sorted(categories_of)
    }


def membership_changes(
    result: Taxonomy,
    most_specific: Mapping[str, tuple[str, ...]],
    told: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[IndirectLink, ...], tuple[DrugMembership, ...]]:
    """Compare each drug's most specific categories with its told memberships.

    Parameters
    ----------
    result : Taxonomy
        Subsumption among the named categories.
    most_specific : Mapping[str, tuple[str, ...]]
        Drug -> its most specific categories (:func:`most_specific_categories`).
    told : Mapping[str, tuple[str, ...]]
        Drug -> the categories of its told memberships.

    Returns
    -------
    tuple[tuple[IndirectLink, ...], tuple[DrugMembership, ...]]
        The told memberships that are no longer most specific, each with the most specific
        categories below the told one; and the most specific memberships that were not told.
    """
    made_indirect = []
    new_direct = []
    for drug_id in sorted(most_specific):
        minimal = most_specific[drug_id]
        told_here = told.get(drug_id, ())
        for category_id in sorted(set(told_here) - set(minimal)):
            via = tuple(m for m in minimal if category_id in result.subsumers[m])
            made_indirect.append(IndirectLink(drug_id, category_id, via))
        new_direct += [DrugMembership(drug_id, c) for c in minimal if c not in told_here]
    return tuple(made_indirect), tuple(new_direct)


def deduced_taxonomy(reasoned: ReasonedOntology, result: Taxonomy) -> DeducedTaxonomy:
    """Compute the deduced taxonomy of the named categories and compare it with the told one.

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.
    result : Taxonomy
        Subsumption among the named categories (:func:`taxonomy`).

    Returns
    -------
    DeducedTaxonomy
        Direct edges, told links made indirect or redundant, equivalences, members and most
        specific categories.

    Raises
    ------
    CyclicTaxonomyError
        If two named categories are equivalent: the direct-parent graph is then not unique.
    """
    members = category_members(reasoned)
    most_specific = most_specific_categories(result, members)
    ontology = reasoned.ontology
    told = {d: ontology.told_categories_of(d) for d in ontology.drug_ids}
    made_indirect, new_direct = membership_changes(result, most_specific, told)
    return DeducedTaxonomy(
        direct_edges=direct_edges(result),
        told_edges_made_indirect=told_edges_made_indirect(result),
        redundant_told_edges=redundant_told_edges(result),
        equivalences=equivalences(result),
        members=members,
        same_members_not_equivalent=same_members_not_equivalent(result, members),
        most_specific_categories=most_specific,
        memberships_made_indirect=made_indirect,
        new_direct_memberships=new_direct,
    )


def _object_ids(facts: Iterable[Expr], position: int, predicate: str, prefix: str) -> list[str]:
    return [
        identifier(f.args[position])
        for f in facts
        if f.op == predicate and str(f.args[position].op).startswith(prefix)
    ]


def inherited_by_new_member(reasoned: ReasonedOntology, category_id: str) -> Inheritance:
    """Return the formulary rows the prototype of ``category_id`` derives (AIMA's inheritance).

    Raises
    ------
    UnknownIdentifierError
        If ``category_id`` is neither a told nor a defined category.
    """
    closure = prototype_closure(reasoned, category_id)
    prototype = prototype_symbol(category_id)
    about = [f for f in closure.facts if f.args and f.args[0] == prototype]
    categories = _categories_of(closure, prototype)
    families = [c for c in reasoned.ontology.data.families if c in categories]
    family_pairs = [
        (family, drug_id)
        for family in families
        for drug_id in drug_members(reasoned.closure, family)
    ]
    coprescriptions = [(identifier(f.args[1]), identifier(f.args[2])) for f in about if f.op == COP]
    return Inheritance(
        category_id=category_id,
        categories=categories,
        candidates=tuple(sorted(_object_ids(about, 1, "Trata", "H_"))),
        contraindications=tuple(sorted(_object_ids(about, 1, "CI", "R_"))),
        interactions=tuple(sorted(_object_ids(about, 1, INT, _DRUG_CONSTANT_PREFIX))),
        coprescriptions=tuple(sorted(coprescriptions)),
        family_pairs=tuple(sorted(family_pairs)),
        clashes=clashes(closure),
    )


def add_membership(reasoned: ReasonedOntology, drug_id: str, category_id: str) -> Closure:
    """Return Cl(KB + {d ∈ K}): the closure after telling one extra membership (a mutation).

    Raises
    ------
    UnknownIdentifierError
        If the drug or the category is unknown.
    """
    _check_drug(reasoned.ontology, drug_id)
    _check_category(reasoned.ontology, category_id)
    fact = atom(MEMBER, drug_symbol(drug_id), category_symbol(category_id))
    return fc_extend(reasoned.closure, reasoned.kb.rules, [fact])


def partition_coverage(reasoned: ReasonedOntology) -> tuple[PartitionCoverage, ...]:
    """Check, closed-world, that every told object of a partitioned category is in one part.

    The exhaustive half of a partition is a disjunction, not a definite clause, so forward chaining
    cannot use it; it is checked on the objects of Cl(KB) instead (``README.md`` §4.4).

    Returns
    -------
    tuple[PartitionCoverage, ...]
        One record per partition, sorted by set identifier.
    """
    closure = reasoned.closure
    results = []
    for disjoint_set in reasoned.ontology.data.disjoint_sets:
        if disjoint_set.partition_of is None:
            continue
        parent = category_symbol(disjoint_set.partition_of)
        answers = ask(closure, atom(MEMBER, _VARIABLE_X, parent))
        members = sorted((theta[_VARIABLE_X] for theta in answers), key=str)
        uncovered = [
            m
            for m in members
            if not any(
                atom(MEMBER, m, category_symbol(part)) in closure.facts
                for part in disjoint_set.category_ids
            )
        ]
        results.append(
            PartitionCoverage(
                set_id=disjoint_set.set_id,
                parent_id=disjoint_set.partition_of,
                parts=disjoint_set.category_ids,
                members=tuple(str(m.op) for m in members),
                uncovered=tuple(str(m.op) for m in uncovered),
            )
        )
    return tuple(results)


# --- provenance of derived rows -----------------------------------------------------------------


def _covers(reasoned: ReasonedOntology, subject_id: str, drug_id: str) -> bool:
    """Whether a link written on ``subject_id`` applies to ``drug_id`` in Cl(KB)."""
    if reasoned.ontology.is_drug(subject_id):
        return subject_id == drug_id
    fact = atom(MEMBER, drug_symbol(drug_id), category_symbol(subject_id))
    return fact in reasoned.closure.facts


def _level(reasoned: ReasonedOntology, subject_id: str) -> LinkLevel:
    return LinkLevel.OBJECT if reasoned.ontology.is_drug(subject_id) else LinkLevel.CATEGORY


def candidate_sources(
    reasoned: ReasonedOntology, condition_id: str, drug_id: str
) -> tuple[Source, ...]:
    """Return the told indication links that make ``drug_id`` a candidate for ``condition_id``."""
    return tuple(
        Source(link.subject_id, _level(reasoned, link.subject_id), "")
        for link in reasoned.ontology.data.indications
        if link.condition_id == condition_id and _covers(reasoned, link.subject_id, drug_id)
    )


def contraindication_sources(
    reasoned: ReasonedOntology, risk_factor_id: str, drug_id: str
) -> tuple[Source, ...]:
    """Return the told contraindication links that contraindicate ``drug_id`` under a factor."""
    return tuple(
        Source(link.subject_id, _level(reasoned, link.subject_id), link.reason)
        for link in reasoned.ontology.data.contraindications
        if link.risk_factor_id == risk_factor_id and _covers(reasoned, link.subject_id, drug_id)
    )


def interaction_sources(reasoned: ReasonedOntology, drug_a: str, drug_b: str) -> tuple[Source, ...]:
    """Return the told interaction links whose two subjects cover the pair, in either order."""
    sources = []
    for link in reasoned.ontology.data.interactions:
        forward = _covers(reasoned, link.subject_a, drug_a) and _covers(
            reasoned, link.subject_b, drug_b
        )
        backward = _covers(reasoned, link.subject_a, drug_b) and _covers(
            reasoned, link.subject_b, drug_a
        )
        if forward or backward:
            subject = f"{link.subject_a}+{link.subject_b}"
            sources.append(
                Source(subject, _level(reasoned, link.subject_a), f"{link.effect}|{link.severity}")
            )
    return tuple(sources)


def coprescription_sources(
    reasoned: ReasonedOntology, drug_id: str, risk_factor_id: str, companion_id: str
) -> tuple[Source, ...]:
    """Return the told coprescription links that require ``companion_id`` for ``drug_id``."""
    return tuple(
        Source(link.subject_id, _level(reasoned, link.subject_id), link.reason)
        for link in reasoned.ontology.data.coprescriptions
        if link.risk_factor_id == risk_factor_id
        and link.companion_drug_id == companion_id
        and _covers(reasoned, link.subject_id, drug_id)
    )


def family_sources(reasoned: ReasonedOntology, drug_a: str, drug_b: str) -> tuple[Source, ...]:
    """Return the families (told ``Familia(c)``) that contain both drugs in Cl(KB)."""
    return tuple(
        Source(family, LinkLevel.CATEGORY, "")
        for family in reasoned.ontology.data.families
        if _covers(reasoned, family, drug_a) and _covers(reasoned, family, drug_b)
    )


# --- the derived formulary ----------------------------------------------------------------------


def _joined(texts: Iterable[str]) -> str:
    return "; ".join(sorted(set(texts)))


def _derived_contraindications(reasoned: ReasonedOntology) -> tuple[Contraindication, ...]:
    rows = []
    for defined in reasoned.ontology.defined_categories:
        if defined.kind is not DefinedKind.CONTRAINDICATED:
            continue
        for drug_id in drug_members(reasoned.closure, defined.category_id):
            sources = contraindication_sources(reasoned, defined.target_id, drug_id)
            reason_text = _joined(s.text for s in sources)
            rows.append(Contraindication(defined.target_id, drug_id, reason_text))
    return tuple(sorted(rows, key=lambda c: (c.risk_factor_id, c.drug_id)))


def _derived_interactions(reasoned: ReasonedOntology) -> tuple[AdverseInteraction, ...]:
    rows = []
    for fact in reasoned.closure.with_predicate(INT):
        a, b = identifier(fact.args[0]), identifier(fact.args[1])
        if not (reasoned.ontology.is_drug(a) and reasoned.ontology.is_drug(b)) or not a < b:
            continue
        sources = interaction_sources(reasoned, a, b)
        effect = _joined(s.text.split("|")[0] for s in sources)
        severity = _joined(s.text.split("|")[1] for s in sources)
        rows.append(AdverseInteraction(a, b, effect, severity))
    return tuple(sorted(rows, key=lambda i: (i.drug_a, i.drug_b)))


def _derived_coprescriptions(reasoned: ReasonedOntology) -> tuple[Coprescription, ...]:
    rows = []
    for fact in reasoned.closure.with_predicate(COP):
        drug_id = identifier(fact.args[0])
        if not reasoned.ontology.is_drug(drug_id):
            continue
        risk_factor_id, companion_id = identifier(fact.args[1]), identifier(fact.args[2])
        sources = coprescription_sources(reasoned, drug_id, risk_factor_id, companion_id)
        reason_text = _joined(s.text for s in sources)
        rows.append(Coprescription(drug_id, risk_factor_id, companion_id, reason_text))
    return tuple(sorted(rows, key=lambda c: (c.drug_id, c.risk_factor_id, c.companion_drug_id)))


def _derived_candidates(reasoned: ReasonedOntology) -> dict[str, frozenset[str]]:
    candidates: dict[str, frozenset[str]] = {}
    for defined in reasoned.ontology.defined_categories:
        if defined.kind is DefinedKind.CANDIDATE:
            members = drug_members(reasoned.closure, defined.category_id)
            if members:
                candidates[defined.target_id] = frozenset(members)
    return {condition_id: candidates[condition_id] for condition_id in sorted(candidates)}


def _derived_families(
    reasoned: ReasonedOntology,
) -> tuple[tuple[DrugClass, ...], dict[str, frozenset[str]]]:
    families = sorted(identifier(fact.args[0]) for fact in reasoned.closure.with_predicate(FAMILIA))
    classes = []
    members: dict[str, frozenset[str]] = {}
    for family in families:
        category = reasoned.ontology.category(family)
        classes.append(
            DrugClass(
                class_id=family,
                name_es=category.name_es,
                name_en=category.name_en,
                atc_code=category.atc_code,
                exclusive=True,
            )
        )
        members[family] = frozenset(drug_members(reasoned.closure, family))
    return tuple(classes), members


def derive_formulary(reasoned: ReasonedOntology) -> Formulary:
    """Read EJ1's drug-level formulary off Cl(KB) (``README.md`` §5.3).

    Candidates come from the defined categories ``candidate_<h>``, contraindications from
    ``contraindicated_<r>``, interactions from ``Int`` (unordered, ``drug_a < drug_b``),
    coprescriptions from ``Cop``, and the families (A6) from ``Familia`` with inherited members.
    Reasons and effects are those of the told links each row comes from.

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.

    Returns
    -------
    Formulary
        The formulary EJ1 consumes, with version ``ontology@<data version>``.
    """
    data = reasoned.ontology.data
    classes, class_members = _derived_families(reasoned)
    return Formulary(
        version=f"ontology@{data.version}",
        conditions=data.conditions,
        drugs=data.drugs,
        risk_factors=data.risk_factors,
        candidates=_derived_candidates(reasoned),
        interactions=_derived_interactions(reasoned),
        contraindications=_derived_contraindications(reasoned),
        coprescriptions=_derived_coprescriptions(reasoned),
        drug_classes=classes,
        class_members=class_members,
    )


def family_pairs(formulary: Formulary) -> tuple[tuple[str, str], ...]:
    """Return every unordered pair of drugs that share an exclusive class (EJ1's A6), sorted."""
    pairs = {
        pair
        for family in formulary.exclusive_families()
        for pair in itertools.combinations(sorted(family), 2)
    }
    return tuple(sorted(pairs))
