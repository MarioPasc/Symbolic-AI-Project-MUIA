"""EJ2 experiments of the report's *Resultados*: P3 (derived formulary) and P4 (DL tasks).

P3 derives EJ1's formulary and compares it with 1.0.0; P4 covers the taxonomy, what a new member
inherits, consistency (with the mutation test) and the cost of the fixed point. Every function is
deterministic and returns frozen dataclasses; ``main.py`` loads the data, calls them and writes
``results.json``. Given HermiT's runs, each answer is also compared with the oracle on the OWL
export (:mod:`symbolic_ai.p1_ej2_ontology.owl_oracle`), which shares only the data tables with
forward chaining. The expected label of every mutation is computed by reachability over the told
subcategory and definition edges (networkx), independently of forward chaining.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

from aima.logic import is_variable, parse_definite_clause
from aima.utils import Expr
from symbolic_ai.dataloader.models import Formulary, OntologyData
from symbolic_ai.p1_ej2_ontology import owl_oracle
from symbolic_ai.p1_ej2_ontology.errors import CyclicTaxonomyError
from symbolic_ai.p1_ej2_ontology.forward_chaining import DISTINCT, fc_closure
from symbolic_ai.p1_ej2_ontology.ontology import (
    SUBSET,
    Ontology,
    identifier,
    tagged_rules,
)
from symbolic_ai.p1_ej2_ontology.reasoner import (
    Clash,
    DirectEdge,
    DrugMembership,
    IndirectLink,
    ReasonedOntology,
    SameMembers,
    Source,
    add_membership,
    candidate_sources,
    clashes,
    classify,
    contraindication_sources,
    coprescription_sources,
    deduced_taxonomy,
    derive_formulary,
    drug_members,
    family_pairs,
    family_sources,
    inherited_by_new_member,
    interaction_sources,
    partition_coverage,
    taxonomy,
    transitive_reduction,
)

__all__ = [
    "MUTATION_SAMPLE_SIZE",
    "ConsistencyResult",
    "DerivedRow",
    "FixedPointResult",
    "FormularyDerivationResult",
    "InheritanceResult",
    "InheritanceRow",
    "MutationRow",
    "OracleAgreement",
    "PartitionRow",
    "TableComparison",
    "TaxonomyResult",
    "as_jsonable",
    "expected_clash_sets",
    "mutation_candidates",
    "run_consistency",
    "run_fixed_point_cost",
    "run_formulary_derivation",
    "run_inheritance",
    "run_taxonomy",
    "told_knowledge",
]

#: Mutations of each kind (clash expected / not expected) re-checked by a real HermiT run each.
MUTATION_SAMPLE_SIZE = 20

_Key = tuple[str, ...]


# --- result records -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleAgreement:
    """How many items were compared with HermiT, and every disagreement found (empty = agree)."""

    compared: int
    disagreements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableComparison:
    """P3: one EJ1 table, hand-written (1.0.0) against derived from the ontology.

    For the families, rows are the unordered pairs of drugs that share a family (A6), and the
    class counts are given separately.
    """

    table: str
    rows_reference: int
    statements: int
    statements_on_categories: int
    statements_on_objects: int
    derived: int
    in_both: int
    only_derived: tuple[_Key, ...]
    only_reference: tuple[_Key, ...]
    classes_reference: int | None = None
    classes_derived: int | None = None


@dataclass(frozen=True, slots=True)
class DerivedRow:
    """P3: one derived row with the told statements it comes from.

    ``also_family_pair`` is set on interaction rows only: whether the two drugs also share an
    exclusive class in the reference formulary (EJ1's A6). EJ1 encodes A3 and A6 with the same
    clause ``~T_a | ~T_b``, so such a pair adds no new constraint to EJ1.
    """

    table: str
    key: _Key
    in_reference: bool
    sources: tuple[str, ...]
    level: str
    oracle_agrees: bool | None
    also_family_pair: bool | None = None


@dataclass(frozen=True, slots=True)
class FormularyDerivationResult:
    """P3: the derived formulary compared with the hand-written one, table by table.

    ``defined_category_members`` lists, for each category defined by conjuncts, the drugs Cl(KB)
    classifies into it (none is told there).
    """

    derived_version: str
    reference_version: str
    tables: tuple[TableComparison, ...]
    rows: tuple[DerivedRow, ...]
    oracle_memberships: OracleAgreement | None
    oracle_rows: OracleAgreement | None
    defined_category_members: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class TaxonomyResult:
    """P4.1: subsumption among the named categories (proper pairs, reflexive ones excluded).

    The defined categories are the 12 of ``README.md`` §4.5 and the ones defined by conjuncts
    (``defined_by_conjuncts``); every other drug category is primitive. The fields from
    ``direct_edges`` on are the deduced taxonomy (:class:`~.reasoner.DeducedTaxonomy`, Fig. 3);
    ``oracle_direct_edges_agree`` says whether the transitive reduction of HermiT's classified
    hierarchy has the same edges (``None`` without the oracle).
    """

    n_primitive: int
    n_defined: int
    defined_by_conjuncts: tuple[str, ...]
    told_pairs: int
    primitive_pairs: int
    primitive_to_defined: int
    defined_to_primitive: int
    defined_to_defined: int
    total_pairs: int
    subsumed_by_defined: Mapping[str, tuple[str, ...]]
    pairs: tuple[tuple[str, str], ...]
    oracle_classification: OracleAgreement | None
    oracle_prototypes: OracleAgreement | None
    direct_edges: tuple[DirectEdge, ...]
    told_edges_made_indirect: tuple[IndirectLink, ...]
    redundant_told_edges: tuple[IndirectLink, ...]
    equivalences: tuple[tuple[str, str], ...]
    members: Mapping[str, tuple[str, ...]]
    same_members_not_equivalent: tuple[SameMembers, ...]
    most_specific_categories: Mapping[str, tuple[str, ...]]
    memberships_made_indirect: tuple[IndirectLink, ...]
    new_direct_memberships: tuple[DrugMembership, ...]
    oracle_direct_edges_agree: bool | None


@dataclass(frozen=True, slots=True)
class InheritanceRow:
    """P4.2: the rows a new member of one drug category would receive."""

    category_id: str
    candidates: tuple[str, ...]
    contraindications: tuple[str, ...]
    interactions: tuple[str, ...]
    coprescriptions: tuple[str, ...]
    family_pairs: tuple[str, ...]
    rows: int


@dataclass(frozen=True, slots=True)
class InheritanceResult:
    """P4.2: inheritance by a prototype of each of the drug categories."""

    rows: tuple[InheritanceRow, ...]
    min_rows: int
    max_rows: int
    oracle: OracleAgreement | None


@dataclass(frozen=True, slots=True)
class PartitionRow:
    """P4.3(c): the closed-world exhaustiveness check of one partition."""

    set_id: str
    parent_id: str
    parts: tuple[str, ...]
    members: int
    uncovered: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MutationRow:
    """P4.3(d): one mutation d ∈ K with its expected label and both reasoners' verdicts.

    ``expected_by`` lists the disjoint sets that make a clash expected (networkx reachability);
    ``detected`` lists the ``Incons`` atoms forward chaining derives; ``oracle_clash`` is HermiT's
    verdict (single run); ``oracle_rechecked`` the verdict of a real per-mutation run when this
    mutation is in the re-checked sample.
    """

    drug_id: str
    category_id: str
    expected_clash: bool
    expected_by: tuple[str, ...]
    detected: tuple[str, ...]
    oracle_clash: bool | None
    oracle_rechecked: bool | None


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """P4.3: consistency of the KB, of each named category, partitions and mutations.

    ``n_clash_expected_from_upper`` counts the expected clashes due to a disjoint set named
    ``upper``; it is kept for the schema and is 0 since the upper ontology was removed (R3a).
    ``n_clash_expected_by_set`` counts the expected clashes by disjoint set (a mutation may be
    counted under several sets).
    """

    kb_clashes: tuple[str, ...]
    oracle_kb_consistent: bool | None
    categories_checked: int
    inconsistent_categories: tuple[str, ...]
    oracle_unsatisfiable_classes: tuple[str, ...] | None
    partitions: tuple[PartitionRow, ...]
    n_mutations: int
    n_clash_expected: int
    n_clash_expected_from_upper: int
    n_clash_expected_by_set: Mapping[str, int]
    n_clash_detected: int
    n_no_clash_expected: int
    n_false_alarms: int
    oracle_single_run: OracleAgreement | None
    oracle_rechecked: OracleAgreement | None
    mutations: tuple[MutationRow, ...]


@dataclass(frozen=True, slots=True)
class FixedPointResult:
    """P4.4: the size of Cl(KB) and the iterations of forward chaining, against AIMA's p·n^k."""

    told_facts: int
    rules: int
    closure_facts: int
    derived_facts: int
    iterations: int
    new_per_iteration: tuple[int, ...]
    facts_per_predicate: Mapping[str, int]
    predicates: int
    constants: int
    max_arity: int
    bound: int
    naive_equals_incremental: bool
    subset_closure_matches_networkx: bool


# --- helpers ------------------------------------------------------------------------------------


def _agreement(
    compared: int, left: Iterable[_Key], right: Iterable[_Key], label: str
) -> OracleAgreement:
    """Compare two key sets: forward chaining (left) with HermiT (right)."""
    left_set, right_set = set(left), set(right)
    disagreements = [f"{label} only in forward chaining: {k}" for k in sorted(left_set - right_set)]
    disagreements += [f"{label} only in HermiT: {k}" for k in sorted(right_set - left_set)]
    return OracleAgreement(compared=compared, disagreements=tuple(disagreements))


def _merge(agreements: Sequence[OracleAgreement]) -> OracleAgreement:
    return OracleAgreement(
        compared=sum(a.compared for a in agreements),
        disagreements=tuple(d for a in agreements for d in a.disagreements),
    )


def _level(sources: Sequence[Source]) -> str:
    levels = sorted({s.level.value for s in sources})
    return levels[0] if len(levels) == 1 else "+".join(levels)


def _subcategory_graph(data: OntologyData) -> nx.DiGraph[str]:
    """Return the told subcategory links, child -> parent (networkx; independent of FC)."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(c.category_id for c in data.categories)
    graph.add_edges_from((e.category_id, e.parent_id) for e in data.subcategories)
    return graph


def _reachability_graph(data: OntologyData) -> nx.DiGraph[str]:
    """Return the subcategory links plus one parent edge D -> k per conjunct k of a definition."""
    graph = _subcategory_graph(data)
    graph.add_edges_from((d.category_id, k) for d in data.definitions for k in d.conjunct_ids)
    return graph


def _up(graph: nx.DiGraph[str], category_id: str) -> set[str]:
    """Return the category and every category above it in ``graph``."""
    return {category_id} | set(nx.descendants(graph, category_id))


# --- P3: the derived formulary ------------------------------------------------------------------


def _formulary_keys(formulary: Formulary) -> dict[str, set[_Key]]:
    return {
        "candidates": {(h, d) for h, drugs in formulary.candidates.items() for d in drugs},
        "contraindications": {(c.risk_factor_id, c.drug_id) for c in formulary.contraindications},
        "interactions": {(i.drug_a, i.drug_b) for i in formulary.interactions},
        "coprescriptions": {
            (c.drug_id, c.risk_factor_id, c.companion_drug_id) for c in formulary.coprescriptions
        },
        "families": set(family_pairs(formulary)),
    }


def _statement_counts(reasoned: ReasonedOntology) -> dict[str, tuple[int, int]]:
    """(on categories, on objects) told statements per EJ1 table."""
    data = reasoned.ontology.data

    def split(subjects: Iterable[str]) -> tuple[int, int]:
        listed = list(subjects)
        on_categories = sum(reasoned.ontology.is_category(s) for s in listed)
        return on_categories, len(listed) - on_categories

    return {
        "candidates": split(link.subject_id for link in data.indications),
        "contraindications": split(link.subject_id for link in data.contraindications),
        "interactions": split(link.subject_a for link in data.interactions),
        "coprescriptions": split(link.subject_id for link in data.coprescriptions),
        "families": (len(data.families), 0),
    }


def _sources_of(reasoned: ReasonedOntology, table: str, key: _Key) -> tuple[Source, ...]:
    finders: dict[str, Callable[..., tuple[Source, ...]]] = {
        "candidates": candidate_sources,
        "contraindications": contraindication_sources,
        "interactions": interaction_sources,
        "coprescriptions": coprescription_sources,
        "families": family_sources,
    }
    return finders[table](reasoned, *key)


def _oracle_formulary_keys(run: owl_oracle.OracleRun, data: OntologyData) -> dict[str, set[_Key]]:
    """Read EJ1's tables off HermiT's inferences, as :func:`derive_formulary` does off Cl(KB)."""
    views = {d.drug_id: run.individuals["D_" + d.drug_id] for d in data.drugs}
    families = set(data.families)
    family_members = {f: sorted(d for d, v in views.items() if f in v.categories) for f in families}
    return {
        "candidates": {
            (h, d)
            for d, v in views.items()
            for h in (c.condition_id for c in data.conditions)
            if f"candidate_{h}" in v.categories
        },
        "contraindications": {
            (r, d)
            for d, v in views.items()
            for r in (f.risk_factor_id for f in data.risk_factors)
            if f"contraindicated_{r}" in v.categories
        },
        "interactions": {
            (a, identifier_name[2:])
            for a, v in views.items()
            for identifier_name in v.interacts
            if identifier_name.startswith("D_") and a < identifier_name[2:]
        },
        "coprescriptions": {(d, r, e) for d, v in views.items() for r, e in v.requires},
        "families": {
            pair
            for members in family_members.values()
            for pair in itertools.combinations(members, 2)
        },
    }


def _oracle_property_keys(
    reasoned: ReasonedOntology, run: owl_oracle.OracleRun
) -> tuple[set[_Key], set[_Key]]:
    """Drug property values (Trata, CI, Int, Cop) by forward chaining and by HermiT."""
    fc_keys: set[_Key] = set()
    for fact in reasoned.closure.facts:
        if fact.op in ("Trata", "CI", "Int", "Cop") and str(fact.args[0].op).startswith("D_"):
            fc_keys.add((str(fact.op), *(identifier(a) for a in fact.args)))
    oracle_keys: set[_Key] = set()
    for drug_id in reasoned.ontology.drug_ids:
        view = run.individuals["D_" + drug_id]
        oracle_keys |= {("Trata", drug_id, h) for h in view.treats}
        oracle_keys |= {("CI", drug_id, r) for r in view.contraindicated_by}
        oracle_keys |= {("Int", drug_id, b[2:]) for b in view.interacts if b.startswith("D_")}
        oracle_keys |= {("Cop", drug_id, r, e) for r, e in view.requires}
    return fc_keys, oracle_keys


def run_formulary_derivation(
    reasoned: ReasonedOntology, reference: Formulary, *, oracle_run: owl_oracle.OracleRun | None
) -> FormularyDerivationResult:
    """P3: derive EJ1's formulary from Cl(KB) and compare it with ``reference``, table by table.

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.
    reference : Formulary
        The hand-written formulary (``load_formulary(version="1.0.0")``).
    oracle_run : owl_oracle.OracleRun | None
        HermiT's base run on the same data, or ``None`` to skip the comparison.

    Returns
    -------
    FormularyDerivationResult
        Per-table counts, every derived row with its provenance (an interaction row also says
        whether its pair is an A6 family pair of ``reference``), and HermiT's agreement.
    """
    derived = derive_formulary(reasoned)
    derived_keys, reference_keys = _formulary_keys(derived), _formulary_keys(reference)
    reference_family_pairs = reference_keys["families"]
    statements = _statement_counts(reasoned)
    oracle_keys = _oracle_formulary_keys(oracle_run, reasoned.ontology.data) if oracle_run else None
    tables, rows = [], []
    for table, keys in derived_keys.items():
        on_categories, on_objects = statements[table]
        tables.append(
            TableComparison(
                table=table,
                rows_reference=len(reference_keys[table]),
                statements=on_categories + on_objects,
                statements_on_categories=on_categories,
                statements_on_objects=on_objects,
                derived=len(keys),
                in_both=len(keys & reference_keys[table]),
                only_derived=tuple(sorted(keys - reference_keys[table])),
                only_reference=tuple(sorted(reference_keys[table] - keys)),
                classes_reference=len(reference.drug_classes) if table == "families" else None,
                classes_derived=len(derived.drug_classes) if table == "families" else None,
            )
        )
        for key in sorted(keys):
            sources = _sources_of(reasoned, table, key)
            rows.append(
                DerivedRow(
                    table=table,
                    key=key,
                    in_reference=key in reference_keys[table],
                    sources=tuple(s.subject_id for s in sources),
                    level=_level(sources),
                    oracle_agrees=None if oracle_keys is None else key in oracle_keys[table],
                    also_family_pair=(
                        key in reference_family_pairs if table == "interactions" else None
                    ),
                )
            )
    oracle_memberships = oracle_rows = None
    if oracle_run is not None and oracle_keys is not None:
        oracle_memberships = _membership_agreement(reasoned, oracle_run)
        row_checks = [
            _agreement(len(derived_keys[t] | oracle_keys[t]), derived_keys[t], oracle_keys[t], t)
            for t in derived_keys
        ]
        fc_properties, oracle_properties = _oracle_property_keys(reasoned, oracle_run)
        row_checks.append(
            _agreement(
                len(fc_properties | oracle_properties),
                fc_properties,
                oracle_properties,
                "property value",
            )
        )
        oracle_rows = _merge(row_checks)
    return FormularyDerivationResult(
        derived_version=derived.version,
        reference_version=reference.version,
        tables=tuple(tables),
        rows=tuple(rows),
        oracle_memberships=oracle_memberships,
        oracle_rows=oracle_rows,
        defined_category_members={
            c: drug_members(reasoned.closure, c) for c in reasoned.ontology.conjunctive_ids
        },
    )


def _membership_agreement(reasoned: ReasonedOntology, run: owl_oracle.OracleRun) -> OracleAgreement:
    """Compare every drug's classification (told and defined categories) with HermiT's."""
    ontology = reasoned.ontology
    universe = len(ontology.category_ids) + len(ontology.defined_ids)
    fc = {(d, c) for d in ontology.drug_ids for c in classify(reasoned, d)}
    oracle = {(d, c) for d in ontology.drug_ids for c in run.individuals["D_" + d].categories}
    return _agreement(len(ontology.drug_ids) * universe, fc, oracle, "membership")


# --- P4.1: taxonomy -----------------------------------------------------------------------------


def run_taxonomy(
    reasoned: ReasonedOntology,
    *,
    oracle_run: owl_oracle.OracleRun | None,
    oracle_prototypes: owl_oracle.OracleRun | None,
) -> TaxonomyResult:
    """P4.1: proper subsumption pairs among the named categories, decided by prototypes.

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.
    oracle_run : owl_oracle.OracleRun | None
        HermiT's base run (its classified class hierarchy), or ``None``.
    oracle_prototypes : owl_oracle.OracleRun | None
        HermiT's run with one prototype individual per named category, or ``None``.

    Returns
    -------
    TaxonomyResult
        Pair counts by kind, the pairs, HermiT's agreement on every cell of the named x named
        matrix, and the deduced taxonomy (direct edges, told links made indirect, members, most
        specific categories). A category defined by conjuncts counts as defined, not primitive.

    Raises
    ------
    CyclicTaxonomyError
        If forward chaining finds two named categories equivalent (no unique direct edges).
    """
    ontology = reasoned.ontology
    result = taxonomy(reasoned)
    deduced = deduced_taxonomy(reasoned, result)
    primitive = set(ontology.primitive_categories)
    defined = set(ontology.defined_ids) | set(ontology.conjunctive_ids)
    pairs = result.pairs()

    def count(sub: set[str], sup: set[str]) -> int:
        return sum(c in sub and d in sup for c, d in pairs)

    named = list(ontology.named_categories)
    cells = len(named) * len(named)
    oracle_classification = oracle_by_prototypes = None
    oracle_direct_agree = None
    if oracle_run is not None:
        hermit = {(c, d) for c in named for d in oracle_run.subsumers[c] if d in set(named)}
        oracle_classification = _agreement(cells, pairs, hermit, "subsumption")
        oracle_direct_agree = _same_direct_edges(named, hermit, deduced.direct_edges)
    if oracle_prototypes is not None:
        by_prototype = {
            (c, d)
            for c in named
            for d in oracle_prototypes.individuals["P_" + c].categories
            if d in set(named) and d != c
        }
        oracle_by_prototypes = _agreement(cells, pairs, by_prototype, "prototype subsumption")
    return TaxonomyResult(
        n_primitive=len(primitive),
        n_defined=len(defined),
        defined_by_conjuncts=ontology.conjunctive_ids,
        told_pairs=len(result.told),
        primitive_pairs=count(primitive, primitive),
        primitive_to_defined=count(primitive, defined),
        defined_to_primitive=count(defined, primitive),
        defined_to_defined=count(defined, defined),
        total_pairs=len(pairs),
        subsumed_by_defined={
            d: tuple(sorted(c for c, sup in pairs if sup == d)) for d in sorted(defined)
        },
        pairs=pairs,
        oracle_classification=oracle_classification,
        oracle_prototypes=oracle_by_prototypes,
        direct_edges=deduced.direct_edges,
        told_edges_made_indirect=deduced.told_edges_made_indirect,
        redundant_told_edges=deduced.redundant_told_edges,
        equivalences=deduced.equivalences,
        members=deduced.members,
        same_members_not_equivalent=deduced.same_members_not_equivalent,
        most_specific_categories=deduced.most_specific_categories,
        memberships_made_indirect=deduced.memberships_made_indirect,
        new_direct_memberships=deduced.new_direct_memberships,
        oracle_direct_edges_agree=oracle_direct_agree,
    )


def _same_direct_edges(
    named: Sequence[str], hermit_pairs: Iterable[tuple[str, str]], fc_edges: Sequence[DirectEdge]
) -> bool:
    """Whether the transitive reduction of HermiT's hierarchy equals forward chaining's edges.

    A cycle in HermiT's pairs (two equivalent classes) has no unique reduction: it disagrees.
    """
    try:
        hermit_direct = transitive_reduction(named, hermit_pairs)
    except CyclicTaxonomyError:
        return False
    return hermit_direct == {(e.child, e.parent) for e in fc_edges}


# --- P4.2: inheritance --------------------------------------------------------------------------


def run_inheritance(
    reasoned: ReasonedOntology, *, oracle_prototypes: owl_oracle.OracleRun | None
) -> InheritanceResult:
    """P4.2: the formulary rows a new member of each drug category inherits (its prototype).

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.
    oracle_prototypes : owl_oracle.OracleRun | None
        HermiT's run with one prototype individual per category, or ``None``.

    Returns
    -------
    InheritanceResult
        One row per drug category, the range of row counts, and HermiT's agreement.
    """
    rows = []
    fc_keys: set[_Key] = set()
    for category_id in reasoned.ontology.drug_categories:
        inherited = inherited_by_new_member(reasoned, category_id)
        row = InheritanceRow(
            category_id=category_id,
            candidates=inherited.candidates,
            contraindications=inherited.contraindications,
            interactions=inherited.interactions,
            coprescriptions=tuple(f"{r}->{e}" for r, e in inherited.coprescriptions),
            family_pairs=tuple(f"{f}:{d}" for f, d in inherited.family_pairs),
            rows=inherited.rows,
        )
        rows.append(row)
        fc_keys |= _inheritance_keys(row)
    oracle = None
    if oracle_prototypes is not None:
        oracle_keys = _oracle_inheritance_keys(reasoned, oracle_prototypes)
        oracle = _agreement(len(fc_keys | oracle_keys), fc_keys, oracle_keys, "inherited row")
    counts = [row.rows for row in rows]
    return InheritanceResult(
        rows=tuple(rows), min_rows=min(counts), max_rows=max(counts), oracle=oracle
    )


def _inheritance_keys(row: InheritanceRow) -> set[_Key]:
    fields = ("candidates", "contraindications", "interactions", "coprescriptions", "family_pairs")
    return {(row.category_id, field, value) for field in fields for value in getattr(row, field)}


def _oracle_inheritance_keys(reasoned: ReasonedOntology, run: owl_oracle.OracleRun) -> set[_Key]:
    """Return the rows HermiT infers for each prototype individual, as InheritanceRow keys."""
    data = reasoned.ontology.data
    drug_views = {d: run.individuals["D_" + d] for d in reasoned.ontology.drug_ids}
    keys: set[_Key] = set()
    for category_id in reasoned.ontology.drug_categories:
        view = run.individuals["P_" + category_id]
        row = InheritanceRow(
            category_id=category_id,
            candidates=tuple(sorted(view.treats)),
            contraindications=tuple(sorted(view.contraindicated_by)),
            interactions=tuple(sorted(b[2:] for b in view.interacts if b.startswith("D_"))),
            coprescriptions=tuple(sorted(f"{r}->{e}" for r, e in view.requires)),
            family_pairs=tuple(
                sorted(
                    f"{f}:{d}"
                    for f in data.families
                    if f in view.categories
                    for d, drug_view in drug_views.items()
                    if f in drug_view.categories
                )
            ),
            rows=0,
        )
        keys |= _inheritance_keys(row)
    return keys


# --- P4.3: consistency --------------------------------------------------------------------------


def mutation_candidates(reasoned: ReasonedOntology) -> tuple[tuple[str, str], ...]:
    """Return every mutation (d, K): a drug and a drug category it does not belong to in Cl(KB).

    The categories d belongs to include those it is classified into by a definition (tramadol in
    ``serotonergic_opioids``). The category of categories ``therapeutic_families`` is not a drug
    category: its members are categories.
    """
    categories = reasoned.ontology.drug_categories
    mutations = []
    for drug_id in reasoned.ontology.drug_ids:
        member_of = set(classify(reasoned, drug_id))
        mutations += [(drug_id, k) for k in categories if k not in member_of]
    return tuple(mutations)


def expected_clash_sets(data: OntologyData, drug_id: str, category_id: str) -> tuple[str, ...]:
    """Return the disjoint sets that make d ∈ K a clash, by reachability in the taxonomy.

    A clash is expected iff some disjoint set holds K₁ ≠ K₂ with K₁ at or above one of d's told
    categories and K₂ at or above K. "Above" follows subcategory edges and, for a category
    defined by conjuncts, an edge to each conjunct. Computed with networkx, independently of
    forward chaining. The label ignores classification *into* a defined category, which is exact
    as long as no defined category is itself in a disjoint set (true in the data: its ancestors
    are then exactly those of its conjuncts).
    """
    graph = _reachability_graph(data)
    above_told: set[str] = set()
    for membership in data.memberships:
        if membership.object_id == drug_id:
            above_told |= _up(graph, membership.category_id)
    above_k = _up(graph, category_id)
    return tuple(
        disjoint_set.set_id
        for disjoint_set in data.disjoint_sets
        if any(
            a != b
            for a in above_told.intersection(disjoint_set.category_ids)
            for b in above_k.intersection(disjoint_set.category_ids)
        )
    )


def _clash_texts(found: Iterable[Clash]) -> tuple[str, ...]:
    return tuple(f"{c.object_id}:{c.category_a}/{c.category_b}" for c in found)


def _sample(items: Sequence[tuple[str, str]], size: int) -> list[tuple[str, str]]:
    """Return ``size`` items spread evenly over ``items`` (all of them if there are fewer)."""
    if len(items) <= size:
        return list(items)
    return [items[(i * len(items)) // size] for i in range(size)]


def run_consistency(
    reasoned: ReasonedOntology, *, oracle_run: owl_oracle.OracleRun | None
) -> ConsistencyResult:
    """P4.3: consistency of the KB and of each named category, partitions, and the mutation test.

    Parameters
    ----------
    reasoned : ReasonedOntology
        The ontology and its fixed point.
    oracle_run : owl_oracle.OracleRun | None
        HermiT's base run, or ``None`` to skip the oracle. With it, HermiT also checks the
        mutations in one run (one class told(d) ⊓ K each, told(d) the intersection of d's told
        categories) and in real per-mutation runs on an evenly spread sample of each kind.

    Returns
    -------
    ConsistencyResult
        The counts the report quotes and one row per mutation.
    """
    ontology = reasoned.ontology
    data = ontology.data
    inconsistent = tuple(
        c for c in ontology.named_categories if inherited_by_new_member(reasoned, c).clashes
    )
    mutations = mutation_candidates(reasoned)
    expected = {m: expected_clash_sets(data, *m) for m in mutations}
    detected = {m: _clash_texts(clashes(add_membership(reasoned, *m))) for m in mutations}
    oracle_kb = oracle_unsat = single = rechecked = None
    oracle_clash: Mapping[tuple[str, str], bool] = {}
    rechecked_verdicts: dict[tuple[str, str], bool] = {}
    if oracle_run is not None:
        oracle_kb, oracle_unsat = oracle_run.consistent, oracle_run.unsatisfiable
        unsat = owl_oracle.unsatisfiable_mutations(data, mutations)
        oracle_clash = {m: m in unsat for m in mutations}
        fc_positive = [m for m in mutations if detected[m]]
        single = _agreement(len(mutations), fc_positive, sorted(unsat), "mutation clash")
        positives = [m for m in mutations if expected[m]]
        negatives = [m for m in mutations if not expected[m]]
        sample = _sample(positives, MUTATION_SAMPLE_SIZE) + _sample(negatives, MUTATION_SAMPLE_SIZE)
        for m in sample:
            rechecked_verdicts[m] = not owl_oracle.mutation_is_consistent(data, *m)
        rechecked = _agreement(
            len(rechecked_verdicts),
            [m for m in rechecked_verdicts if oracle_clash[m]],
            [m for m, clash in rechecked_verdicts.items() if clash],
            "per-mutation run vs single run",
        )
    rows = tuple(
        MutationRow(
            drug_id=d,
            category_id=k,
            expected_clash=bool(expected[(d, k)]),
            expected_by=expected[(d, k)],
            detected=detected[(d, k)],
            oracle_clash=oracle_clash.get((d, k)),
            oracle_rechecked=rechecked_verdicts.get((d, k)),
        )
        for d, k in mutations
    )
    positive_rows = [r for r in rows if r.expected_clash]
    negative_rows = [r for r in rows if not r.expected_clash]
    by_set = Counter(set_id for r in positive_rows for set_id in r.expected_by)
    return ConsistencyResult(
        kb_clashes=_clash_texts(clashes(reasoned.closure)),
        oracle_kb_consistent=oracle_kb,
        categories_checked=len(ontology.named_categories),
        inconsistent_categories=inconsistent,
        oracle_unsatisfiable_classes=oracle_unsat,
        partitions=tuple(
            PartitionRow(p.set_id, p.parent_id, p.parts, len(p.members), p.uncovered)
            for p in partition_coverage(reasoned)
        ),
        n_mutations=len(rows),
        n_clash_expected=len(positive_rows),
        n_clash_expected_from_upper=sum("upper" in r.expected_by for r in positive_rows),
        n_clash_expected_by_set={s.set_id: by_set[s.set_id] for s in data.disjoint_sets},
        n_clash_detected=sum(bool(r.detected) for r in positive_rows),
        n_no_clash_expected=len(negative_rows),
        n_false_alarms=sum(bool(r.detected) for r in negative_rows),
        oracle_single_run=single,
        oracle_rechecked=rechecked,
        mutations=rows,
    )


# --- P4.4: cost of the fixed point --------------------------------------------------------------


def run_fixed_point_cost(reasoned: ReasonedOntology) -> FixedPointResult:
    """P4.4: deterministic counts of the fixed point, against AIMA's bound p·n^k (§9.3.2).

    p is the number of predicates of the KB (the built-in ``Distinct`` excluded), k their maximum
    arity and n the number of constants of the KB (facts and rules). The naive loop (Fig. 9.3) is
    run once more to check that it adds the same facts at every iteration, and the ⊂ facts of
    Cl(KB) are compared with the transitive closure of the told subcategory links computed by
    networkx (a definition yields memberships through O6, never ``Subset`` facts).
    """
    closure, kb = reasoned.closure, reasoned.kb
    atoms = [*closure.facts, *(a for rule in kb.rules for a in _rule_atoms(rule))]
    predicates = {(str(a.op), len(a.args)) for a in atoms if a.op != DISTINCT}
    constants = {arg for a in atoms for arg in a.args if not is_variable(arg)}
    max_arity = max(arity for _, arity in predicates)
    naive = fc_closure(kb, incremental=False)
    graph = _subcategory_graph(reasoned.ontology.data)
    networkx_pairs = {(c, p) for c in graph.nodes for p in nx.descendants(graph, c)}
    fc_pairs = {
        (identifier(f.args[0]), identifier(f.args[1])) for f in closure.with_predicate(SUBSET)
    }
    per_predicate = Counter(str(f.op) for f in closure.facts)
    return FixedPointResult(
        told_facts=len(closure.told),
        rules=len(kb.rules),
        closure_facts=len(closure.facts),
        derived_facts=len(closure.derived),
        iterations=closure.iterations,
        new_per_iteration=closure.new_per_iteration,
        facts_per_predicate={p: per_predicate.get(p, 0) for p in sorted(n for n, _ in predicates)},
        predicates=len(predicates),
        constants=len(constants),
        max_arity=max_arity,
        bound=len(predicates) * len(constants) ** max_arity,
        naive_equals_incremental=naive.new_by_iteration == closure.new_by_iteration,
        subset_closure_matches_networkx=fc_pairs == networkx_pairs,
    )


def _rule_atoms(rule: Expr) -> list[Expr]:
    premises, conclusion = parse_definite_clause(rule)
    return [*premises, conclusion]


def told_knowledge(ontology: Ontology) -> dict[str, object]:
    """Return the told knowledge (no inference) as JSON-ready data, and the clauses O1-O6.

    It is the ``ontology`` section of ``results.json``, from which Fig. 2 is drawn.

    Parameters
    ----------
    ontology : Ontology
        The ontology.

    Returns
    -------
    dict[str, object]
        Categories (with told parents, family and drug-category flags), drugs with their told
        categories, conditions, risk factors, every link table (a self-link has
        ``subject_a == subject_b``), the disjoint sets, the definitions by conjuncts, the defined
        categories of §4.5 and the text of every clause with its axiom tag.
    """
    data = ontology.data
    parents = ontology.parents()
    families = set(data.families)
    return {
        "categories": [
            {
                "id": c.category_id,
                "name_es": c.name_es,
                "name_en": c.name_en,
                "atc_code": c.atc_code,
                "parents": list(parents.get(c.category_id, ())),
                "family": c.category_id in families,
                "drug_category": c.category_id in ontology.drug_categories,
            }
            for c in data.categories
        ],
        "drugs": [
            {
                "id": d.drug_id,
                "name_es": d.name_es,
                "categories": list(ontology.told_categories_of(d.drug_id)),
            }
            for d in data.drugs
        ],
        "conditions": [{"id": c.condition_id, "name_es": c.name_es} for c in data.conditions],
        "risk_factors": [{"id": r.risk_factor_id, "name_es": r.name_es} for r in data.risk_factors],
        "indications": [
            {"subject": i.subject_id, "condition": i.condition_id} for i in data.indications
        ],
        "contraindications": [
            {"subject": c.subject_id, "risk_factor": c.risk_factor_id, "reason": c.reason}
            for c in data.contraindications
        ],
        "interactions": [
            {
                "subject_a": i.subject_a,
                "subject_b": i.subject_b,
                "effect": i.effect,
                "severity": i.severity,
            }
            for i in data.interactions
        ],
        "coprescriptions": [
            {
                "subject": c.subject_id,
                "risk_factor": c.risk_factor_id,
                "companion": c.companion_drug_id,
            }
            for c in data.coprescriptions
        ],
        "families": list(data.families),
        "disjoint_sets": [
            {"set_id": s.set_id, "categories": list(s.category_ids), "partition_of": s.partition_of}
            for s in data.disjoint_sets
        ],
        "definitions": [
            {"category": d.category_id, "conjuncts": list(d.conjunct_ids)} for d in data.definitions
        ],
        "defined_categories": [
            {"id": d.category_id, "kind": d.kind.value, "target": d.target_id}
            for d in ontology.defined_categories
        ],
        "clauses": [{"axiom": r.tag, "clause": str(r.clause)} for r in tagged_rules(ontology)],
    }


# --- serialisation ------------------------------------------------------------------------------


def as_jsonable(result: object) -> object:
    """Convert a result dataclass (or a tuple of them) into JSON-ready built-in types.

    Parameters
    ----------
    result : object
        A result dataclass instance, or a tuple or list of them.

    Returns
    -------
    object
        Nested ``dict``/``list``/``str``/``int``/``bool``/``None`` values.

    Raises
    ------
    TypeError
        If ``result`` is neither a dataclass instance nor a sequence of them.
    """
    if isinstance(result, tuple | list):
        return [as_jsonable(item) for item in result]
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return _plain(dataclasses.asdict(result))
    raise TypeError(f"cannot serialise {type(result).__name__}")


def _plain(value: object) -> object:
    """Turn mappings into dicts and tuples into lists, recursively."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value
