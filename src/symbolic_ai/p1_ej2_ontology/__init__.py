"""EJ2: an ontology of drug categories and forward chaining over it, checked against HermiT.

The category-level knowledge of the formulary (data 1.1.0) is written as first-order definite
clauses (axioms O1-O7); forward chaining (AIMA Fig. 9.3 with the incremental rule of §9.3.3)
computes its fixed point, on which classification, subsumption and consistency are queries, and
from which EJ1's drug-level formulary is derived. The figure module (:mod:`.plot`, Graphviz) and
the oracle's dependency (owlready2) are optional and are not imported here.
"""

from __future__ import annotations

from symbolic_ai.p1_ej2_ontology.errors import (
    CyclicTaxonomyError,
    NotDefiniteClauseError,
    OntologyError,
    OracleUnavailableError,
    ResultsFormatError,
    UnknownIdentifierError,
)
from symbolic_ai.p1_ej2_ontology.forward_chaining import (
    Closure,
    KnowledgeBase,
    ask,
    fc_closure,
    fc_extend,
)
from symbolic_ai.p1_ej2_ontology.ontology import (
    DefinedCategory,
    DefinedKind,
    Ontology,
    build_ontology,
    tagged_rules,
    to_knowledge_base,
)
from symbolic_ai.p1_ej2_ontology.reasoner import (
    Clash,
    Inheritance,
    ReasonedOntology,
    Taxonomy,
    add_membership,
    clashes,
    classify,
    derive_formulary,
    inherited_by_new_member,
    partition_coverage,
    prototype_closure,
    reason,
    subsumes,
    taxonomy,
)

__all__ = [
    "Clash",
    "Closure",
    "CyclicTaxonomyError",
    "DefinedCategory",
    "DefinedKind",
    "Inheritance",
    "KnowledgeBase",
    "NotDefiniteClauseError",
    "Ontology",
    "OntologyError",
    "OracleUnavailableError",
    "ReasonedOntology",
    "ResultsFormatError",
    "Taxonomy",
    "UnknownIdentifierError",
    "add_membership",
    "ask",
    "build_ontology",
    "clashes",
    "classify",
    "derive_formulary",
    "fc_closure",
    "fc_extend",
    "inherited_by_new_member",
    "partition_coverage",
    "prototype_closure",
    "reason",
    "subsumes",
    "tagged_rules",
    "taxonomy",
    "to_knowledge_base",
]
