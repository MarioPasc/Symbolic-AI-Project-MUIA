"""Exceptions raised by the EJ2 ontology, its forward-chaining reasoner and the OWL oracle."""

from __future__ import annotations


class OntologyError(Exception):
    """Base class of every exception raised by ``symbolic_ai.p1_ej2_ontology``."""


class UnknownIdentifierError(OntologyError, KeyError):
    """A category, drug, condition or risk factor identifier is not part of the ontology."""


class CyclicTaxonomyError(OntologyError):
    """The told subcategory links contain a cycle, so ⊂ would not be a strict order."""


class NotDefiniteClauseError(OntologyError):
    """A sentence given to forward chaining is not a function-free, range-restricted Horn clause.

    Forward chaining is complete only for definite clauses (AIMA §9.3), and it terminates on
    Datalog: no function symbols, and every variable of the conclusion occurs in a premise.
    """


class OracleUnavailableError(OntologyError):
    """The OWL oracle cannot run: owlready2 is not installed or no Java runtime was found."""


class ResultsFormatError(OntologyError):
    """A saved results file lacks a field the figures need, or has the wrong schema."""
