"""A small hand-built ontology for EJ2 tests (not a test module; never reads ``data/``).

Taxonomy: drugs ⊃ a ⊃ {a1, a2} (a partition of a) and drugs ⊃ b. Objects: x1 ∈ a1, x2 ∈ a2,
y ∈ b. Links: a treats H1; a1 is contraindicated by R1; a1 with R1 requires y; every member of
a interacts with every member of b; a is a family.
"""

from __future__ import annotations

from symbolic_ai.dataloader.models import (
    Category,
    Condition,
    ContraindicationLink,
    CoprescriptionLink,
    DisjointSet,
    Drug,
    IndicationLink,
    InteractionLink,
    Membership,
    OntologyData,
    RiskFactor,
    SubcategoryEdge,
)


def toy_ontology_data(
    *,
    interactions: tuple[InteractionLink, ...] | None = None,
    subcategories: tuple[SubcategoryEdge, ...] | None = None,
) -> OntologyData:
    """Return the toy ontology; ``interactions`` and ``subcategories`` may be replaced."""
    default_interactions = (InteractionLink("a", "b", "bleeding", "major"),)
    default_subcategories = (
        SubcategoryEdge("a", "drugs"),
        SubcategoryEdge("a1", "a"),
        SubcategoryEdge("a2", "a"),
        SubcategoryEdge("b", "drugs"),
    )
    return OntologyData(
        version="toy",
        drugs=(Drug("x1", "X1", "X1"), Drug("x2", "X2", "X2"), Drug("y", "Y", "Y")),
        conditions=(Condition("H1", "Uno", "One"),),
        risk_factors=(RiskFactor("R1", "Uno", "One"),),
        categories=tuple(
            Category(c, c.upper(), c.upper()) for c in ("a", "a1", "a2", "b", "drugs")
        ),
        subcategories=default_subcategories if subcategories is None else subcategories,
        memberships=(Membership("x1", "a1"), Membership("x2", "a2"), Membership("y", "b")),
        indications=(IndicationLink("a", "H1"),),
        contraindications=(ContraindicationLink("a1", "R1", "toy reason"),),
        interactions=default_interactions if interactions is None else interactions,
        coprescriptions=(CoprescriptionLink("a1", "R1", "y", "toy reason"),),
        families=("a",),
        disjoint_sets=(DisjointSet("parts", ("a1", "a2"), "a"),),
    )
