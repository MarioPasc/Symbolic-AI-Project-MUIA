"""Plain (non-pytest) builders of the EJ3 test data: the drug costs and small toy formularies.

``COSTS`` is transcribed from ``data/formulary/drug_costs.csv`` (data 1.2.0); the integration test
of ``test_ej3_main.py`` checks that the two agree. The full formulary and the encounters are those
of EJ1's ``fixture_data``. Kept free of pytest so the determinism subprocess can import it too.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from symbolic_ai.dataloader.models import (
    AdverseInteraction,
    Condition,
    Contraindication,
    Coprescription,
    Drug,
    DrugClass,
    Formulary,
    RiskFactor,
)

#: Monthly cost of each drug of the 1.0.0 formulary, in euro cents (data 1.2.0).
COSTS: dict[str, int] = {
    "amlodipine": 180,
    "apixaban": 5400,
    "bisoprolol": 250,
    "enalapril": 150,
    "gliclazide": 350,
    "hydrochlorothiazide": 140,
    "ibuprofen": 230,
    "linagliptin": 3800,
    "losartan": 320,
    "metformin": 170,
    "mirtazapine": 610,
    "omeprazole": 270,
    "paracetamol": 190,
    "propranolol": 290,
    "sertraline": 420,
    "tramadol": 480,
    "verapamil": 520,
    "warfarin": 260,
}


def toy_formulary(
    candidates: Mapping[str, Iterable[str]],
    *,
    extra_drugs: Iterable[str] = (),
    risk_factors: Iterable[str] = (),
    interactions: Iterable[tuple[str, str]] = (),
    contraindications: Iterable[tuple[str, str]] = (),
    coprescriptions: Iterable[tuple[str, str, str]] = (),
    families: Iterable[Iterable[str]] = (),
) -> Formulary:
    """Build a small formulary from bare identifiers.

    ``candidates`` maps a condition to its drugs; ``contraindications`` holds (risk factor, drug)
    pairs, ``coprescriptions`` (drug, risk factor, companion) triples and ``families`` the member
    sets of the exclusive classes. ``extra_drugs`` are drugs that treat no condition.
    """
    candidate_sets = {c: frozenset(drugs) for c, drugs in candidates.items()}
    drug_ids = sorted(set().union(*candidate_sets.values(), extra_drugs))
    family_sets = [frozenset(family) for family in families]
    return Formulary(
        version="toy",
        conditions=tuple(Condition(c, c, c) for c in sorted(candidate_sets)),
        drugs=tuple(Drug(d, d, d) for d in drug_ids),
        risk_factors=tuple(RiskFactor(r, r, r) for r in sorted(risk_factors)),
        candidates=candidate_sets,
        interactions=tuple(
            AdverseInteraction(*sorted(pair), "toy", "major") for pair in sorted(interactions)
        ),
        contraindications=tuple(
            Contraindication(r, d, "toy") for r, d in sorted(contraindications)
        ),
        coprescriptions=tuple(
            Coprescription(d, r, companion, "toy") for d, r, companion in sorted(coprescriptions)
        ),
        drug_classes=tuple(
            DrugClass(f"family{i}", "f", "f", exclusive=True) for i in range(len(family_sets))
        ),
        class_members={f"family{i}": family for i, family in enumerate(family_sets)},
    )
