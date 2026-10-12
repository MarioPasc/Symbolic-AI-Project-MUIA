"""EJ4's instance enumeration on a hand-made formulary (unit) and on the real data (integration).

The tiny formulary has conditions A (a1, a2) and B (b1, b2) and risk factors R and S: R
contraindicates a1, a2 needs the companion b2 under S (so, by D23, S excludes a2), and a2 interacts
with b1. Costs: a1 = b1 = 1, a2 = b2 = 2. Every expected value below is worked out by hand from
these tables in the comment next to it.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from symbolic_ai.dataloader import (
    AdverseInteraction,
    Condition,
    Contraindication,
    Coprescription,
    Drug,
    DrugCost,
    Formulary,
    Pharmacokinetics,
    RiskFactor,
    load_drug_costs,
    load_formulary,
    load_pharmacokinetics,
)
from symbolic_ai.p2_ej4_transition.errors import InstanceError
from symbolic_ai.p2_ej4_transition.instances import (
    Category,
    CensusRow,
    History,
    ProblemEntry,
    _Enumeration,
    _start_regimens,
    _tables,
    _transition_for,
    acceptable_drugs,
    case_study_problem,
    census,
    cheapest_safe_regimen,
    enumerate_problems,
)
from symbolic_ai.p2_ej4_transition.problem import Event, EventKind

RF = EventKind.NEW_RISK_FACTOR
COND = EventKind.NEW_CONDITION
_DRUGS = ("a1", "a2", "b1", "b2")


def _tiny_formulary() -> Formulary:
    return Formulary(
        version="test",
        conditions=(Condition("A", "A", "A"), Condition("B", "B", "B")),
        drugs=tuple(Drug(d, d, d) for d in _DRUGS),
        risk_factors=(RiskFactor("R", "R", "R"), RiskFactor("S", "S", "S")),
        candidates={"A": frozenset({"a1", "a2"}), "B": frozenset({"b1", "b2"})},
        interactions=(AdverseInteraction("a2", "b1", "effect", "major"),),
        contraindications=(Contraindication("R", "a1", "reason"),),
        coprescriptions=(Coprescription("a2", "S", "b2", "reason"),),
    )


def _costs(**cents: int) -> dict[str, DrugCost]:
    table = {"a1": 1, "a2": 2, "b1": 1, "b2": 2} | cents
    return {d: DrugCost(d, c) for d, c in table.items()}


def _pk() -> dict[str, Pharmacokinetics]:
    return {d: Pharmacokinetics(d, washout_days=1, taper_days=0) for d in _DRUGS}


@pytest.fixture(scope="module")
def tiny_entries() -> tuple[ProblemEntry, ...]:
    return enumerate_problems(_tiny_formulary(), _costs(), _pk())


def _entry_of(entries: tuple[ProblemEntry, ...], history: History, event: Event) -> ProblemEntry:
    matches = [e for e in entries if (history, event) in e.occurrences]
    assert len(matches) == 1
    return matches[0]


# --- acceptability (D23) ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "risk_factors", "expected"),
    [
        ("A", (), ("a1", "a2")),
        ("A", ("R",), ("a2",)),  # contraindication
        ("A", ("S",), ("a1",)),  # D23: a2 needs a companion under S
        ("A", ("R", "S"), ()),
        ("B", ("R", "S"), ("b1", "b2")),
    ],
)
def test_acceptable_drugs_apply_contraindications_and_d23(
    condition: str, risk_factors: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert acceptable_drugs(_tiny_formulary(), condition, risk_factors) == expected


@pytest.mark.parametrize(("condition", "risk_factors"), [("Z", ()), ("A", ("Q",))])
def test_acceptable_drugs_reject_unknown_identifiers(
    condition: str, risk_factors: tuple[str, ...]
) -> None:
    with pytest.raises(InstanceError):
        acceptable_drugs(_tiny_formulary(), condition, risk_factors)


# --- cheapest safe regimen ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (History(("A", "B"), ()), {"A": "a1", "B": "b1"}),  # a1+b1 = 2 is the cheapest
        (History(("A", "B"), ("R",)), {"A": "a2", "B": "b2"}),  # a2+b1 interact
        (History(("A",), ("R", "S")), None),  # A has no acceptable drug
    ],
)
def test_cheapest_safe_regimen(history: History, expected: dict[str, str] | None) -> None:
    assert cheapest_safe_regimen(_tiny_formulary(), _costs(), history) == expected


def test_cheapest_safe_regimen_breaks_ties_by_sorted_drug_ids() -> None:
    # a1+b2 and a2+b2 both cost 3 (a1+b1 costs 6); ("a1", "b2") < ("a2", "b2").
    costs = _costs(a1=1, a2=1, b1=5, b2=2)
    assert cheapest_safe_regimen(_tiny_formulary(), costs, History(("A", "B"), ())) == {
        "A": "a1",
        "B": "b2",
    }


def test_cheapest_safe_regimen_needs_every_cost() -> None:
    costs = _costs()
    del costs["b2"]
    with pytest.raises(InstanceError, match="b2"):
        cheapest_safe_regimen(_tiny_formulary(), costs, History(("A",), ()))


@pytest.mark.parametrize(
    ("conditions", "risk_factors"), [((), ()), (("B", "A"), ()), (("A",), ("S", "R"))]
)
def test_history_rejects_empty_or_unsorted_tuples(
    conditions: tuple[str, ...], risk_factors: tuple[str, ...]
) -> None:
    with pytest.raises(InstanceError):
        History(conditions, risk_factors)


# --- classification and census on the tiny formulary --------------------------------------------


@pytest.mark.parametrize(
    ("history", "event", "category"),
    [
        # R_0 = a1; after R only a2: a1 is forced out, nothing else to keep.
        (History(("A",), ()), Event(RF, "R"), Category.FORCED_ONLY),
        # R_0 = a1; a1 is still acceptable under S.
        (History(("A",), ()), Event(RF, "S"), Category.TRIVIAL),
        # R_0 = a2; under R and S, A has no acceptable drug.
        (History(("A",), ("R",)), Event(RF, "S"), Category.REFER),
        # R_0 = b1; under R only a2 treats A, and a2 interacts with b1: b1 must change.
        (History(("B",), ("R",)), Event(COND, "A"), Category.KNOCK_ON),
        # R_0 = a1+b1; after R, a2 is forced in and b1 (still acceptable) must leave.
        (History(("A", "B"), ()), Event(RF, "R"), Category.KNOCK_ON),
        # R_0 = b1; a1+b1 is safe, so only the new condition changes.
        (History(("B",), ()), Event(COND, "A"), Category.FORCED_ONLY),
    ],
)
def test_each_category_on_the_tiny_formulary(
    tiny_entries: tuple[ProblemEntry, ...], history: History, event: Event, category: Category
) -> None:
    assert _entry_of(tiny_entries, history, event).category is category


def test_tiny_census_matches_the_hand_count(tiny_entries: tuple[ProblemEntry, ...]) -> None:
    # 10 starting histories; worked out pair by pair in the module docstring's tables.
    assert census(tiny_entries) == (
        CensusRow(RF, events=12, refer=4, trivial=6, forced_only=1, knock_on=1),
        CensusRow(COND, events=7, refer=1, trivial=0, forced_only=5, knock_on=1),
    )


def test_tiny_starting_histories() -> None:
    run = _Enumeration(_tables(_tiny_formulary()), {"a1": 1, "a2": 2, "b1": 1, "b2": 2})
    starts = list(_start_regimens(run))
    assert len(starts) == 10  # 12 histories minus ({A}, {R, S}) and ({A, B}, {R, S})
    assert starts[0] == (History(("A",), ()), {"A": "a1"})


def test_deduplication_merges_identical_csps(tiny_entries: tuple[ProblemEntry, ...]) -> None:
    # B on b1 under any new risk factor: four pairs, one CSP ({B}, b1, {b1, b2}).
    entry = _entry_of(tiny_entries, History(("B",), ()), Event(RF, "R"))
    assert entry.category is Category.TRIVIAL
    assert entry.occurrences == (
        (History(("B",), ()), Event(RF, "R")),
        (History(("B",), ()), Event(RF, "S")),
        (History(("B",), ("R",)), Event(RF, "S")),
        (History(("B",), ("S",)), Event(RF, "R")),
    )
    assert entry.problem is not None
    assert entry.problem.problem_id == "B|-|+rf:R"  # the first occurrence represents the entry


def test_refer_entries_have_no_problem_and_others_have_one(
    tiny_entries: tuple[ProblemEntry, ...],
) -> None:
    for entry in tiny_entries:
        assert (entry.problem is None) == (entry.category is Category.REFER)


def test_events_already_present_are_rejected() -> None:
    with pytest.raises(InstanceError, match="already present"):
        _transition_for(_tiny_formulary(), _costs(), _pk(), History(("A",), ()), Event(COND, "A"))


def test_missing_pharmacokinetics_is_reported() -> None:
    pk = _pk()
    del pk["b2"]
    with pytest.raises(InstanceError, match="b2"):
        enumerate_problems(_tiny_formulary(), _costs(), pk)


def test_problem_fields_of_a_new_condition(tiny_entries: tuple[ProblemEntry, ...]) -> None:
    problem = _entry_of(tiny_entries, History(("B",), ("R",)), Event(COND, "A")).problem
    assert problem is not None
    assert problem.problem_id == "B|R|+cond:A"
    assert problem.conditions == ("A", "B")
    assert problem.risk_factors == ("R",)
    assert dict(problem.start) == {"A": None, "B": "b1"}
    assert dict(problem.candidates) == {"A": ("a1", "a2"), "B": ("b1", "b2")}
    assert dict(problem.acceptable) == {"A": ("a2",), "B": ("b1", "b2")}
    assert problem.interactions == frozenset({frozenset({"a2", "b1"})})
    assert dict(problem.washout_days) == dict.fromkeys(_DRUGS, 1)


# --- invariants shared by both formularies ------------------------------------------------------


def _check_invariants(
    entries: tuple[ProblemEntry, ...],
    formulary: Formulary,
    costs: Mapping[str, DrugCost],
    pk: Mapping[str, Pharmacokinetics],
) -> None:
    """Every occurrence rebuilds to its entry's CSP key and category; keys are unique."""
    keys = set()
    for entry in entries:
        for history, event in entry.occurrences:
            problem, category = _transition_for(formulary, costs, pk, history, event)
            assert category is entry.category
            if entry.problem is None:
                assert problem is None
                continue
            assert problem is not None
            key = (problem.conditions, dict(problem.start), dict(problem.acceptable))
            assert key == (
                entry.problem.conditions,
                dict(entry.problem.start),
                dict(entry.problem.acceptable),
            )
        if entry.problem is not None:
            keys.add(
                (
                    entry.problem.conditions,
                    tuple(entry.problem.start.items()),
                    tuple(entry.problem.acceptable.items()),
                )
            )
    assert len(keys) == sum(e.problem is not None for e in entries)


def test_tiny_invariants(tiny_entries: tuple[ProblemEntry, ...]) -> None:
    _check_invariants(tiny_entries, _tiny_formulary(), _costs(), _pk())


# --- the real data 1.3.0 ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_data() -> tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]]:
    return load_formulary(), load_drug_costs(), load_pharmacokinetics()


@pytest.fixture(scope="module")
def real_entries(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
) -> tuple[ProblemEntry, ...]:
    return enumerate_problems(*real_data)


@pytest.mark.integration
def test_real_census_is_the_design_table(real_entries: tuple[ProblemEntry, ...]) -> None:
    assert census(real_entries) == (
        CensusRow(RF, events=8896, refer=1664, trivial=5472, forced_only=1600, knock_on=160),
        CensusRow(COND, events=8512, refer=1236, trivial=0, forced_only=7084, knock_on=192),
    )


@pytest.mark.integration
def test_real_starting_histories(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
    real_entries: tuple[ProblemEntry, ...],
) -> None:
    formulary, costs, _pk_table = real_data
    run = _Enumeration(_tables(formulary), {d: c.monthly_cost_cents for d, c in costs.items()})
    assert sum(1 for _ in _start_regimens(run)) == 2688
    assert len({h for e in real_entries for h, _event in e.occurrences}) == 2688


@pytest.mark.integration
def test_real_enumeration_is_deterministic(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
    real_entries: tuple[ProblemEntry, ...],
) -> None:
    assert enumerate_problems(*real_data) == real_entries


@pytest.mark.integration
def test_real_problem_ids_are_unique(real_entries: tuple[ProblemEntry, ...]) -> None:
    ids = [e.problem.problem_id for e in real_entries if e.problem is not None]
    assert len(ids) == len(set(ids))


@pytest.mark.integration
def test_real_invariants(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
    real_entries: tuple[ProblemEntry, ...],
) -> None:
    _check_invariants(real_entries, *real_data)


@pytest.mark.integration
def test_d23_on_the_real_data(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
) -> None:
    formulary = real_data[0]
    assert acceptable_drugs(formulary, "PAIN", ()) == ("ibuprofen", "paracetamol", "tramadol")
    assert acceptable_drugs(formulary, "PAIN", ("AGE65",)) == ("paracetamol", "tramadol")


@pytest.mark.integration
def test_case_study(
    real_data: tuple[Formulary, Mapping[str, DrugCost], Mapping[str, Pharmacokinetics]],
    real_entries: tuple[ProblemEntry, ...],
) -> None:
    problem = case_study_problem(*real_data)
    assert problem.problem_id == "DEP|LIVER|+cond:PAIN"
    assert problem.event == Event(COND, "PAIN")
    assert problem.conditions == ("DEP", "PAIN")
    assert problem.risk_factors == ("LIVER",)
    assert dict(problem.start) == {"DEP": "sertraline", "PAIN": None}
    assert problem.acceptable["PAIN"] == ("ibuprofen", "tramadol")
    assert problem.acceptable["DEP"] == ("mirtazapine", "sertraline")
    assert problem.taper_days["sertraline"] == 4
    assert problem.washout_days["sertraline"] == 7
    entry = _entry_of(real_entries, History(("DEP",), ("LIVER",)), Event(COND, "PAIN"))
    assert entry.category is Category.KNOCK_ON
