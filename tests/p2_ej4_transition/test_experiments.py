"""Tests of the experiments of EJ4 (``experiments.py``): R2 validation, mutations, R3, context."""

from __future__ import annotations

import dataclasses
from collections import Counter

import ej4_toys
import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary, load_pharmacokinetics
from symbolic_ai.p2_ej4_transition import EJ4_DATA_VERSION, EventKind, TransitionProblem
from symbolic_ai.p2_ej4_transition.checker import RULES, check_schedule
from symbolic_ai.p2_ej4_transition.experiments import (
    DESIGN_CENSUS,
    EXTENDED_CASE_ID,
    MUTATIONS,
    R3_CONFIGS,
    census_matches_design,
    context_numbers,
    entries_in_category,
    find_entry,
    mutate_contraindicated_drug,
    mutate_day0,
    mutate_delete_key,
    mutate_drop_taper_day,
    mutate_final,
    mutate_start_early,
    mutate_untreated_day,
    run_algorithm_table,
    run_min_conflicts_rows,
    run_validation,
)
from symbolic_ai.p2_ej4_transition.instances import (
    Category,
    CensusRow,
    History,
    ProblemEntry,
    case_study_problem,
    census,
    enumerate_problems,
)
from symbolic_ai.p2_ej4_transition.solve import (
    MAIN_CONFIG,
    SolverConfig,
    find_min_horizon,
    solve_horizon,
)


@pytest.fixture(scope="module")
def entries() -> tuple[ProblemEntry, ...]:
    """Every problem of the real data (1.3.0); enumeration takes about 0.2 s."""
    version = EJ4_DATA_VERSION
    return enumerate_problems(
        load_formulary(version=version),
        load_drug_costs(version=version),
        load_pharmacokinetics(version=version),
    )


@pytest.fixture(scope="module")
def case_study() -> TransitionProblem:
    """The worked example (design §1)."""
    version = EJ4_DATA_VERSION
    return case_study_problem(
        load_formulary(version=version),
        load_drug_costs(version=version),
        load_pharmacokinetics(version=version),
    )


def _toy_entry(toy: ej4_toys.KnownAnswer) -> ProblemEntry:
    """A one-occurrence entry around a toy problem (the history is a placeholder)."""
    problem = toy.problem
    return ProblemEntry(problem, Category.FORCED_ONLY, ((History(("X",), ()), problem.event),))


def _schedule(problem: TransitionProblem) -> tuple[dict[tuple[str, int], object], int]:
    result = find_min_horizon(problem, MAIN_CONFIG)
    assert result.schedule is not None and result.h_star is not None
    return dict(result.schedule), result.h_star


# --- R2 on the toys: the seven known answers, by solver and oracle ------------------------------


def test_known_answers_agree_with_solver_and_oracle() -> None:
    validation = run_validation([_toy_entry(t) for t in ej4_toys.KNOWN_ANSWERS])
    assert validation.problems == validation.events == 7
    assert validation.oracle is not None
    assert (validation.oracle.compared, validation.oracle.agree) == (7, 7)
    assert validation.oracle.disagreements == ()
    expected = Counter(t.h_star for t in ej4_toys.KNOWN_ANSWERS)
    assert dict(validation.context.histogram) == dict(expected)
    assert validation.checker.valid == validation.checker.schedules == 5


def test_every_mutant_of_the_toys_is_rejected_with_its_rule() -> None:
    validation = run_validation([_toy_entry(t) for t in ej4_toys.KNOWN_ANSWERS], use_oracle=False)
    assert validation.oracle is None
    assert [m.name for m in validation.mutations] == [m.name for m in MUTATIONS]
    for summary in validation.mutations:
        assert summary.rejected_with_expected_rule == summary.applied, summary.misses
        assert summary.misses == ()


def test_mutations_cover_every_checker_rule() -> None:
    assert {m.expected_rule for m in MUTATIONS} == set(RULES)


def test_refer_entries_are_skipped(entries: tuple[ProblemEntry, ...]) -> None:
    refer = entries_in_category(entries, Category.REFER)[:3]
    assert refer and all(e.problem is None for e in refer)
    validation = run_validation(refer, use_oracle=False)
    assert (validation.problems, validation.checker.schedules) == (0, 0)


# --- mutations on the worked example (real data) ------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    ("generate", "rule", "places"),
    [
        (mutate_drop_taper_day, "transition", ["DEP day 4"]),
        (mutate_start_early, "interaction", ["PAIN starts ibuprofen on day 11"]),
        (mutate_untreated_day, "transition", [f"DEP day {d}" for d in range(1, 12)]),
        (
            mutate_contraindicated_drug,
            "domain",
            [f"PAIN day {d} paracetamol" for d in range(1, 12)],
        ),
        (mutate_day0, "day0", ["DEP day 0", "PAIN day 0"]),
        (mutate_final, "final", ["DEP ends on mirtazapine↓1", "PAIN ends on paracetamol"]),
        (mutate_delete_key, "shape", ["DEP day 12 deleted", "PAIN day 12 deleted"]),
    ],
)
def test_mutations_of_the_worked_example(
    case_study: TransitionProblem, generate: object, rule: str, places: list[str]
) -> None:
    schedule, horizon = _schedule(case_study)
    assert callable(generate)
    mutants = generate(case_study, schedule, horizon)
    assert [place for place, _ in mutants] == places
    for _place, mutant in mutants:
        assert mutant != schedule
        assert rule in {v.rule for v in check_schedule(case_study, mutant, horizon)}
    assert check_schedule(case_study, schedule, horizon) == ()  # the original is untouched


def test_start_early_needs_a_present_interacting_drug() -> None:
    # T5: z1 starts on day 4, when x1 and y1 have just left the body, and both interact with z1.
    schedule, horizon = _schedule(ej4_toys.T5.problem)
    mutants = mutate_start_early(ej4_toys.T5.problem, schedule, horizon)
    assert [place for place, _ in mutants] == ["Z starts z1 on day 3"]
    # T1 has no interactions, so no start can be shown to be blocked.
    schedule, horizon = _schedule(ej4_toys.T1.problem)
    assert mutate_start_early(ej4_toys.T1.problem, schedule, horizon) == []


def test_mutations_without_intermediate_days_do_not_apply() -> None:
    schedule, horizon = _schedule(ej4_toys.T1.problem)  # H* = 1: no day strictly inside
    for generate in (mutate_untreated_day, mutate_contraindicated_drug, mutate_drop_taper_day):
        assert generate(ej4_toys.T1.problem, schedule, horizon) == []


# --- census and context (real data) -------------------------------------------------------------


def _design_rows() -> list[CensusRow]:
    """The census of :data:`DESIGN_CENSUS` as rows (no real data: its counts may change)."""
    kinds = {"new_risk_factor": EventKind.NEW_RISK_FACTOR, "new_condition": EventKind.NEW_CONDITION}
    return [CensusRow(kinds[kind], *counts) for kind, counts in DESIGN_CENSUS.items()]


def test_census_matches_design_compares_every_count() -> None:
    rows = _design_rows()
    assert census_matches_design(rows)
    changed = dataclasses.replace(rows[0], trivial=rows[0].trivial + 1)
    assert not census_matches_design([changed, rows[1]])
    assert not census_matches_design(rows[:1])


@pytest.mark.integration
def test_census_counts_every_event_once(entries: tuple[ProblemEntry, ...]) -> None:
    rows = census(entries)
    assert sum(row.events for row in rows) == sum(len(e.occurrences) for e in entries)
    for row in rows:
        assert row.events == row.refer + row.trivial + row.forced_only + row.knock_on


def test_context_numbers_count_events_by_kind_and_category() -> None:
    t0, t1 = _toy_entry(ej4_toys.T0), _toy_entry(ej4_toys.T1)
    doubled = dataclasses.replace(t1, occurrences=t1.occurrences * 2)
    results = [(e, find_min_horizon(e.problem)) for e in (t0, doubled) if e.problem is not None]
    context = context_numbers(results)
    assert context.histogram == ((0, 1), (1, 2))
    assert [(c.h_star, c.entries, c.events) for c in context.by_kind_and_category] == [
        (0, 1, 1),
        (1, 1, 2),
    ]


@pytest.mark.integration
def test_find_entry(entries: tuple[ProblemEntry, ...]) -> None:
    extended = find_entry(entries, EXTENDED_CASE_ID)
    assert extended is not None and extended.problem is not None
    assert extended.category is Category.KNOCK_ON
    assert len(extended.problem.conditions) == 5
    assert find_entry(entries, "no such problem") is None


# --- R3 -----------------------------------------------------------------------------------------


@pytest.mark.integration
def test_algorithm_table_on_the_worked_example(case_study: TransitionProblem) -> None:
    runs = run_algorithm_table([(case_study, 12)])
    assert [(r.configuration, r.horizon) for r in runs] == [
        (key, h) for key, _config in R3_CONFIGS for h in (11, 12)
    ]
    assert [(r.status, r.assignments, r.checks, r.ac3_removed) for r in runs] == [
        ("infeasible", 218, 6558, 0),
        ("solved", 244, 7546, 0),
        ("infeasible", 25, 499, 0),
        ("solved", 51, 1206, 0),
        ("infeasible", 26, 918, 0),
        ("solved", 45, 1694, 0),
        ("infeasible", 0, 1234, 116),
        ("solved", 26, 2264, 133),
    ]
    assert "seconds" not in runs[0].to_dict()
    assert runs[-1].label == MAIN_CONFIG.label


def test_algorithm_table_reports_an_exhausted_budget() -> None:
    tiny = (("bt", SolverConfig("none", "static", "keep_first", False, max_assignments=1)),)
    runs = run_algorithm_table([(ej4_toys.T3.problem, 4)], tiny)
    assert [r.status for r in runs] == ["budget", "budget"]
    assert all(r.assignments == 2 for r in runs)


@pytest.mark.integration
def test_min_conflicts_finds_12_but_cannot_refute_11(case_study: TransitionProblem) -> None:
    runs = run_min_conflicts_rows(case_study, (11, 12), max_steps=10_000, seed=0)
    assert [(r.horizon, r.status, r.assignments, r.checks) for r in runs] == [
        (11, "not_found", 10024, 3043513),
        (12, "solved", 47, 7088),
    ]
    assert all(r.configuration == "min_conflicts" for r in runs)


@pytest.mark.slow
@pytest.mark.integration
def test_extended_case_at_horizon_11(entries: tuple[ProblemEntry, ...]) -> None:
    extended = find_entry(entries, EXTENDED_CASE_ID)
    assert extended is not None and extended.problem is not None
    runs = {key: solve_horizon(extended.problem, 11, config) for key, config in R3_CONFIGS}
    assert (runs["bt"].status, runs["bt"].stats.assignments) == ("budget", 1_000_001)
    assert (runs["bt_fc"].status, runs["bt_fc"].stats.assignments) == ("infeasible", 63439)
    assert (runs["bt_fc_mrv"].status, runs["bt_fc_mrv"].stats.assignments) == ("infeasible", 40)
    assert (runs["main"].status, runs["main"].stats.assignments) == ("infeasible", 0)


# --- R2 on every real problem (slow) ------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_validation_on_every_problem(entries: tuple[ProblemEntry, ...]) -> None:
    # Structural properties only: the population counts change with the data and with D23.
    validation = run_validation(entries)
    solvable = [e for e in entries if e.problem is not None]
    assert validation.problems == len(solvable)
    assert validation.events == sum(len(e.occurrences) for e in solvable)
    assert validation.oracle is not None
    assert validation.oracle.agree == validation.oracle.compared > 0
    assert validation.oracle.disagreements == ()
    assert validation.checker.valid == validation.checker.schedules == validation.problems
    assert all(m.rejected_with_expected_rule == m.applied > 0 for m in validation.mutations)
    assert sum(n for _h, n in validation.context.histogram) == validation.events
