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
    SolvedEntry,
    census_matches_design,
    check_categories,
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
    run_knock_on_population,
    run_min_conflicts_rows,
    run_synthetic,
    run_validation,
    solve_entries,
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
    assert validation.three_way_agree == validation.problems
    assert all(m.rejected_with_expected_rule == m.applied > 0 for m in validation.mutations)
    assert sum(n for _h, n in validation.context.histogram) == validation.events


@pytest.mark.slow
def test_synthetic_slow_batch_three_way_agreement() -> None:
    result = run_synthetic()  # seed 2027, 1500 problems, h_max 20 (design §7 R2)
    _assert_synthetic_structure(result)
    assert result.three_way_agree == result.compared == result.count == 1500


# --- design v2.3 rows: categories, synthetic batch, knock-on population ------------------------


@pytest.mark.integration
def test_every_real_category_agrees_with_its_h_star(entries: tuple[ProblemEntry, ...]) -> None:
    solved = solve_entries(entries)
    rows = check_categories(solved)
    assert [r.category for r in rows] == ["trivial", "forced_only", "knock_on"]
    assert sum(r.entries for r in rows) == len(solved)
    for row in rows:
        assert row.agree == row.entries and row.disagreements == ()
        assert sum(n for _h, n in row.h_stars) == row.entries
    trivial = rows[0]
    assert trivial.h_stars == ((0, trivial.entries),)
    assert all(h is not None and h >= 1 for r in rows[1:] for h, _n in r.h_stars)


def test_a_category_that_contradicts_h_star_is_reported() -> None:
    problem = ej4_toys.T1.problem  # H* = 1, so it cannot be trivial
    occurrences = ((History(("X",), ()), problem.event),)
    trivial = ProblemEntry(problem, Category.TRIVIAL, occurrences)
    knock_on = ProblemEntry(problem, Category.KNOCK_ON, occurrences)  # x1 is not acceptable
    solved = [SolvedEntry(e, problem, find_min_horizon(problem)) for e in (trivial, knock_on)]
    rows = {r.category: r for r in check_categories(solved)}
    assert (rows["trivial"].agree, rows["trivial"].disagreements) == (0, (problem.problem_id,))
    assert (rows["knock_on"].agree, rows["knock_on"].disagreements) == (0, (problem.problem_id,))


def test_validation_counts_the_three_way_agreement() -> None:
    entries = [_toy_entry(t) for t in ej4_toys.KNOWN_ANSWERS]
    assert run_validation(entries).three_way_agree == 7
    assert run_validation(entries, use_oracle=False).three_way_agree is None
    solved = solve_entries(entries)
    shared = run_validation(entries, solved_entries=solved)
    assert shared == run_validation(entries)


def _assert_synthetic_structure(result: object) -> None:
    """Structural properties of a synthetic run (no hard-coded population counts)."""
    from symbolic_ai.p2_ej4_transition.experiments import SyntheticResult

    assert isinstance(result, SyntheticResult)
    assert result.coverage.problems == result.count
    assert sum(n for _s, n in result.oracle_statuses) == result.count
    assert sum(n for _h, n in result.h_star_histogram) == result.count
    assert result.agree == result.compared and result.disagreements == ()
    assert result.checker_valid == result.checker_schedules == result.solved
    assert result.ac3_decides_before <= result.with_h_star_minus_one <= result.solved
    assert set(result.backtracking) <= {f"S{i:04d}" for i in range(result.count)}
    assert result.median_assignments <= result.max_assignments


def test_small_synthetic_batch_agrees_and_is_deterministic() -> None:
    first = run_synthetic(seed=2026, count=40)
    _assert_synthetic_structure(first)
    assert first.three_way_agree == first.count == 40
    assert run_synthetic(seed=2026, count=40) == first


@pytest.mark.integration
def test_knock_on_population_row(entries: tuple[ProblemEntry, ...]) -> None:
    solved = solve_entries(entries)
    knock_on = [s for s in solved if s.entry.category is Category.KNOCK_ON]
    rows = run_knock_on_population(solved, budget=200)  # a small budget keeps it fast
    assert [r.configuration for r in rows] == [key for key, _c in R3_CONFIGS]
    for row in rows:
        assert row.entries == len(knock_on) and row.budget == 200
        assert sum(n for _s, n in row.statuses) == row.entries
        assert row.budget_hits == dict(row.statuses).get("budget", 0)
        # the budget is checked when the next variable is chosen, so a stopped run may overshoot
        assert (row.max_assignments > row.budget) == (row.budget_hits > 0)
        assert "seconds" not in row.to_dict()
    main = rows[-1]
    # AC-3 alone refutes H* - 1 on every real knock-on problem
    assert (main.median_assignments, main.budget_hits, main.statuses) == (
        0.0,
        0,
        (("infeasible", main.entries),),
    )
