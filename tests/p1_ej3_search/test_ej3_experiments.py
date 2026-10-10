"""Tests for symbolic_ai.p1_ej3_search.experiments: the scenarios (P6) and the comparison (P7)."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from symbolic_ai.dataloader.models import Encounter, Formulary
from symbolic_ai.p1_ej1_logic import RegimenSpace
from symbolic_ai.p1_ej1_logic.experiments import as_jsonable
from symbolic_ai.p1_ej3_search import Algorithm, effective_branching_factor
from symbolic_ai.p1_ej3_search.experiments import (
    CONFIGURATIONS,
    ComparisonResult,
    ScenarioResult,
    run_scenarios,
    run_search_comparison,
)


def test_configurations_are_every_algorithm_plain_and_pruned() -> None:
    assert [c.name for c in CONFIGURATIONS] == [
        "astar",
        "astar_pruned",
        "idastar",
        "idastar_pruned",
        "greedy",
        "greedy_pruned",
        "uniform_cost",
        "uniform_cost_pruned",
    ]
    assert all(c.pruned == c.name.endswith("_pruned") for c in CONFIGURATIONS)
    assert {c.algorithm for c in CONFIGURATIONS} == set(Algorithm)


# --- P6 on E001-E005 ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios(
    formulary: Formulary, costs: Mapping[str, int], space: RegimenSpace
) -> dict[str, ScenarioResult]:
    import fixture_data  # on sys.path through conftest

    encounters: Mapping[str, Encounter] = fixture_data.encounters()
    ordered = [encounters[k] for k in sorted(encounters)]
    return {r.encounter_id: r for r in run_scenarios(formulary, costs, ordered, space)}


@pytest.mark.parametrize(
    ("encounter_id", "regimen", "cost", "baseline_cost", "models", "effort"),
    [
        (
            "E001",
            ("enalapril", "linagliptin", "mirtazapine", "paracetamol", "warfarin"),
            5010,
            5330,
            115,
            (5, 12),
        ),
        ("E003", ("hydrochlorothiazide", "warfarin"), 400, 440, 94, (2, 9)),
        ("E004", ("ibuprofen", "mirtazapine", "omeprazole"), 1110, 1110, 1, (2, 2)),
    ],
)
def test_prescribe_row(
    scenarios: dict[str, ScenarioResult],
    encounter_id: str,
    regimen: tuple[str, ...],
    cost: int,
    baseline_cost: int,
    models: int,
    effort: tuple[int, int],
) -> None:
    row = scenarios[encounter_id]
    assert row.action == "prescribe"
    assert (row.regimen, row.cost) == (regimen, cost)
    assert (row.baseline_cost, row.saving) == (baseline_cost, baseline_cost - cost)
    assert (row.gamma_plus_models, row.minimum_cost) == (models, cost)
    assert (row.algorithm, row.expanded, row.generated) == ("astar", *effort)


def test_e001_row_names_agent_1s_regimen_and_the_unknown_risk_factor(
    scenarios: dict[str, ScenarioResult],
) -> None:
    row = scenarios["E001"]
    assert row.baseline_regimen == (
        "amlodipine",
        "linagliptin",
        "mirtazapine",
        "tramadol",
        "warfarin",
    )
    assert (row.present_risk_factors, row.unknown_risk_factors) == (("AGE65",), ("CKD",))
    assert row.chosen == ("warfarin", "mirtazapine", "enalapril", "paracetamol", "linagliptin")


@pytest.mark.parametrize(
    ("encounter_id", "action", "tests"),
    [("E002", "request_test", ("PREG",)), ("E005", "refer", ())],
)
def test_rows_without_a_prescription_have_no_cost(
    scenarios: dict[str, ScenarioResult], encounter_id: str, action: str, tests: tuple[str, ...]
) -> None:
    row = scenarios[encounter_id]
    assert (row.action, row.tests) == (action, tests)
    assert (row.regimen, row.cost, row.baseline_cost, row.saving) == ((), None, None, None)
    assert (row.gamma_plus_models, row.minimum_cost) == (0, None)
    assert (row.algorithm, row.expanded, row.generated) == (None, None, None)


def test_scenarios_serialise_to_json(scenarios: dict[str, ScenarioResult]) -> None:
    text = json.dumps(as_jsonable(tuple(scenarios.values())), sort_keys=True)
    assert json.loads(text)[0]["encounter_id"] == "E001"


# --- P7 on the mini formulary (28 records) ------------------------------------------------------


@pytest.fixture
def comparison(elena_formulary: Formulary, elena_costs: Mapping[str, int]) -> ComparisonResult:
    return run_search_comparison(elena_formulary, elena_costs, RegimenSpace(elena_formulary))


def test_comparison_headline_numbers_of_the_mini_formulary(comparison: ComparisonResult) -> None:
    assert (comparison.n_instances, comparison.n_satisfiable) == (28, 24)
    assert comparison.verdicts_agree_with_oracle
    assert comparison.n_unique_optimum == 24
    assert comparison.heuristic_admissible_at_root
    # Pain at 65 with liver disease: h sees ibuprofen (2.30) but not its omeprazole (2.70).
    assert comparison.n_root_heuristic_exact == 23


def test_every_optimal_configuration_finds_the_minimum_everywhere(
    comparison: ComparisonResult,
) -> None:
    by_name = {c.configuration: c for c in comparison.configurations}
    assert list(by_name) == [c.name for c in CONFIGURATIONS]
    for configuration in CONFIGURATIONS:
        summary = by_name[configuration.name]
        assert (summary.algorithm, summary.pruned) == (
            configuration.algorithm.value,
            configuration.pruned,
        )
        if configuration.algorithm.optimal:
            assert (summary.n_optimal, summary.max_excess_cost) == (24, 0)
    assert (by_name["greedy"].n_optimal, by_name["greedy"].max_excess_cost) == (8, 5140)
    assert (by_name["astar"].mean_expanded, by_name["astar"].max_generated) == (1.5833, 4)


def test_agent_1_is_already_cheapest_on_the_mini_formulary(comparison: ComparisonResult) -> None:
    baseline = comparison.baseline
    assert (baseline.n_optimal, baseline.max_excess_cost) == (24, 0)
    assert baseline.mean_cost == baseline.mean_minimum_cost == 400.4167


def test_instances_carry_the_oracle_and_one_run_per_configuration(
    comparison: ComparisonResult,
) -> None:
    by_record = {(i.conditions, i.present_risk_factors): i for i in comparison.instances}
    forced = by_record[(("PAIN",), ("AGE65", "LIVER"))]
    assert (forced.models, forced.minimum_cost, forced.optimal_regimens) == (1, 500, 1)
    assert (forced.root_heuristic, forced.baseline_cost) == (230.0, 500)
    assert [run.configuration for run in forced.runs] == [c.name for c in CONFIGURATIONS]
    assert {run.cost for run in forced.runs} == {500}

    untreatable = by_record[(("AF", "PAIN"), ("LIVER",))]
    assert (untreatable.models, untreatable.minimum_cost) == (0, None)
    assert not untreatable.search_finds_regimen
    assert (untreatable.root_heuristic, untreatable.baseline_cost, untreatable.runs) == (
        None,
        None,
        (),
    )


def test_effort_by_number_of_conditions(comparison: ComparisonResult) -> None:
    rows = comparison.by_n_conditions
    assert [(r.n_conditions, r.n_satisfiable) for r in rows] == [(1, 12), (2, 10), (3, 2)]
    assert [r.mean_minimum_cost for r in rows] == [269.1667, 494.0, 720.0]
    assert rows[0].mean_expanded["idastar"] == 1.0833
    assert set(rows[0].mean_expanded) == {c.name for c in CONFIGURATIONS}


def test_every_run_carries_its_depth_and_effective_branching_factor(
    comparison: ComparisonResult,
) -> None:
    runs = [run for instance in comparison.instances for run in instance.runs]
    assert runs
    for run in runs:
        assert 1 <= run.depth <= run.generated
        assert run.effective_branching_factor == effective_branching_factor(
            run.generated, run.depth
        )
    by_record = {(i.conditions, i.present_risk_factors): i for i in comparison.instances}
    forced = by_record[(("PAIN",), ("AGE65", "LIVER"))]
    for run in forced.runs:
        assert (run.depth, run.effective_branching_factor) == (1, float(run.generated))


def _mean_of(values: list[float]) -> float:
    return round(sum(values) / len(values), 4)


def test_summaries_carry_the_mean_effective_branching_factor(comparison: ComparisonResult) -> None:
    satisfiable = [i for i in comparison.instances if i.minimum_cost is not None]
    for index, summary in enumerate(comparison.configurations):
        b_stars = [i.runs[index].effective_branching_factor for i in satisfiable]
        assert summary.mean_effective_branching_factor == _mean_of(
            [b for b in b_stars if b is not None]
        )
        assert summary.mean_effective_branching_factor >= 1.0
    for row in comparison.by_n_conditions:
        group = [i for i in satisfiable if len(i.conditions) == row.n_conditions]
        assert set(row.mean_effective_branching_factor) == {c.name for c in CONFIGURATIONS}
        for index, configuration in enumerate(CONFIGURATIONS):
            b_stars = [i.runs[index].effective_branching_factor for i in group]
            assert row.mean_effective_branching_factor[configuration.name] == _mean_of(
                [b for b in b_stars if b is not None]
            )


def test_comparison_does_not_depend_on_workers(
    elena_formulary: Formulary, elena_costs: Mapping[str, int], comparison: ComparisonResult
) -> None:
    space = RegimenSpace(elena_formulary)
    parallel = run_search_comparison(elena_formulary, elena_costs, space, workers=2)
    assert json.dumps(as_jsonable(parallel), sort_keys=True) == json.dumps(
        as_jsonable(comparison), sort_keys=True
    )


def test_comparison_rejects_a_worker_count_below_one(
    elena_formulary: Formulary, elena_costs: Mapping[str, int]
) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1, got 0"):
        run_search_comparison(
            elena_formulary, elena_costs, RegimenSpace(elena_formulary), workers=0
        )
