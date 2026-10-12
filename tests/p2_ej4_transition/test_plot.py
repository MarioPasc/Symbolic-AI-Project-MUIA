"""Tests of the Gantt figures of EJ4 (``plot.py``): bands, colours, layout, reproducible files."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import ej4_toys
import matplotlib as mpl
import pytest

from symbolic_ai.p2_ej4_transition.plot import (
    GANTT_EXTENDED_FIGURE,
    GANTT_FIGURE,
    drug_colours,
    gantt_figure,
    presence_bands,
    save_gantt,
)
from symbolic_ai.p2_ej4_transition.solve import find_min_horizon
from symbolic_ai.p2_ej4_transition.worked_example import ScheduleTable
from symbolic_ai.viz import TOL

#: The worked example's schedule (design §1): sertraline tapered on days 1-4, mirtazapine from
#: day 5, the analgesic on day 12.
CASE_STUDY = ScheduleTable(
    problem_id="DEP|LIVER|+cond:PAIN",
    conditions=("DEP", "PAIN"),
    horizon=12,
    rows=(
        (
            ("sertraline", 0),
            *((("sertraline", k)) for k in range(1, 5)),
            *([("mirtazapine", 0)] * 8),
        ),
        (*([None] * 12), ("ibuprofen", 0)),
    ),
    washout_days=(("ibuprofen", 1), ("mirtazapine", 4), ("sertraline", 7)),
    taper_days=(("ibuprofen", 0), ("mirtazapine", 3), ("sertraline", 4)),
)
NAMES = {"sertraline": "Sertralina", "mirtazapine": "Mirtazapina", "ibuprofen": "Ibuprofeno"}


def _toy_table(toy: ej4_toys.KnownAnswer) -> ScheduleTable:
    result = find_min_horizon(toy.problem)
    assert result.schedule is not None and result.h_star is not None
    return ScheduleTable.from_schedule(toy.problem, result.schedule, result.h_star)


def test_sertraline_is_present_until_day_11() -> None:
    assert presence_bands(CASE_STUDY) == [(0, "sertraline", 5, 11)]


def test_bands_follow_the_last_dose_and_skip_drugs_still_given() -> None:
    # T5 (solver's schedule): x1 last given on day 0 (washout 3), y1 on day 1 (washout 2); z1 is
    # given on the last day, so it has no band.
    table = _toy_table(ej4_toys.T5)
    assert [row[-1] for row in table.rows] == [("x2", 0), ("y2", 0), ("z1", 0)]
    assert presence_bands(table) == [(0, "x1", 1, 3), (1, "y1", 2, 3)]


def test_bands_are_cut_at_the_horizon() -> None:
    long_washout = dataclasses.replace(
        CASE_STUDY, washout_days=(("ibuprofen", 1), ("mirtazapine", 4), ("sertraline", 30))
    )
    assert presence_bands(long_washout) == [(0, "sertraline", 5, 12)]


def test_one_colour_per_drug_in_order_of_appearance() -> None:
    colours = drug_colours(CASE_STUDY)
    assert list(colours) == ["sertraline", "mirtazapine", "ibuprofen"]
    assert colours["sertraline"] == TOL["blue"]
    assert len(set(colours.values())) == 3


def test_figure_labels_are_spanish_condition_names() -> None:
    fig = gantt_figure(CASE_STUDY, NAMES)
    (ax,) = fig.axes
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Depresión", "Dolor crónico"]
    assert ax.get_xlabel() == "Día"
    legend_texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend_texts[:3] == ["Sertralina", "Mirtazapina", "Ibuprofeno"]
    assert "sin tratar" in legend_texts


def test_save_gantt_is_reproducible_and_restores_rc_params(tmp_path: Path) -> None:
    before = dict(mpl.rcParams)
    first = save_gantt(CASE_STUDY, NAMES, tmp_path / "a" / GANTT_FIGURE)
    second = save_gantt(CASE_STUDY, NAMES, tmp_path / "b" / GANTT_FIGURE)
    assert [p.suffix for p in first] == [".pdf", ".png"]
    for one, other in zip(first, second, strict=True):
        assert one.read_bytes() == other.read_bytes()
    assert dict(mpl.rcParams) == before


@pytest.mark.parametrize("toy", [ej4_toys.T3, ej4_toys.T5], ids=["T3", "T5"])
def test_toy_schedules_can_be_drawn(toy: ej4_toys.KnownAnswer, tmp_path: Path) -> None:
    written = save_gantt(_toy_table(toy), {}, tmp_path / GANTT_EXTENDED_FIGURE)
    assert all(p.stat().st_size > 0 for p in written)
