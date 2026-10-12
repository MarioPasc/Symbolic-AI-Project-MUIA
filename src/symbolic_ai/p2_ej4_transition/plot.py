"""Gantt figures of EJ4 for the beamer slides (design §7 R1.5 and R3).

``fig_ej4_gantt`` draws the worked example's schedule and ``fig_ej4_gantt_extended`` the extended
case of R3: one row per condition, days ``0..H`` on the x axis, one colour per drug, taper days
hatched and labelled ``↓k``, untreated days outlined, and, below each row, a light band for the
days a stopped drug is still in the body (its washout after the last dose). Day 0 is the event; a
dashed line marks ``H*``.

The figures are sized for a 16:9 slide in a sans-serif font, with a local ``rc_context``: the
shared IEEE style of ``symbolic_ai.viz`` (serif, for the P1 report) is not used or changed; only
its colour palette and its reproducible ``save_figure`` are reused. Labels come from
``tables.LABELS_ES``. Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §7 (private).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.typing import RcKeyType

from symbolic_ai.p2_ej4_transition.problem import drug_of
from symbolic_ai.p2_ej4_transition.tables import LABELS_ES
from symbolic_ai.p2_ej4_transition.worked_example import ScheduleTable
from symbolic_ai.viz import TOL, save_figure

__all__ = [
    "GANTT_EXTENDED_FIGURE",
    "GANTT_FIGURE",
    "SLIDE_RCPARAMS",
    "drug_colours",
    "gantt_figure",
    "presence_bands",
    "save_gantt",
]

#: File stem of the worked example's Gantt (R1).
GANTT_FIGURE = "fig_ej4_gantt"
#: File stem of the extended case's Gantt (R3).
GANTT_EXTENDED_FIGURE = "fig_ej4_gantt_extended"

#: Matplotlib settings for a figure about 14 cm wide on a 16:9 beamer slide (sans-serif).
SLIDE_RCPARAMS: Mapping[RcKeyType, object] = MappingProxyType(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.hashsalt": "symai",
        "savefig.dpi": 300,
    }
)

#: Width of the figures, in inches (about 14 cm, the text width of a 16:9 slide).
_WIDTH_IN = 5.6
#: Colours of the drugs, in order of first appearance (Paul Tol's bright palette, then grey).
_PALETTE = ("blue", "green", "orange", "purple", "cyan", "yellow", "red", "grey")
#: Height of one row, in inches; a 16:9 slide leaves about 7 cm of height below the title.
_ROW_IN = 0.34
#: Up to this many rows the legend goes below the chart; with more it goes to the right.
_ROWS_WITH_LEGEND_BELOW = 3
#: Vertical layout of a row: the bar, then the washout band below it.
_BAR_HALF = 0.28
_BAND_TOP, _BAND_BOTTOM = 0.32, 0.44


def drug_colours(table: ScheduleTable) -> dict[str, str]:
    """Return one colour per drug of the schedule, in order of first appearance (row by row).

    Parameters
    ----------
    table
        The schedule.

    Returns
    -------
    dict[str, str]
        ``drug_id -> hex colour``.
    """
    drugs = list(dict.fromkeys(d for row in table.rows for v in row if (d := drug_of(v))))
    return {drug: TOL[_PALETTE[i % len(_PALETTE)]] for i, drug in enumerate(drugs)}


def presence_bands(table: ScheduleTable) -> list[tuple[int, str, int, int]]:
    """Return the washout bands: a drug stopped before ``H`` is still present after its last dose.

    Parameters
    ----------
    table
        The schedule.

    Returns
    -------
    list[tuple[int, str, int, int]]
        ``(row, drug, first_day, last_day)``: the drug of row ``row`` was last given on
        ``first_day - 1`` and is present until ``last_day`` (its washout, cut at ``H``).
    """
    washout = dict(table.washout_days)
    bands = []
    for index, row in enumerate(table.rows):
        last_given: dict[str, int] = {}
        for day, value in enumerate(row):
            drug = drug_of(value)
            if drug is not None:
                last_given[drug] = day
        for drug, day in last_given.items():
            end = min(day + washout[drug], table.horizon)
            if day < table.horizon and end > day:
                bands.append((index, drug, day + 1, end))
    return bands


def gantt_figure(
    table: ScheduleTable,
    names: Mapping[str, str],
    labels: Mapping[str, str] = LABELS_ES,
) -> Figure:
    """Draw the Gantt chart of a schedule (design §7 R1.5).

    Parameters
    ----------
    table
        The schedule (``worked_example.ScheduleTable``).
    names
        Display name per drug (the formulary's ``name_es``); missing ones use the identifier.
    labels
        The label dictionary (condition names, legend entries).

    Returns
    -------
    Figure
        The figure (object-oriented API, no ``pyplot``).
    """
    colours = drug_colours(table)
    rows = len(table.conditions)
    below = rows <= _ROWS_WITH_LEGEND_BELOW
    height = _ROW_IN * rows + (1.6 if below else 1.0)
    fig = Figure(figsize=(_WIDTH_IN, height), layout="constrained")
    ax = fig.add_subplot()
    for index, row in enumerate(table.rows):
        for day, value in enumerate(row):
            _draw_cell(ax, index, day, value, colours)
    for index, drug, first, last in presence_bands(table):
        ax.add_patch(
            Rectangle(
                (first - 0.5, index + _BAND_TOP),
                last - first + 1,
                _BAND_BOTTOM - _BAND_TOP,
                facecolor=colours[drug],
                alpha=0.35,
                linewidth=0,
            )
        )
    _decorate(ax, table, labels)
    _legend(fig, table, colours, names, labels, below=below)
    return fig


def _draw_cell(
    ax: Axes, row: int, day: int, value: tuple[str, int] | None, colours: Mapping[str, str]
) -> None:
    """Draw one day of one condition: a bar, a hatched taper day, or an outlined untreated day."""
    origin = (day - 0.5, row - _BAR_HALF)
    if value is None:
        ax.add_patch(
            Rectangle(
                origin,
                1,
                2 * _BAR_HALF,
                facecolor="none",
                edgecolor="#999999",
                linestyle=(0, (2, 2)),
                linewidth=0.6,
            )
        )
        return
    drug, phase = value
    ax.add_patch(
        Rectangle(
            origin,
            1,
            2 * _BAR_HALF,
            facecolor=colours[drug],
            alpha=0.55 if phase else 1.0,
            hatch="////" if phase else None,
            edgecolor="white",
            linewidth=0.8,
        )
    )
    if phase:
        ax.text(day, row, f"↓{phase}", ha="center", va="center", fontsize=7, color="black")


def _decorate(ax: Axes, table: ScheduleTable, labels: Mapping[str, str]) -> None:
    """Set the axes: days, condition names, the event on day 0 and the line at ``H*``."""
    rows = len(table.conditions)
    ax.set_xlim(-0.6, table.horizon + 0.6)
    ax.set_ylim(rows - 0.45, -0.75)
    ax.set_xticks(range(table.horizon + 1))
    ax.set_yticks(range(rows))
    ax.set_yticklabels([labels.get(f"condition.{c}", c) for c in table.conditions])
    ax.set_xlabel(labels["day"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.annotate(
        labels["event"],
        xy=(0, -0.38),
        xytext=(0, -0.68),
        ha="center",
        va="center",
        fontsize=8,
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.8},
    )
    ax.axvline(table.horizon - 0.5, color="black", linestyle="--", linewidth=0.8)
    ax.text(
        table.horizon - 0.55,
        -0.62,
        labels["h_star"].format(h=table.horizon),
        ha="right",
        va="center",
        fontsize=8,
    )


def _legend(
    fig: Figure,
    table: ScheduleTable,
    colours: Mapping[str, str],
    names: Mapping[str, str],
    labels: Mapping[str, str],
    *,
    below: bool,
) -> None:
    """Add the legend: one entry per drug, then taper, washout and untreated (when they occur)."""
    values = [v for row in table.rows for v in row]
    handles = [Patch(facecolor=c, label=names.get(d, d)) for d, c in colours.items()]
    if any(v is not None and v[1] > 0 for v in values):
        handles.append(
            Patch(facecolor="#cccccc", hatch="////", edgecolor="white", label=labels["taper"])
        )
    if presence_bands(table):
        handles.append(Patch(facecolor="#888888", alpha=0.35, label=labels["presence"]))
    if any(v is None for v in values):
        handles.append(
            Patch(
                facecolor="none",
                edgecolor="#999999",
                linestyle=(0, (2, 2)),
                label=labels["untreated"],
            )
        )
    fig.legend(
        handles=handles,
        loc="outside lower center" if below else "outside right center",
        ncols=3 if below else 1,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )


def save_gantt(
    table: ScheduleTable,
    names: Mapping[str, str],
    out_stem: Path,
    labels: Mapping[str, str] = LABELS_ES,
) -> tuple[Path, ...]:
    """Draw the Gantt chart of ``table`` and write it as PDF and PNG, reproducibly.

    Parameters
    ----------
    table
        The schedule.
    names
        Display name per drug.
    out_stem
        Output path without extension (e.g. ``outputs/ej4/fig_ej4_gantt``).
    labels
        The label dictionary.

    Returns
    -------
    tuple[Path, ...]
        The PDF and PNG written.
    """
    # rc_context() restores every rcParam on exit, including those updated inside it.
    with mpl.rc_context():
        mpl.rcParams.update(SLIDE_RCPARAMS)
        fig = gantt_figure(table, names, labels)
        return save_figure(fig, out_stem)
