"""Report figures of EJ1, drawn from the saved results file (never from values held in memory).

Figure labels are report text and therefore in Spanish, the report's language.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator

from symbolic_ai.p1_ej1_logic.errors import ResultsFormatError
from symbolic_ai.viz import TOL, figure_size, ieee_style, save_figure

__all__ = ["VALUE_ORDERING_FIGURE", "plot_value_ordering"]

#: File stem of the value-ordering figure inside the output directory.
VALUE_ORDERING_FIGURE = "fig_ej1_value_ordering"

#: Bars shorter than this fraction of the tallest one are labelled with their count.
_LABEL_BELOW = 0.4

#: The P2 configurations drawn, in legend order: (results key, label, colour, marker).
_ORDERINGS = (
    ("false_first", "Falso primero", TOL["blue"], "o"),
    ("true_first", "Verdadero primero", TOL["red"], "s"),
)


def plot_value_ordering(value_ordering: Mapping[str, object], out_stem: Path) -> tuple[Path, ...]:
    """Draw Fig. 1 of EJ1: redundant drugs under each DPLL configuration of experiment P2.

    The excess of a regimen is its number of drugs minus the oracle's minimum for the same record.
    Panel (a): mean excess by number of present conditions. Panel (b): number of satisfiable
    records by excess. Tick labels use the decimal comma of the (Spanish) report.

    Parameters
    ----------
    value_ordering : Mapping[str, object]
        The ``value_ordering`` section of ``results.json`` (schema ``symai.ej1.results/3``).
    out_stem : Path
        Output path without extension; a PDF and a PNG are written.

    Returns
    -------
    tuple[Path, ...]
        The files written.

    Raises
    ------
    ResultsFormatError
        If the section lacks a field the figure needs.
    """
    by_conditions = _records(value_ordering, "by_n_conditions")
    summaries = {str(_field(s, "ordering")): s for s in _records(value_ordering, "orderings")}
    with ieee_style():
        fig = Figure(figsize=figure_size(columns=1, aspect=0.5), layout="constrained")
        ax_a, ax_b = fig.subplots(1, 2, gridspec_kw={"width_ratios": (1.15, 1.0)})
        _panel_by_conditions(ax_a, by_conditions)
        _panel_histogram(ax_b, summaries)
        handles, labels = ax_a.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2)
        return save_figure(fig, out_stem)


def _panel_by_conditions(ax: Axes, rows: Sequence[Mapping[str, object]]) -> None:
    """Panel (a): mean excess over the minimum against the number of present conditions."""
    n_conditions = [_int(r, "n_conditions") for r in rows]
    minimum = np.array([_float(r, "mean_minimum") for r in rows])
    for key, label, colour, marker in _ORDERINGS:
        mean_size = np.array([_float(r, f"mean_{key}") for r in rows])
        ax.plot(n_conditions, mean_size - minimum, marker=marker, color=colour, label=label)
    ax.set_xlabel("Condiciones presentes")
    ax.set_ylabel("Exceso medio (fármacos)")
    ax.set_xticks(n_conditions)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_comma))
    ax.set_title("(a)", loc="left")


def _panel_histogram(ax: Axes, summaries: Mapping[str, Mapping[str, object]]) -> None:
    """Panel (b): satisfiable instances by excess over the minimum, one bar group per excess."""
    histograms = {key: _histogram(summaries, key) for key, *_ in _ORDERINGS}
    excess_values = sorted({k for h in histograms.values() for k in h})
    tallest = max(max(h.values()) for h in histograms.values())
    width = 0.8 / len(_ORDERINGS)
    offsets = [(i - (len(_ORDERINGS) - 1) / 2) * width for i in range(len(_ORDERINGS))]
    for offset, (key, label, colour, _) in zip(offsets, _ORDERINGS, strict=True):
        counts = [histograms[key].get(k, 0) for k in excess_values]
        x = np.array(excess_values) + offset
        bars = ax.bar(x, counts, width=width, color=colour, label=label)
        # Only bars too short to read off the axis get a label; tall ones would overlap.
        labels = [str(c) if 0 < c < _LABEL_BELOW * tallest else "" for c in counts]
        ax.bar_label(bars, labels=labels, fontsize=5.5, padding=1.5, rotation=90)
    ax.set_xlabel("Exceso (fármacos)")
    ax.set_ylabel("Historias")
    ax.set_xticks(excess_values)
    ax.set_ylim(0, tallest * 1.1)
    ax.set_title("(b)", loc="left")


def _decimal_comma(value: float, _position: int | None) -> str:
    """Format a tick value with a decimal comma (``0,5``), as the report writes numbers."""
    return f"{value:g}".replace(".", ",")


def _histogram(summaries: Mapping[str, Mapping[str, object]], key: str) -> dict[int, int]:
    """Return the excess histogram of ordering ``key`` with integer keys."""
    if key not in summaries:
        raise ResultsFormatError(f"value_ordering.orderings has no entry for {key}")
    raw = _field(summaries[key], "excess_histogram")
    if not isinstance(raw, Mapping):
        raise ResultsFormatError(f"excess_histogram of {key} is not a mapping")
    return {int(k): int(v) for k, v in raw.items()}


def _field(record: Mapping[str, object], name: str) -> object:
    """Return ``record[name]`` or raise :class:`ResultsFormatError` naming the missing field."""
    if name not in record:
        raise ResultsFormatError(f"results field {name!r} is missing")
    return record[name]


def _records(section: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    """Return the list of objects ``section[name]``, checking its type."""
    value = _field(section, name)
    if not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value):
        raise ResultsFormatError(f"results field {name!r} is not a list of objects")
    return value


def _int(record: Mapping[str, object], name: str) -> int:
    """Return an integer field of ``record``."""
    value = _field(record, name)
    if not isinstance(value, int):
        raise ResultsFormatError(f"results field {name!r} is not an integer: {value!r}")
    return value


def _float(record: Mapping[str, object], name: str) -> float:
    """Return a numeric field of ``record`` as a float."""
    value = _field(record, name)
    if not isinstance(value, int | float):
        raise ResultsFormatError(f"results field {name!r} is not a number: {value!r}")
    return float(value)
