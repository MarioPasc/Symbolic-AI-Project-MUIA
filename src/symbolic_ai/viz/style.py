"""IEEE figure style: rcParams, the Paul Tol bright palette, IEEE widths and a reproducible saver.

The rcParams and the palette are ported from the team's earlier figure scripts (IsalHG
``viz/style.py``, as reused in the GenAI proposal figure), with Times-like STIX fonts added so that
figures match IEEEtran's Times body text. Every figure of the practical is drawn inside
:func:`ieee_style` and written with :func:`save_figure`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType

__all__ = [
    "IEEE_COLUMN_WIDTH_IN",
    "IEEE_RCPARAMS",
    "IEEE_TEXT_WIDTH_IN",
    "TOL",
    "figure_size",
    "ieee_style",
    "save_figure",
]

#: Width of one column of the IEEE two-column conference template, in inches.
IEEE_COLUMN_WIDTH_IN = 3.5
#: Width of the full text block (both columns and the gutter), in inches.
IEEE_TEXT_WIDTH_IN = 7.16

#: rcParams for IEEE single-column figures; read-only so no caller can mutate the shared style.
IEEE_RCPARAMS: Mapping[RcKeyType, object] = MappingProxyType(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        # TrueType (not Type 3) fonts in PDF and PS output, as IEEE PDF eXpress requires.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

#: Paul Tol's "bright" qualitative palette (colour-blind safe), by colour name.
TOL: Mapping[str, str] = MappingProxyType(
    {
        "blue": "#4477AA",
        "red": "#EE6677",
        "green": "#228833",
        "yellow": "#CCBB44",
        "cyan": "#66CCEE",
        "purple": "#AA3377",
        "grey": "#BBBBBB",
        "orange": "#EE7733",
    }
)


@contextmanager
def ieee_style() -> Iterator[None]:
    """Apply :data:`IEEE_RCPARAMS` inside a ``with`` block, then restore the previous rcParams.

    Yields
    ------
    None
        Control returns to the caller with the IEEE style active.
    """
    # rc_context() restores every rcParam on exit, including those updated inside it.
    with mpl.rc_context():
        mpl.rcParams.update(IEEE_RCPARAMS)
        yield


def figure_size(columns: int = 1, aspect: float = 0.62) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for a figure spanning one or two IEEE columns.

    Parameters
    ----------
    columns : int
        ``1`` for a single column (:data:`IEEE_COLUMN_WIDTH_IN`), ``2`` for the full text width.
    aspect : float
        Height divided by width.

    Returns
    -------
    tuple[float, float]
        The figure size to pass as ``figsize``.

    Raises
    ------
    ValueError
        If ``columns`` is not 1 or 2, or ``aspect`` is not positive.
    """
    if columns not in (1, 2):
        raise ValueError(f"columns must be 1 or 2, got {columns}")
    if aspect <= 0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    width = IEEE_COLUMN_WIDTH_IN if columns == 1 else IEEE_TEXT_WIDTH_IN
    return width, width * aspect


def save_figure(
    fig: Figure, out_stem: Path, formats: Sequence[str] = ("pdf", "png")
) -> tuple[Path, ...]:
    """Write ``fig`` as ``out_stem.<format>`` for each format, reproducibly.

    The PDF is written without a creation date and the PNG without a software tag, so the same
    figure saved twice gives byte-identical files (the results of the report can be re-checked by
    hash).

    Parameters
    ----------
    fig : Figure
        The figure to save.
    out_stem : Path
        Output path without extension; parent directories are created.
    formats : Sequence[str]
        File formats to write, e.g. ``("pdf", "png")``; PDF is the vector version for LaTeX.

    Returns
    -------
    tuple[Path, ...]
        The paths written, in the order of ``formats``.

    Raises
    ------
    ValueError
        If ``formats`` is empty or contains a format other than ``pdf``, ``png`` or ``svg``.
    """
    supported = {"pdf", "png", "svg"}
    if not formats or not set(formats) <= supported:
        raise ValueError(
            f"formats must be a non-empty subset of {sorted(supported)}, got {formats}"
        )
    metadata: dict[str, dict[str, str | None]] = {
        "pdf": {"CreationDate": None, "Producer": None},
        "png": {"Software": None},
        "svg": {"Date": None},
    }
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, metadata=metadata[fmt])
        written.append(path)
    return tuple(written)
