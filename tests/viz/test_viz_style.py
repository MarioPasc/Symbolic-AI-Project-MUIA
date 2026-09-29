"""Tests for symbolic_ai.viz.style: sizes, the style context and reproducible saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from symbolic_ai.viz import (
    IEEE_COLUMN_WIDTH_IN,
    IEEE_RCPARAMS,
    IEEE_TEXT_WIDTH_IN,
    TOL,
    figure_size,
    ieee_style,
    save_figure,
)


def _small_figure() -> Figure:
    fig = Figure(figsize=figure_size())
    ax = fig.subplots()
    ax.plot([1, 2, 3], [1, 4, 9], color=TOL["blue"])
    return fig


class TestFigureSize:
    def test_one_column_uses_the_ieee_column_width(self) -> None:
        width, height = figure_size(columns=1, aspect=0.5)
        assert width == IEEE_COLUMN_WIDTH_IN
        assert height == pytest.approx(IEEE_COLUMN_WIDTH_IN * 0.5)

    def test_two_columns_use_the_text_width(self) -> None:
        assert figure_size(columns=2)[0] == IEEE_TEXT_WIDTH_IN

    @pytest.mark.parametrize(("columns", "aspect"), [(0, 0.6), (3, 0.6), (1, 0.0), (1, -1.0)])
    def test_invalid_arguments_raise(self, columns: int, aspect: float) -> None:
        with pytest.raises(ValueError, match="must be"):
            figure_size(columns=columns, aspect=aspect)


class TestIeeeStyle:
    def test_style_is_active_inside_and_restored_after(self) -> None:
        before = mpl.rcParams["font.size"]
        with ieee_style():
            assert mpl.rcParams["font.size"] == IEEE_RCPARAMS["font.size"]
            assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["font.size"] == before

    def test_shared_constants_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            TOL["blue"] = "#000000"  # type: ignore[index]


class TestSaveFigure:
    def test_writes_every_requested_format(self, tmp_path: Path) -> None:
        with ieee_style():
            written = save_figure(_small_figure(), tmp_path / "sub" / "fig")
        assert [p.suffix for p in written] == [".pdf", ".png"]
        assert all(p.stat().st_size > 0 for p in written)

    def test_saving_twice_gives_identical_bytes(self, tmp_path: Path) -> None:
        with ieee_style():
            first = save_figure(_small_figure(), tmp_path / "a")
            second = save_figure(_small_figure(), tmp_path / "b")
        for path_a, path_b in zip(first, second, strict=True):
            assert path_a.read_bytes() == path_b.read_bytes()

    @pytest.mark.parametrize("formats", [(), ("jpg",), ("pdf", "eps")])
    def test_unsupported_formats_raise(self, tmp_path: Path, formats: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="formats"):
            save_figure(_small_figure(), tmp_path / "fig", formats=formats)
