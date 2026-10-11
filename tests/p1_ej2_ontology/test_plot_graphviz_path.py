"""Tests of ``ensure_graphviz_on_path``: Graphviz is found when the environment is not activated."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from symbolic_ai.p1_ej2_ontology.plot import ensure_graphviz_on_path

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX executable bits")


def _fake_dot(directory: Path) -> Path:
    directory.mkdir(parents=True)
    dot = directory / "dot"
    dot.write_text("#!/bin/sh\n", encoding="utf-8")
    dot.chmod(0o755)
    return dot


def test_dot_in_the_environment_is_put_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dot = _fake_dot(tmp_path / "env" / "bin")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    assert ensure_graphviz_on_path()
    assert shutil.which("dot") == str(dot)


def test_dot_already_on_path_leaves_path_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_dot(tmp_path / "system")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setenv("PATH", str(tmp_path / "system"))
    assert ensure_graphviz_on_path()
    assert os.environ["PATH"] == str(tmp_path / "system")


def test_no_dot_anywhere_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    assert not ensure_graphviz_on_path()
    assert os.environ["PATH"] == str(tmp_path / "elsewhere")
