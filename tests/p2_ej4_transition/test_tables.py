"""Tests of the LaTeX tables of EJ4 (``tables.py``): codes, escaping, content, compilation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ej4_toys
import pytest

from symbolic_ai.dataloader import load_drug_costs, load_formulary, load_pharmacokinetics
from symbolic_ai.p2_ej4_transition import EJ4_DATA_VERSION
from symbolic_ai.p2_ej4_transition.experiments import (
    AlgorithmRun,
    CategoryAgreement,
    PopulationRow,
    SyntheticResult,
    ValidationResult,
    run_algorithm_table,
    run_min_conflicts_rows,
    run_synthetic,
    run_validation,
)
from symbolic_ai.p2_ej4_transition.instances import (
    Category,
    ContinuityReport,
    History,
    ProblemEntry,
    case_study_problem,
)
from symbolic_ai.p2_ej4_transition.tables import (
    LABELS_ES,
    algorithm_legend,
    algorithm_table,
    compact_values,
    domain_legend,
    domain_table,
    drug_codes,
    latex_text,
    validation_table,
    value_code,
)
from symbolic_ai.p2_ej4_transition.worked_example import WorkedExample, build_worked_example

NAMES = {
    "sertraline": "Sertralina",
    "mirtazapine": "Mirtazapina",
    "ibuprofen": "Ibuprofeno",
    "tramadol": "Tramadol",
    "paracetamol": "Paracetamol",
    "metformin": "Metformina",
}
CODES = {"sertraline": "S", "mirtazapine": "M", "ibuprofen": "I", "tramadol": "T"}


@pytest.fixture(scope="module")
def example() -> WorkedExample:
    """R1 on the real case study."""
    version = EJ4_DATA_VERSION
    formulary = load_formulary(version=version)
    problem = case_study_problem(
        formulary, load_drug_costs(version=version), load_pharmacokinetics(version=version)
    )
    return build_worked_example(problem, formulary.contraindications)


@pytest.fixture(scope="module")
def validation() -> ValidationResult:
    """R2 on the seven toys (fast)."""
    toys = [
        ProblemEntry(t.problem, Category.FORCED_ONLY, ((History(("X",), ()), t.problem.event),))
        for t in ej4_toys.KNOWN_ANSWERS
    ]
    return run_validation(toys)


# --- codes and escaping -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "latex"),
    [
        ("50% & #1_a", r"50\% \& \#1\_a"),
        ("S↓2", r"S$\downarrow$2"),
        ("∅", r"$\emptyset$"),
        ("H* = 12", "H$^*$ = 12"),
        ("{x}", r"\{x\}"),
        ("Día", "Día"),
    ],
)
def test_latex_text(text: str, latex: str) -> None:
    assert latex_text(text) == latex


def test_drug_codes_use_one_letter_unless_they_clash() -> None:
    assert drug_codes(["sertraline", "ibuprofen"], NAMES) == {"ibuprofen": "I", "sertraline": "S"}
    assert drug_codes(["metformin", "mirtazapine"], NAMES) == {
        "metformin": "Me",
        "mirtazapine": "Mi",
    }
    assert drug_codes(["zz"], {}) == {"zz": "Z"}


@pytest.mark.parametrize(
    ("value", "code"), [(None, "∅"), (("sertraline", 0), "S"), (("sertraline", 2), "S↓2")]
)
def test_value_code(value: tuple[str, int] | None, code: str) -> None:
    assert value_code(value, CODES) == code


def test_compact_values_groups_by_drug_and_ranges_taper_days() -> None:
    values = [("sertraline", 0), ("sertraline", 2), ("sertraline", 3), ("sertraline", 4)]
    values += [("mirtazapine", 1), ("mirtazapine", 3), None]
    assert compact_values(values, CODES) == ("S, S↓2-4", "M↓1, M↓3", "∅")
    assert compact_values([], CODES) == ()


# --- content ------------------------------------------------------------------------------------


@pytest.mark.integration
def test_domain_table_at_11_marks_the_emptied_domain(example: WorkedExample) -> None:
    codes = drug_codes([r.drug_id for r in example.instance.drugs], NAMES)
    table = domain_table(example.at(11), ("DEP", "PAIN"), codes)
    assert table.count(r"\textcolor{red}{\textbf{vacío}}") == 1
    dep_row = next(line for line in table.splitlines() if line.startswith("Depresión"))
    cells = dep_row.removesuffix(r" \\ \addlinespace").split(" & ")
    assert len(cells) == 1 + 12
    assert "vacío" in cells[1 + 2]  # day 2
    assert r"\begin{tabular}{l*{12}{c}}" in table
    assert r"\toprule" in table and r"\bottomrule" in table


@pytest.mark.integration
def test_domain_table_at_12_has_13_columns_and_no_emptied_domain(example: WorkedExample) -> None:
    codes = drug_codes([r.drug_id for r in example.instance.drugs], NAMES)
    table = domain_table(example.at(12), ("DEP", "PAIN"), codes)
    assert r"\begin{tabular}{l*{13}{c}}" in table
    assert "vacío" not in table
    pain_row = next(line for line in table.splitlines() if line.startswith("Dolor crónico"))
    assert pain_row.removesuffix(r" \\ \addlinespace").split(" & ")[-1] == "I, T"


def test_domain_legend_defines_every_code() -> None:
    legend = domain_legend(CODES, NAMES)
    for code in CODES.values():
        assert f"{code}: " in legend
    assert r"$\emptyset$: sin tratar" in legend
    assert "gris" in legend


#: The continuity counts of design §5 (v2.3), as a report (identifiers are placeholders).
CONTINUITY = ContinuityReport(
    histories=4032,
    equal=2720,
    different=(),
    companion=tuple(f"PAIN|C{i}" for i in range(32)),
    refer=1280,
    unexpected=(),
)
CATEGORIES = (
    CategoryAgreement("trivial", 607, 607, ((0, 607),), ()),
    CategoryAgreement("knock_on", 162, 161, ((2, 24), (12, 138)), ("X",)),
)
POPULATION = tuple(
    PopulationRow(key, key, 100_000, 162, (), median, 100_001, hits, 1.0)
    for key, median, hits in (
        ("bt", 100_001.0, 122),
        ("bt_fc", 899.0, 0),
        ("bt_fc_mrv", 27.5, 0),
        ("main", 0.0, 0),
    )
)


@pytest.fixture(scope="module")
def synthetic() -> SyntheticResult:
    """A small synthetic batch (fast)."""
    return run_synthetic(seed=2026, count=30)


def _rows(table: str) -> list[str]:
    """The body rows of the R2 table (between the rules)."""
    body = table.split(r"\midrule")[1].split(r"\bottomrule")[0]
    return [row for row in body.strip().splitlines() if row]


def test_validation_table_has_one_row_per_check(
    validation: ValidationResult, synthetic: SyntheticResult
) -> None:
    table = validation_table(
        validation,
        census_ok=True,
        synthetic=synthetic,
        continuity=CONTINUITY,
        categories=CATEGORIES,
        census_events=17_552,
    )
    rows = _rows(table)
    assert len(rows) == 6  # real, synthetic, mutations, continuity, categories, census
    assert rows[0].endswith(r"7/7 \tabularnewline")  # CSP = oracle = checker on the toys
    assert rows[1].endswith(rf"{synthetic.three_way_agree}/{synthetic.count} \tabularnewline")
    applied = sum(m.applied for m in validation.mutations)
    assert rows[2].endswith(rf"{applied}/{applied} \tabularnewline")
    assert r"2\,720/2\,720" in rows[3] and "32 con acompañante" in rows[3]
    assert rows[4].endswith(r"768/769 \tabularnewline")
    assert r"17\,552 eventos" in rows[5] and rows[5].endswith(r"sí \tabularnewline")
    # the per-rule breakdown stays in results.json, not on the slide
    assert latex_text(LABELS_ES["mutation.drop_taper_day"]) not in table


def test_validation_table_without_oracle_and_optional_rows(validation: ValidationResult) -> None:
    table = validation_table(
        ValidationResult(**{**_fields(validation), "oracle": None, "three_way_agree": None}),
        census_ok=False,
    )
    rows = _rows(table)
    assert len(rows) == 4  # real, synthetic (not run), mutations, census
    assert "sin oráculo" in rows[0]
    assert LABELS_ES["r2.not_run"] in rows[1]
    assert rows[3].endswith(r"no \tabularnewline")


def _fields(validation: ValidationResult) -> dict[str, object]:
    return {name: getattr(validation, name) for name in ValidationResult.__dataclass_fields__}


def _run(configuration: str, horizon: int, status: str, assignments: int) -> AlgorithmRun:
    return AlgorithmRun("P", configuration, configuration, horizon, status, assignments, 9, 0, 0.1)


def test_algorithm_table_spells_out_unexpected_outcomes() -> None:
    runs = [
        _run("bt", 11, "budget", 1_000_001),
        _run("bt", 12, "solved", 266),
        _run("min_conflicts", 11, "not_found", 10_024),
    ]
    table = algorithm_table(runs, [("P", "Ejemplo", 12)])
    assert r"1\,000\,001 & \multicolumn{2}{l}{presupuesto agotado}" in table
    assert r"10\,024 & \multicolumn{2}{l}{no encontrada}" in table
    assert r"266 & 9 & 0" in table
    assert r"\multicolumn{3}{c}{---}" in table  # min-conflicts was not run at H = 12


def test_algorithm_table_adds_the_population_column_and_the_synthetic_row(
    synthetic: SyntheticResult,
) -> None:
    runs = [
        _run("bt", 11, "budget", 1_000_001),
        _run("bt", 12, "solved", 266),
        _run("min_conflicts", 11, "not_found", 10_024),
    ]
    table = algorithm_table(
        runs, [("P", "Ejemplo", 12)], population=POPULATION, synthetic=synthetic
    )
    assert r"\begin{tabular}{lrrrrr}" in table
    assert latex_text(LABELS_ES["r3.population"].format(entries=162)) in table
    bt_row = next(line for line in table.splitlines() if line.startswith("BT &"))
    assert bt_row.endswith(r"100\,001 (122) \\")
    min_conflicts_row = next(line for line in table.splitlines() if line.startswith("Min-"))
    assert min_conflicts_row.endswith(r"--- \\")  # not run on the population
    synthetic_row = next(line for line in table.splitlines() if "p{12cm}" in line)
    expected = f"{synthetic.ac3_decides_before}/{synthetic.with_h_star_minus_one}"
    assert r"\multicolumn{6}" in synthetic_row and expected in synthetic_row


def test_algorithm_legend() -> None:
    legend = algorithm_legend(12, 1_000_000)
    assert r"1\,000\,000" in legend
    assert "H$^*$ $-$ 1 = 11" in legend
    assert "Knock-on" not in legend
    with_population = algorithm_legend(12, 1_000_000, population_budget=100_000)
    assert "Knock-on reales" in with_population and r"100\,000" in with_population


# --- compilation in a minimal beamer document ---------------------------------------------------

#: The preamble of the talk's slides (Overleaf ``practica2/main.tex``): the width check is only
#: meaningful with its fonts, so it runs only when they are installed.
_SLIDES_PREAMBLE = r"""\usepackage[spanish,es-noshorthands,es-nodecimaldot]{babel}
\usepackage[sfdefault,scaled=.92]{FiraSans}
\usepackage{newtxsf}
\usetheme[progressbar=frametitle,numbering=fraction,sectionpage=none]{metropolis}
"""
_SLIDES_PACKAGES = ("FiraSans.sty", "newtxsf.sty", "beamerthememetropolis.sty")
_DOCUMENT = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
%s
\begin{document}
%s
\end{document}
"""


def _has_slide_fonts() -> bool:
    """Whether kpsewhich finds every package of the slides' preamble."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return False
    found = subprocess.run(
        [kpsewhich, *_SLIDES_PACKAGES], capture_output=True, text=True, check=False
    ).stdout.split()
    return len(found) == len(_SLIDES_PACKAGES)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_fragments_compile_in_beamer_and_fit_the_slide(
    example: WorkedExample,
    validation: ValidationResult,
    synthetic: SyntheticResult,
    tmp_path: Path,
) -> None:
    codes = drug_codes([r.drug_id for r in example.instance.drugs], NAMES)
    runs = run_algorithm_table([(ej4_toys.T3.problem, 4)]) + run_min_conflicts_rows(
        ej4_toys.T3.problem, (3, 4), max_steps=100
    )
    fragments = {
        "h11.tex": domain_table(example.at(11), ("DEP", "PAIN"), codes),
        "h12.tex": domain_table(example.at(12), ("DEP", "PAIN"), codes),
        "r2.tex": validation_table(
            validation,
            census_ok=True,
            synthetic=synthetic,
            continuity=CONTINUITY,
            categories=CATEGORIES,
            census_events=17_552,
        ),
        "r3.tex": algorithm_table(
            runs, [("T3", "T3", 4)], population=POPULATION, synthetic=synthetic
        ),
    }
    for name, content in fragments.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    frames = "\n".join(
        rf"\begin{{frame}}{{{name}}}\centering\input{{{name}}}\end{{frame}}" for name in fragments
    )
    slides = _has_slide_fonts()
    preamble = _SLIDES_PREAMBLE if slides else ""
    (tmp_path / "doc.tex").write_text(_DOCUMENT % (preamble, frames), encoding="utf-8")
    completed = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    log = (tmp_path / "doc.log").read_text(encoding="utf-8", errors="replace")
    assert completed.returncode == 0, log[-2000:]
    if slides:  # with the default fonts the widths say nothing about the talk
        assert "Overfull \\hbox" not in log
        assert "Overfull \\vbox" not in log
