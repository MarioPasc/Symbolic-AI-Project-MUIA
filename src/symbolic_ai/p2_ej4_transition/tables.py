r"""LaTeX tables of EJ4 for the beamer slides (booktabs fragments to ``\input``, no preamble).

Four fragments (design §7):

- ``tab_ej4_domains_h11.tex`` and ``tab_ej4_domains_h12.tex``: the worked example's domains after
  AC-3 at ``H* - 1`` and ``H*``, conditions by days, with the values AC-3 removed in gray and the
  emptied domain marked (R1);
- ``tab_ej4_r2.tex``: the validation summary and the mutations per rule (R2);
- ``tab_ej4_r3.tex``: the algorithm table on the two real instances (R3).

Every visible word comes from one label dictionary (:data:`LABELS_ES`, Spanish, the language of
the talk), so another dictionary with the same keys switches the language. The fragments need
``booktabs`` and ``xcolor`` (beamer loads ``xcolor``). Drug names come from the formulary's
``name_es`` column; values are written with short codes (``S``, ``S↓1``, ``∅``) defined by
:func:`domain_legend`. Design: ``code/docs/SPECIFICATIONS/EJ4-transition/README.md`` §7 (private).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from symbolic_ai.p2_ej4_transition.experiments import (
    AlgorithmRun,
    CategoryAgreement,
    PopulationRow,
    SyntheticResult,
    ValidationResult,
)
from symbolic_ai.p2_ej4_transition.instances import ContinuityReport
from symbolic_ai.p2_ej4_transition.problem import Value, drug_of
from symbolic_ai.p2_ej4_transition.worked_example import DomainCell, HorizonDomains

__all__ = [
    "LABELS_ES",
    "algorithm_legend",
    "algorithm_table",
    "compact_values",
    "domain_legend",
    "domain_table",
    "drug_codes",
    "latex_text",
    "validation_table",
    "value_code",
]

#: Every label of the figures and tables of EJ4, in Spanish. Keys ``condition.<id>``,
#: ``status.<status>``, ``config.<key>``, ``rule.<rule>`` and ``mutation.<name>`` name things.
LABELS_ES: Mapping[str, str] = MappingProxyType(
    {
        "day": "Día",
        "condition": "Condición",
        "event": "evento",
        "h_star": "H* = {h}",
        "untreated": "sin tratar",
        "taper": "reducción de dosis (↓k: día k)",
        "presence": "presente tras la última dosis",
        "emptied": "vacío",
        "removed": "en gris: valores eliminados por AC-3",
        "emptied_legend": "vacío: dominio que AC-3 deja sin valores",
        "untreated_code": "∅: sin tratar",
        "taper_code": "X↓k: día k de reducción de X",
        "condition.AF": "Fibrilación auricular",
        "condition.DEP": "Depresión",
        "condition.GERD": "Reflujo",
        "condition.HTN": "Hipertensión",
        "condition.PAIN": "Dolor crónico",
        "condition.T2D": "Diabetes tipo 2",
        "status.solved": "resuelto",
        "status.infeasible": "imposible",
        "status.budget": "presupuesto agotado",
        "status.timeout": "tiempo agotado",
        "status.not_found": "no encontrada",
        "config.bt": "BT",
        "config.bt_fc": "BT + FC",
        "config.bt_fc_mrv": "BT + FC + MRV",
        "config.main": "AC-3 + BT + FC + MRV",
        "config.min_conflicts": "Min-conflicts",
        "rule.shape": "forma",
        "rule.day0": "día 0",
        "rule.domain": "dominio",
        "rule.transition": "transición",
        "rule.interaction": "interacción",
        "rule.final": "final",
        "mutation.drop_taper_day": "Quitar el último día de reducción",
        "mutation.start_early": "Empezar un fármaco un día antes",
        "mutation.untreated_day": "Dejar sin tratar un día intermedio",
        "mutation.contraindicated_drug": "Dar un fármaco contraindicado",
        "mutation.day0": "Cambiar el día 0",
        "mutation.final": "Terminar reducido o no aceptable",
        "mutation.delete_key": "Borrar un día del calendario",
        "r2.check": "Comprobación",
        "r2.result": "Resultado",
        "r2.census": "Censo de eventos reproducido",
        "r2.yes": "sí",
        "r2.no": "no",
        "r2.cases": "Casos",
        "r2.real": "Reales: CSP = oráculo = verificador",
        "r2.real_cases": "{problems} problemas ({events} eventos)",
        "r2.real_no_oracle": "{valid}/{schedules} válidos (sin oráculo)",
        "r2.synthetic": "Sintéticas (semilla {seed}): CSP = oráculo = verificador",
        "r2.synthetic_cases": "{count} problemas, {impossible} sin transición",
        "r2.mutations": "Mutaciones dirigidas ({rules} reglas), rechazadas con su regla",
        "r2.mutations_cases": "{applied} mutantes",
        "r2.continuity": "Continuidad con EJ3: R₀ = régimen de EJ3 (un fármaco por condición)",
        "r2.continuity_cases": (
            "{histories} historias; aparte, {companion} con acompañante, {refer} derivadas, "
            "{unexpected} inesperadas"
        ),
        "r2.categories": "Categoría (trivial, forzado, knock-on) coherente con H*",
        "r2.categories_cases": "{entries} problemas",
        "r2.census_cases": "{events} eventos",
        "r2.not_run": "no ejecutado",
        "r3.configuration": "Configuración",
        "r3.horizon": "H",
        "r3.assignments": "Asign.",
        "r3.checks": "Compr.",
        "r3.ac3": "Elim. AC-3",
        "r3.not_run": "---",
        "r3.population": "Knock-on reales ({entries})",
        "r3.population_header": "Mediana (agotan)",
        "r3.population_cell": "{median} ({hits})",
        "r3.synthetic": (
            "Sintéticas ({count}): AC-3 decide H*\u22121 sin asignar en {decided}/{before}; la "
            "búsqueda retrocede en H* en {backtracking}/{solved}; asign. en H*: mediana "
            "{median}, máx. {max}."
        ),
        "r3.worked": "Ejemplo resuelto",
        "r3.extended": "Caso extendido",
        "r3.legend": (
            "Con H* \u2212 1 = {before} días ninguna configuración completa encuentra solución "
            "(imposible) y con H* = {h} la encuentran (resuelto), salvo indicación. Asign.: "
            "asignaciones; Compr.: comprobaciones de restricciones; Elim. AC-3: valores "
            "eliminados por AC-3 antes de buscar. Presupuesto: {budget} asignaciones."
        ),
        "r3.population_legend": (
            " Knock-on reales: mediana de asignaciones de cada configuración en H* \u2212 1 sobre "
            "todos los problemas knock-on, con un presupuesto de {budget} asignaciones; entre "
            "paréntesis, cuántos lo agotan."
        ),
    }
)

#: Unicode symbols of the labels and codes, and how pdfLaTeX writes them.
_LATEX_SYMBOLS = {
    "↓": r"$\downarrow$",
    "∅": r"$\emptyset$",
    "*": "$^*$",
    "\u2212": "$-$",
    "₀": "$_0$",
}
#: Characters LaTeX treats specially, escaped.
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_HEADER = (
    "% Generated by symbolic_ai.p2_ej4_transition (python -m symbolic_ai.p2_ej4_transition.main"
)
_HEADER += " --experiments). Do not edit: regenerate.\n"


def latex_text(text: str) -> str:
    """Return ``text`` escaped for pdfLaTeX, with the arrows and symbols of the codes in math mode.

    Parameters
    ----------
    text
        Plain text (UTF-8 letters such as ``í`` are kept; pdfLaTeX reads UTF-8 by default).

    Returns
    -------
    str
        The LaTeX source.
    """
    return "".join(_LATEX_SPECIAL.get(c, _LATEX_SYMBOLS.get(c, c)) for c in text)


def drug_codes(drug_ids: Sequence[str], names: Mapping[str, str]) -> dict[str, str]:
    """Return a short code per drug: the first letter of its name, or two letters on a clash.

    Parameters
    ----------
    drug_ids
        The drugs to code.
    names
        Display name per drug (missing ones use the identifier).

    Returns
    -------
    dict[str, str]
        ``drug_id -> code``, e.g. ``sertraline -> S``; codes are unique.
    """
    shown = {d: names.get(d, d) for d in sorted(set(drug_ids))}
    for length in (1, 2):
        codes = {d: name[:length].capitalize() for d, name in shown.items()}
        if len(set(codes.values())) == len(codes):
            return codes
    return {d: d for d in shown}


def value_code(value: Value, codes: Mapping[str, str]) -> str:
    """Return the short code of a value: ``S`` (full dose), ``S↓2`` (taper day 2) or ``∅``.

    Parameters
    ----------
    value
        A value of the contract.
    codes
        The drug codes (:func:`drug_codes`).

    Returns
    -------
    str
        The code.
    """
    if value is None:
        return "∅"
    drug, phase = value
    return codes[drug] if phase == 0 else f"{codes[drug]}↓{phase}"


def compact_values(values: Sequence[Value], codes: Mapping[str, str]) -> tuple[str, ...]:
    """Return the values as one compact group per drug, e.g. ``("S, S↓2-4", "M, M↓1-3")``.

    Consecutive taper days are written as a range; groups keep the order of first appearance.

    Parameters
    ----------
    values
        The values (a domain or the values removed from it).
    codes
        The drug codes (:func:`drug_codes`).

    Returns
    -------
    tuple[str, ...]
        One string per drug (and ``"∅"`` for the untreated value).
    """
    phases: dict[str | None, list[int]] = {}
    for value in values:
        phases.setdefault(drug_of(value), []).append(0 if value is None else value[1])
    groups = []
    for drug, drug_phases in phases.items():
        if drug is None:
            groups.append("∅")
            continue
        code = codes[drug]
        parts = [code] if 0 in drug_phases else []
        parts += [f"{code}↓{run}" for run in _ranges(sorted(p for p in drug_phases if p > 0))]
        groups.append(", ".join(parts))
    return tuple(groups)


def _ranges(numbers: Sequence[int]) -> list[str]:
    """Return sorted integers as ranges: ``[1, 2, 3, 5] -> ["1-3", "5"]``."""
    runs: list[list[int]] = []
    for n in numbers:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return [str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs]


def domain_legend(
    codes: Mapping[str, str], names: Mapping[str, str], labels: Mapping[str, str] = LABELS_ES
) -> str:
    """Return the caption-ready legend of the domain tables (LaTeX).

    Parameters
    ----------
    codes
        The drug codes.
    names
        Display name per drug.
    labels
        The label dictionary.

    Returns
    -------
    str
        E.g. ``S: Sertralina; ...; X↓k: día k de reducción de X; ∅: sin tratar; ...``.
    """
    drugs = [f"{codes[d]}: {names.get(d, d)}" for d in sorted(codes, key=lambda d: codes[d])]
    items = [*drugs, labels["taper_code"], labels["untreated_code"], labels["removed"]]
    items.append(labels["emptied_legend"])
    return latex_text("; ".join(items) + ".")


def domain_table(
    domains: HorizonDomains,
    conditions: Sequence[str],
    codes: Mapping[str, str],
    labels: Mapping[str, str] = LABELS_ES,
) -> str:
    """Return the domain table at one horizon: conditions by days, values left after AC-3.

    Each cell lists the values AC-3 left (black) and, below them in small gray type, the values it
    removed, one line per drug. The domain AC-3 emptied reads ``vacío`` in red (design §7 R1.3).

    Parameters
    ----------
    domains
        The domains at one horizon (``worked_example.horizon_domains``).
    conditions
        The conditions, in row order.
    codes
        The drug codes (:func:`drug_codes`).
    labels
        The label dictionary.

    Returns
    -------
    str
        A ``tabular`` fragment.
    """
    days = range(domains.horizon + 1)
    lines = [
        _HEADER.rstrip("\n"),
        r"{\footnotesize\setlength{\tabcolsep}{2pt}%",
        rf"\begin{{tabular}}{{l*{{{len(days)}}}{{c}}}}",
        r"\toprule",
        " & ".join([latex_text(labels["day"]), *map(str, days)]) + r" \\",
        r"\midrule",
    ]
    for condition in conditions:
        cells = [_domain_cell(domains, domains.cell(condition, day), codes, labels) for day in days]
        name = labels.get(f"condition.{condition}", condition)
        lines.append(" & ".join([latex_text(name), *cells]) + r" \\ \addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(lines)


def _domain_cell(
    domains: HorizonDomains, cell: DomainCell, codes: Mapping[str, str], labels: Mapping[str, str]
) -> str:
    """Return one cell of a domain table (see :func:`domain_table`)."""
    if domains.emptied == (cell.condition_id, cell.day):
        kept = [rf"\textcolor{{red}}{{\textbf{{{latex_text(labels['emptied'])}}}}}"]
    else:
        kept = [_latex_codes(", ".join(compact_values(cell.after_ac3, codes)))]
    removed = [
        rf"\textcolor{{gray}}{{\tiny {_latex_codes(group)}}}"
        for group in compact_values(cell.removed_by_ac3, codes)
    ]
    if not removed:
        return kept[0]
    return r"\begin{tabular}[t]{@{}c@{}}" + r"\\".join(kept + removed) + r"\end{tabular}"


def _latex_codes(text: str) -> str:
    """Return value codes in LaTeX, with a range of taper days as an en dash (``S↓2--4``)."""
    return re.sub(r"(\d)-(\d)", r"\1--\2", latex_text(text))


def validation_table(
    validation: ValidationResult,
    census_ok: bool,
    labels: Mapping[str, str] = LABELS_ES,
    *,
    synthetic: SyntheticResult | None = None,
    continuity: ContinuityReport | None = None,
    categories: Sequence[CategoryAgreement] = (),
    census_events: int | None = None,
) -> str:
    """Return the R2 table: one row per check, sized for one 16:9 slide (design §7 R2).

    Rows, in order: real instances (CSP, oracle and checker agree), synthetic instances (the same
    three-way agreement), the targeted mutations aggregated over the rules (the per-rule
    breakdown stays in ``results.json``), continuity with EJ3, categories against the solved
    H*, and the census. A check that was not run is omitted, or says so.

    Parameters
    ----------
    validation
        The result of ``experiments.run_validation``.
    census_ok
        Whether the census equals design §5 (``experiments.census_matches_design``).
    labels
        The label dictionary.
    synthetic
        The result of ``experiments.run_synthetic``, or ``None`` if it was not run.
    continuity
        The result of ``instances.continuity_with_ej3``, or ``None`` if it was not run.
    categories
        The result of ``experiments.check_categories`` (empty: the row is omitted).
    census_events
        The events the census counts (``None``: the cell is left empty).

    Returns
    -------
    str
        A ``tabular`` fragment.
    """
    rows = [_real_row(validation, labels)]
    rows.append(_synthetic_row(synthetic, labels))
    applied = sum(m.applied for m in validation.mutations)
    rows.append(
        (
            labels["r2.mutations"].format(rules=len(validation.mutations)),
            labels["r2.mutations_cases"].format(applied=_plain(applied)),
            _ratio(sum(m.rejected_with_expected_rule for m in validation.mutations), applied),
        )
    )
    if continuity is not None:
        rows.append(_continuity_row(continuity, labels))
    if categories:
        entries = sum(c.entries for c in categories)
        rows.append(
            (
                labels["r2.categories"],
                labels["r2.categories_cases"].format(entries=_plain(entries)),
                _ratio(sum(c.agree for c in categories), entries),
            )
        )
    events = (
        ""
        if census_events is None
        else labels["r2.census_cases"].format(events=_plain(census_events))
    )
    rows.append((labels["r2.census"], events, labels["r2.yes"] if census_ok else labels["r2.no"]))
    header = (labels["r2.check"], labels["r2.cases"], labels["r2.result"])
    # p columns wrap long cells; \raggedright in them redefines \\, so rows end with
    # \tabularnewline (no array package needed).
    lines = [
        _HEADER.rstrip("\n"),
        r"{\scriptsize",
        r"\begin{tabular}{p{5.4cm}p{5.0cm}r}",
        r"\toprule",
        " & ".join(latex_text(h) for h in header) + r" \tabularnewline",
        r"\midrule",
    ]
    for name, cases, result in rows:
        cells = [
            r"\raggedright " + _thousands(latex_text(name)),
            r"\raggedright " + _thousands(latex_text(cases)),
            result,
        ]
        lines.append(" & ".join(cells) + r" \tabularnewline")
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(lines)


def _real_row(validation: ValidationResult, labels: Mapping[str, str]) -> tuple[str, str, str]:
    """Return the R2 row of the real instances (the result cell is already LaTeX)."""
    cases = labels["r2.real_cases"].format(
        problems=_plain(validation.problems), events=_plain(validation.events)
    )
    if validation.three_way_agree is None:
        checker = validation.checker
        text = labels["r2.real_no_oracle"].format(
            valid=_plain(checker.valid), schedules=_plain(checker.schedules)
        )
        return labels["r2.real"], cases, _thousands(latex_text(text))
    return labels["r2.real"], cases, _ratio(validation.three_way_agree, validation.problems)


def _synthetic_row(
    synthetic: SyntheticResult | None, labels: Mapping[str, str]
) -> tuple[str, str, str]:
    """Return the R2 row of the synthetic instances (the result cell is already LaTeX)."""
    if synthetic is None:
        name = labels["r2.synthetic"].format(seed="-")
        return name, "", latex_text(labels["r2.not_run"])
    impossible = dict(synthetic.oracle_statuses).get("impossible", 0)
    cases = labels["r2.synthetic_cases"].format(
        count=_plain(synthetic.count), impossible=_plain(impossible)
    )
    name = labels["r2.synthetic"].format(seed=synthetic.seed)
    return name, cases, _ratio(synthetic.three_way_agree, synthetic.count)


def _continuity_row(
    continuity: ContinuityReport, labels: Mapping[str, str]
) -> tuple[str, str, str]:
    """Return the R2 row of the continuity with EJ3 (the result cell is already LaTeX).

    The result is ``equal / (equal + different)``: the histories where EJ3 prescribes one drug per
    condition; the companion, referral and unexpected histories are counted in the cases cell.
    """
    cases = labels["r2.continuity_cases"].format(
        histories=_plain(continuity.histories),
        companion=len(continuity.companion),
        refer=_plain(continuity.refer),
        unexpected=len(continuity.unexpected),
    )
    result = _ratio(continuity.equal, continuity.equal + len(continuity.different))
    return labels["r2.continuity"], cases, result


def _ratio(numerator: int, denominator: int) -> str:
    """Return ``numerator/denominator`` with thin spaces between thousands (LaTeX)."""
    return f"{_int(numerator)}/{_int(denominator)}"


def _plain(number: int) -> str:
    """Return an integer with commas between thousands, for :func:`_thousands` after escaping."""
    return f"{number:,}"


def algorithm_table(
    runs: Sequence[AlgorithmRun],
    instances: Sequence[tuple[str, str, int]],
    labels: Mapping[str, str] = LABELS_ES,
    *,
    population: Sequence[PopulationRow] = (),
    synthetic: SyntheticResult | None = None,
) -> str:
    """Return the R3 table: configurations by horizon, one column group per instance (§7 R3).

    A run with the expected outcome (``infeasible`` at ``H* - 1``, ``solved`` at ``H*``) shows its
    assignments, checks and AC-3 removals; any other outcome shows its assignments and then its
    status in words (``presupuesto agotado`` for the budget) across the other two columns. The two
    summary rows of design §7 R3 are a last column (the real knock-on population at ``H* - 1``:
    median assignments and budget hits, on each configuration's first row) and a last row (the
    synthetic batch).

    Parameters
    ----------
    runs
        The rows of ``experiments.run_algorithm_table`` and ``run_min_conflicts_rows``.
    instances
        ``(problem_id, title, H*)`` per column group, in order.
    labels
        The label dictionary.
    population
        The rows of ``experiments.run_knock_on_population`` (empty: no population column).
    synthetic
        The result of ``experiments.run_synthetic`` (``None``: no synthetic row).

    Returns
    -------
    str
        A ``tabular`` fragment.
    """
    by_key = {(r.instance_id, r.configuration, r.horizon): r for r in runs}
    by_config = {row.configuration: row for row in population}
    configurations = list(dict.fromkeys(r.configuration for r in runs))
    horizons = sorted({h for _id, _title, h_star in instances for h in (h_star - 1, h_star)})
    extra = 1 if population else 0
    columns = 2 + 3 * len(instances) + extra
    titles = [rf"\multicolumn{{3}}{{c}}{{{latex_text(t)}}}" for _i, t, _h in instances]
    rules = [rf"\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(instances))]
    names = [
        latex_text(labels[k])
        for k in (
            "r3.configuration",
            "r3.horizon",
            *(("r3.assignments", "r3.checks", "r3.ac3") * len(instances)),
        )
    ]
    if population:
        titles.append(latex_text(labels["r3.population"].format(entries=population[0].entries)))
        rules.append(rf"\cmidrule(lr){{{columns}-{columns}}}")
        names.append(latex_text(labels["r3.population_header"]))
    lines = [
        _HEADER.rstrip("\n"),
        r"{\scriptsize\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabular}{lr" + "rrr" * len(instances) + "r" * extra + "}",
        r"\toprule",
        " & ".join(["", "", *titles]) + r" \\",
        " ".join(rules),
        " & ".join(names) + r" \\",
        r"\midrule",
    ]
    for configuration in configurations:
        name = latex_text(labels.get(f"config.{configuration}", configuration))
        for index, horizon in enumerate(horizons):
            cells = [name if index == 0 else "", str(horizon)]
            cells += [
                _algorithm_cells(by_key.get((problem_id, configuration, horizon)), h_star, labels)
                for problem_id, _title, h_star in instances
            ]
            if population:
                row = by_config.get(configuration)
                cells.append(_population_cell(row, labels) if index == 0 else "")
            lines.append(" & ".join(cells) + r" \\")
    if synthetic is not None:
        text = _thousands(latex_text(_synthetic_text(synthetic, labels)))
        lines += [r"\midrule", rf"\multicolumn{{{columns}}}{{p{{12cm}}}}{{{text}}} \\"]
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(lines)


def _population_cell(row: PopulationRow | None, labels: Mapping[str, str]) -> str:
    """Return the population cell of one configuration (``---`` when it was not run)."""
    if row is None:
        return latex_text(labels["r3.not_run"])
    text = labels["r3.population_cell"].format(
        median=_number(row.median_assignments), hits=row.budget_hits
    )
    return _thousands(latex_text(text))


def _synthetic_text(synthetic: SyntheticResult, labels: Mapping[str, str]) -> str:
    """Return the synthetic row of the R3 table, in words."""
    return labels["r3.synthetic"].format(
        count=f"{synthetic.count:,}",
        decided=synthetic.ac3_decides_before,
        before=synthetic.with_h_star_minus_one,
        backtracking=len(synthetic.backtracking),
        solved=synthetic.solved,
        median=_number(synthetic.median_assignments),
        max=f"{synthetic.max_assignments:,}",
    )


def _number(value: float) -> str:
    """Return a median as an integer when it is one, else with one decimal (commas: thousands)."""
    return f"{int(value):,}" if value == int(value) else f"{value:,.1f}"


def _algorithm_cells(run: AlgorithmRun | None, h_star: int, labels: Mapping[str, str]) -> str:
    """Return the three cells of one run in the R3 table (see :func:`algorithm_table`)."""
    if run is None:
        return rf"\multicolumn{{3}}{{c}}{{{latex_text(labels['r3.not_run'])}}}"
    expected = "solved" if run.horizon == h_star else "infeasible"
    if run.status == expected:
        return f"{_int(run.assignments)} & {_int(run.checks)} & {_int(run.ac3_removed)}"
    status = latex_text(labels.get(f"status.{run.status}", run.status))
    return rf"{_int(run.assignments)} & \multicolumn{{2}}{{l}}{{{status}}}"


def algorithm_legend(
    h_star: int,
    budget: int,
    labels: Mapping[str, str] = LABELS_ES,
    population_budget: int | None = None,
) -> str:
    """Return the caption-ready legend of the R3 table (LaTeX).

    Parameters
    ----------
    h_star
        The minimum horizon of the instances (both are 12 in R3).
    budget
        The assignment budget of each run.
    labels
        The label dictionary.
    population_budget
        The budget of the knock-on population column, if the table has one.

    Returns
    -------
    str
        The legend.
    """
    text = labels["r3.legend"].format(before=h_star - 1, h=h_star, budget=f"{budget:,}")
    if population_budget is not None:
        text += labels["r3.population_legend"].format(budget=f"{population_budget:,}")
    return _thousands(latex_text(text))


def _int(number: int) -> str:
    """Return an integer with thin spaces between thousands (Spanish style)."""
    return f"{number:,}".replace(",", r"\,")


def _thousands(text: str) -> str:
    """Replace the thousands commas of formatted numbers by thin spaces."""
    out = []
    for i, char in enumerate(text):
        between_digits = 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()
        out.append(r"\," if char == "," and between_digits else char)
    return "".join(out)
