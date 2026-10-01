"""Fig. 2 of EJ2: the ontology drawn as a semantic network by Graphviz, from the saved results file.

Notation of AIMA §10.5.1: categories are rounded boxes, drugs are ellipses, a solid arrow is a
subcategory link (⊂) and a dashed arrow a membership (∈); a member-property link (AIMA's boxed
link) is written in small grey type inside the label of the category or drug it is stated on.
Figure labels are report text and therefore in Spanish: names come from ``name_es`` and condition
and risk-factor identifiers from the report's abbreviations (:data:`ABBREVIATIONS`, presentation
only). The figure is laid out for the full IEEE text width (7.16 in) and at most 3.2 in of height,
so that it is printed at its natural size: every leaf category and every drug takes one line.
"""

from __future__ import annotations

import html
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import graphviz

from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError
from symbolic_ai.viz import TOL

__all__ = [
    "ABBREVIATIONS",
    "INFERRED_FIGURE",
    "ONTOLOGY_FIGURE",
    "build_ontology_graph",
    "plot_ontology",
]

#: File stems of the two variants inside the output directory.
ONTOLOGY_FIGURE = "fig_ej2_ontology"
INFERRED_FIGURE = "fig_ej2_ontology_inferred"

#: Identifier -> Spanish abbreviation of the report (conditions and risk factors).
ABBREVIATIONS: Mapping[str, str] = {
    "HTN": "HTA",
    "AF": "FA",
    "T2D": "DM2",
    "DEP": "DEP",
    "PAIN": "DOL",
    "GERD": "ERGE",
    "PREG": "EMB",
    "CKD": "ERC",
    "LIVER": "HEP",
    "EPI": "EPI",
    "ASTHMA": "ASMA",
    "AGE65": "EDAD65",
}

#: Graphviz's PostScript name, mapped to Times New Roman; "Times New Roman" itself is misread by
#: Pango, which takes "Roman" for a style and falls back to a sans-serif font.
_FONT = "Times-Roman"
_NAME_PT = 7.0
_SMALL_PT = 6.0
_INK = "#000000"
_LINK_COLOUR = "#4D4D4D"
_INFERRED_COLOUR = TOL["blue"]
_PNG_DPI = "300"
_SEPARATOR = " · "
#: Categories not drawn: the category of categories is shown by the families' bold border.
_HIDDEN = frozenset({"therapeutic_families"})
#: Drug ellipses are drawn flat, at a fixed height, and wider than their text by this factor, so
#: that 18 of them fit in the 3.2 in of the figure.
_DRUG_HEIGHT_IN = 0.13
_DRUG_WIDTH_FACTOR = 1.45
_DRUG_WIDTH_PAD_IN = 0.04


# --- reading the results ------------------------------------------------------------------------


def _section(results: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = results.get(key)
    if not isinstance(value, Mapping):
        raise ResultsFormatError(f"results file has no {key!r} section")
    return value


def _records(section: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = section.get(key)
    if not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value):
        raise ResultsFormatError(f"section field {key!r} is missing or is not a list of records")
    return list(value)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ResultsFormatError(f"record field {key!r} is missing or is not text: {record}")
    return value


def _texts(record: Mapping[str, object], key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ResultsFormatError(f"record field {key!r} is missing or is not a list of text")
    return list(value)


# --- labels -------------------------------------------------------------------------------------


def _abbreviation(identifier: str) -> str:
    return ABBREVIATIONS.get(identifier, identifier)


def _lower_first(name: str) -> str:
    return name[:1].lower() + name[1:]


def _small(text: str, colour: str = _LINK_COLOUR) -> str:
    return f'<FONT POINT-SIZE="{_SMALL_PT}" COLOR="{colour}">{html.escape(text)}</FONT>'


def _head(name: str, code: str | None) -> str:
    """Return the name, followed by a code (ATC) in small type when there is one."""
    head = html.escape(name)
    return head + (f" {_small(code, _INK)}" if code else "")


class _Knowledge:
    """The told knowledge of the ``ontology`` section, indexed for drawing."""

    def __init__(self, ontology: Mapping[str, object]) -> None:
        self.categories = {_text(c, "id"): c for c in _records(ontology, "categories")}
        self.drugs = {_text(d, "id"): d for d in _records(ontology, "drugs")}
        self.families = {c for c, record in self.categories.items() if record.get("family")}
        self.upper = frozenset(
            c for c, record in self.categories.items() if not record.get("drug_category")
        )
        self.with_children = {
            parent for record in self.categories.values() for parent in _texts(record, "parents")
        }
        self.links: dict[str, list[str]] = {}
        self._collect_links(ontology)
        self.tags = self._disjointness_tags(_records(ontology, "disjoint_sets"))

    def _add(self, subject: str, line: str) -> None:
        self.links.setdefault(subject, []).append(line)

    def _collect_links(self, ontology: Mapping[str, object]) -> None:
        for link in _records(ontology, "indications"):
            self._add(_text(link, "subject"), f"trata {_abbreviation(_text(link, 'condition'))}")
        for link in _records(ontology, "contraindications"):
            self._add(_text(link, "subject"), f"CI: {_abbreviation(_text(link, 'risk_factor'))}")
        for link in _records(ontology, "coprescriptions"):
            companion = _lower_first(_text(self.drugs[_text(link, "companion")], "name_es"))
            risk_factor = _abbreviation(_text(link, "risk_factor"))
            self._add(_text(link, "subject"), f"cop: {risk_factor} → {companion}")

    def _disjointness_tags(self, sets: Sequence[Mapping[str, object]]) -> dict[str, str]:
        """Tag the parent of each disjoint set: ``partición`` or ``disjuntas``."""
        tags = {}
        for disjoint_set in sets:
            parent = disjoint_set.get("partition_of")
            if isinstance(parent, str):
                tags[parent] = "partición"
                continue
            members = _texts(disjoint_set, "categories")
            parents = {tuple(_texts(self.categories[m], "parents")) for m in members}
            if len(parents) == 1 and len(next(iter(parents))) == 1:
                tags[next(iter(parents))[0]] = "disjuntas"
        return tags

    def name(self, category_id: str) -> str:
        return _text(self.categories[category_id], "name_es")

    def depths(self) -> dict[str, int]:
        """Return each category's depth: 0 for a root, else one more than its deepest parent."""
        depths: dict[str, int] = {}

        def depth(category_id: str) -> int:
            if category_id not in depths:
                parents = _texts(self.categories[category_id], "parents")
                depths[category_id] = 1 + max((depth(p) for p in parents), default=-1)
            return depths[category_id]

        for category_id in sorted(self.categories):
            depth(category_id)
        return depths


# --- the graph ----------------------------------------------------------------------------------


def _category_node(graph: graphviz.Digraph, knowledge: _Knowledge, category_id: str) -> None:
    """Draw a rounded box: leaves on one line, categories with children on several."""
    code = knowledge.categories[category_id].get("atc_code")
    head = _head(knowledge.name(category_id), code if isinstance(code, str) else None)
    links = knowledge.links.get(category_id, [])
    tag = knowledge.tags.get(category_id)
    if category_id in knowledge.with_children:
        lines = [_small(link) for link in links] + ([_small(tag)] if tag else [])
        label = "<" + "<BR/>".join([head, *lines]) + ">"
    elif len(links) > 1:  # a leaf with several links: name and first link, then the rest
        rest = _small("; ".join(links[1:]))
        label = "<" + head + _small(_SEPARATOR + links[0]) + "<BR/>" + rest + ">"
    else:
        label = "<" + head + (_small(_SEPARATOR + links[0]) if links else "") + ">"
    family = category_id in knowledge.families
    graph.node(
        category_id,
        label=label,
        shape="box",
        style="rounded,bold" if family else "rounded",
        penwidth="1.5" if family else "0.5",
    )


def _drug_label(knowledge: _Knowledge, drug_id: str, inferred: Sequence[str]) -> str:
    """One line: the name, the told object-level links, then the inherited ones in colour."""
    label = html.escape(_text(knowledge.drugs[drug_id], "name_es"))
    links = knowledge.links.get(drug_id, [])
    if links:
        label += _small(_SEPARATOR + "; ".join(links))
    if inferred:
        label += _small(_SEPARATOR + "; ".join(inferred), _INFERRED_COLOUR)
    return f"<{label}>"


def _label_widths(labels: Mapping[str, str]) -> dict[str, float]:
    """Measure HTML-like labels, in inches, with Graphviz itself (one ``dot`` run, JSON output)."""
    probe = graphviz.Digraph()
    probe.attr("node", fontname=_FONT, fontsize=str(_NAME_PT), shape="plaintext")
    probe.attr("node", margin="0", height="0", width="0")
    for name, label in labels.items():
        probe.node(name, label=label)
    layout = json.loads(probe.pipe(format="json"))
    return {
        str(obj["name"]): float(obj["width"])
        for obj in layout.get("objects", [])
        if obj.get("name") in labels
    }


def _drug_node(graph: graphviz.Digraph, drug_id: str, label: str, text_width: float) -> None:
    """Draw a flat ellipse of fixed height around a one-line label."""
    width = _DRUG_WIDTH_FACTOR * text_width + _DRUG_WIDTH_PAD_IN
    graph.node(
        drug_id,
        label=label,
        shape="ellipse",
        penwidth="0.5",
        fixedsize="true",
        width=f"{width:.3f}",
        height=f"{_DRUG_HEIGHT_IN:.3f}",
    )


def _inferred_lines(results: Mapping[str, object]) -> dict[str, list[str]]:
    """Return the candidates and contraindications each drug inherits from a category."""
    rows = _records(_section(results, "formulary_derivation"), "rows")
    by_drug: dict[str, list[str]] = {}
    for row in rows:
        table, key = _text(row, "table"), _texts(row, "key")
        if _text(row, "level") != "category":
            continue
        if table == "candidates":
            by_drug.setdefault(key[1], []).append(f"trata {_abbreviation(key[0])}")
        elif table == "contraindications":
            by_drug.setdefault(key[1], []).append(f"CI: {_abbreviation(key[0])}")
    return by_drug


def _add_edges(graph: graphviz.Digraph, knowledge: _Knowledge, hidden: frozenset[str]) -> None:
    for category_id in sorted(set(knowledge.categories) - hidden):
        for parent in sorted(set(_texts(knowledge.categories[category_id], "parents")) - hidden):
            # Drawn parent -> child with dir=back: the arrow points from the subcategory to the
            # category while the parent stays on the left (rankdir=LR).
            graph.edge(parent, category_id, dir="back")
    for drug_id in sorted(knowledge.drugs):
        for category_id in _texts(knowledge.drugs[drug_id], "categories"):
            graph.edge(category_id, drug_id, dir="back", style="dashed")


def build_ontology_graph(results: Mapping[str, object], *, inferred: bool) -> graphviz.Digraph:
    """Build the Graphviz source of Fig. 2 from the ``ontology`` section of ``results.json``.

    There is no legend: the notation and the five category pairs with an adverse interaction are
    given in the report's caption. The inferred variant leaves out the upper ontology (clinical
    objects, conditions, risk factors) to make room for the inherited properties of each drug.

    Parameters
    ----------
    results : Mapping[str, object]
        The content of ``results.json`` (schema ``symai.ej2.results/1``).
    inferred : bool
        Add, in a second colour, the candidates and contraindications each drug inherits
        (rows of the ``formulary_derivation`` section that come from a category-level link).

    Returns
    -------
    graphviz.Digraph
        The graph; its ``source`` is deterministic.

    Raises
    ------
    ResultsFormatError
        If a section or field the figure needs is missing.
    """
    knowledge = _Knowledge(_section(results, "ontology"))
    inherited = _inferred_lines(results) if inferred else {}
    graph = graphviz.Digraph(name=INFERRED_FIGURE if inferred else ONTOLOGY_FIGURE)
    graph.attr(rankdir="LR", nodesep="0.032", ranksep="0.18", margin="0", pad="0.01")
    graph.attr(newrank="true", fontname=_FONT)
    graph.attr("node", fontname=_FONT, fontsize=str(_NAME_PT), fontcolor=_INK, color=_INK)
    graph.attr("node", margin="0.04,0.008", height="0.05", width="0.05")
    graph.attr("edge", arrowsize="0.3", penwidth="0.5", color=_INK)
    hidden = _HIDDEN | (knowledge.upper if inferred else frozenset())
    depths = knowledge.depths()
    for level in sorted(set(depths.values())):
        # One column per depth of the told taxonomy, so that a column is a level of the tree.
        with graph.subgraph(name=f"depth_{level}") as column:
            column.attr(rank="same")
            for category_id in sorted(c for c, d in depths.items() if d == level):
                if category_id not in hidden:
                    _category_node(column, knowledge, category_id)
    labels = {d: _drug_label(knowledge, d, inherited.get(d, [])) for d in sorted(knowledge.drugs)}
    widths = _label_widths(labels)
    with graph.subgraph(name="drug_objects") as objects:
        objects.attr(rank="max")
        for drug_id, label in labels.items():
            _drug_node(objects, drug_id, label, widths[drug_id])
    _add_edges(graph, knowledge, hidden)
    return graph


@contextmanager
def _source_date_epoch() -> Iterator[None]:
    """Set ``SOURCE_DATE_EPOCH=0`` while rendering, so cairo writes a fixed PDF creation date."""
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous


def plot_ontology(
    results: Mapping[str, object], out_stem: Path, *, inferred: bool
) -> tuple[Path, ...]:
    """Draw Fig. 2 (``inferred=False``) or its variant with inherited properties, as .dot/.pdf/.png.

    Parameters
    ----------
    results : Mapping[str, object]
        The content of ``results.json``.
    out_stem : Path
        Output path without extension; parent directories are created.
    inferred : bool
        Draw the variant with the inherited candidates and contraindications of each drug.

    Returns
    -------
    tuple[Path, ...]
        The ``.dot``, ``.pdf`` and ``.png`` files written.

    Raises
    ------
    ResultsFormatError
        If a section or field the figure needs is missing.
    """
    graph = build_ontology_graph(results, inferred=inferred)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_stem.with_suffix(".dot")
    dot_path.write_text(graph.source, encoding="utf-8")
    pdf_path, png_path = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    with _source_date_epoch():
        graphviz.render("dot", "pdf", dot_path, outfile=pdf_path)
        png_source = graphviz.Source(graph.source.replace("{", f"{{\n\tdpi={_PNG_DPI}", 1))
        png_source.render(outfile=png_path, format="png", cleanup=True)
    return dot_path, pdf_path, png_path
