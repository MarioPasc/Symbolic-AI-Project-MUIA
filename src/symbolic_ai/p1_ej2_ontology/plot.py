"""Fig. 2 of EJ2: the ontology drawn as a semantic network by Graphviz, from the saved results file.

Notation of AIMA §10.5.1: categories are rounded boxes, drugs are ellipses, a solid arrow is a
subcategory link (⊂) and a dashed arrow a membership (∈); a member-property link (AIMA's boxed
link) is written in small grey type inside the label of the category or drug it is stated on. The
taxonomy is a DAG: a category may have several parent arrows (multiple inheritance). A category
defined by the conjunction of others has a dashed border and a double arrow to each conjunct. An
interaction self-link (any two members of the category interact) is a loop on its category, in
colour, with the effect inside the label. Figure labels are report text and therefore in Spanish:
names come from ``name_es``, condition and risk-factor identifiers from the report's abbreviations
(:data:`ABBREVIATIONS`) and effects from :data:`EFFECTS_ES` (both presentation only). The figure is
laid out for the full IEEE text width (7.16 in) and at most 3.2 in of height, so that it is printed
at its natural size.
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
    "EFFECTS_ES",
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

#: Effect of an interaction link -> its wording in the report (presentation only).
EFFECTS_ES: Mapping[str, str] = {
    "bleeding": "hemorragia",
    "bradycardia, AV block": "bradicardia, BAV",
    "serotonin syndrome": "sínd. serotoninérgico",
}

#: Graphviz's PostScript name, mapped to Times New Roman; "Times New Roman" itself is misread by
#: Pango, which takes "Roman" for a style and falls back to a sans-serif font.
_FONT = "Times-Roman"
_NAME_PT = 7.0
_SMALL_PT = 6.0
_INK = "#000000"
_LINK_COLOUR = "#4D4D4D"
_INFERRED_COLOUR = TOL["blue"]
_INTERACTION_COLOUR = TOL["purple"]
_PNG_DPI = "300"
_SEPARATOR = " · "
#: Categories not drawn: the category of categories is shown by the families' bold border.
_HIDDEN = frozenset({"therapeutic_families"})
#: Drug ellipses are drawn flat, at a fixed height, and wider than their text by this factor, so
#: that 18 of them fit in the 3.2 in of the figure.
_DRUG_HEIGHT_IN = 0.13
#: Clearance (points) kept around a defined category by an invisible cluster.
_CLEARANCE_PT = "2"
#: Rank separation (in) of the told and the inferred variant: the inferred drug labels are longer,
#: so the told variant spends the same width (6.99 in) on looser columns instead.
_RANKSEP_IN: Mapping[bool, str] = {False: "0.45", True: "0.18"}
#: Weight of the longest membership arrow of a drug told in several categories (kept straight).
_LONG_MEMBERSHIP_WEIGHT = "3"
_LOOP_PENWIDTH = "0.8"
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
        self.conjuncts = {
            _text(d, "category"): _texts(d, "conjuncts") for d in _records(ontology, "definitions")
        }
        self.with_children = {p for c in self.categories for p in self.above(c)}
        self.links: dict[str, list[str]] = {}
        self.self_links: dict[str, str] = {}
        self.pairs: list[tuple[str, str]] = []
        self._collect_links(ontology)
        self.tags = self._disjointness_tags(_records(ontology, "disjoint_sets"))

    def above(self, category_id: str) -> list[str]:
        """Return the told parents and, for a defined category, its conjuncts."""
        parents = _texts(self.categories[category_id], "parents")
        return sorted({*parents, *self.conjuncts.get(category_id, [])})

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
        for link in _records(ontology, "interactions"):
            a, b = _text(link, "subject_a"), _text(link, "subject_b")
            if a == b:
                effect = _text(link, "effect")
                self.self_links[a] = EFFECTS_ES.get(effect, effect)
            else:
                self.pairs.append((a, b))

    def _disjointness_tags(self, sets: Sequence[Mapping[str, object]]) -> dict[str, str]:
        """Tag the parent of each disjoint set: ``partición`` or ``disjuntas``.

        A set that is not a partition tags the one told parent its members have in common (with
        multiple inheritance a member may have other parents too).
        """
        tags = {}
        for disjoint_set in sets:
            parent = disjoint_set.get("partition_of")
            if isinstance(parent, str):
                tags[parent] = "partición"
                continue
            members = _texts(disjoint_set, "categories")
            common = set.intersection(
                *(set(_texts(self.categories[m], "parents")) for m in members)
            )
            if len(common) == 1:
                tags[common.pop()] = "disjuntas"
        return tags

    def name(self, category_id: str) -> str:
        return _text(self.categories[category_id], "name_es")

    def depths(self) -> dict[str, int]:
        """Return each category's depth: 0 for a root, else one more than its deepest parent.

        A conjunct of a defined category counts as a parent, so the defined category is drawn to
        the right of both conjuncts.
        """
        depths: dict[str, int] = {}

        def depth(category_id: str) -> int:
            if category_id not in depths:
                above = self.above(category_id)
                depths[category_id] = 1 + max((depth(p) for p in above), default=-1)
            return depths[category_id]

        for category_id in sorted(self.categories):
            depth(category_id)
        return depths


# --- the graph ----------------------------------------------------------------------------------


def _category_label(knowledge: _Knowledge, category_id: str) -> str:
    """Return the label: the name, then the links (and a parent's disjointness tag) in small type.

    A category with children puts all of them on a second line; a leaf keeps its first link on
    the name's line and the rest, if any, on a second line, so that most leaves take one line.
    """
    code = knowledge.categories[category_id].get("atc_code")
    head = _head(knowledge.name(category_id), code if isinstance(code, str) else None)
    links = [_small(link) for link in knowledge.links.get(category_id, [])]
    if category_id in knowledge.self_links:
        effect = knowledge.self_links[category_id]
        links.append(_small(f"int.: {effect}", _INTERACTION_COLOUR))
    if category_id in knowledge.conjuncts:
        # The definition itself is the second line, D ≡ k₁ ∩ … ∩ kₙ (categories as sets of
        # members); "⊓" is not in Times New Roman and would pull in a fallback font.
        definition = " ∩ ".join(knowledge.name(k) for k in knowledge.conjuncts[category_id])
        first = head + "".join(_small(_SEPARATOR) + link for link in links)
        return f"<{first}<BR/>{_small(f'≡ {definition}')}>"
    if category_id in knowledge.with_children:
        tag = knowledge.tags.get(category_id)
        second = _small(_SEPARATOR).join([*links, *([_small(tag)] if tag else [])])
        return "<" + head + (f"<BR/>{second}" if second else "") + ">"
    if len(links) > 1:
        rest = _small("; ").join(links[1:])
        return "<" + head + _small(_SEPARATOR) + links[0] + "<BR/>" + rest + ">"
    return "<" + head + (_small(_SEPARATOR) + links[0] if links else "") + ">"


def _category_node(graph: graphviz.Digraph, knowledge: _Knowledge, category_id: str) -> None:
    """Draw a rounded box: bold for a family, dashed for a category defined by conjuncts."""
    family = category_id in knowledge.families
    style = ["rounded", "filled"]
    if family:
        style.append("bold")
    if category_id in knowledge.conjuncts:
        style.append("dashed")
    graph.node(
        category_id,
        label=_category_label(knowledge, category_id),
        shape="box",
        style=",".join(style),
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
        style="filled",
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


def _membership_edges(graph: graphviz.Digraph, knowledge: _Knowledge) -> None:
    """Draw the dashed membership arrows; a drug's longest one is kept straight (higher weight).

    The longest arrow of a drug told in several categories goes to its shallowest category; left
    to its default weight, dot bends it round the boxes in between.
    """
    depths = knowledge.depths()
    for drug_id in sorted(knowledge.drugs):
        told = _texts(knowledge.drugs[drug_id], "categories")
        longest = min(told, key=lambda c: (depths[c], c)) if len(told) > 1 else None
        for category_id in told:
            weight = {"weight": _LONG_MEMBERSHIP_WEIGHT} if category_id == longest else {}
            graph.edge(category_id, drug_id, dir="back", style="dashed", **weight)


def _classification_edges(graph: graphviz.Digraph, results: Mapping[str, object]) -> None:
    """Draw, in the inferred colour, each drug's derived membership in a defined category.

    These memberships are never told: forward chaining classifies the drug by the definition.
    The edges do not constrain the layout, so both variants keep the same columns.
    """
    members = _section(_section(results, "formulary_derivation"), "defined_category_members")
    for category_id in sorted(members):
        for drug_id in _texts(members, category_id):
            graph.edge(
                category_id,
                drug_id,
                dir="back",
                style="dashed",
                color=_INFERRED_COLOUR,
                constraint="false",
            )


def _add_edges(graph: graphviz.Digraph, knowledge: _Knowledge, hidden: frozenset[str]) -> None:
    for category_id in sorted(set(knowledge.categories) - hidden):
        for parent in sorted(set(_texts(knowledge.categories[category_id], "parents")) - hidden):
            # Drawn parent -> child with dir=back: the arrow points from the subcategory to the
            # category while the parent stays on the left (rankdir=LR).
            graph.edge(parent, category_id, dir="back")
        for conjunct in knowledge.conjuncts.get(category_id, []):
            # A hollow arrowhead: D ⊑ k follows from the definition D ≡ k₁ ⊓ … ⊓ kₙ.
            graph.edge(conjunct, category_id, dir="back", arrowtail="empty")
    _membership_edges(graph, knowledge)
    for category_id in sorted(knowledge.self_links):
        graph.edge(
            category_id,
            category_id,
            dir="none",
            color=_INTERACTION_COLOUR,
            penwidth=_LOOP_PENWIDTH,
        )
    for a, b in knowledge.pairs:
        graph.edge(a, b, dir="none", color=_INTERACTION_COLOUR, constraint="false")


def build_ontology_graph(results: Mapping[str, object], *, inferred: bool) -> graphviz.Digraph:
    """Build the Graphviz source of Fig. 2 from the ``ontology`` section of ``results.json``.

    There is no legend: the notation is given in the report's caption.

    Parameters
    ----------
    results : Mapping[str, object]
        The content of ``results.json`` (schema ``symai.ej2.results/1``).
    inferred : bool
        Add, in a second colour, the candidates and contraindications each drug inherits (rows
        of the ``formulary_derivation`` section that come from a category-level link) and each
        drug's derived membership in a category defined by conjuncts.

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
    graph.attr(rankdir="LR", nodesep="0.032", ranksep=_RANKSEP_IN[inferred], margin="0", pad="0.01")
    # Edges first, under white-filled nodes: a line that grazes a box passes behind it instead
    # of crossing its text (the separations are at the minimum that fits 3.2 in).
    graph.attr(newrank="true", fontname=_FONT, outputorder="edgesfirst")
    graph.attr("node", fontname=_FONT, fontsize=str(_NAME_PT), fontcolor=_INK, color=_INK)
    graph.attr("node", fillcolor="white")
    graph.attr("node", margin="0.04,0.008", height="0.05", width="0.05")
    graph.attr("edge", arrowsize="0.3", penwidth="0.5", color=_INK)
    hidden = _HIDDEN
    depths = knowledge.depths()
    for level in sorted(set(depths.values())):
        # One column per depth of the told taxonomy, so that a column is a level of the tree.
        with graph.subgraph(name=f"depth_{level}") as column:
            column.attr(rank="same")
            for category_id in sorted(c for c, d in depths.items() if d == level):
                if category_id not in hidden:
                    _category_node(column, knowledge, category_id)
    for category_id in sorted(knowledge.conjuncts):
        # dot routes other edges outside a cluster: the membership arrows of the members' told
        # categories would otherwise run along the defined category's dashed border.
        with graph.subgraph(name=f"cluster_{category_id}") as clearance:
            clearance.attr(style="invis", margin=_CLEARANCE_PT)
            clearance.node(category_id)
    labels = {d: _drug_label(knowledge, d, inherited.get(d, [])) for d in sorted(knowledge.drugs)}
    widths = _label_widths(labels)
    with graph.subgraph(name="drug_objects") as objects:
        objects.attr(rank="max")
        for drug_id, label in labels.items():
            _drug_node(objects, drug_id, label, widths[drug_id])
    _add_edges(graph, knowledge, hidden)
    if inferred:
        _classification_edges(graph, results)
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
