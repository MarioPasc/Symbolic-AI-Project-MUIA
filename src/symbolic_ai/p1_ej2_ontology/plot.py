"""Fig. 2 of EJ2: the ontology drawn as a semantic network by Graphviz, from the saved results file.

The figure follows AIMA §10.5.1 (3rd ed. Fig. 12.5-12.6): every first-order predicate of the
ontology is an edge between two nodes, and the ternary coprescription link is reified as a node
with one role edge per argument (AIMA's *Fly17*). Relations are told apart by colour only
(:data:`RELATION_COLOURS`, frozen by the results design §12.1 because the report's caption
reproduces the same hex values); every edge is solid. Categories are rounded boxes, drugs are
ellipses, and conditions and risk factors are value nodes (plain boxes with the border colour of
the relation that points to them). Figure labels are report text and therefore in Spanish: names
come from ``name_es`` and condition and risk-factor identifiers from the report's abbreviations
(:data:`ABBREVIATIONS`, presentation only). There is no legend: the notation is in the caption.

The layout takes two Graphviz passes. ``dot`` places the nodes in columns (the depth of the told
taxonomy, then the risk factors, then the drugs), in the top-to-bottom order of
:data:`CATEGORY_ORDER`, from the edges that shape the hierarchy; ``neato -n2`` then keeps those
positions and routes every edge around the nodes. ``dot`` alone cannot draw the long
non-hierarchical edges (families, definitions) without moving them round the ends of the columns.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations, pairwise
from pathlib import Path
from types import MappingProxyType

import graphviz

from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError

__all__ = [
    "ABBREVIATIONS",
    "CATEGORY_ORDER",
    "INFERRED_FIGURE",
    "INFERRED_MEMBER_COLOUR",
    "ONTOLOGY_FIGURE",
    "RELATION_COLOURS",
    "Relation",
    "build_ontology_graph",
    "condition_node",
    "plot_ontology",
    "risk_factor_node",
]

logger = logging.getLogger(__name__)

#: File stems of the two variants inside the output directory.
ONTOLOGY_FIGURE = "fig_ej2_ontology"
INFERRED_FIGURE = "fig_ej2_ontology_inferred"


class Relation(StrEnum):
    """A relation drawn as an edge, named by its predicate in the knowledge base."""

    SUBSET = "Subset"
    MEMBER = "Member"
    TREATS = "TrataCat"
    CONTRAINDICATED = "CICat"
    COPRESCRIPTION = "CopCat"
    INTERACTION = "IntCat"
    DISJOINT = "Disj"
    FAMILY = "Familia"
    DEFINITION = "Definition"


#: Edge colour of each relation: Paul Tol's bright palette (:data:`symbolic_ai.viz.TOL`) plus
#: black. Frozen by the results design §12.1: the report's caption draws the same hex values.
#: The definition grey is the darker one §12.1 allows (#888888 for Tol's #BBBBBB): at print size
#: a 0.8 pt line in #BBBBBB was the faintest ink in the figure and vanishes in greyscale.
RELATION_COLOURS: Mapping[Relation, str] = MappingProxyType(
    {
        Relation.SUBSET: "#000000",
        Relation.MEMBER: "#4477AA",
        Relation.TREATS: "#228833",
        Relation.CONTRAINDICATED: "#EE6677",
        Relation.COPRESCRIPTION: "#EE7733",
        Relation.INTERACTION: "#AA3377",
        Relation.DISJOINT: "#66CCEE",
        Relation.FAMILY: "#CCBB44",
        Relation.DEFINITION: "#888888",
    }
)

#: The inferred variant's derived memberships: the membership blue at 50 % opacity (alpha 0x80).
INFERRED_MEMBER_COLOUR = RELATION_COLOURS[Relation.MEMBER] + "80"

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

#: Top-to-bottom order of the categories (presentation only, chosen by looking at the PNG). Each
#: column draws its categories in this order; a drug follows its told category, a condition its
#: category, the reified coprescription its subject and a risk factor the median of its sources.
#: It keeps the categories that share a drug, a risk factor or a reified link next to each other:
#: AINE next to IBP (coprescription with omeprazol), Opioides next to Serotoninérgicos (tramadol),
#: ACOD next to Bloqueantes del SRAA (EMB) and Tiazidas next to Sulfonilureas (ERC). Riesgo
#: hemorrágico comes before Antidepresivos because ``dot`` keeps it there (its child AINE pulls
#: it up), and the order must be one ``dot`` keeps. A category missing from it (a newer data
#: version) is drawn last, in identifier order.
CATEGORY_ORDER: tuple[str, ...] = (
    "proton_pump_inhibitors",
    "analgesics",
    "nsaids",
    "anilides",
    "opioids",
    "serotonergic_opioids",
    "serotonergic_drugs",
    "bleeding_risk_drugs",
    "antidepressants",
    "ssris",
    "other_antidepressants",
    "anticoagulants",
    "vitamin_k_antagonists",
    "low_molecular_weight_heparins",
    "direct_oral_anticoagulants",
    "antihypertensives",
    "raas_blockers",
    "ace_inhibitors",
    "arbs",
    "calcium_channel_blockers",
    "dihydropyridines",
    "non_dihydropyridines",
    "bradycardic_drugs",
    "beta_blockers",
    "cardioselective_beta_blockers",
    "nonselective_beta_blockers",
    "thiazides",
    "antidiabetics",
    "sulfonylureas",
    "biguanides",
    "dpp4_inhibitors",
    "drugs",
    "therapeutic_families",
)

#: Relations whose edges have no arrowhead (symmetric predicates).
_UNDIRECTED = frozenset({Relation.INTERACTION, Relation.DISJOINT})
#: Relations whose edges set the columns (ranks); every other edge has ``constraint=false``.
_RANKING = frozenset({Relation.SUBSET, Relation.MEMBER})
#: Relations that ``dot`` sees when it places the nodes; the long ones are left to the router.
_PLACING = frozenset(
    {
        Relation.SUBSET,
        Relation.MEMBER,
        Relation.TREATS,
        Relation.CONTRAINDICATED,
        Relation.COPRESCRIPTION,
        Relation.DISJOINT,
        Relation.DEFINITION,
        Relation.FAMILY,
    }
)
#: Pull of a membership in the placement: a drug stands level with its category, so that the
#: membership arrows run straight and the other links leave them at a visible angle.
_PLACING_WEIGHTS: Mapping[Relation, str] = {Relation.MEMBER: "8"}

#: The category of categories that every therapeutic family is a member of (``Familia(c)``).
_FAMILY_CLASS = "therapeutic_families"
#: Label of the reified coprescription node.
_COP_LABEL = "cop"

#: Graphviz's PostScript name, mapped to Times New Roman; "Times New Roman" itself is misread by
#: Pango, which takes "Roman" for a style and falls back to a sans-serif font.
_FONT = "Times-Roman"
_NAME_PT = 8.0
_SMALL_PT = 7.0
_INK = "#000000"
_EDGE_PT = "0.8"
_BORDER_PT = "0.6"
_ARROW_SIZE = "0.4"
_PNG_DPI = "300"
_NODESEP_IN = "0.06"
_RANKSEP_IN = "0.27"
#: Clearance (points) the edge router keeps around every node.
_EDGE_CLEARANCE = "+1"
#: ``neato`` sizes a self-loop by ``nodesep`` (in); this draws a flat arc about 5 pt high.
_LOOP_SIZE_IN = "0.12"
#: Height (in) kept free above a category with a self-link (for its arc) and between a category
#: and the condition below it (so that the arrow has a visible shaft).
_LOOP_ROOM_IN = 0.12
_ARROW_ROOM_IN = 0.08
_POINTS_PER_IN = 72.0
#: Drug ellipses are drawn flat, at a fixed height, and wider than their text by this factor.
_DRUG_HEIGHT_IN = 0.15
_DRUG_WIDTH_FACTOR = 1.3
_DRUG_WIDTH_PAD_IN = 0.04
_PINNED_COMMENT = "Pinned layout (positions computed by dot): render with `neato -n2`."

_Attributes = dict[str, str]


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


# --- the told knowledge ---------------------------------------------------------------------------


def condition_node(condition_id: str) -> str:
    """Return the node identifier of a condition (EJ1's ``H_`` prefix)."""
    return f"H_{condition_id}"


def risk_factor_node(risk_factor_id: str) -> str:
    """Return the node identifier of a risk factor (EJ1's ``R_`` prefix)."""
    return f"R_{risk_factor_id}"


@dataclass(frozen=True, slots=True)
class _Coprescription:
    """A told ``CopCat(c, r, d')``, drawn as a reified node with three role edges."""

    subject: str
    risk_factor: str
    companion: str

    @property
    def node(self) -> str:
        return f"cop_{self.subject}_{self.risk_factor}_{self.companion}"


@dataclass(frozen=True, slots=True)
class _Fact:
    """One told fact, drawn as an edge from ``tail`` to ``head``."""

    relation: Relation
    tail: str
    head: str


@dataclass(frozen=True, slots=True)
class _Knowledge:
    """The told facts of the ``ontology`` section, as the figure draws them."""

    categories: Mapping[str, Mapping[str, object]]
    drugs: Mapping[str, Mapping[str, object]]
    definitions: Mapping[str, tuple[str, ...]]
    treats: tuple[tuple[str, str], ...]
    contraindicated: tuple[tuple[str, str], ...]
    coprescriptions: tuple[_Coprescription, ...]
    interactions: tuple[tuple[str, str], ...]
    disjoint_pairs: tuple[tuple[str, str], ...]
    families: tuple[str, ...]

    @classmethod
    def from_section(cls, ontology: Mapping[str, object]) -> _Knowledge:
        """Index the ``ontology`` section of ``results.json``.

        Raises
        ------
        ResultsFormatError
            If a field is missing or malformed, or if families are told but the category of
            categories they belong to (``therapeutic_families``) is not.
        """
        categories = {_text(c, "id"): c for c in _records(ontology, "categories")}
        families = _texts(ontology, "families")
        if families and _FAMILY_CLASS not in categories:
            raise ResultsFormatError(
                f"families are told but there is no category {_FAMILY_CLASS!r}"
            )
        return cls(
            categories=categories,
            drugs={_text(d, "id"): d for d in _records(ontology, "drugs")},
            definitions={
                _text(d, "category"): tuple(_texts(d, "conjuncts"))
                for d in _records(ontology, "definitions")
            },
            treats=tuple(
                (_text(link, "subject"), _text(link, "condition"))
                for link in _records(ontology, "indications")
            ),
            contraindicated=tuple(
                (_text(link, "subject"), _text(link, "risk_factor"))
                for link in _records(ontology, "contraindications")
            ),
            coprescriptions=tuple(
                _Coprescription(
                    _text(link, "subject"), _text(link, "risk_factor"), _text(link, "companion")
                )
                for link in _records(ontology, "coprescriptions")
            ),
            interactions=tuple(
                (_text(link, "subject_a"), _text(link, "subject_b"))
                for link in _records(ontology, "interactions")
            ),
            disjoint_pairs=tuple(
                pair
                for disjoint_set in _records(ontology, "disjoint_sets")
                for pair in combinations(sorted(_texts(disjoint_set, "categories")), 2)
            ),
            families=tuple(sorted(families)),
        )

    def parents(self, category_id: str) -> list[str]:
        """Return the told parents of a category."""
        return sorted(_texts(self.categories[category_id], "parents"))

    def depths(self) -> dict[str, int]:
        """Return each category's depth: 0 for a root, else one more than its deepest parent.

        A conjunct of a defined category counts as a parent, so the defined category is drawn to
        the right of both conjuncts.
        """
        depths: dict[str, int] = {}

        def depth(category_id: str) -> int:
            if category_id not in depths:
                above = [*self.parents(category_id), *self.definitions.get(category_id, ())]
                depths[category_id] = 1 + max((depth(p) for p in above), default=-1)
            return depths[category_id]

        for category_id in sorted(self.categories):
            depth(category_id)
        return depths

    def facts(self) -> list[_Fact]:
        """Return every told fact the figure draws, one per edge, in a fixed order."""
        facts = [
            _Fact(Relation.SUBSET, c, p) for c in sorted(self.categories) for p in self.parents(c)
        ]
        facts += [
            _Fact(Relation.MEMBER, d, c)
            for d in sorted(self.drugs)
            for c in _texts(self.drugs[d], "categories")
        ]
        facts += [
            _Fact(Relation.DEFINITION, c, k)
            for c in sorted(self.definitions)
            for k in self.definitions[c]
        ]
        facts += [_Fact(Relation.TREATS, s, condition_node(h)) for s, h in self.treats]
        facts += [
            _Fact(Relation.CONTRAINDICATED, s, risk_factor_node(r)) for s, r in self.contraindicated
        ]
        for cop in self.coprescriptions:
            facts += [
                _Fact(Relation.COPRESCRIPTION, cop.subject, cop.node),
                _Fact(Relation.COPRESCRIPTION, cop.node, risk_factor_node(cop.risk_factor)),
                _Fact(Relation.COPRESCRIPTION, cop.node, cop.companion),
            ]
        facts += [_Fact(Relation.INTERACTION, a, b) for a, b in self.interactions]
        facts += [_Fact(Relation.DISJOINT, a, b) for a, b in self.disjoint_pairs]
        facts += [_Fact(Relation.FAMILY, f, _FAMILY_CLASS) for f in self.families]
        return facts


# --- nodes ----------------------------------------------------------------------------------------


def _category_attributes(knowledge: _Knowledge, category_id: str) -> _Attributes:
    """Return a category's box: rounded, with the Spanish name only (no ATC code, no jargon).

    A defined category has the definition colour as its border, and the category of the
    therapeutic families a double border and its name on two lines (it would otherwise set the
    width of the leftmost column).
    """
    attributes = {
        "label": _text(knowledge.categories[category_id], "name_es"),
        "shape": "box",
        "style": "rounded,filled",
    }
    if category_id in knowledge.definitions:
        attributes["color"] = RELATION_COLOURS[Relation.DEFINITION]
    if category_id == _FAMILY_CLASS:
        attributes["peripheries"] = "2"
        attributes["label"] = attributes["label"].replace(" ", r"\n", 1)
    return attributes


def _value_attributes(identifier: str, relation: Relation) -> _Attributes:
    """Return a value node (condition, risk factor): a plain box in its relation's colour."""
    return {
        "label": ABBREVIATIONS.get(identifier, identifier),
        "shape": "box",
        "style": "filled",
        "color": RELATION_COLOURS[relation],
    }


def _reified_attributes() -> _Attributes:
    """Return the reified coprescription node: a small diamond in the coprescription colour."""
    return {
        "label": _COP_LABEL,
        "shape": "diamond",
        "style": "filled",
        "color": RELATION_COLOURS[Relation.COPRESCRIPTION],
        "fontsize": str(_SMALL_PT),
        "margin": "0",
    }


def _label_widths(labels: Mapping[str, str]) -> dict[str, float]:
    """Measure labels, in inches, with Graphviz itself (one ``dot`` run, JSON output)."""
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


def _drug_attributes(knowledge: _Knowledge) -> dict[str, _Attributes]:
    """Return the drugs' flat ellipses: fixed height, wider than the measured name by a factor."""
    labels = {d: _text(knowledge.drugs[d], "name_es") for d in sorted(knowledge.drugs)}
    widths = _label_widths(labels)
    return {
        drug_id: {
            "label": label,
            "shape": "ellipse",
            "style": "filled",
            "fixedsize": "true",
            "width": f"{_DRUG_WIDTH_FACTOR * widths[drug_id] + _DRUG_WIDTH_PAD_IN:.3f}",
            "height": f"{_DRUG_HEIGHT_IN:.3f}",
        }
        for drug_id, label in labels.items()
    }


def _node_attributes(knowledge: _Knowledge) -> dict[str, _Attributes]:
    """Return the drawing attributes of every node, by node identifier."""
    nodes = {c: _category_attributes(knowledge, c) for c in sorted(knowledge.categories)}
    nodes.update(
        {condition_node(h): _value_attributes(h, Relation.TREATS) for _, h in knowledge.treats}
    )
    risk_factors = {r for _, r in knowledge.contraindicated}
    risk_factors |= {cop.risk_factor for cop in knowledge.coprescriptions}
    nodes.update(
        {
            risk_factor_node(r): _value_attributes(r, Relation.CONTRAINDICATED)
            for r in sorted(risk_factors)
        }
    )
    nodes.update({cop.node: _reified_attributes() for cop in knowledge.coprescriptions})
    nodes.update(_drug_attributes(knowledge))
    return nodes


# --- placement ----------------------------------------------------------------------------------


def _sort_keys(knowledge: _Knowledge) -> dict[str, float]:
    """Return each node's position in the top-to-bottom order of :data:`CATEGORY_ORDER`.

    A drug takes its first told category's key, a condition just below its category, the
    reified coprescription just above its subject, and a risk factor the median of its sources.
    """
    known = [c for c in CATEGORY_ORDER if c in knowledge.categories]
    extra = sorted(set(knowledge.categories) - set(known))
    keys = {c: float(i) for i, c in enumerate([*known, *extra])}
    for subject, condition in knowledge.treats:
        keys[condition_node(condition)] = keys[subject] + 0.5
    for cop in knowledge.coprescriptions:
        keys[cop.node] = keys[cop.subject] - 0.5
    for drug_id, record in knowledge.drugs.items():
        keys[drug_id] = min(keys[c] for c in _texts(record, "categories"))
    sources: dict[str, list[float]] = {}
    for subject, risk_factor in knowledge.contraindicated:
        sources.setdefault(risk_factor, []).append(keys[subject])
    for cop in knowledge.coprescriptions:
        sources.setdefault(cop.risk_factor, []).append(keys[cop.node])
    for risk_factor, source_keys in sources.items():
        keys[risk_factor_node(risk_factor)] = statistics.median(source_keys)
    return keys


def _columns(knowledge: _Knowledge, nodes: Mapping[str, _Attributes]) -> list[list[str]]:
    """Return the columns from left to right, each sorted top to bottom.

    A category stands in the column of its depth, a condition beside its category, the reified
    coprescription one column right of its subject, then come the risk factors and the drugs.
    """
    depths = knowledge.depths()
    risk_column, drug_column = max(depths.values()) + 1, max(depths.values()) + 2
    column_of = dict(depths)
    column_of.update({condition_node(h): depths[s] for s, h in knowledge.treats})
    column_of.update({cop.node: depths[cop.subject] + 1 for cop in knowledge.coprescriptions})
    column_of.update({d: drug_column for d in knowledge.drugs})
    keys = _sort_keys(knowledge)
    columns: list[list[str]] = [[] for _ in range(drug_column + 1)]
    for node_id in sorted(nodes, key=lambda n: (keys[n], n)):
        columns[column_of.get(node_id, risk_column)].append(node_id)
    return [column for column in columns if column]


def _edge(graph: graphviz.Digraph, fact: _Fact) -> None:
    """Draw one fact in the colour of its relation; only ⊂ and ∈ edges set the columns.

    A self-link is an arc over the top of its node, from the top-right to the top-left corner.
    """
    attributes = {"color": RELATION_COLOURS[fact.relation]}
    if fact.relation in _UNDIRECTED:
        attributes["dir"] = "none"
    if fact.relation not in _RANKING:
        attributes["constraint"] = "false"
    if fact.tail == fact.head:
        attributes.update(tailport="ne", headport="nw")
    graph.edge(fact.tail, fact.head, **attributes)


def _common_attributes(graph: graphviz.Digraph) -> None:
    """Set the font, sizes and colours shared by both passes."""
    graph.attr(margin="0", pad="0.01", fontname=_FONT, outputorder="edgesfirst")
    graph.attr("node", fontname=_FONT, fontsize=str(_NAME_PT), fontcolor=_INK, color=_INK)
    graph.attr("node", fillcolor="white", penwidth=_BORDER_PT)
    graph.attr("node", margin="0.04,0.008", height="0.05", width="0.05")
    graph.attr("edge", arrowsize=_ARROW_SIZE, penwidth=_EDGE_PT)


def _positions(
    knowledge: _Knowledge, nodes: Mapping[str, _Attributes], columns: Sequence[Sequence[str]]
) -> dict[str, str]:
    """Place the nodes with ``dot`` and return each node's position, in points.

    This layout is never drawn. With ``rankdir=RL`` the drugs are the rightmost column. Edges fix
    the order inside each column (a chain from top to bottom; ``dot`` treats it as a preference,
    and :data:`CATEGORY_ORDER` is chosen so that it holds) and the order of the columns (weight
    0: they do not pull nodes); the facts of :data:`_PLACING` pull related nodes together, each
    turned to point left (or down, inside a column) so that none contradicts the columns. A node
    that needs free room above it (:func:`_room_above`) is placed taller by that room and then
    moved down by half of it.
    """
    room = _room_above(knowledge)
    heights = _node_heights(nodes, sorted(room))
    graph = graphviz.Digraph(name="placement")
    graph.attr(rankdir="RL", newrank="true", nodesep=_NODESEP_IN, ranksep=_RANKSEP_IN)
    _common_attributes(graph)
    for index, column in enumerate(columns):
        with graph.subgraph(name=f"column_{index}") as rank:
            rank.attr(rank="min" if index == len(columns) - 1 else "same")
            for node_id in column:
                attributes = dict(nodes[node_id])
                if node_id in room:
                    attributes["height"] = f"{heights[node_id] + room[node_id]:.3f}"
                rank.node(node_id, **attributes)
            for upper, lower in pairwise(column):
                rank.edge(upper, lower)
    for left, right in pairwise(columns):
        graph.edge(right[0], left[0], weight="0")
    place = {n: (c, -r) for c, column in enumerate(columns) for r, n in enumerate(column)}
    for fact in _placing_facts(knowledge):
        tail, head = sorted((fact.tail, fact.head), key=place.__getitem__, reverse=True)
        graph.edge(tail, head, weight=_PLACING_WEIGHTS.get(fact.relation, "1"))
    layout = json.loads(graph.pipe(format="json"))
    points: dict[str, tuple[float, float]] = {}
    for obj in layout.get("objects", []):
        if obj.get("name") in nodes:
            x, y = (float(v) for v in str(obj["pos"]).split(","))
            points[str(obj["name"])] = (x, y - _POINTS_PER_IN * room.get(obj["name"], 0.0) / 2)
    for column in columns:
        heights_down = [points[n][1] for n in column]
        if heights_down != sorted(heights_down, reverse=True):
            logger.warning("dot did not keep the top-to-bottom order of column %s", column)
    return {n: f"{x:.2f},{y:.2f}" for n, (x, y) in points.items()}


def _placing_facts(knowledge: _Knowledge) -> list[_Fact]:
    """Return the facts that pull nodes together in the placement (:data:`_PLACING`).

    A risk factor is pulled only by its median source or sources (the middle one or two in the
    vertical order). Pulled by all of them, a factor told of distant categories (ERC: AINE,
    Tiazidas, Sulfonilureas, Biguanidas) opens empty bands in every column its pull crosses.
    """
    keys = _sort_keys(knowledge)
    facts = [f for f in knowledge.facts() if f.relation in _PLACING]
    risk_nodes = {risk_factor_node(r) for _, r in knowledge.contraindicated}
    risk_nodes |= {risk_factor_node(cop.risk_factor) for cop in knowledge.coprescriptions}
    sources: dict[str, list[str]] = {}
    for fact in facts:
        if fact.head in risk_nodes:
            sources.setdefault(fact.head, []).append(fact.tail)
    median: set[tuple[str, str]] = set()
    for risk_node, tails in sources.items():
        ordered = sorted(tails, key=lambda t: (keys[t], t))
        middle = (len(ordered) - 1) / 2
        median |= {(t, risk_node) for i, t in enumerate(ordered) if abs(i - middle) <= 0.5}
    return [f for f in facts if f.head not in risk_nodes or (f.tail, f.head) in median]


def _room_above(knowledge: _Knowledge) -> dict[str, float]:
    """Return the free height (in) to keep above some nodes.

    A category with a self-link needs room for its arc, and a condition (drawn below its
    category) room for the shaft of the arrow that points to it.
    """
    room = {a: _LOOP_ROOM_IN for a, b in knowledge.interactions if a == b}
    room.update({condition_node(h): _ARROW_ROOM_IN for _, h in knowledge.treats})
    return room


def _node_heights(nodes: Mapping[str, _Attributes], node_ids: Sequence[str]) -> dict[str, float]:
    """Measure the drawn height (in) of some nodes with Graphviz (one ``dot`` run, JSON output)."""
    probe = graphviz.Digraph()
    _common_attributes(probe)
    for node_id in node_ids:
        probe.node(node_id, **nodes[node_id])
    layout = json.loads(probe.pipe(format="json"))
    return {
        str(obj["name"]): float(obj["height"])
        for obj in layout.get("objects", [])
        if obj.get("name") in node_ids
    }


def build_ontology_graph(results: Mapping[str, object], *, inferred: bool) -> graphviz.Digraph:
    """Build the Graphviz source of Fig. 2 from the ``ontology`` section of ``results.json``.

    Parameters
    ----------
    results : Mapping[str, object]
        The content of ``results.json`` (schema ``symai.ej2.results/1``).
    inferred : bool
        Add each drug's derived membership in a category defined by conjuncts (never told:
        forward chaining classifies the drug by the definition), as a membership edge at 50 %
        opacity. The nodes keep the positions of the told variant.

    Returns
    -------
    graphviz.Digraph
        The graph, with every node pinned (render it with ``neato -n2``); its ``source`` is
        deterministic.

    Raises
    ------
    ResultsFormatError
        If a section or field the figure needs is missing.
    """
    knowledge = _Knowledge.from_section(_section(results, "ontology"))
    nodes = _node_attributes(knowledge)
    columns = _columns(knowledge, nodes)
    positions = _positions(knowledge, nodes, columns)
    graph = graphviz.Digraph(
        name=INFERRED_FIGURE if inferred else ONTOLOGY_FIGURE,
        comment=_PINNED_COMMENT,
        engine="neato",
    )
    graph.attr(splines="true", esep=_EDGE_CLEARANCE, nodesep=_LOOP_SIZE_IN)
    _common_attributes(graph)
    for column in columns:
        for node_id in column:
            graph.node(node_id, pos=f"{positions[node_id]}!", **nodes[node_id])
    for fact in knowledge.facts():
        _edge(graph, fact)
    if inferred:
        _classification_edges(graph, results)
    return graph


def _classification_edges(graph: graphviz.Digraph, results: Mapping[str, object]) -> None:
    """Draw each drug's derived membership in a defined category, at 50 % opacity."""
    members = _section(_section(results, "formulary_derivation"), "defined_category_members")
    for category_id in sorted(members):
        for drug_id in _texts(members, category_id):
            graph.edge(drug_id, category_id, color=INFERRED_MEMBER_COLOUR)


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
    """Draw Fig. 2 (``inferred=False``) or its variant with derived memberships, as .dot/.pdf/.png.

    Parameters
    ----------
    results : Mapping[str, object]
        The content of ``results.json``.
    out_stem : Path
        Output path without extension; parent directories are created.
    inferred : bool
        Draw the variant with each drug's derived membership in a defined category.

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
        graphviz.render("neato", "pdf", dot_path, outfile=pdf_path, neato_no_op=2)
        png_source = graphviz.Source(
            graph.source.replace("{", f"{{\n\tdpi={_PNG_DPI}", 1), engine="neato"
        )
        png_source.render(outfile=png_path, format="png", cleanup=True, neato_no_op=2)
    return dot_path, pdf_path, png_path
