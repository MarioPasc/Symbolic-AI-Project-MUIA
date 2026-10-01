"""Command-line entry point for EJ2 (``python -m symbolic_ai.p1_ej2_ontology.main``).

The only module of this exercise that prints or writes files. It reads the ontology and the
reference formulary through :mod:`symbolic_ai.dataloader` and has five modes: classify one drug
(``--classify``), print the taxonomy (``--taxonomy``) or the derived formulary against 1.0.0
(``--formulary``), run the experiments of the report and save them with both figures
(``--experiments``), or redraw the figures from a saved results file (``--plot``).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import symbolic_ai
from symbolic_ai.dataloader import load_formulary, load_ontology
from symbolic_ai.p1_ej2_ontology import owl_oracle
from symbolic_ai.p1_ej2_ontology.errors import ResultsFormatError, UnknownIdentifierError
from symbolic_ai.p1_ej2_ontology.experiments import (
    ConsistencyResult,
    FixedPointResult,
    FormularyDerivationResult,
    InheritanceResult,
    OracleAgreement,
    TaxonomyResult,
    as_jsonable,
    run_consistency,
    run_fixed_point_cost,
    run_formulary_derivation,
    run_inheritance,
    run_taxonomy,
)
from symbolic_ai.p1_ej2_ontology.ontology import Ontology, build_ontology, tagged_rules
from symbolic_ai.p1_ej2_ontology.plot import INFERRED_FIGURE, ONTOLOGY_FIGURE, plot_ontology
from symbolic_ai.p1_ej2_ontology.reasoner import (
    ReasonedOntology,
    classify,
    inherited_by_new_member,
    reason,
    taxonomy,
)

__all__ = ["REFERENCE_VERSION", "RESULTS_SCHEMA", "build_arg_parser", "main"]

#: Identifier and version of the layout of ``results.json``; bump it when a field changes meaning.
RESULTS_SCHEMA = "symai.ej2.results/1"
#: The hand-written formulary the derived one is compared with (EJ1 stays pinned to it).
REFERENCE_VERSION = "1.0.0"
#: Commit of the vendored aima-python (``src/aima/VENDORED.md``).
AIMA_PYTHON_COMMIT = "f104e03fdc1582f014b3a4c93f1a9b7d2da70114"

_EXPERIMENTS_COMMAND = "python -m symbolic_ai.p1_ej2_ontology.main --experiments"
_DEFAULT_OUT_DIR = Path("outputs") / "ej2"
_RESULTS_FILE = "results.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``(--classify DRUG | --taxonomy | --formulary | --experiments |
        --plot RESULTS_JSON) [--no-oracle] [--data-dir PATH] [--out-dir DIR]``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m symbolic_ai.p1_ej2_ontology.main",
        description="EJ2: forward chaining over the ontology of drug categories.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--classify", metavar="DRUG", help="Print the categories of one drug.")
    mode.add_argument(
        "--taxonomy", action="store_true", help="Print the subsumers of every named category."
    )
    mode.add_argument(
        "--formulary",
        action="store_true",
        help=f"Print the derived formulary, table by table, against {REFERENCE_VERSION}.",
    )
    mode.add_argument(
        "--experiments",
        action="store_true",
        help="Run P3 and P4, write results.json and both figures to --out-dir.",
    )
    mode.add_argument(
        "--plot",
        type=Path,
        metavar="RESULTS_JSON",
        help="Only redraw the figures from a saved results.json, into --out-dir.",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip HermiT (owlready2 + Java); the oracle fields of results.json are null.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Database directory (default: $SYMAI_DATA_DIR or <repository root>/data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory of --experiments and --plot (default: {_DEFAULT_OUT_DIR}).",
    )
    return parser


# --- simple modes -------------------------------------------------------------------------------


def _print_classification(reasoned: ReasonedOntology, drug_id: str) -> None:
    categories = classify(reasoned, drug_id)
    defined = set(reasoned.ontology.defined_ids)
    print(f"{drug_id}: leaf category {reasoned.ontology.leaf_of(drug_id)}")
    print(f"  categories: {', '.join(c for c in categories if c not in defined)}")
    print(f"  defined categories: {', '.join(c for c in categories if c in defined) or '-'}")


def _print_taxonomy(reasoned: ReasonedOntology) -> None:
    result = taxonomy(reasoned)
    for category_id in result.categories:
        above = ", ".join(result.subsumers[category_id]) or "-"
        print(f"{category_id} ⊑ {above}")
    print(
        f"{len(result.pairs())} proper subsumption pairs among {len(result.categories)} categories"
    )


def _print_formulary(result: FormularyDerivationResult) -> None:
    print(f"derived {result.derived_version} vs hand-written {result.reference_version}")
    print("table              reference  statements  derived  in both  only derived  only ref.")
    for t in result.tables:
        print(
            f"{t.table:<18} {t.rows_reference:>9}  {t.statements:>10}  {t.derived:>7}  "
            f"{t.in_both:>7}  {len(t.only_derived):>12}  {len(t.only_reference):>9}"
        )
        for key in t.only_derived:
            print(f"  only derived: {' / '.join(key)}")


# --- experiments --------------------------------------------------------------------------------


def _git(*arguments: str) -> str | None:
    """Run ``git`` in the repository root; ``None`` if git or the repository is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    """Installed version of distribution ``name``, or ``None``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _graphviz_version() -> str | None:
    """Version of the Graphviz ``dot`` executable, or ``None`` if it is not installed."""
    try:
        import graphviz

        return ".".join(str(part) for part in graphviz.version())
    except (ImportError, OSError, RuntimeError):
        return None


def _java_version() -> str | None:
    """First line of ``java -version``, or ``None`` if Java is not installed."""
    try:
        completed = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = completed.stderr.strip().splitlines()
    return lines[0] if lines else None


def _provenance(ontology: Ontology, with_oracle: bool) -> dict[str, object]:
    """Record what produced the results: command, code and data versions, libraries."""
    status = _git("status", "--porcelain")
    command = _EXPERIMENTS_COMMAND + ("" if with_oracle else " --no-oracle")
    return {
        "command": command,
        "package_version": symbolic_ai.__version__,
        "data_version": ontology.data.version,
        "reference_formulary_version": REFERENCE_VERSION,
        "aima_python_commit": AIMA_PYTHON_COMMIT,
        "oracle": "HermiT (bundled with owlready2)" if with_oracle else None,
        "owlready2": owl_oracle.owlready2_version() if with_oracle else None,
        "java": _java_version() if with_oracle else None,
        "graphviz": _graphviz_version(),
        "python_graphviz": _package_version("graphviz"),
        "networkx": _package_version("networkx"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
    }


def _ontology_section(ontology: Ontology) -> dict[str, object]:
    """Return the told knowledge (no inference), enough to draw Fig. 2, and the clauses."""
    data = ontology.data
    parents = ontology.parents()
    leaf = {m.object_id: m.category_id for m in data.memberships}
    families = set(data.families)
    return {
        "categories": [
            {
                "id": c.category_id,
                "name_es": c.name_es,
                "name_en": c.name_en,
                "atc_code": c.atc_code,
                "parents": list(parents.get(c.category_id, ())),
                "family": c.category_id in families,
                "drug_category": c.category_id in ontology.drug_categories,
            }
            for c in data.categories
        ],
        "drugs": [
            {"id": d.drug_id, "name_es": d.name_es, "leaf": leaf[d.drug_id]} for d in data.drugs
        ],
        "conditions": [{"id": c.condition_id, "name_es": c.name_es} for c in data.conditions],
        "risk_factors": [{"id": r.risk_factor_id, "name_es": r.name_es} for r in data.risk_factors],
        "indications": [
            {"subject": i.subject_id, "condition": i.condition_id} for i in data.indications
        ],
        "contraindications": [
            {"subject": c.subject_id, "risk_factor": c.risk_factor_id, "reason": c.reason}
            for c in data.contraindications
        ],
        "interactions": [
            {
                "subject_a": i.subject_a,
                "subject_b": i.subject_b,
                "effect": i.effect,
                "severity": i.severity,
            }
            for i in data.interactions
        ],
        "coprescriptions": [
            {
                "subject": c.subject_id,
                "risk_factor": c.risk_factor_id,
                "companion": c.companion_drug_id,
            }
            for c in data.coprescriptions
        ],
        "families": list(data.families),
        "disjoint_sets": [
            {"set_id": s.set_id, "categories": list(s.category_ids), "partition_of": s.partition_of}
            for s in data.disjoint_sets
        ],
        "defined_categories": [
            {"id": d.category_id, "kind": d.kind.value, "target": d.target_id}
            for d in ontology.defined_categories
        ],
        "clauses": [{"axiom": r.tag, "clause": str(r.clause)} for r in tagged_rules(ontology)],
    }


def _agreement_text(agreement: OracleAgreement | None) -> str:
    if agreement is None:
        return "oracle skipped"
    return f"HermiT: {agreement.compared} compared, {len(agreement.disagreements)} disagreements"


def _print_summary(
    p3: FormularyDerivationResult,
    tax: TaxonomyResult,
    inheritance: InheritanceResult,
    consistency: ConsistencyResult,
    cost: FixedPointResult,
) -> None:
    """Print the headline numbers of each experiment."""
    _print_formulary(p3)
    print(f"  memberships: {_agreement_text(p3.oracle_memberships)}")
    print(f"  rows and property values: {_agreement_text(p3.oracle_rows)}")
    print(
        f"P4.1 taxonomy: told {tax.told_pairs}, primitive {tax.primitive_pairs}, "
        f"primitive->defined {tax.primitive_to_defined}, defined->primitive "
        f"{tax.defined_to_primitive}, defined->defined {tax.defined_to_defined}, "
        f"total {tax.total_pairs}"
    )
    print(f"  classification: {_agreement_text(tax.oracle_classification)}")
    print(f"  prototypes: {_agreement_text(tax.oracle_prototypes)}")
    nsaids = next(r for r in inheritance.rows if r.category_id == "nsaids")
    print(
        f"P4.2 inheritance: {inheritance.min_rows}-{inheritance.max_rows} rows per new member; "
        f"nsaids {nsaids.rows}; {_agreement_text(inheritance.oracle)}"
    )
    print(
        f"P4.3 consistency: KB clashes {len(consistency.kb_clashes)}, inconsistent categories "
        f"{len(consistency.inconsistent_categories)}/{consistency.categories_checked}, "
        f"HermiT consistent {consistency.oracle_kb_consistent}"
    )
    print(
        f"  mutations {consistency.n_mutations}: clash expected {consistency.n_clash_expected} "
        f"(upper partition {consistency.n_clash_expected_from_upper}), detected "
        f"{consistency.n_clash_detected}; no clash expected {consistency.n_no_clash_expected}, "
        f"false alarms {consistency.n_false_alarms}"
    )
    print(f"  single run: {_agreement_text(consistency.oracle_single_run)}")
    print(f"  per-mutation runs: {_agreement_text(consistency.oracle_rechecked)}")
    print(
        f"P4.4 fixed point: {cost.told_facts} told + {cost.derived_facts} derived = "
        f"{cost.closure_facts} facts in {cost.iterations} iterations {cost.new_per_iteration}; "
        f"p={cost.predicates}, n={cost.constants}, k={cost.max_arity}, p*n^k={cost.bound}"
    )


def _experiment_results(args: argparse.Namespace, ontology: Ontology) -> dict[str, object]:
    """Run P3 and P4 (with the oracle unless ``--no-oracle``); return ``results.json``'s content."""
    with_oracle = not args.no_oracle
    reasoned = reason(ontology)
    reference = load_formulary(data_dir=args.data_dir, version=REFERENCE_VERSION)
    base_run = prototypes_run = None
    if with_oracle:
        base_run = owl_oracle.reason_base(ontology.data)
        prototypes_run = owl_oracle.reason_with_prototypes(ontology.data, ontology.named_categories)
    p3 = run_formulary_derivation(reasoned, reference, oracle_run=base_run)
    tax = run_taxonomy(reasoned, oracle_run=base_run, oracle_prototypes=prototypes_run)
    inheritance = run_inheritance(reasoned, oracle_prototypes=prototypes_run)
    consistency = run_consistency(reasoned, oracle_run=base_run)
    cost = run_fixed_point_cost(reasoned)
    _print_summary(p3, tax, inheritance, consistency, cost)
    return {
        "schema": RESULTS_SCHEMA,
        "provenance": _provenance(ontology, with_oracle),
        "ontology": _ontology_section(ontology),
        "fixed_point": as_jsonable(cost),
        "formulary_derivation": as_jsonable(p3),
        "taxonomy": as_jsonable(tax),
        "inheritance": as_jsonable(inheritance),
        "consistency": as_jsonable(consistency),
    }


def _load_results(path: Path) -> Mapping[str, object]:
    """Read a saved ``results.json`` and check its schema."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("schema") != RESULTS_SCHEMA:
        raise ResultsFormatError(f"{path} is not a {RESULTS_SCHEMA} results file")
    return loaded


def _draw_figures(results_path: Path, out_dir: Path) -> None:
    """Draw both variants of Fig. 2 from the saved results file and print the paths written."""
    results = _load_results(results_path)
    for stem, inferred in ((ONTOLOGY_FIGURE, False), (INFERRED_FIGURE, True)):
        for path in plot_ontology(results, out_dir / stem, inferred=inferred):
            print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI in the mode selected by the arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments, excluding the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.plot is not None:
        _draw_figures(args.plot, args.out_dir)
        return 0
    ontology = build_ontology(load_ontology(data_dir=args.data_dir))
    if args.classify is not None:
        reasoned = reason(ontology)
        try:
            _print_classification(reasoned, args.classify)
        except UnknownIdentifierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        inherited = inherited_by_new_member(reasoned, ontology.leaf_of(args.classify))
        print(f"  a new member of its leaf would receive {inherited.rows} formulary rows")
        return 0
    if args.taxonomy:
        _print_taxonomy(reason(ontology))
        return 0
    if args.formulary:
        reference = load_formulary(data_dir=args.data_dir, version=REFERENCE_VERSION)
        _print_formulary(run_formulary_derivation(reason(ontology), reference, oracle_run=None))
        return 0
    results = _experiment_results(args, ontology)
    results_path = args.out_dir / _RESULTS_FILE
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {results_path}")
    # The figures are drawn from the file just written, so they can only show saved results.
    _draw_figures(results_path, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
