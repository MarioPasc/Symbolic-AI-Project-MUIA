# symnbolic_ai_project

Code for the practicals of *IA simbólica* (Máster Universitario en Inteligencia Artificial, UAM,
2026/27): a knowledge-based agent that prescribes safely for patients with several conditions,
built over one shared drug formulary and patient database (`data/`).

## Install

```bash
conda env create -f environment.yml          # creates the `symai` environment, editable install
# after a change to environment.yml or pyproject.toml:
conda env update -f environment.yml --prune
```

Check the install:

```bash
conda run -n symai python -c "import aima.logic, symbolic_ai; print('ok')"
```

## Run — EJ1, the SAT prescribing agent

```bash
conda activate symai

python -m symbolic_ai.p1_ej1_logic.main --encounter E001
python -m symbolic_ai.p1_ej1_logic.main --all
python -m symbolic_ai.p1_ej1_logic.main --all --first-value true        # textbook DPLL value ordering
python -m symbolic_ai.p1_ej1_logic.main --encounter E004 --no-classify  # skip essential/excluded/optional
python -m symbolic_ai.p1_ej1_logic.main --all --json outputs/decisions.json
python -m symbolic_ai.p1_ej1_logic.main --all --data-dir /path/to/data

# equivalently, the console script declared in pyproject.toml:
symai-ej1 --all
```

### The experiments of the report (EJ1, *Resultados*)

```bash
python -m symbolic_ai.p1_ej1_logic.main --experiments --workers 8   # scenarios; verdicts and value ordering
python -m symbolic_ai.p1_ej1_logic.main --plot outputs/ej1/results.json   # redraw the figure only
```

`--experiments` writes `outputs/ej1/results.json` (schema `symai.ej1.results/3`) and
`outputs/ej1/fig_ej1_value_ordering.{pdf,png}`. The run is deterministic: the files are byte-identical for any
`--workers`. It decides the five encounters of the database (P1). On the 4,032 fully observed records, every
base the agent can query up to clause order, it checks DPLL's verdict against an independent oracle
(`p1_ej1_logic/semantics.py`, the axioms evaluated on sets of drugs) and compares DPLL's two value orderings with
the oracle's minimum regimen (P2).

The problem (axioms A1-A6, the bounds Γ⁺/Γ⁻, the decision rule) and the reasons the agent calls
`aima.logic.dpll` directly, with `T_d` tried false first by default, are in the report (EJ1,
Methodology) and in the team's design notes, which are kept outside this repository.

## Run — EJ2, the ontology of drug categories

EJ2 writes the formulary's knowledge once per drug *category* (`data/ontology/`, data 1.1.0: a
taxonomy of 32 drug categories with multiple inheritance, one category defined by its conjuncts,
the told categories of each drug, and links written on categories, the interactions being
self-links on three adverse-effect categories), translates it into first-order definite clauses
(axioms O1-O6, with both directions of every definition), and computes their fixed point by
forward chaining (AIMA Fig. 9.3 with the incremental rule of §9.3.3, over aima-python's `Expr` and
unification). Classification, subsumption (by a prototype of each category) and consistency are
queries on that fixed point, and the drug-level formulary of EJ1 is read off it and compared with
the hand-written formulary 1.0.0. Every answer is checked against an independent oracle: an OWL 2
export of the same tables reasoned by HermiT through owlready2.

```bash
conda activate symai

python -m symbolic_ai.p1_ej2_ontology.main --classify tramadol       # categories of one drug
python -m symbolic_ai.p1_ej2_ontology.main --taxonomy                # subsumers of the 44 named categories
python -m symbolic_ai.p1_ej2_ontology.main --formulary               # derived formulary vs 1.0.0
python -m symbolic_ai.p1_ej2_ontology.main --experiments             # P3, P4 and both figures (~20 s)
python -m symbolic_ai.p1_ej2_ontology.main --experiments --no-oracle # without HermiT (no Java needed)
python -m symbolic_ai.p1_ej2_ontology.main --plot outputs/ej2/results.json   # redraw the figures only
```

`--experiments` writes `outputs/ej2/results.json` (schema `symai.ej2.results/1`: provenance, the
told ontology, the fixed point, P3 `formulary_derivation`, P4 `taxonomy`, `inheritance`,
`consistency`) and `outputs/ej2/fig_ej2_ontology{,_inferred}.{dot,pdf,png}`, the ontology drawn as
a semantic network by Graphviz from that file. Two runs give byte-identical files (no timestamps;
the PDFs are rendered with `SOURCE_DATE_EPOCH=0`).

**Optional requirements.** The oracle needs `owlready2` (it bundles HermiT) and a Java runtime
(≥ 11) on the system; the figures need Graphviz's `dot`. Both come with `environment.yml` (Java
excepted); with pip, `pip install -e ".[owl]"`. Without Java, run with `--no-oracle`: the
forward-chaining results are the same, the oracle fields of `results.json` are `null`, and the
oracle tests are skipped.

## Test

```bash
pytest -m "not slow"                              # fast suite, run often
pytest                                             # full suite (includes the brute-force checks)
pytest -m "not integration"                        # skip the tests that read the real database
ruff format --check src/symbolic_ai tests && ruff check src/symbolic_ai tests
mypy
python -m symbolic_ai.dataloader.validate          # check data/ against its schema and rules
```

## Layout

```
symnbolic_ai_project/
├── pyproject.toml, environment.yml   package metadata, dependencies, tool configuration
├── data/                             the formulary and patient database (01-database.md)
├── outputs/                          run artefacts (JSON, figures), git-ignored, recreated by main.py
├── src/
│   ├── aima/                         vendored aima-python (read-only; see src/aima/VENDORED.md)
│   └── symbolic_ai/
│       ├── dataloader/               the only reader of data/
│       ├── viz/                      shared IEEE figure style for every exercise
│       ├── p1_ej1_logic/             EJ1: the SAT prescribing agent (this practical's core)
│       ├── p1_ej2_ontology/          EJ2: ontology, forward chaining, HermiT oracle, figure
│       └── p1_ej3_search/            EJ3: placeholder
└── tests/                            mirrors src/symbolic_ai/
```

`docs/SPECIFICATIONS/02-code-architecture.md` (private TFM repository) is the full design
reference for this layout and for `p1_ej1_logic`'s file roles.

## Credits

- **aima-python**, https://github.com/aimacode/aima-python, commit
  `f104e03fdc1582f014b3a4c93f1a9b7d2da70114`, MIT licence — vendored unmodified in `src/aima/`
  (`src/aima/VENDORED.md`). `symbolic_ai.p1_ej1_logic.solver.DPLLSolver` wraps `aima.logic.dpll`
  (Fig. 7.17 of *Artificial Intelligence: A Modern Approach*).
- Russell, S., Norvig, P. *Artificial Intelligence: A Modern Approach*, 4th ed., 2022 — §7.6.1
  (DPLL, value ordering), §7.7 (knowledge-based agents).
- Figure style (`src/symbolic_ai/viz/style.py`): matplotlib; Paul Tol's *bright* colour-blind-safe palette
  (P. Tol, "Colour schemes", SRON technical note SRON/EPS/TN/09-002).
- EJ2: aima-python's `Expr`, `expr`, `unify`, `subst`, `variables`, `is_definite_clause`,
  `parse_definite_clause` (same vendored copy); its `fol_fc_ask` is used only as a test reference.
  The forward-chaining loop follows Fig. 9.3 and §9.3.2-9.3.3 of *Artificial Intelligence: A
  Modern Approach*; the ontology follows §10.1-10.2 and §10.5 (4th ed.).
- **owlready2** 0.51, https://pypi.org/project/owlready2/ (J.-B. Lamy, *Artificial Intelligence in
  Medicine* 80, 2017), LGPL-3.0-or-later, with the bundled **HermiT** 1.3.8 OWL 2 reasoner
  (B. Glimm, I. Horrocks, B. Motik, G. Stoilos, Z. Wang, *J. Automated Reasoning* 53(3), 2014),
  LGPL-3.0-or-later — the EJ2 oracle (optional dependency).
- **networkx** 3.7, https://networkx.org, BSD-3-Clause — acyclicity, reachability and transitive
  closure checks of the taxonomy.
- **Graphviz** 14.1.2, https://graphviz.org, EPL-1.0, through **python-graphviz** 0.21,
  https://github.com/xflr6/graphviz, MIT — the EJ2 ontology figure.
