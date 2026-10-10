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

## Reproduce every result of the report

One command runs the experiments of EJ1, EJ2 and EJ3, in that order, from the repository root:

```bash
conda activate symai
python -m symbolic_ai.reproduce --workers 8      # or: symai-reproduce --workers 8
```

It calls each exercise's `--experiments` (the commands of the sections below) with the same output
directories, prints what each exercise prints, and ends with a summary: exit code, wall time and
results file per exercise. It keeps going if one exercise fails and then exits with code 1.
Measured on a 24-core workstation with `--workers 12`: EJ1 9 s, EJ2 23 s, EJ3 88 s, about
2 minutes in total. Every file is byte-identical for any `--workers` (only `provenance` records the
git commit).

Options: `--only ej1 ej3` runs a subset; `--out-root DIR` writes to `DIR/ej{1,2,3}` instead of
`outputs/ej{1,2,3}`; `--data-dir PATH` uses another database; `--workers N` goes to EJ1 and EJ3
(EJ2 runs in one process); `--report-figures DIR` also copies the report's figure PDFs into `DIR`
under the names the report's LaTeX uses (`ej1_value_ordering.pdf`, `ej2_ontology.pdf`,
`ej2_taxonomy_a.pdf`, `ej2_taxonomy_b.pdf`).

**Prerequisites.** EJ2's oracle (HermiT, bundled with `owlready2`) needs a Java runtime ≥ 11 on the
system, and its figures need Graphviz's `dot` (installed by `environment.yml`). Without Java, add
`--no-oracle`: it is passed to EJ2 only, the forward-chaining results are the same, and the oracle
fields of `outputs/ej2/results.json` are `null`.

Files written (`outputs/` is git-ignored):

| exercise | files |
|---|---|
| EJ1 | `outputs/ej1/results.json`, `outputs/ej1/fig_ej1_value_ordering.{pdf,png}` |
| EJ2 | `outputs/ej2/results.json`, `outputs/ej2/fig_ej2_ontology{,_inferred}.{dot,pdf,png}` |
| EJ3 | `outputs/ej3/results.json` |

Where each table and figure of the report comes from:

| report | file | JSON section |
|---|---|---|
| EJ1, Table II (decisions of P1) | `outputs/ej1/results.json` | `scenarios` (Γ⁺/Γ⁻ model counts from the oracle) |
| EJ1, Fig. 1 (value ordering, P2) | `outputs/ej1/fig_ej1_value_ordering.pdf` | drawn from `value_ordering` |
| EJ2, Fig. 2 (the ontology as a semantic network) | `outputs/ej2/fig_ej2_ontology.pdf` | drawn from `ontology` and `formulary_derivation.defined_category_members` |
| EJ2, Fig. 3 (deduced against told taxonomy) | `outputs/ej2/fig_ej2_taxonomy_{a,b}.pdf` (not yet drawn by this version of the code; `--report-figures` reports them missing) | `taxonomy` |
| EJ3, Table III (P6 scenarios) | `outputs/ej3/results.json` | `scenarios` |
| EJ3, Table IV (P7 cost and effort) | `outputs/ej3/results.json` | `search_comparison.configurations` and `search_comparison.baseline` |

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

# equivalently, the console script declared in pyproject.toml:
symai-ej2 --experiments
```

`--experiments` writes `outputs/ej2/results.json` (schema `symai.ej2.results/1`: provenance, the
told ontology, the fixed point, P3 `formulary_derivation`, P4 `taxonomy`, `inheritance`,
`consistency`) and `outputs/ej2/fig_ej2_ontology{,_inferred}.{dot,pdf,png}`, the ontology drawn as
a semantic network by Graphviz from that file. The `taxonomy` section also holds the deduced
taxonomy (`direct_edges` told/deduced, `told_edges_made_indirect`, `redundant_told_edges`,
`equivalences`, `members`, `same_members_not_equivalent`, `most_specific_categories`,
`memberships_made_indirect`, `new_direct_memberships`, `oracle_direct_edges_agree`), drawn as the
two panels of Fig. 3, `outputs/ej2/fig_ej2_taxonomy_{a,b}.{dot,pdf,png}` (line style = told,
deduced or made indirect). Two runs give byte-identical files (no timestamps; the PDFs are
rendered with `SOURCE_DATE_EPOCH=0`).

**Optional requirements.** The oracle needs `owlready2` (it bundles HermiT) and a Java runtime
(≥ 11) on the system; the figures need Graphviz's `dot`. Both come with `environment.yml` (Java
excepted); with pip, `pip install -e ".[owl]"`. Without Java, run with `--no-oracle`: the
forward-chaining results are the same, the oracle fields of `results.json` are `null`, and the
oracle tests are skipped.

## Run — EJ3, the cost-aware agent (informed search)

EJ3 keeps the agent of EJ1 and adds a search on top of it. Agent 1 runs unchanged and still decides
PRESCRIBE / REQUEST_TEST / REFER; when it prescribes, the EJ3 agent searches the regimens of the
worst-case record Γ⁺ and returns the cheapest one instead of the first model DPLL found. The drug
costs are a table of their own (`data/formulary/drug_costs.csv`, data 1.2.0: illustrative monthly
costs in euro cents, not real prices).

- **State**: a partial regimen, a set of drugs that breaks none of A2-A6. **Action**: prescribe one
  candidate for the first uncovered condition, with the companions A5 requires. **Goal**: every
  present condition is covered (A1). **Path cost**: the cost of the drugs added.
- **Heuristic**: for each uncovered condition, the cheapest candidate still compatible with the
  drugs already prescribed (shared among the conditions a drug treats; infinite when a condition
  has none left). It is consistent, so A* graph search returns a cheapest regimen.
- **Algorithms** (`aima.search`, called unchanged): A* by default; IDA*, greedy best-first and
  uniform-cost search for comparison.
- **From Agent 1** the search takes the proof that a regimen exists, the essential and excluded
  drugs (to prune; `--no-prune` skips them) and Agent 1's regimen (an upper bound on the cost).
  Every regimen is checked against the clauses of Γ⁺ before it is returned.

```bash
conda activate symai

python -m symbolic_ai.p1_ej3_search.main --encounter E001
python -m symbolic_ai.p1_ej3_search.main --all
python -m symbolic_ai.p1_ej3_search.main --all --algorithm greedy    # astar | idastar | greedy | uniform_cost
python -m symbolic_ai.p1_ej3_search.main --all --no-prune            # without Agent 1's classification
python -m symbolic_ai.p1_ej3_search.main --all --json outputs/ej3_decisions.json
python -m symbolic_ai.p1_ej3_search.main --experiments --workers 8   # P6 and P7 (88 s with 12 workers)

# equivalently, the console script declared in pyproject.toml:
symai-ej3 --all
```

`--experiments` writes `outputs/ej3/results.json` (schema `symai.ej3.results/2`); the file is
byte-identical for any `--workers`. P6 decides the five encounters of the database. P7 takes the
4,032 fully observed records and, on the 2,752 satisfiable ones, runs the four algorithms with and
without the pruning, checks every cost against the cheapest regimen of the EJ1 oracle
(`RegimenSpace.regimens`, which the agent never uses) and measures how far Agent 1's own regimen is
from it. Nearly all of the run time is Agent 1 classifying the drugs of each record (37 SAT calls);
the searches themselves take seconds. Each run also reports its solution depth d and its effective branching factor
b*, the root of N + 1 = 1 + b* + ... + (b*)^d for the N nodes generated (AIMA 4th ed. §3.6.1),
averaged per configuration and per number of conditions.

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
│       ├── reproduce.py              one command for every experiment of the report
│       ├── dataloader/               the only reader of data/
│       ├── viz/                      shared IEEE figure style for every exercise
│       ├── p1_ej1_logic/             EJ1: the SAT prescribing agent (this practical's core)
│       ├── p1_ej2_ontology/          EJ2: ontology, forward chaining, HermiT oracle, figure
│       └── p1_ej3_search/            EJ3: the cost-aware agent (A* for the cheapest regimen)
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
- EJ3: aima-python's `astar_search`, `iterative_deepening_astar_search`,
  `best_first_graph_search`, `uniform_cost_search` and `InstrumentedProblem` (same vendored copy),
  called unchanged. The problem formulation follows §3.1 of *Artificial Intelligence: A Modern
  Approach* (4th ed.), best-first search Fig. 3.7, uniform-cost search §3.4.2, greedy search §3.5.1,
  A* and consistency §3.5.2, IDA* §3.5.5, heuristics from relaxed problems §3.6.2, and one
  condition branched per node as in backtracking search for CSPs §5.3.
- **owlready2** 0.51, https://pypi.org/project/owlready2/ (J.-B. Lamy, *Artificial Intelligence in
  Medicine* 80, 2017), LGPL-3.0-or-later, with the bundled **HermiT** 1.3.8 OWL 2 reasoner
  (B. Glimm, I. Horrocks, B. Motik, G. Stoilos, Z. Wang, *J. Automated Reasoning* 53(3), 2014),
  LGPL-3.0-or-later — the EJ2 oracle (optional dependency).
- **networkx** 3.7, https://networkx.org, BSD-3-Clause — acyclicity, reachability and transitive
  closure checks of the taxonomy.
- **Graphviz** 14.1.2, https://graphviz.org, EPL-1.0, through **python-graphviz** 0.21,
  https://github.com/xflr6/graphviz, MIT — the EJ2 ontology figure.
