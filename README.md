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
│       ├── p1_ej2_ontology/          EJ2: placeholder
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
