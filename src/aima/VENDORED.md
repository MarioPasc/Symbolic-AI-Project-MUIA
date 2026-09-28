# Vendored `aima-python`

This directory is an unmodified, byte-for-byte copy of six files from
[aimacode/aima-python](https://github.com/aimacode/aima-python), vendored to avoid the package's
`requirements.txt` (which pulls TensorFlow, Keras and OpenCV) and to pin the code exactly so the
grader can run the repository offline against conda and PyPI only.

- **Source**: `https://github.com/aimacode/aima-python`
- **Commit**: `f104e03fdc1582f014b3a4c93f1a9b7d2da70114`
- **Date vendored**: 2026-09-28
- **Licence**: MIT (see `LICENSE` in this directory, copied from the same commit)

## Files

Downloaded from
`https://raw.githubusercontent.com/aimacode/aima-python/f104e03fdc1582f014b3a4c93f1a9b7d2da70114/aima/<file>`
(and `.../LICENSE` for the licence), byte-for-byte, **unmodified**.

| file | bytes | sha256 |
|---|---|---|
| `__init__.py` | 104 | `b3df28b87438852fb485f33044b18b622cec18143f552845edb4b9708c598cdb` |
| `logic.py` | 82337 | `217bbb63bfec21354a2ad0511f92593773b78ef134406dbfbf3ce2061c5bec0a` |
| `utils.py` | 26367 | `8c26445d4ece99884a53ea19b72ca13796ed03bd514bd079a8815439e76ecca2` |
| `agents.py` | 39252 | `20e1038ab72f23836295b986d1497f666fe065d928f33184d490f98ddcfd84aa` |
| `csp.py` | 59767 | `d955e1d2689fa22ff0d4ffa458b40299aa0c2c579eeaeb3d4fc0761a1b679487` |
| `search.py` | 78367 | `05180b631a702259d14bb2368b56a18f7b88f2e445629d64e5628d1cbe84ca3a` |
| `LICENSE` | 1091 | `20d3931305eed48a9287938cce469ae9482a998d073b828c6e9cbcf60916a0f1` |

Hashes computed with Python's `hashlib.sha256`; verified stable across two independent fetches of
each URL in the same session (network-level determinism check, since the sandboxed environment this
copy was made in blocks `curl`/`wget` and a byte-exact `cmp` against a fresh download; re-run the
"How to update" command below to reproduce the hashes independently with `curl` and `sha256sum`).

## Why these six files

`aima.logic` (the module used by `symbolic_ai.p1_ej1_logic.solver`) imports from `aima.utils`, and
`aima.utils`/`aima.agents`/`aima.csp` are imported transitively by the package's own `__init__.py`
import graph. Vendoring all six keeps `import aima.logic` working exactly as it would from a full
checkout, without pulling in `aima.search` on its own merits beyond that same import graph, or any
of the modules that require TensorFlow/Keras/OpenCV (`deep_learning4e.py`, `nlp.py`,
`reinforcement_learning4e.py`, etc., none of which are vendored).

## Unmodified

Nothing in this directory is edited. Any behavioural adaptation needed by the exercise (e.g. a
custom `branching_heuristic` for `aima.logic.dpll`) lives in
`src/symbolic_ai/p1_ej1_logic/solver.py`, which imports these modules but never patches their
source. `src/aima/` is excluded from `ruff`, `mypy` and coverage (see `pyproject.toml`).

## How to update

1. Pick a new commit of `aimacode/aima-python`.
2. Re-download the same six files and `LICENSE` from
   `https://raw.githubusercontent.com/aimacode/aima-python/<new-commit>/aima/<file>` (and
   `.../LICENSE`), e.g.:

   ```bash
   commit=<new-commit>
   for f in __init__.py logic.py utils.py agents.py csp.py search.py; do
       curl -sL "https://raw.githubusercontent.com/aimacode/aima-python/${commit}/aima/${f}" -o "src/aima/${f}"
   done
   curl -sL "https://raw.githubusercontent.com/aimacode/aima-python/${commit}/LICENSE" -o src/aima/LICENSE
   sha256sum src/aima/*.py src/aima/LICENSE
   ```

3. Update the commit, date, byte counts and hashes in this file.
4. Run `pytest -m "not slow"`, `ruff check src/symbolic_ai tests`, `mypy` (all unaffected by
   `src/aima`, which stays excluded) and re-verify the reference outcomes of
   `docs/SPECIFICATIONS/01-database.md` §6 still hold, since `aima.logic.dpll`'s tie-breaking
   behaviour on ties between the pure-symbol rule, unit propagation and the branching heuristic is
   what those numbers depend on.
5. Commit the update alone, `chore(vendor): update aima-python to <new-commit>`.
