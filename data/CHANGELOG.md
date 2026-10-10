# Changelog

All notable changes to the course database are documented here, one entry per data version
(semantic versioning: MINOR for additive changes, MAJOR only by explicit team decision).

## 1.2.0 — 2026-10-09

Additive: the drug costs for EJ3 (cost-aware search). No existing file or row changed; EJ1 stays
pinned to 1.0.0 and `load_formulary(version="1.0.0")` returns exactly what it returned before. Loads
that take the default version (`load_formulary()`, `load_ontology()`) return the same content,
labelled 1.2.0.

- **New table `formulary/drug_costs.csv`** (resource `drug_costs`, every row `since = 1.2.0`):
  `drug_id`, `monthly_cost_cents` (one month of treatment, in euro cents), `source` (left empty, as
  in the other formulary tables). 18 rows, one per drug.
- The costs are **illustrative, not real prices**. Long-established generics cost between 1.40 and
  6.10 euros a month; apixaban (54.00) and linagliptin (38.00) cost an order of magnitude more.
  Omeprazole (2.70) is priced so that ibuprofen with its companion (5.00) is dearer than tramadol
  alone (4.80), which makes the coprescription of A5 matter to the cost of a regimen.
- **Access**: `load_drug_costs(data_dir, version)`, a loader of its own that returns one `DrugCost`
  per drug; `Formulary` has no new field.
- **Validation** (`python -m symbolic_ai.dataloader.validate`): no cost is negative. That every
  drug has a cost is checked by `load_drug_costs`, not by the validation, so that a drug added later
  without a cost does not invalidate the database for the exercises that read none.

## 1.1.0 — 2026-10-01, amended 2026-10-01 (R1, R2, R3a)

Additive: the ontology of drug categories for EJ2. No existing file or row changed; EJ1 stays
pinned to 1.0.0 and `load_formulary(version="1.0.0")` returns exactly what it returned before.

### Amended 2026-10-01 (R1, R2, R3a)

Version 1.1.0 was edited in place by team decision (it had been pushed the same day and only EJ2
reads `ontology/`). `formulary/` and `patients/` are untouched. Every row keeps `since = 1.1.0`.

- **R1, multiple inheritance.** Three adverse-effect categories under `drugs`:
  `bleeding_risk_drugs` (Riesgo hemorrágico), `bradycardic_drugs` (Bradicardizantes) and
  `serotonergic_drugs` (Serotoninérgicos). They are second parents of `anticoagulants`, `nsaids`
  and `ssris`; of `beta_blockers` and `non_dihydropyridines`; and of `ssris`. The taxonomy is now a
  DAG (`ssris` has three told parents). The 5 category-pair interactions become 3 **self-links**
  (`subject_a = subject_b`), one per adverse-effect category: any two different members of it
  interact.
- **R2, a defined category.** New table `definitions.csv` (`category_id`, `conjunct_id`, `since`):
  `serotonergic_opioids` ≡ `opioids` ⊓ `serotonergic_drugs`. Its `subcategories` row is removed;
  tramadol is told in `opioids` and `serotonergic_drugs` (two rows) and is classified into
  `serotonergic_opioids` by the reasoner. The epilepsy contraindication moves from `tramadol` to
  `serotonergic_opioids`, so all 10 contraindication links are on categories.
- **R3a, no upper ontology.** `clinical_objects`, `conditions` and `risk_factors` are removed, with
  their 3 subcategory links and the `upper` disjoint set. Conditions and risk factors stay the
  values of the links.
- Result: 33 categories (32 drug categories and `therapeutic_families`), 36 subcategory links,
  1 definition with 2 conjuncts, 19 told memberships, 6 indications, 10 contraindications (all on
  categories), 3 interaction self-links, 1 coprescription, 5 families, 7 disjoint sets (4 of them
  partitions).
- **Validation**: every drug has *at least one* told membership (checked by `load_ontology`; the
  "at most one" check is gone); `subject_a <= subject_b`; a defined category has at least two
  conjuncts, no `subcategories` row as child and no told member; the graph of subcategory and
  definition edges together is acyclic.

### As first published

- **New folder `ontology/`** (resources `ontology_*` in `datapackage.json`, every row
  `since = 1.1.0`): 33 categories (29 drug categories, the upper categories `clinical_objects`,
  `conditions`, `risk_factors`, and the category of categories `therapeutic_families`), 31
  subcategory links, 18 told memberships (one leaf category per drug), 6 indication links,
  10 contraindication links (9 on categories, 1 on a drug), 5 category-pair interactions,
  1 coprescription, 5 therapeutic families and 8 disjoint sets (5 of them partitions).
- The ontology never repeats the drug-level formulary tables (candidates, contraindications,
  adverse interactions, coprescriptions, drug classes): EJ2 derives them and compares the result
  with 1.0.0.
- **Validation** (`python -m symbolic_ai.dataloader.validate`): link subjects resolve to a category
  or a drug; category and drug identifiers never coincide; the subcategory graph is acyclic;
  `subject_a < subject_b`; every drug has exactly one told membership (the "at least one" half is
  checked by `load_ontology`, so that a drug added later does not invalidate the database for EJ1);
  every disjoint set has at least two members and one consistent `partition_of`.

## 1.0.0 — 2026-09-28

Initial database, designed for EJ1 (SAT prescribing agent) and shared by EJ2/EJ3 and Práctica 2.

- **Formulary**: 6 conditions, 18 drugs, 6 risk factors, 18 candidate pairs (A1), 7 adverse
  interactions (A3, all `major`), 11 contraindications (A4), 1 coprescription (A5), 5 exclusive
  drug classes with 11 members (A6) — 50 rule clauses in total for EJ1.
- **Patients**: 3 patients (P001–P003), 5 encounters (E001–E005) covering the EJ1 scenarios V1–V3
  of `EJ1-sat/README.md` §10.
- `source` columns of the formulary tables are left empty; they are filled with the CIMA-AEMPS /
  STOPP-START reference of each entry before the report cites the formulary.
