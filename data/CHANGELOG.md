# Changelog

All notable changes to the course database are documented here, one entry per data version
(semantic versioning: MINOR for additive changes, MAJOR only by explicit team decision).

## 1.1.0 — 2026-10-01

Additive: the ontology of drug categories for EJ2. No existing file or row changed; EJ1 stays
pinned to 1.0.0 and `load_formulary(version="1.0.0")` returns exactly what it returned before.

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
