# Course database: formulary and patients

One source of truth for the practical part of IA simbólica (MUIA, UAM 2026/27): a didactic drug
formulary and a handful of fictitious patient encounters. Práctica 1 (EJ1 SAT agent, EJ2 ontology,
EJ3 search) and Práctica 2 all read the same tables, described here.

## Disclaimer

The patients are invented. The formulary is a didactic simplification of real pharmacology, built
for a symbolic-AI exercise; it is **not clinical guidance** and must not be used to inform an actual
prescribing decision. ATC and ICD-10 codes are written from memory and have not been checked against
the WHO index. The drug costs are illustrative values chosen for the exercise, not real prices.

## Layout

```
data/
├── datapackage.json            descriptor: name, version, resources, schemas, keys
├── README.md                   this file
├── CHANGELOG.md                one entry per data version
├── formulary/                  stable knowledge, the same for every patient
├── ontology/                   the same knowledge per drug category (EJ2, since 1.1.0)
└── patients/                   the cases
```

`ontology/` holds the categories (`categories.csv`), their subcategory links (`subcategories.csv`;
a category may have several parents), the categories defined by the conjunction of others
(`definitions.csv`), the told categories of each drug (`memberships.csv`; at least one per drug),
the links written once per category (`indications.csv`, `contraindications.csv`,
`interactions.csv`, where a self-link `subject_a = subject_b` relates any two members of one
category, `coprescriptions.csv`), the therapeutic families (`families.csv`) and the disjoint sets
and partitions (`disjoint_sets.csv`). EJ2 derives the drug-level formulary from these tables; it
never reads the drug-level tables of `formulary/` except to compare with them.

`formulary/drug_costs.csv` (EJ3, since 1.2.0) gives each drug the cost of one month of treatment,
in euro cents. It is a table of its own, read with `load_drug_costs`: `load_formulary` and the
`Formulary` it returns are unchanged. Every drug must have one cost, none negative.

See `../docs` (private TFM repository, not in this code repository) for the full schema
specification, `SPECIFICATIONS/01-database.md`.

## Access

Only `symbolic_ai.dataloader` reads these files. Any exercise module that needs data asks the
dataloader for immutable domain objects; it never opens a CSV directly.

## Validating the database

```bash
python -m symbolic_ai.dataloader.validate
```

Exits 0 and prints a summary of table counts when every check passes; otherwise lists every
violation found (primary/foreign keys, enums, `since` versions, completeness and consistency rules)
and exits 1.

The descriptor also follows the [Frictionless Data Package](https://specs.frictionlessdata.io/data-package/)
specification, so it can optionally be checked with the reference tool:

```bash
frictionless validate data/datapackage.json
```

## Compatibility

The database changes additively only: new tables, nullable or defaulted columns, and new rows.
Existing rows and columns are never renamed, removed or reinterpreted. Every formulary row carries
`since`, the data version in which it was added; `load_formulary(version=...)` reconstructs the
formulary as of any earlier version. See `CHANGELOG.md` for the version history.
