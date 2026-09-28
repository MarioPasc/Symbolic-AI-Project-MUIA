# Course database: formulary and patients

One source of truth for the practical part of IA simbólica (MUIA, UAM 2026/27): a didactic drug
formulary and a handful of fictitious patient encounters. Práctica 1 (EJ1 SAT agent, EJ2 ontology,
EJ3 search) and Práctica 2 all read the same tables, described here.

## Disclaimer

The patients are invented. The formulary is a didactic simplification of real pharmacology, built
for a symbolic-AI exercise; it is **not clinical guidance** and must not be used to inform an actual
prescribing decision. ATC and ICD-10 codes are written from memory and have not been checked against
the WHO index.

## Layout

```
data/
├── datapackage.json            descriptor: name, version, resources, schemas, keys
├── README.md                   this file
├── CHANGELOG.md                one entry per data version
├── formulary/                  stable knowledge, the same for every patient
└── patients/                   the cases
```

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
