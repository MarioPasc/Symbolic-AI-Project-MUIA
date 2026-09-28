# Changelog

All notable changes to the course database are documented here, one entry per data version
(semantic versioning: MINOR for additive changes, MAJOR only by explicit team decision).

## 1.0.0 — 2026-09-28

Initial database, designed for EJ1 (SAT prescribing agent) and shared by EJ2/EJ3 and Práctica 2.

- **Formulary**: 6 conditions, 18 drugs, 6 risk factors, 18 candidate pairs (A1), 7 adverse
  interactions (A3, all `major`), 11 contraindications (A4), 1 coprescription (A5), 5 exclusive
  drug classes with 11 members (A6) — 50 rule clauses in total for EJ1.
- **Patients**: 3 patients (P001–P003), 5 encounters (E001–E005) covering the EJ1 scenarios V1–V3
  of `EJ1-sat/README.md` §10.
- `source` columns of the formulary tables are left empty; they are filled with the CIMA-AEMPS /
  STOPP-START reference of each entry before the report cites the formulary.
