"""Plain (non-pytest) builders of the EJ1 test data, transcribed from the specifications.

Kept free of any pytest dependency so the same data can be built both from ``conftest.py``
fixtures and from a standalone script run in a subprocess (the determinism test,
``test_ej1_determinism.py``), without importing the test package itself.

Sources: the full 1.0.0 formulary and encounters E001-E005 are transcribed from
``docs/SPECIFICATIONS/01-database.md`` §6; Elena's and Pablo's mini world is transcribed from
``docs/SPECIFICATIONS/EJ1-sat/README.md`` §5.
"""

from __future__ import annotations

from datetime import date

from symbolic_ai.dataloader.models import (
    AdverseInteraction,
    Condition,
    Contraindication,
    Coprescription,
    Drug,
    DrugClass,
    Encounter,
    Formulary,
    RiskFactor,
    RiskStatus,
)

FORMULARY_VERSION = "1.0.0"


def full_formulary() -> Formulary:
    """Build the full 1.0.0 formulary: 6 conditions, 18 drugs, 6 factors (01-database.md §6)."""
    conditions = (
        Condition("AF", "Fibrilación auricular", "Atrial fibrillation", "I48"),
        Condition("DEP", "Depresión", "Depression", "F32"),
        Condition(
            "GERD",
            "Enfermedad por reflujo gastroesofágico",
            "Gastro-oesophageal reflux disease",
            "K21",
        ),
        Condition("HTN", "Hipertensión arterial", "Hypertension", "I10"),
        Condition("PAIN", "Dolor crónico", "Chronic pain", "R52.2"),
        Condition("T2D", "Diabetes mellitus tipo 2", "Type 2 diabetes mellitus", "E11"),
    )
    risk_factors = (
        RiskFactor("AGE65", "Edad de 65 años o más", "Age 65 or older", "age_years >= 65"),
        RiskFactor("ASTHMA", "Asma", "Asthma", "diagnosed asthma"),
        RiskFactor(
            "CKD",
            "Enfermedad renal crónica avanzada",
            "Advanced chronic kidney disease",
            "eGFR < 30 mL/min/1.73 m2 (stages G4-G5)",
        ),
        RiskFactor("EPI", "Epilepsia", "Epilepsy", "diagnosed epilepsy"),
        RiskFactor("LIVER", "Hepatopatía grave", "Severe liver disease", "Child-Pugh class C"),
        RiskFactor("PREG", "Embarazo", "Pregnancy", "confirmed pregnancy"),
    )
    drugs = (
        Drug("amlodipine", "Amlodipino", "Amlodipine", "C08CA01"),
        Drug("apixaban", "Apixabán", "Apixaban", "B01AF02"),
        Drug("bisoprolol", "Bisoprolol", "Bisoprolol", "C07AB07"),
        Drug("enalapril", "Enalapril", "Enalapril", "C09AA02"),
        Drug("gliclazide", "Gliclazida", "Gliclazide", "A10BB09"),
        Drug("hydrochlorothiazide", "Hidroclorotiazida", "Hydrochlorothiazide", "C03AA03"),
        Drug("ibuprofen", "Ibuprofeno", "Ibuprofen", "M01AE01"),
        Drug("linagliptin", "Linagliptina", "Linagliptin", "A10BH05"),
        Drug("losartan", "Losartán", "Losartan", "C09CA01"),
        Drug("metformin", "Metformina", "Metformin", "A10BA02"),
        Drug("mirtazapine", "Mirtazapina", "Mirtazapine", "N06AX11"),
        Drug("omeprazole", "Omeprazol", "Omeprazole", "A02BC01"),
        Drug("paracetamol", "Paracetamol", "Paracetamol", "N02BE01"),
        Drug("propranolol", "Propranolol", "Propranolol", "C07AA05"),
        Drug("sertraline", "Sertralina", "Sertraline", "N06AB06"),
        Drug("tramadol", "Tramadol", "Tramadol", "N02AX02"),
        Drug("verapamil", "Verapamilo", "Verapamil", "C08DA01"),
        Drug("warfarin", "Warfarina", "Warfarin", "B01AA03"),
    )
    candidates = {
        "HTN": frozenset(
            {
                "enalapril",
                "losartan",
                "amlodipine",
                "hydrochlorothiazide",
                "bisoprolol",
                "propranolol",
                "verapamil",
            }
        ),
        "AF": frozenset({"warfarin", "apixaban"}),
        "T2D": frozenset({"metformin", "gliclazide", "linagliptin"}),
        "DEP": frozenset({"sertraline", "mirtazapine"}),
        "PAIN": frozenset({"paracetamol", "ibuprofen", "tramadol"}),
        "GERD": frozenset({"omeprazole"}),
    }
    interactions = (
        AdverseInteraction("apixaban", "ibuprofen", "bleeding", "major"),
        AdverseInteraction("bisoprolol", "verapamil", "bradycardia, AV block", "major"),
        AdverseInteraction("ibuprofen", "sertraline", "gastrointestinal bleeding", "major"),
        AdverseInteraction("ibuprofen", "warfarin", "bleeding", "major"),
        AdverseInteraction("propranolol", "verapamil", "bradycardia, AV block", "major"),
        AdverseInteraction("sertraline", "tramadol", "serotonin syndrome", "major"),
        AdverseInteraction("sertraline", "warfarin", "bleeding", "major"),
    )
    contraindications = (
        Contraindication("ASTHMA", "propranolol", "bronchospasm, non-selective beta blocker"),
        Contraindication("CKD", "gliclazide", "hypoglycaemia"),
        Contraindication("CKD", "hydrochlorothiazide", "ineffective at low eGFR"),
        Contraindication("CKD", "ibuprofen", "nephrotoxicity"),
        Contraindication("CKD", "metformin", "lactic acidosis"),
        Contraindication("EPI", "tramadol", "lowers the seizure threshold"),
        Contraindication("LIVER", "paracetamol", "hepatotoxicity"),
        Contraindication("PREG", "apixaban", "no safety data in pregnancy"),
        Contraindication("PREG", "enalapril", "fetotoxicity"),
        Contraindication("PREG", "losartan", "fetotoxicity"),
        Contraindication("PREG", "warfarin", "teratogenicity"),
    )
    coprescriptions = (Coprescription("ibuprofen", "AGE65", "omeprazole", "gastroprotection"),)
    drug_classes = (
        DrugClass("analgesics", "Analgésicos", "Analgesics", exclusive=True),
        DrugClass(
            "anticoagulants", "Anticoagulantes", "Anticoagulants", atc_code="B01A", exclusive=True
        ),
        DrugClass(
            "antidepressants", "Antidepresivos", "Antidepressants", atc_code="N06A", exclusive=True
        ),
        DrugClass(
            "beta_blockers", "Betabloqueantes", "Beta blockers", atc_code="C07", exclusive=True
        ),
        DrugClass(
            "raas_blockers",
            "Bloqueantes del SRAA (IECA/ARA-II)",
            "RAAS blockers",
            atc_code="C09",
            exclusive=True,
        ),
    )
    class_members = {
        "raas_blockers": frozenset({"enalapril", "losartan"}),
        "beta_blockers": frozenset({"bisoprolol", "propranolol"}),
        "anticoagulants": frozenset({"warfarin", "apixaban"}),
        "antidepressants": frozenset({"sertraline", "mirtazapine"}),
        "analgesics": frozenset({"paracetamol", "ibuprofen", "tramadol"}),
    }
    return Formulary(
        version=FORMULARY_VERSION,
        conditions=conditions,
        drugs=drugs,
        risk_factors=risk_factors,
        candidates=candidates,
        interactions=interactions,
        contraindications=contraindications,
        coprescriptions=coprescriptions,
        drug_classes=drug_classes,
        class_members=class_members,
    )


def _all_absent(present: set[str], unknown: set[str]) -> dict[str, RiskStatus]:
    """Build a complete 6-risk-factor status map: ``present``/``unknown`` override, rest absent."""
    all_ids = ("AGE65", "ASTHMA", "CKD", "EPI", "LIVER", "PREG")
    statuses: dict[str, RiskStatus] = {}
    for risk_factor_id in all_ids:
        if risk_factor_id in present:
            statuses[risk_factor_id] = RiskStatus.PRESENT
        elif risk_factor_id in unknown:
            statuses[risk_factor_id] = RiskStatus.UNKNOWN
        else:
            statuses[risk_factor_id] = RiskStatus.ABSENT
    return statuses


def encounters() -> dict[str, Encounter]:
    """Build encounters E001-E005 of ``01-database.md`` §6."""
    return {
        "E001": Encounter(
            encounter_id="E001",
            patient_id="P001",
            seq=1,
            visit_date=date(2026, 9, 1),
            age_years=78,
            conditions=frozenset({"HTN", "AF", "T2D", "DEP", "PAIN"}),
            risk_factors=_all_absent(present={"AGE65"}, unknown={"CKD"}),
        ),
        "E002": Encounter(
            encounter_id="E002",
            patient_id="P002",
            seq=1,
            visit_date=date(2026, 9, 2),
            age_years=34,
            conditions=frozenset({"AF", "HTN"}),
            risk_factors=_all_absent(present=set(), unknown={"PREG"}),
        ),
        "E003": Encounter(
            encounter_id="E003",
            patient_id="P002",
            seq=2,
            visit_date=date(2026, 9, 9),
            age_years=34,
            conditions=frozenset({"AF", "HTN"}),
            risk_factors=_all_absent(present=set(), unknown=set()),
        ),
        "E004": Encounter(
            encounter_id="E004",
            patient_id="P003",
            seq=1,
            visit_date=date(2026, 9, 3),
            age_years=70,
            conditions=frozenset({"PAIN", "DEP"}),
            risk_factors=_all_absent(present={"AGE65", "LIVER", "EPI"}, unknown=set()),
        ),
        "E005": Encounter(
            encounter_id="E005",
            patient_id="P003",
            seq=2,
            visit_date=date(2026, 9, 17),
            age_years=70,
            conditions=frozenset({"PAIN", "DEP"}),
            risk_factors=_all_absent(present={"AGE65", "LIVER", "EPI", "CKD"}, unknown=set()),
        ),
    }


def elena_formulary() -> Formulary:
    """Build the 5-drug mini formulary of ``EJ1-sat/README.md`` §5.1: Elena's and Pablo's world."""
    conditions = (
        Condition("AF", "Fibrilación auricular", "Atrial fibrillation"),
        Condition(
            "GERD", "Enfermedad por reflujo gastroesofágico", "Gastro-oesophageal reflux disease"
        ),
        Condition("PAIN", "Dolor crónico", "Chronic pain"),
    )
    risk_factors = (
        RiskFactor("AGE65", "Edad de 65 años o más", "Age 65 or older"),
        RiskFactor("LIVER", "Hepatopatía grave", "Severe liver disease"),
    )
    drugs = (
        Drug("apixaban", "Apixabán", "Apixaban"),
        Drug("ibuprofen", "Ibuprofeno", "Ibuprofen"),
        Drug("omeprazole", "Omeprazol", "Omeprazole"),
        Drug("paracetamol", "Paracetamol", "Paracetamol"),
        Drug("warfarin", "Warfarina", "Warfarin"),
    )
    candidates = {
        "AF": frozenset({"warfarin", "apixaban"}),
        "PAIN": frozenset({"paracetamol", "ibuprofen"}),
        "GERD": frozenset({"omeprazole"}),
    }
    interactions = (
        AdverseInteraction("apixaban", "ibuprofen", "bleeding", "major"),
        AdverseInteraction("ibuprofen", "warfarin", "bleeding", "major"),
    )
    contraindications = (Contraindication("LIVER", "paracetamol", "hepatotoxicity"),)
    coprescriptions = (Coprescription("ibuprofen", "AGE65", "omeprazole", "gastroprotection"),)
    drug_classes = (
        DrugClass("anticoagulants", "Anticoagulantes", "Anticoagulants", exclusive=True),
        DrugClass("analgesics", "Analgésicos", "Analgesics", exclusive=True),
    )
    class_members = {
        "anticoagulants": frozenset({"warfarin", "apixaban"}),
        "analgesics": frozenset({"paracetamol", "ibuprofen"}),
    }
    return Formulary(
        version=FORMULARY_VERSION,
        conditions=conditions,
        drugs=drugs,
        risk_factors=risk_factors,
        candidates=candidates,
        interactions=interactions,
        contraindications=contraindications,
        coprescriptions=coprescriptions,
        drug_classes=drug_classes,
        class_members=class_members,
    )


def elena_encounter() -> Encounter:
    """Elena: 72, atrial fibrillation and chronic pain, no reflux, liver fine (README §5.3)."""
    return Encounter(
        encounter_id="ELENA",
        patient_id="P-ELENA",
        seq=1,
        visit_date=date(2026, 1, 1),
        age_years=72,
        conditions=frozenset({"AF", "PAIN"}),
        risk_factors={"AGE65": RiskStatus.PRESENT, "LIVER": RiskStatus.ABSENT},
    )


def pablo_encounter() -> Encounter:
    """Pablo: 70, only chronic pain, no liver disease (README §5.6)."""
    return Encounter(
        encounter_id="PABLO",
        patient_id="P-PABLO",
        seq=1,
        visit_date=date(2026, 1, 1),
        age_years=70,
        conditions=frozenset({"PAIN"}),
        risk_factors={"AGE65": RiskStatus.PRESENT, "LIVER": RiskStatus.ABSENT},
    )
