"""Reference oracle for EJ1: axioms A1-A6 evaluated as set conditions on regimens, by enumeration.

The agent reasons over the CNF encoding of :mod:`symbolic_ai.p1_ej1_logic.encoding` with DPLL. This
module restates the six axioms as the report's prose defines them, directly on a regimen (a set of
drugs), and enumerates every regimen of the formulary. It shares only the formulary data with the
agent (no clause, no CNF, no solver), so an error in the encoding, in the solver wrapper or in the
decision rule shows up as a disagreement with it. It is used to check DPLL's verdicts and to measure
what DPLL cannot give (model counts and minimum regimen sizes).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from symbolic_ai.dataloader.models import Formulary
from symbolic_ai.p1_ej1_logic.errors import OracleError

__all__ = ["MAX_ORACLE_DRUGS", "FullRecord", "RegimenSpace"]

#: Largest formulary the oracle enumerates: 2**22 regimens is about 4 million rows.
MAX_ORACLE_DRUGS = 22

BoolMatrix = npt.NDArray[np.bool_]
BoolVector = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FullRecord:
    """A fully observed clinical record: the present conditions and the present risk factors.

    Every formulary condition or risk factor not listed is absent. The bounds of the agent map
    onto full records: Γ⁺ assumes every unknown risk factor present, Γ⁻ assumes them absent.

    Parameters
    ----------
    conditions : frozenset[str]
        Identifiers of the present conditions.
    present_risk_factors : frozenset[str]
        Identifiers of the present risk factors.
    """

    conditions: frozenset[str]
    present_risk_factors: frozenset[str]


@dataclass(frozen=True, slots=True)
class _DrugRules:
    """The formulary's axioms as column indices of the regimen matrix (precomputed once)."""

    candidates: dict[str, tuple[int, ...]]
    indications: tuple[frozenset[str], ...]
    requirers: tuple[tuple[int, ...], ...]
    adverse_pairs: tuple[tuple[int, int], ...]
    family_pairs: tuple[tuple[int, int], ...]
    contraindications: tuple[tuple[str, int], ...]
    coprescriptions: tuple[tuple[int, str, int], ...]


class RegimenSpace:
    """Every regimen of a formulary, with the axioms A1-A6 evaluated by enumeration.

    The patient-independent axioms A3 (adverse interactions) and A6 (one drug per family) are
    applied once at construction; A1, A2, A4 and A5 depend on the record and are applied per query.

    Parameters
    ----------
    formulary : Formulary
        The formulary whose regimens (subsets of its drugs) are enumerated.

    Raises
    ------
    OracleError
        If the formulary has more than :data:`MAX_ORACLE_DRUGS` drugs.
    """

    def __init__(self, formulary: Formulary) -> None:
        drug_ids = formulary.drug_ids
        if len(drug_ids) > MAX_ORACLE_DRUGS:
            raise OracleError(
                f"formulary {formulary.version} has {len(drug_ids)} drugs; the oracle enumerates "
                f"2**n regimens and accepts at most {MAX_ORACLE_DRUGS}"
            )
        self._formulary = formulary
        self._drug_ids = drug_ids
        self._rules = _drug_rules(formulary)
        all_regimens = _all_regimens(len(drug_ids))
        keep = _a3(all_regimens, self._rules) & _a6(all_regimens, self._rules)
        self._regimens: BoolMatrix = all_regimens[keep]
        self._sizes = self._regimens.sum(axis=1)

    @property
    def drug_ids(self) -> tuple[str, ...]:
        """The formulary's drug identifiers, in the column order of the regimen matrix."""
        return self._drug_ids

    @property
    def n_candidate_regimens(self) -> int:
        """Number of regimens that satisfy A3 and A6 (the only ones any record can accept)."""
        return int(self._regimens.shape[0])

    def count(self, record: FullRecord) -> int:
        """Return the number of regimens that satisfy A1-A6 for ``record`` (its model count).

        Parameters
        ----------
        record : FullRecord
            The fully observed record.

        Returns
        -------
        int
            Number of models of Γ for that record, counted over the decision symbols.

        Raises
        ------
        OracleError
            If the record names a condition or risk factor outside the formulary.
        """
        return int(self._accepted(record).sum())

    def minimum_size(self, record: FullRecord) -> int | None:
        """Return the size of the smallest regimen accepted for ``record``, or ``None`` if none is.

        Parameters
        ----------
        record : FullRecord
            The fully observed record.

        Returns
        -------
        int | None
            The minimum number of drugs over all models, or ``None`` when Γ is unsatisfiable.

        Raises
        ------
        OracleError
            If the record names a condition or risk factor outside the formulary.
        """
        accepted = self._accepted(record)
        if not accepted.any():
            return None
        return int(self._sizes[accepted].min())

    def violations(self, record: FullRecord, regimen: Iterable[str]) -> tuple[str, ...]:
        """Return the axioms (``"A1"`` ... ``"A6"``) that ``regimen`` violates for ``record``.

        Parameters
        ----------
        record : FullRecord
            The fully observed record.
        regimen : Iterable[str]
            Drug identifiers of the regimen to check.

        Returns
        -------
        tuple[str, ...]
            The violated axioms in order; empty when the regimen satisfies all six.

        Raises
        ------
        OracleError
            If the record or the regimen names an identifier outside the formulary.
        """
        self._check_record(record)
        drugs = frozenset(regimen)
        unknown = sorted(drugs - set(self._drug_ids))
        if unknown:
            raise OracleError(f"regimen names drugs outside the formulary: {unknown}")
        row = np.array([[drug_id in drugs for drug_id in self._drug_ids]], dtype=bool)
        checks = (
            ("A1", _a1(row, self._rules, record)),
            ("A2", _a2(row, self._rules, record)),
            ("A3", _a3(row, self._rules)),
            ("A4", _a4(row, self._rules, record)),
            ("A5", _a5(row, self._rules, record)),
            ("A6", _a6(row, self._rules)),
        )
        return tuple(axiom for axiom, satisfied in checks if not satisfied[0])

    def is_valid(self, record: FullRecord, regimen: Iterable[str]) -> bool:
        """Return whether ``regimen`` satisfies A1-A6 for ``record``."""
        return not self.violations(record, regimen)

    def _accepted(self, record: FullRecord) -> BoolVector:
        """Mask of the (A3, A6)-regimens that also satisfy A1, A2, A4 and A5 for ``record``."""
        self._check_record(record)
        regimens = self._regimens
        return (
            _a1(regimens, self._rules, record)
            & _a2(regimens, self._rules, record)
            & _a4(regimens, self._rules, record)
            & _a5(regimens, self._rules, record)
        )

    def _check_record(self, record: FullRecord) -> None:
        """Raise :class:`OracleError` if ``record`` names an identifier outside the formulary."""
        unknown_conditions = sorted(record.conditions - set(self._formulary.condition_ids))
        unknown_risks = sorted(record.present_risk_factors - set(self._formulary.risk_factor_ids))
        if unknown_conditions or unknown_risks:
            raise OracleError(
                f"record names identifiers outside formulary {self._formulary.version}: "
                f"conditions {unknown_conditions}, risk factors {unknown_risks}"
            )


def _all_regimens(n_drugs: int) -> BoolMatrix:
    """Return the ``(2**n, n)`` matrix whose row ``k`` is the binary expansion of ``k``."""
    masks = np.arange(1 << n_drugs, dtype=np.uint32)
    shifts = np.arange(n_drugs, dtype=np.uint32)
    return ((masks[:, None] >> shifts) & 1).astype(bool)


def _drug_rules(formulary: Formulary) -> _DrugRules:
    """Translate the formulary into column indices of the regimen matrix."""
    index = {drug_id: i for i, drug_id in enumerate(formulary.drug_ids)}
    requirers = tuple(
        tuple(
            sorted(
                index[cp.drug_id]
                for cp in formulary.coprescriptions
                if cp.companion_drug_id == drug_id
            )
        )
        for drug_id in formulary.drug_ids
    )
    family_pairs = tuple(
        (index[a], index[b])
        for family in formulary.exclusive_families()
        for a, b in combinations(sorted(family), 2)
    )
    return _DrugRules(
        candidates={
            c: tuple(sorted(index[d] for d in formulary.candidates.get(c, frozenset())))
            for c in formulary.condition_ids
        },
        indications=tuple(formulary.indications(d) for d in formulary.drug_ids),
        requirers=requirers,
        adverse_pairs=tuple((index[i.drug_a], index[i.drug_b]) for i in formulary.interactions),
        family_pairs=family_pairs,
        contraindications=tuple(
            (x.risk_factor_id, index[x.drug_id]) for x in formulary.contraindications
        ),
        coprescriptions=tuple(
            (index[p.drug_id], p.risk_factor_id, index[p.companion_drug_id])
            for p in formulary.coprescriptions
        ),
    )


def _a1(regimens: BoolMatrix, rules: _DrugRules, record: FullRecord) -> BoolVector:
    """A1 coverage: every present condition c has some candidate of D_c in the regimen."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for condition_id in sorted(record.conditions):
        ok &= regimens[:, list(rules.candidates[condition_id])].any(axis=1)
    return ok


def _a2(regimens: BoolMatrix, rules: _DrugRules, record: FullRecord) -> BoolVector:
    """A2 justification: each prescribed drug treats a present condition or is a companion."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for column, indications in enumerate(rules.indications):
        if indications & record.conditions:
            continue
        accompanies = regimens[:, list(rules.requirers[column])].any(axis=1)
        ok &= ~regimens[:, column] | accompanies
    return ok


def _a3(regimens: BoolMatrix, rules: _DrugRules) -> BoolVector:
    """A3 adverse interaction: no interacting pair is prescribed together."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for a, b in rules.adverse_pairs:
        ok &= ~(regimens[:, a] & regimens[:, b])
    return ok


def _a4(regimens: BoolMatrix, rules: _DrugRules, record: FullRecord) -> BoolVector:
    """A4 contraindication: no drug excluded by a present risk factor is prescribed."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for risk_factor_id, column in rules.contraindications:
        if risk_factor_id in record.present_risk_factors:
            ok &= ~regimens[:, column]
    return ok


def _a5(regimens: BoolMatrix, rules: _DrugRules, record: FullRecord) -> BoolVector:
    """A5 co-prescription: a drug prescribed under its risk factor comes with its companion."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for drug, risk_factor_id, companion in rules.coprescriptions:
        if risk_factor_id in record.present_risk_factors:
            ok &= ~regimens[:, drug] | regimens[:, companion]
    return ok


def _a6(regimens: BoolMatrix, rules: _DrugRules) -> BoolVector:
    """A6 duplication: at most one drug of each exclusive family."""
    ok = np.ones(regimens.shape[0], dtype=bool)
    for a, b in rules.family_pairs:
        ok &= ~(regimens[:, a] & regimens[:, b])
    return ok
