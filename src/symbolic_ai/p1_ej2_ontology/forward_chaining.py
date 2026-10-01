"""Forward chaining to a fixed point over function-free definite clauses (AIMA Fig. 9.3, §9.3.3).

Clauses, unification and substitution are those of the vendored aima-python (``aima.logic``:
``Expr``, ``is_definite_clause``, ``parse_definite_clause``, ``unify``, ``subst``, ``variables``).
The loop is ours: FOL-FC-ASK of AIMA Fig. 9.3 run until no new fact appears, with two of the
improvements of AIMA §9.3.2-9.3.3. Facts are indexed by predicate and by constant argument, so a
premise is matched only against the facts that can unify with it; and, in incremental mode, a rule
fires at iteration *t* only for substitutions in which at least one premise matches a fact first
derived at iteration *t* - 1. Both modes add the same facts at every iteration (AIMA §9.3.3).

``Distinct(a, b)`` is a built-in premise evaluated by procedural attachment (AIMA §10.5.1): it holds
when its two arguments, bound by the other premises, are different constants (unique names).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from aima.logic import (
    is_definite_clause,
    is_symbol,
    is_variable,
    parse_definite_clause,
    subst,
    unify,
    variables,
)
from aima.utils import Expr
from symbolic_ai.p1_ej2_ontology.errors import NotDefiniteClauseError

__all__ = [
    "DISTINCT",
    "Closure",
    "KnowledgeBase",
    "ask",
    "fc_closure",
    "fc_extend",
]

logger = logging.getLogger(__name__)

#: Name of the built-in inequality premise, evaluated by procedural attachment.
DISTINCT = "Distinct"

Substitution: TypeAlias = dict[Expr, Expr]
_PredicateKey: TypeAlias = tuple[str, int]
_ArgumentKey: TypeAlias = tuple[str, int, Expr]
_EMPTY: frozenset[Expr] = frozenset()


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    """A Datalog knowledge base: ground atomic facts and definite-clause rules (aima ``Expr``).

    Parameters
    ----------
    facts : tuple[Expr, ...]
        Ground atoms (no variables, no function symbols).
    rules : tuple[Expr, ...]
        Sentences ``p1 & ... & pn ==> q``; variables are lower-case symbols (aima convention).
    """

    facts: tuple[Expr, ...]
    rules: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class _Rule:
    """A rule split into ordinary premises, built-in tests and its conclusion."""

    clause: Expr
    premises: tuple[Expr, ...]
    tests: tuple[Expr, ...]
    conclusion: Expr


class _FactIndex:
    """Ground facts indexed by predicate and by (predicate, argument position, constant)."""

    __slots__ = ("_by_argument", "_by_predicate", "facts")

    def __init__(self, facts: Iterable[Expr] = ()) -> None:
        self.facts: set[Expr] = set()
        self._by_predicate: defaultdict[_PredicateKey, set[Expr]] = defaultdict(set)
        self._by_argument: defaultdict[_ArgumentKey, set[Expr]] = defaultdict(set)
        for fact in facts:
            self.add(fact)

    def __contains__(self, fact: Expr) -> bool:
        return fact in self.facts

    def add(self, fact: Expr) -> None:
        """Add one ground fact to every index it belongs to."""
        self.facts.add(fact)
        self._by_predicate[(fact.op, len(fact.args))].add(fact)
        for position, argument in enumerate(fact.args):
            self._by_argument[(fact.op, position, argument)].add(fact)

    def candidates(self, pattern: Expr) -> Collection[Expr]:
        """Return the smallest indexed set that contains every fact unifiable with ``pattern``."""
        best: Collection[Expr] = self._by_predicate.get((pattern.op, len(pattern.args)), _EMPTY)
        for position, argument in enumerate(pattern.args):
            if is_variable(argument):
                continue
            bucket = self._by_argument.get((pattern.op, position, argument), _EMPTY)
            if len(bucket) < len(best):
                best = bucket
        return best

    def copy(self) -> _FactIndex:
        """Return an independent copy (the sets are copied, the immutable facts are shared)."""
        duplicate = _FactIndex()
        duplicate.facts = set(self.facts)
        for key, facts in self._by_predicate.items():
            duplicate._by_predicate[key] = set(facts)
        for argument_key, facts in self._by_argument.items():
            duplicate._by_argument[argument_key] = set(facts)
        return duplicate


@dataclass(frozen=True, slots=True)
class Closure:
    """The fixed point Cl(KB) of forward chaining, with the facts added at each iteration.

    Parameters
    ----------
    told : frozenset[Expr]
        The facts given to forward chaining (the KB's facts, plus any facts added by
        :func:`fc_extend`).
    facts : frozenset[Expr]
        Every fact of the fixed point, told ones included.
    new_by_iteration : tuple[frozenset[Expr], ...]
        The facts first derived at iterations 1, 2, ...; the pass that finds no new fact (the
        fixed-point test) is not counted.
    """

    told: frozenset[Expr]
    facts: frozenset[Expr]
    new_by_iteration: tuple[frozenset[Expr], ...]
    _index: _FactIndex = field(repr=False, compare=False)

    @property
    def iterations(self) -> int:
        """Number of iterations that added at least one fact."""
        return len(self.new_by_iteration)

    @property
    def new_per_iteration(self) -> tuple[int, ...]:
        """Number of facts first derived at each iteration."""
        return tuple(len(new) for new in self.new_by_iteration)

    @property
    def derived(self) -> frozenset[Expr]:
        """The facts of the fixed point that were not told."""
        return self.facts - self.told

    def with_predicate(self, predicate: str) -> tuple[Expr, ...]:
        """Return the facts of one predicate, sorted by their text."""
        return tuple(sorted((f for f in self.facts if f.op == predicate), key=str))


# --- validation of the input --------------------------------------------------------------------


def _check_atom(atom: Expr, sentence: Expr) -> None:
    """Reject an atom whose arguments are not constants or variables (function symbols)."""
    if not isinstance(atom, Expr) or not is_symbol(atom.op):
        raise NotDefiniteClauseError(f"{sentence}: {atom} is not an atom")
    for argument in atom.args:
        if not isinstance(argument, Expr) or argument.args:
            raise NotDefiniteClauseError(
                f"{sentence}: argument {argument} of {atom} is not a constant or a variable "
                "(function symbols are not allowed in Datalog)"
            )


def _check_fact(fact: Expr) -> Expr:
    """Return ``fact`` if it is a ground atom that is not a built-in."""
    if not isinstance(fact, Expr) or not is_definite_clause(fact) or fact.op == "==>":
        raise NotDefiniteClauseError(f"not a ground atomic fact: {fact}")
    _check_atom(fact, fact)
    if fact.op == DISTINCT:
        raise NotDefiniteClauseError(f"{fact}: the built-in {DISTINCT} cannot be told as a fact")
    if variables(fact):
        raise NotDefiniteClauseError(f"fact {fact} is not ground")
    return fact


def _variables_of(atoms: Iterable[Expr]) -> set[Expr]:
    found: set[Expr] = set()
    for atom in atoms:
        found |= variables(atom)
    return found


def _compile_rule(clause: Expr) -> _Rule:
    """Split a rule, rejecting anything that is not a range-restricted Datalog definite clause."""
    if not isinstance(clause, Expr) or not is_definite_clause(clause) or clause.op != "==>":
        raise NotDefiniteClauseError(f"not a definite-clause rule (p1 & ... & pn ==> q): {clause}")
    antecedents, conclusion = parse_definite_clause(clause)
    for atom in [*antecedents, conclusion]:
        _check_atom(atom, clause)
    if conclusion.op == DISTINCT:
        raise NotDefiniteClauseError(f"{clause}: the built-in {DISTINCT} cannot be a conclusion")
    premises = tuple(atom for atom in antecedents if atom.op != DISTINCT)
    tests = tuple(atom for atom in antecedents if atom.op == DISTINCT)
    if any(len(test.args) != 2 for test in tests):
        raise NotDefiniteClauseError(f"{clause}: {DISTINCT} takes exactly two arguments")
    unbound = (_variables_of([conclusion, *tests])) - _variables_of(premises)
    if unbound:
        names = ", ".join(sorted(str(v) for v in unbound))
        raise NotDefiniteClauseError(
            f"{clause}: variable(s) {names} occur in no ordinary premise (not range-restricted)"
        )
    return _Rule(clause=clause, premises=premises, tests=tests, conclusion=conclusion)


# --- matching -----------------------------------------------------------------------------------


def _join(
    premises: Sequence[Expr], index: _FactIndex, theta: Substitution
) -> Iterator[Substitution]:
    """Yield every extension of ``theta`` that matches all ``premises`` against indexed facts."""
    if not premises:
        yield theta
        return
    pattern = subst(theta, premises[0])
    for fact in index.candidates(pattern):
        extended = unify(pattern, fact, theta)
        if extended is not None:
            yield from _join(premises[1:], index, extended)


def _tests_hold(tests: Sequence[Expr], theta: Substitution) -> bool:
    """Evaluate the built-in ``Distinct`` premises under ``theta`` (procedural attachment)."""
    return all(subst(theta, test.args[0]) != subst(theta, test.args[1]) for test in tests)


def _rule_matches(
    rule: _Rule, index: _FactIndex, frontier: _FactIndex | None
) -> Iterator[Substitution]:
    """Yield the substitutions that satisfy ``rule``'s premises and tests.

    With a ``frontier`` (incremental mode), only substitutions in which at least one premise
    matches a frontier fact are produced: premise *i* is matched against the frontier and the
    others against every fact. A substitution with several frontier premises may be yielded more
    than once; the caller collects conclusions in a set.
    """
    if frontier is None:
        matches: Iterable[Substitution] = _join(rule.premises, index, {})
        yield from (theta for theta in matches if _tests_hold(rule.tests, theta))
        return
    for position, premise in enumerate(rule.premises):
        others = rule.premises[:position] + rule.premises[position + 1 :]
        for fact in frontier.candidates(premise):
            theta = unify(premise, fact, {})
            if theta is None:
                continue
            for extended in _join(others, index, theta):
                if _tests_hold(rule.tests, extended):
                    yield extended


def _saturate(
    rules: Sequence[_Rule], index: _FactIndex, frontier: frozenset[Expr], *, incremental: bool
) -> list[frozenset[Expr]]:
    """Run FOL-FC-ASK's loop until no rule adds a fact; ``index`` is extended in place.

    As in AIMA Fig. 9.3, the facts found during one iteration are added to the knowledge base only
    when the iteration ends, and a conclusion is new when it is not already a fact (for ground
    atoms, "does not unify with a sentence already in KB or new" is set membership).
    """
    history: list[frozenset[Expr]] = []
    while True:
        frontier_index = _FactIndex(frontier) if incremental else None
        new: set[Expr] = set()
        for rule in rules:
            for theta in _rule_matches(rule, index, frontier_index):
                conclusion = subst(theta, rule.conclusion)
                if conclusion not in index:
                    new.add(conclusion)
        if not new:
            return history
        for fact in new:
            index.add(fact)
        history.append(frozenset(new))
        logger.debug("iteration %d: %d new facts", len(history), len(new))
        frontier = frozenset(new)


# --- public API ---------------------------------------------------------------------------------


def fc_closure(kb: KnowledgeBase, *, incremental: bool = True) -> Closure:
    """Compute Cl(KB), the fixed point of forward chaining (AIMA Fig. 9.3 and §9.3.3).

    Parameters
    ----------
    kb : KnowledgeBase
        Ground facts and Datalog definite-clause rules.
    incremental : bool
        ``True`` (default): a rule fires only on substitutions that use at least one fact derived
        in the previous iteration (AIMA §9.3.3). ``False``: every rule is matched against every
        fact at every iteration, as in Fig. 9.3. Both add the same facts at every iteration.

    Returns
    -------
    Closure
        The fixed point, with the facts first derived at each iteration.

    Raises
    ------
    NotDefiniteClauseError
        If a fact is not a ground atom, or a rule is not a function-free, range-restricted
        definite clause.
    """
    rules = tuple(_compile_rule(rule) for rule in kb.rules)
    told = frozenset(_check_fact(fact) for fact in kb.facts)
    index = _FactIndex(told)
    history = _saturate(rules, index, told, incremental=incremental)
    return Closure(
        told=told, facts=frozenset(index.facts), new_by_iteration=tuple(history), _index=index
    )


def fc_extend(closure: Closure, rules: Sequence[Expr], facts: Iterable[Expr]) -> Closure:
    """Compute Cl(KB + F) from Cl(KB) by incremental forward chaining seeded with F.

    Definite-clause entailment is monotone, so Cl(KB + F) = Cl(Cl(KB) + F), and every fact that
    is not in Cl(KB) needs at least one premise outside Cl(KB). Seeding the incremental loop
    (AIMA §9.3.3) with the facts of F that are new therefore derives exactly the missing facts.

    Parameters
    ----------
    closure : Closure
        Cl(KB), computed by :func:`fc_closure` with the same ``rules``.
    rules : Sequence[Expr]
        The rules of KB.
    facts : Iterable[Expr]
        The ground facts F to add.

    Returns
    -------
    Closure
        Cl(KB + F); ``new_by_iteration`` counts only the iterations of this extension.

    Raises
    ------
    NotDefiniteClauseError
        If a fact or a rule is malformed (see :func:`fc_closure`).
    """
    compiled = tuple(_compile_rule(rule) for rule in rules)
    delta = frozenset(_check_fact(fact) for fact in facts) - closure.facts
    index = closure._index.copy()
    for fact in delta:
        index.add(fact)
    history = _saturate(compiled, index, delta, incremental=True)
    return Closure(
        told=closure.told | delta,
        facts=frozenset(index.facts),
        new_by_iteration=tuple(history),
        _index=index,
    )


def ask(closure: Closure, query: Expr) -> tuple[Substitution, ...]:
    """Return every substitution under which the atom ``query`` is a fact of ``closure``.

    Parameters
    ----------
    closure : Closure
        A fixed point computed by :func:`fc_closure` or :func:`fc_extend`.
    query : Expr
        An atom, possibly with variables (the ``alpha`` of AIMA Fig. 9.3).

    Returns
    -------
    tuple[Substitution, ...]
        One substitution per matching fact, sorted by the text of the matched fact.
    """
    matched = sorted(closure._index.candidates(query), key=str)
    answers = (unify(query, fact, {}) for fact in matched)
    return tuple(answer for answer in answers if answer is not None)
