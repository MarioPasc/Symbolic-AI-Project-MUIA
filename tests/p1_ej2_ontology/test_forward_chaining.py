"""Forward chaining: AIMA's known answers, naive = incremental, agreement with ``fol_fc_ask``."""

from __future__ import annotations

import itertools

import pytest

from aima.logic import FolKB, constant_symbols, crime_kb, expr, fol_fc_ask, test_kb
from aima.utils import Expr
from symbolic_ai.p1_ej2_ontology.errors import NotDefiniteClauseError
from symbolic_ai.p1_ej2_ontology.forward_chaining import (
    DISTINCT,
    Closure,
    KnowledgeBase,
    ask,
    fc_closure,
    fc_extend,
)
from symbolic_ai.p1_ej2_ontology.ontology import (
    MEMBER,
    Ontology,
    atom,
    category_symbol,
    drug_symbol,
    prototype_symbol,
    to_knowledge_base,
)


def _split(clauses: list[Expr]) -> KnowledgeBase:
    """Separate an aima ``FolKB``'s clauses into facts and rules."""
    facts = tuple(c for c in clauses if c.op != "==>")
    rules = tuple(c for c in clauses if c.op == "==>")
    return KnowledgeBase(facts=facts, rules=rules)


def _texts(facts: frozenset[Expr]) -> list[str]:
    return sorted(str(f) for f in facts)


#: The slides' inheritance example (02, "Herencia y taxonomías"): every food is edible, fruit is a
#: subclass of food, apples a subclass of fruit, so an individual apple is edible.
SLIDES_INHERITANCE = KnowledgeBase(
    facts=tuple(
        map(
            expr,
            [
                "Subset(Frutas, Alimentos)",
                "Subset(Manzanas, Frutas)",
                "Member(Manzana1, Manzanas)",
                "ComestibleCat(Alimentos)",
            ],
        )
    ),
    rules=tuple(
        map(
            expr,
            [
                "(Member(x, c) & Subset(c, d)) ==> Member(x, d)",
                "(Subset(c, d) & Subset(d, e)) ==> Subset(c, e)",
                "(Member(x, c) & ComestibleCat(c)) ==> Comestible(x)",
            ],
        )
    ),
)

SELF_PAIRS = KnowledgeBase(
    facts=tuple(map(expr, ["Member(X1, A)", "Member(X2, A)", "IntCat(A, A)"])),
    rules=(expr("(Member(a, c) & Member(b, d) & IntCat(c, d) & Distinct(a, b)) ==> Int(a, b)"),),
)


# --- known answers --------------------------------------------------------------------------


def test_crime_example_derives_criminal_west_in_two_iterations() -> None:
    closure = fc_closure(_split(list(crime_kb.clauses)))
    assert closure.iterations == 2
    assert [_texts(new) for new in closure.new_by_iteration] == [
        ["Hostile(Nono)", "Sells(West, M1, Nono)", "Weapon(M1)"],
        ["Criminal(West)"],
    ]
    assert ask(closure, expr("Criminal(x)")) == ({expr("x"): expr("West")},)


def test_slides_inheritance_example_makes_an_apple_edible() -> None:
    closure = fc_closure(SLIDES_INHERITANCE)
    assert expr("Comestible(Manzana1)") in closure.facts
    assert expr("Member(Manzana1, Alimentos)") in closure.facts
    assert expr("Subset(Manzanas, Alimentos)") in closure.facts


def test_distinct_blocks_self_pairs() -> None:
    closure = fc_closure(SELF_PAIRS)
    assert _texts(frozenset(closure.with_predicate("Int"))) == ["Int(X1, X2)", "Int(X2, X1)"]


def test_empty_knowledge_base_has_an_empty_closure() -> None:
    closure = fc_closure(KnowledgeBase(facts=(), rules=SLIDES_INHERITANCE.rules))
    assert closure.facts == frozenset()
    assert closure.iterations == 0


def test_rules_without_matching_facts_add_nothing() -> None:
    kb = KnowledgeBase(facts=(expr("Other(A)"),), rules=SLIDES_INHERITANCE.rules)
    closure = fc_closure(kb)
    assert closure.facts == frozenset(kb.facts)
    assert closure.derived == frozenset()


# --- naive (Fig. 9.3) and incremental (§9.3.3) agree iteration by iteration -----------------


def _small_kbs() -> list[KnowledgeBase]:
    return [
        _split(list(crime_kb.clauses)),
        _split(list(test_kb.clauses)),
        SLIDES_INHERITANCE,
        SELF_PAIRS,
    ]


@pytest.mark.parametrize("kb", _small_kbs(), ids=["crime", "aima_test_kb", "slides", "self_pairs"])
def test_naive_and_incremental_add_the_same_facts_at_every_iteration(kb: KnowledgeBase) -> None:
    naive = fc_closure(kb, incremental=False)
    incremental = fc_closure(kb, incremental=True)
    assert naive.new_by_iteration == incremental.new_by_iteration


def test_naive_and_incremental_agree_on_the_toy_ontology(toy_ontology: Ontology) -> None:
    kb = to_knowledge_base(toy_ontology)
    assert fc_closure(kb, incremental=False).new_by_iteration == fc_closure(kb).new_by_iteration


@pytest.mark.integration
def test_naive_and_incremental_agree_on_the_real_ontology(real_kb: KnowledgeBase) -> None:
    naive = fc_closure(real_kb, incremental=False)
    incremental = fc_closure(real_kb, incremental=True)
    assert naive.new_by_iteration == incremental.new_by_iteration


# --- agreement with aima-python's fol_fc_ask ------------------------------------------------


def _fol_fc_ask_closure(kb: KnowledgeBase) -> frozenset[Expr]:
    """Run aima's ``fol_fc_ask`` to its fixed point and return its facts.

    ``fol_fc_ask`` has no built-ins, so ``Distinct`` is told as a fact for every ordered pair of
    different constants (unique names); those facts are removed from the answer.
    """
    clauses = [*kb.facts, *kb.rules]
    constants = sorted({c for clause in clauses for c in constant_symbols(clause)}, key=str)
    distinct = [atom(DISTINCT, a, b) for a, b in itertools.permutations(constants, 2)]
    folkb = FolKB([*clauses, *distinct])
    # A query that never matches makes fol_fc_ask run until no new fact appears.
    assert list(fol_fc_ask(folkb, expr("NeverDerived(q)"))) == []
    return frozenset(c for c in folkb.clauses if c.op not in ("==>", DISTINCT))


@pytest.mark.parametrize("kb", _small_kbs(), ids=["crime", "aima_test_kb", "slides", "self_pairs"])
def test_closure_equals_fol_fc_ask_on_small_kbs(kb: KnowledgeBase) -> None:
    assert fc_closure(kb).facts == _fol_fc_ask_closure(kb)


@pytest.mark.slow
def test_closure_equals_fol_fc_ask_on_the_toy_ontology(toy_ontology: Ontology) -> None:
    kb = to_knowledge_base(toy_ontology)
    assert fc_closure(kb).facts == _fol_fc_ask_closure(kb)


# --- extension of a closure -----------------------------------------------------------------


def _with_fact(kb: KnowledgeBase, fact: Expr) -> KnowledgeBase:
    return KnowledgeBase(facts=(*kb.facts, fact), rules=kb.rules)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fact",
    [
        atom(MEMBER, prototype_symbol("nsaids"), category_symbol("nsaids")),
        atom(MEMBER, drug_symbol("ibuprofen"), category_symbol("opioids")),
        atom(MEMBER, drug_symbol("warfarin"), category_symbol("ssris")),
    ],
    ids=["prototype", "clash", "new_interactions"],
)
def test_extension_equals_the_closure_of_the_extended_kb(
    real_kb: KnowledgeBase, real_closure: Closure, fact: Expr
) -> None:
    extended = fc_extend(real_closure, real_kb.rules, [fact])
    assert extended.facts == fc_closure(_with_fact(real_kb, fact)).facts
    assert extended.told == real_closure.told | {fact}


def test_extending_with_a_known_fact_changes_nothing() -> None:
    closure = fc_closure(SLIDES_INHERITANCE)
    extended = fc_extend(closure, SLIDES_INHERITANCE.rules, [expr("Member(Manzana1, Frutas)")])
    assert extended.facts == closure.facts
    assert extended.iterations == 0


# --- failure paths: only function-free, range-restricted definite clauses -------------------


@pytest.mark.parametrize(
    "rule",
    [
        "P(x) | Q(x)",
        "(~P(x)) ==> Q(x)",
        "P(x) ==> (Q(x) | R(x))",
        "P(F(x)) ==> Q(x)",
        "P(x) ==> Q(y)",
        "P(x) ==> Distinct(x, x)",
        "(P(x) & Distinct(x, y)) ==> Q(x)",
        "(P(x) & Distinct(x)) ==> Q(x)",
        "P(x)",
    ],
)
def test_non_definite_or_non_datalog_rule_raises(rule: str) -> None:
    with pytest.raises(NotDefiniteClauseError):
        fc_closure(KnowledgeBase(facts=(expr("P(A)"),), rules=(expr(rule),)))


@pytest.mark.parametrize(
    "fact", ["P(x)", "P(F(A))", "Distinct(A, B)", "P(A) ==> Q(A)", "P(A) | Q(A)"]
)
def test_malformed_fact_raises(fact: str) -> None:
    with pytest.raises(NotDefiniteClauseError):
        fc_closure(KnowledgeBase(facts=(expr(fact),), rules=()))


def test_malformed_fact_in_an_extension_raises() -> None:
    closure = fc_closure(SLIDES_INHERITANCE)
    with pytest.raises(NotDefiniteClauseError):
        fc_extend(closure, SLIDES_INHERITANCE.rules, [expr("Member(x, Frutas)")])
