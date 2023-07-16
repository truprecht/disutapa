from sdcp.reranking.dop import *


trees = [
    Tree("(S (S (A 0) (B 1)) (S (A 2) (C 3) (S (D 4))))"),
    Tree("(S (A 0) (C 1) (S (D 2)))"),
]
derivations = [
    Derivation(("S", ("S", "S"), lcfrs_composition("01")), 1, (
            Derivation(("S", ("A", "B"), lcfrs_composition("01")), 2, (None, None)),
            Derivation(("S", ("A", "C", "S"), lcfrs_composition("012")), 3, (
                None,
                None,
                Derivation(("S", ("D",), lcfrs_composition("0")), 4, (None,))
            ))
        )
    ),
    Derivation(("S", ("A", "C", "S"), lcfrs_composition("012")), 5, (
        None,
        None,
        Derivation(("S", ("D",), lcfrs_composition("0")), 6, (None,))
    ))
]



def test_into_derivation():
    for t, d in zip(trees, derivations):
        assert into_derivation(t) == (d, SortedSet(t.leaves()))

def test_common_fragments():
    excludes = set()
    
    assert largest_common_fragment(derivations[0], derivations[1], excludes) is None
    assert not excludes
    
    assert largest_common_fragment(derivations[1], derivations[1], excludes) == ImmutableTree(("S", ("A", "C", "S"), lcfrs_composition("012")), [
            None,
            None,
            ImmutableTree(("S", ("D",), lcfrs_composition("0")), [
                None,
            ])
        ])
    assert excludes == {(5,5), (6,6)}


def test_fragment_to_transitions():
    rules = [
        ("S", ("D",), lcfrs_composition("0")),
        ("S", ("A", "C", "S"), lcfrs_composition("012"))
    ]
    fragments = [
        ImmutableTree(0, [None]),
        ImmutableTree(1, [None, None, ImmutableTree(0, [None])])
    ]
    assert set(fragment_to_transitions(fragments[0], 12, rules, 2.0)) == {(("D",), 0, "S", 2.0)}
    assert set(fragment_to_transitions(fragments[1], 12, rules, 2.0)) == {
        (("D",), 0, "12-(2,)", 0.0),
        (("A", "C", "12-(2,)",), 1, "S", 2.0)
    }


def test_dop():
    grammar = Dop(trees, prior=0)
    r1 = 2
    r2 = 1
    assert set(grammar.transitions.keys()) == { r1, r2 }
    assert set(grammar.transitions[r1]) == { 
        (("A", "C", "0-(2,)",), "S", 0.0)
    }
    assert set(grammar.transitions[r2]) == { 
        (("D",), "0-(2,)", 0.0),
    }

    grammar = Dop(trees, prior=1)
    assert set(grammar.transitions.keys()) == { r1, r2 }
    assert set(grammar.transitions[r1]) == { 
        (("A", "C", "0-(2,)",), "S", log(4) - log(3))
    }
    assert set(grammar.transitions[r2]) == { 
        (("D",), "0-(2,)", 0.0),
    }
    assert grammar.fallback_weight == {
        "S": log(4)
    }

    assert grammar.match(trees[0]) == log(4) - log(3) + log(4) + log(4)


def test_parsing():
    parser = DopTreeParser(Dop(trees, prior=0))
    parser.fill_chart(trees[1])

    assert set(parser.chart.keys()) == { (2, '0-(2,)'), (1, 'S'), (2, 'S') }
    assert parser.chart[(1, 'S')] == 0
    assert parser.chart[(2, '0-(2,)')] == 0
    assert parser.chart[(2, 'S')] == float("inf")

    parser = DopTreeParser(Dop(trees, prior=1))
    parser.fill_chart(trees[0])

    assert set(parser.chart.keys()) == {(1, "S"), (2, "S"), (3, "S"), (4, "S"), (4, "0-(2,)")}
    assert parser.chart[(1, "S")] == parser.chart[(2, "S")] + parser.chart[(3, "S")] + log(4)
    assert parser.chart[(2, "S")] == log(4)
    assert parser.chart[(3, "S")] == parser.chart[(4, "0-(2,)")] + log(4) - log(3)
    assert parser.chart[(4, "S")] == log(4)
    assert parser.chart[(4, "0-(2,)")] == 0.0

