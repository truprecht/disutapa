from sdcp.grammar.dop import *


trees = [
    Tree("(S (S (A 0) (B 1)) (S (A 2) (C 3) (S (D 4))))"),
    Tree("(S (A 0) (C 1) (S (D 2)))"),
]
derivations = [
    Derivation(3, 0, (
            Derivation(0, 1, (None, None)),
            Derivation(2, 2, (
                None,
                None,
                Derivation(1, 3, (None,))
            ))
        )
    ),
    Derivation(2, 4, (
        None,
        None,
        Derivation(1, 5, (None,))
    ))
]



def test_into_derivation():
    into_derivation = DerivationFactory(None)
    for t, d in zip(trees, derivations):
        assert into_derivation._produce_with_positions(t) == (d, SortedSet(t.leaves()))

def test_common_fragments():
    excludes = set()
    
    assert largest_common_fragment(derivations[0], derivations[1], excludes) is None
    assert not excludes
    
    assert largest_common_fragment(derivations[1], derivations[1], excludes) == Fragment(2, (
            None,
            None,
            Fragment(1, (None,))
        ))
    assert excludes == {(4,4), (5,5)}


def test_fragment_to_transitions():
    rules = [
        ("S", ("D",), lcfrs_composition("0")),
        ("S", ("A", "C", "S"), lcfrs_composition("012"))
    ]
    fragments = [
        Fragment(0, [None]),
        Fragment(1, [None, None, Fragment(0, [None])])
    ]
    assert set(fragment_to_transitions(fragments[0], rules, 2.0)) == {(("D",), 0, "S", 2.0)}
    assert set(fragment_to_transitions(fragments[1], rules, 2.0)) == {
        (("D",), 0, "F (0 *)", 0.0),
        (("A", "C", "F (0 *)",), 1, "S", 2.0)
    }


def test_dop():
    grammar = Dop(trees, prior=0)
    assert set(grammar.transitions.keys()) == set(range(4))
    print(grammar.transitions)
    assert set(grammar.transitions[0]) == {
        (("A", "B",), "S", log(8))
    }
    assert set(grammar.transitions[1]) == {
        (("D",), "S", log(8) - log(2)),
        (("D",), "F (1 *)", 0),
    }
    assert set(grammar.transitions[2]) == {
        (("A", "C", "S",), "S", log(8) - log(2)),
        (("A", "C", "F (1 *)",), "S", log(8) - log(2))
    }
    assert set(grammar.transitions[3]) == {
        (("S", "S",), "S", log(8))
    }

    assert grammar.match(trees[0]) == log(8) - log(2) + log(8) + log(8)

    grammar = Dop(trees, prior=1)
    assert set(grammar.transitions.keys()) == set(range(4))
    assert set(grammar.transitions[0]) == {
        (("A", "B",), "S", log(14) - log(2))
    }
    assert set(grammar.transitions[1]) == {
        (("D",), "S", log(14) - log(3)),
        (("D",), "F (1 *)", 0),
    }
    assert set(grammar.transitions[2]) == {
        (("A", "C", "S",), "S", log(14) - log(3)),
        (("A", "C", "F (1 *)",), "S", log(14) - log(3))
    }
    assert set(grammar.transitions[3]) == {
        (("S", "S",), "S", log(14) - log(2))
    }
    assert grammar.fallback_weight == {
        "S": log(14)
    }

    assert grammar.match(trees[0]) == log(14) - log(3) + log(14) - log(2) + log(14) - log(2)


def test_parsing():
    grammar = Dop(trees, prior=0)
    parser = DopTreeParser(grammar, "min")
    parser.fill_chart(trees[1], grammar.derivation_factory_state)

    assert set(parser.chart.keys()) == { (1, 'F (1 *)'), (1, 'S'), (0, 'S') }
    assert parser.chart[(0, 'S')] == log(8) - log(2)
    assert parser.chart[(1, 'F (1 *)')] == 0.0
    assert parser.chart[(1, 'S')] == log(8) - log(2)

    grammar = Dop(trees, prior=1)
    parser = DopTreeParser(grammar, "min")
    parser.fill_chart(trees[0], grammar.derivation_factory_state)

    assert set(parser.chart.keys()) == {(0, "S"), (1, "S"), (2, "S"), (3, "S"), (3, "F (1 *)")}
    assert parser.chart[(0, "S")] == parser.chart[(1, "S")] + parser.chart[(2, "S")] + log(14) - log(2)
    assert parser.chart[(1, "S")] == log(14) - log(2)
    assert parser.chart[(2, "S")] == parser.chart[(3, "F (1 *)")] + log(14) - log(3)
    assert parser.chart[(3, "S")] == log(14) - log(3)
    assert parser.chart[(3, "F (1 *)")] == 0.0

