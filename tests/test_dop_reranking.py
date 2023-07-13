from sdcp.reranking.dop import *


trees = [
    Tree("(S (S (A 0) (B 1)) (S (A 2) (C 3) (S (D 4))))"),
    Tree("(S (A 0) (C 1) (S (D 2)))"),
]
derivations = [
    ImmutableTree(("S", ("S", "S"), lcfrs_composition("01")), [
            ImmutableTree(("S", ("A", "B"), lcfrs_composition("01")), [None, None]),
            ImmutableTree(("S", ("A", "C", "S"), lcfrs_composition("012")), [
                None,
                None,
                ImmutableTree(("S", ("D",), lcfrs_composition("0")), [None])
            ])
        ]
    ),
    ImmutableTree(("S", ("A", "C", "S"), lcfrs_composition("012")), [
        None,
        None,
        ImmutableTree(("S", ("D",), lcfrs_composition("0")), [None])
    ])
]



def test_into_derivation():
    for t, d in zip(trees, derivations):
        assert into_derivation(t) == (d, SortedSet(t.leaves()))

def test_common_fragments():
    excludes = set()
    
    assert largest_common_fragment(derivations[0], (), derivations[1], (1,), 1, excludes) is None
    assert not excludes
    
    assert largest_common_fragment(derivations[1], (3,4,5), derivations[1], (1,2,3,), 1, excludes) == ImmutableTree(("S", ("A", "C", "S"), lcfrs_composition("012")), [
            None,
            None,
            ImmutableTree(("S", ("D",), lcfrs_composition("0")), [
                None,
            ])
        ])
    assert excludes == {((3,4,5), 1, (1,2,3)), ((3,4,5,2), 1, (1,2,3,2))}


def test_dop():
    grammar = Dop(trees, prior=0)
    r1 = 2
    r2 = 1
    assert set(grammar.rules.keys()) == { r1 }
    assert set(grammar.rules[r1]) == { 
        ImmutableTree(r1, [None, None, ImmutableTree(r2, [None])])
    }
    assert grammar.weights == {
        ImmutableTree(r1, [None, None, ImmutableTree(r2, [None])]): 0,
    }

    grammar = Dop(trees, prior=1)
    assert grammar.weights == {
        ImmutableTree(r1, [None, None, ImmutableTree(r2, [None])]): log(4) - log(3),
    }
    assert grammar.fallback_weight == {
        "S": log(4)
    }


def test_parsing():
    r1 = 2
    r2 = 1
    fragments = [
        ImmutableTree(r1, [None, None, ImmutableTree(r2, [None])]),
        ImmutableTree(r1, [None, None, None]),
        ImmutableTree(r2, [None]),
    ]
    tree = fragments[0]

    assert list(DopTreeParser._match(fragments[2], tree[2], (2,))) == []
    assert list(DopTreeParser._match(fragments[0], tree, ())) == []
    assert list(DopTreeParser._match(fragments[1], tree, ())) == [(tree[2], (2,))]

    assert DopTreeParser._match(fragments[2], tree, ()) is None

    parser = DopTreeParser(Dop(trees, prior=0))
    parser.fill_chart(trees[1])

    assert set(parser.chart.keys()) == {(), (2,)}
    assert parser.chart[()] == 0
    assert parser.chart[(2,)] == float("-inf")

    parser = DopTreeParser(Dop(trees, prior=1))
    parser.fill_chart(trees[0])

    assert set(parser.chart.keys()) == {(), (0,), (1,), (1,2)}
    assert parser.chart[()] == parser.chart[(0,)] + parser.chart[(1,)] + log(4)
    assert parser.chart[(0,)] == log(4)
    assert parser.chart[(1,)] == log(4) - log(3)
    assert parser.chart[(1,2,)] == log(4)