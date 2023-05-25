from sdcp.grammar.extract import rule, extract, __extract_tree, singleton, sdcp_clause, lcfrs_composition
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree, Tree

from sortedcontainers import SortedSet # type: ignore

def test_singleton():
    tree = AutoTree("(ROOT 0)")
    assert singleton(tree) == ((rule("ROOT", dcp=sdcp_clause.binary_node(None)),), ("ROOT",))

    tree = AutoTree("(SBAR+S 0)")
    assert singleton(tree) == ((rule("ROOT", dcp=sdcp_clause.binary_node("SBAR")),), ("S",))


example_rules = [
    rule("L-VP"),
    rule("ROOT", ("VP", None, "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=lcfrs_composition("0120")),
    rule("NP", dcp=sdcp_clause.binary_node("NP")),
    rule("VP", ("VP", None), dcp=sdcp_clause.binary_node("VP", 1), scomp=lcfrs_composition("0,10")),
    rule("VP", ("L-VP", None, "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=lcfrs_composition("0,12")),
    rule("VP|<>"),
]

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}) == Tree((5, SortedSet([5]), example_rules[5], SortedSet([4,5])), [])
    assert __extract_tree(tree[(0,0)], "VP", set()) == \
        Tree((4, SortedSet([0,4,5]), example_rules[4], SortedSet([0,4,5])), [
            Tree((0, SortedSet([0]), example_rules[0], SortedSet([0])), []),
            Tree((5, SortedSet([5]), example_rules[5], SortedSet([4,5])), [])
        ])
    
    assert extract(tree)[0] == example_rules


def test_corpus_extractor():
    c = corpus_extractor([(Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), "where the survey was carried out".split())], horzmarkov=0)
    c.read()

    assert c.goldrules == [tuple(range(6))]
    assert c.goldtrees == [Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")]
    assert c.sentences == [tuple("where the survey was carried out".split())]
    assert c.goldpos == [tuple("WRB PT NN VBD VBN RP".split())]
    assert list(c.rules) == example_rules
    
    c = corpus_extractor([(Tree("(ROOT (S ($. 0)))"), ".".split())])
    c.read()

    assert c.goldrules == [tuple(range(1))]
    assert c.goldtrees == [Tree("(ROOT (S ($. 0)))")]
    assert c.sentences == [tuple(".".split())]
    assert c.goldpos == [tuple("$.".split())]
    assert list(c.rules) == [
        rule("ROOT", dcp=sdcp_clause.binary_node("ROOT+S")),
    ]


def test_derivations():
    from sdcp.grammar.extract import __extract_tree
    from discodop.tree import Tree # type: ignore

    def eval_derivation(deriv, p = None):
        lex, _, rul, _ = deriv.label
        clause, ps = rul.dcp(lex, p)
        cs = (eval_derivation(c_, p_) for c_, p_ in zip(deriv, ps))
        return clause(*cs)


    t = AutoTree("(SBAR+S (VP (VP (WRB 0) (VP|<> (VBN 4) (RP 5))) (VBD 3)) (NP (PT 1) (NN 2)))")
    assert __extract_tree(t, None, set(), override_lhs="ROOT") == Tree(
        (1, SortedSet(range(6)), example_rules[1], SortedSet([0,1,2,3,4,5])), [
            Tree((3, SortedSet([0,3,4,5]), example_rules[3], SortedSet([0,3,4,5])), [
                Tree((4, SortedSet([0,4,5]), example_rules[4], SortedSet([0,4,5])), [
                    Tree((0, SortedSet([0]), example_rules[0], SortedSet([0])), []),
                    Tree((5, SortedSet([5]), example_rules[5], SortedSet([4,5])), [])
                ])
            ]),
            Tree((2, SortedSet([2]), example_rules[2], SortedSet([1,2])), [])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


    rules = [
        rule("ROOT", ("DU", None), dcp=sdcp_clause.binary_node("ROOT", 1), scomp=lcfrs_composition("010")),
            rule("DU", ("PP", None, "DU|<>"), dcp=sdcp_clause.binary_node("DU", 2), scomp=lcfrs_composition("012,2")),
                rule("PP", ("L-PP", None), dcp=sdcp_clause.binary_node("PP", 1), scomp=lcfrs_composition("01")),
                    rule("L-PP"),
                rule("DU|<>", (None, "SMAIN"), dcp=sdcp_clause.binary_node(None, 1, transport_idx=0), scomp=lcfrs_composition("01,1")),
                    rule("SMAIN", (None, "PP", "NP"), dcp=sdcp_clause.binary_node("SMAIN", 2, transport_idx=0), scomp=lcfrs_composition("01,2")),
                        rule("PP", (None, "NP",), dcp=sdcp_clause.binary_node("PP", 1, transport_idx=0), scomp=lcfrs_composition("01")),
                            rule("NP", dcp=sdcp_clause.binary_node("NP")),
                        rule("NP", dcp=sdcp_clause.binary_node("NP"))
    ]
    yields = [
        SortedSet(range(9)),
            SortedSet([0,1,2,3,4,5,6,8]),
                SortedSet(range(2)),
                    SortedSet([0]),
                SortedSet([2,3,4,5,6,8]),
                    SortedSet([3,4,5,6,8]),
                        SortedSet([4,5,6]),
                            SortedSet([5,6]),
                        SortedSet([3,8])
    ]
    t = AutoTree("(ROOT (DU (PP 0 1) (DU|<> 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6))))) 7)")
    assert __extract_tree(t, None, set()) == Tree(
        (7, yields[0], rules[0], yields[0]), [
            Tree((2, yields[1], rules[1], yields[1]), [
                Tree((1, yields[2], rules[2], yields[2]), [
                    Tree((0, yields[3], rules[3], yields[3]), [])
                ]),
                Tree((3, yields[4]-SortedSet([2]), rules[4], yields[4]), [
                    Tree((4, yields[5]-SortedSet([3]), rules[5], yields[5]), [
                        Tree((5, yields[6]-SortedSet([4]), rules[6], yields[6]), [
                            Tree((6, SortedSet([6]), rules[7], yields[7]), [])
                        ]),
                        Tree((8, SortedSet([8]), rules[8], yields[8]), []),
                    ])
                ])
            ])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(ROOT (DU (PP 0 1) 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6)))) 7)")