from sdcp.grammar.extract import rule, extract, __extract_tree, singleton, sdcp_clause, lcfrs_composition
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree, Tree

from sortedcontainers import SortedSet

def test_singleton():
    tree = AutoTree("(ROOT 0)")
    assert singleton(tree) == ((rule("ROOT", (), dcp=sdcp_clause.binary_node(None)),), ("ROOT",))

    tree = AutoTree("(SBAR+S 0)")
    assert singleton(tree) == ((rule("ROOT", (), dcp=sdcp_clause.binary_node("SBAR")),), ("S",))

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}) == Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
    assert __extract_tree(tree[(0,0)], "VP", set()) == Tree((4, SortedSet([0,4,5]), rule("VP", ("L-VP", "VP|<>"), dcp=sdcp_clause.binary_node("VP", arity=2, transport_idx=1), scomp=lcfrs_composition("1,02"))), [
        Tree((0, SortedSet([0]), rule("L-VP", ())), []),
        Tree((5, SortedSet([5]), rule("VP|<>", ())),[])
    ])
    
    assert extract(tree)[0] == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=lcfrs_composition("1021")),
        rule("NP", (), dcp=sdcp_clause.binary_node("NP")),
        rule("VP", ("VP",), dcp=sdcp_clause.binary_node("VP", 1), scomp=lcfrs_composition("1,01")),
        rule("VP", ("L-VP", "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=lcfrs_composition("1,02")),
        rule("VP|<>", ()),
    ]


def test_corpus_extractor():
    c = corpus_extractor([(Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), "where the survey was carried out".split())], horzmarkov=0)
    c.read()

    assert c.goldrules == [tuple(range(6))]
    assert c.goldtrees == [Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")]
    assert c.sentences == [tuple("where the survey was carried out".split())]
    assert c.goldpos == [tuple("WRB PT NN VBD VBN RP".split())]
    assert list(c.rules) == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=lcfrs_composition("1021")),
        rule("NP", (), dcp=sdcp_clause.binary_node("NP")),
        rule("VP", ("VP",), dcp=sdcp_clause.binary_node("VP", 1), scomp=lcfrs_composition("1,01")),
        rule("VP", ("L-VP", "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=lcfrs_composition("1,02")),
        rule("VP|<>", ()),
    ]
    
    c = corpus_extractor([(Tree("(ROOT (S ($. 0)))"), ".".split())])
    c.read()

    assert c.goldrules == [tuple(range(1))]
    assert c.goldtrees == [Tree("(ROOT (S ($. 0)))")]
    assert c.sentences == [tuple(".".split())]
    assert c.goldpos == [tuple("$.".split())]
    assert list(c.rules) == [
        rule("ROOT", (), dcp=sdcp_clause.binary_node("ROOT+S")),
    ]


def test_derivations():
    from sdcp.grammar.extract import __extract_tree
    from discodop.tree import Tree

    def eval_derivation(deriv, p = None):
        lex, _, rul = deriv.label
        clause, ps = rul.dcp(lex, p)
        cs = (eval_derivation(c_, p_) for c_, p_ in zip(deriv, ps))
        return clause(*cs)


    t = AutoTree("(SBAR+S (VP (VP (WRB 0) (VP|<> (VBN 4) (RP 5))) (VBD 3)) (NP (PT 1) (NN 2)))")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (1, SortedSet(range(6)), rule("SBAR", ("VP", "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=lcfrs_composition("1021"))), [
            Tree((3, SortedSet([0,3,4,5]), rule("VP", ("VP",), dcp=sdcp_clause.binary_node("VP", 1), scomp=lcfrs_composition("1,01"))), [
                Tree((4, SortedSet([0,4,5]), rule("VP", ("L-VP", "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=lcfrs_composition("1,02"))), [
                    Tree((0, SortedSet([0]), rule("L-VP", ()),), []),
                    Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
                ])
            ]),
            Tree((2, SortedSet([2]), rule("NP", (), dcp=sdcp_clause.binary_node("NP"))), [])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


    t = AutoTree("(ROOT (DU (PP 0 1) (DU|<> 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6))))) 7)")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (7, SortedSet(range(9)), rule("ROOT", ("DU",), dcp=sdcp_clause.binary_node("ROOT", 1), scomp=lcfrs_composition("101"))), [
            Tree((2, SortedSet(list(range(7))+[8]), rule("DU", ("PP", "DU|<>"), dcp=sdcp_clause.binary_node("DU", 2), scomp=lcfrs_composition("102,2"))), [
                Tree((1, SortedSet(range(2)), rule("PP", ("L-PP",), dcp=sdcp_clause.binary_node("PP", 1), scomp=lcfrs_composition("10"))), [
                    Tree((0, SortedSet([0]), rule("L-PP", ())), [])
                ]),
                Tree((3, SortedSet([3,4,5,6,8]), rule("DU|<>", ("SMAIN",), dcp=sdcp_clause.binary_node(None, 1, transport_idx=0), scomp=lcfrs_composition("01,1"))), [
                    Tree((4, SortedSet([4,5,6,8]), rule("SMAIN", ("NP", "PP"), dcp=sdcp_clause.binary_node("SMAIN", 2), scomp=lcfrs_composition("02,1"))), [
                        Tree((8, SortedSet([8]), rule("NP", (), dcp=sdcp_clause.binary_node("NP"))), []),
                        Tree((5, SortedSet([5,6]), rule("PP", ("NP",), dcp=sdcp_clause.binary_node("PP", 1, transport_idx=0), scomp=lcfrs_composition("01"))), [
                            Tree((6, SortedSet([6]), rule("NP", (), dcp=sdcp_clause.binary_node("NP"))), [])
                        ]),
                    ])
                ])
            ])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(ROOT (DU (PP 0 1) 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6)))) 7)")