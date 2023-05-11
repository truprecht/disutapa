from sdcp.grammar.extract import rule, extract, __extract_tree, singleton
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree, Tree

from sortedcontainers import SortedSet

def test_singleton():
    tree = AutoTree("(ROOT 0)")
    assert singleton(tree) == ((rule("ROOT", (), fn_node=""),), ("ROOT",))

    tree = AutoTree("(SBAR+S 0)")
    assert singleton(tree) == ((rule("ROOT", (), fn_node="SBAR"),), ("S",))

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}) == Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
    assert __extract_tree(tree[(0,0)], "VP", set()) == Tree((4, SortedSet([0,4,5]), rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition="1,02")), [
        Tree((0, SortedSet([0]), rule("L-VP", ())), []),
        Tree((5, SortedSet([5]), rule("VP|<>", ())),[])
    ])
    
    assert extract(tree)[0] == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S", composition="1021"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", composition="1,01"),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition="1,02"),
        rule("VP|<>", ()),
    ]


def test_corpus_extractor():
    c = corpus_extractor([(Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), "where the survey was carried out".split())])
    c.read()

    assert c.goldrules == [tuple(range(6))]
    assert c.goldtrees == [Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")]
    assert c.sentences == [tuple("where the survey was carried out".split())]
    assert c.goldpos == [tuple("WRB PT NN VBD VBN RP".split())]
    assert list(c.rules) == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S", composition="1021"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", composition="1,01"),
        rule("VP", ("L-VP", "VP|<VBN,RP>"), fn_node="VP", composition="1,02"),
        rule("VP|<VBN,RP>", ()),
    ]
    
    c = corpus_extractor([(Tree("(ROOT (S ($. 0)))"), ".".split())])
    c.read()

    assert c.goldrules == [tuple(range(1))]
    assert c.goldtrees == [Tree("(ROOT (S ($. 0)))")]
    assert c.sentences == [tuple(".".split())]
    assert c.goldpos == [tuple("$.".split())]
    assert list(c.rules) == [
        rule("ROOT", (), fn_node="ROOT+S"),
    ]


def test_derivations():
    from sdcp.grammar.extract import __extract_tree
    from discodop.tree import Tree

    def eval_derivation(deriv, p = None):
        lex, _, rul = deriv.label
        clause, ps = rul.fn(lex, p)
        cs = (eval_derivation(c_, p_) for c_, p_ in zip(deriv, ps))
        return clause(*cs)


    t = AutoTree("(SBAR+S (VP (VP (WRB 0) (VP|<> (VBN 4) (RP 5))) (VBD 3)) (NP (PT 1) (NN 2)))")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (1, SortedSet(range(6)), rule("SBAR", ("VP", "NP"), fn_node="SBAR+S", composition="1021")), [
            Tree((3, SortedSet([0,3,4,5]), rule("VP", ("VP",), fn_node="VP", composition="1,01")), [
                Tree((4, SortedSet([0,4,5]), rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition="1,02")), [
                    Tree((0, SortedSet([0]), rule("L-VP", ()),), []),
                    Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
                ])
            ]),
            Tree((2, SortedSet([2]), rule("NP", (), fn_node="NP")), [])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


    t = AutoTree("(ROOT (DU (PP 0 1) (DU|<> 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6))))) 7)")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (7, SortedSet(range(9)), rule("ROOT", ("DU",), fn_node="ROOT", composition="101")), [
            Tree((2, SortedSet(list(range(7))+[8]), rule("DU", ("PP", "DU|<>"), fn_node="DU", composition="102,2")), [
                Tree((1, SortedSet(range(2)), rule("PP", ("L-PP",), fn_node="PP", composition="10")), [
                    Tree((0, SortedSet([0]), rule("L-PP", ())), [])
                ]),
                Tree((3, SortedSet([3,4,5,6,8]), rule("DU|<>", ("SMAIN",), fn_push=0, composition="01,1")), [
                    Tree((4, SortedSet([4,5,6,8]), rule("SMAIN", ("NP", "PP"), fn_node="SMAIN", composition="02,1")), [
                        Tree((8, SortedSet([8]), rule("NP", (), fn_node="NP")), []),
                        Tree((5, SortedSet([5,6]), rule("PP", ("NP",), fn_node="PP", fn_push=0)), [
                            Tree((6, SortedSet([6]), rule("NP", (), fn_node="NP")), [])
                        ]),
                    ])
                ])
            ])
        ]
    )
    assert AutoTree.convert(eval_derivation(__extract_tree(t, "ROOT", set()))[0]) == AutoTree("(ROOT (DU (PP 0 1) 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6)))) 7)")