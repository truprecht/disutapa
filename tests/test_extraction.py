from sdcp.grammar.extract import rule, sdcp_clause, extract, __extract_tree
from sdcp.grammar.parser import parser, grammar
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree, Tree


def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}) == Tree((5, rule("VP|<>", ())), [])
    assert __extract_tree(tree[(0,0)], "VP", set()) == Tree((4, rule("VP", ("L-VP", "VP|<>"), fn_node="VP")), [
        Tree((0, rule("L-VP", ())), []),
        Tree((5, rule("VP|<>", ())),[])
    ])
    
    assert list(extract(tree)) == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP"),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP"),
        rule("VP|<>", ()),
    ]



def test_corpus_extractor():
    c = corpus_extractor([(AutoTree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), "where the survey was carried out".split())])
    c.read()

    assert c.goldrules == [list(range(6))]
    assert c.goldtrees == [AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]
    assert c.sentences == [tuple("where the survey was carried out".split())]
    assert list(c.rules) == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP"),
        rule("VP", ("L-VP", "VP|<VBN,RP>"), fn_node="VP"),
        rule("VP|<VBN,RP>", ()),
    ]
    
    parse = parser(grammar(list(c.rules)))
    for rs, gold in zip(c.goldrules, c.goldtrees):
        parse.init(*([r] for r in rs))
        parse.fill_chart()
        assert AutoTree(parse.get_best()) == gold

def test_sample():
    c = corpus_extractor("tests/sample.export", vertmarkov=1)
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = parser(grammar(list(c.rules)))
    for rs, gold in zip(c.goldrules, c.goldtrees):
        parse.init(*([r] for r in rs))
        parse.fill_chart()
        print(AutoTree(parse.get_best()), gold)
        assert AutoTree(parse.get_best()) == gold


def test_derivations():
    from sdcp.grammar.extract import __extract_tree
    from discodop.tree import Tree

    def eval_derivation(deriv, p = None):
        lex, rul = deriv.label
        clause, ps = rul.fn(lex, p)
        cs = (eval_derivation(c_, p_) for c_, p_ in zip(deriv, ps))
        return clause(*cs)


    t = AutoTree("(SBAR+S (VP (VP (WRB 0) (VP|<> (VBN 4) (RP 5))) (VBD 3)) (NP (PT 1) (NN 2)))")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (1, rule("SBAR+S", ("VP", "NP"), fn_node="SBAR+S")), [
            Tree((3, rule("VP", ("VP",), fn_node="VP")), [
                Tree((4, rule("VP", ("L-VP", "VP|<>"), fn_node="VP")), [
                    Tree((0, rule("L-VP", ()),), []),
                    Tree((5, rule("VP|<>", ())), [])
                ])
            ]),
            Tree((2, rule("NP", (), fn_node="NP")), [])
        ]
    )
    assert AutoTree(eval_derivation(__extract_tree(t, "ROOT", set()))) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


    t = AutoTree("(ROOT (DU (PP 0 1) (DU|<> 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6))))) 7)")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (7,rule("ROOT", ("DU",), fn_node="ROOT")), [
            Tree((2, rule("DU", ("PP", "DU|<>"), fn_node="DU")), [
                Tree((1, rule("PP", ("L-PP",), fn_node="PP")), [
                    Tree((0, rule("L-PP", ())), [])
                ]),
                Tree((3, rule("DU|<>", ("SMAIN",), fn_push=0)), [
                    Tree((4, rule("SMAIN", ("NP", "PP"), fn_node="SMAIN")), [
                        Tree((8, rule("NP", (), fn_node="NP")), []),
                        Tree((5, rule("PP", ("NP",), fn_node="PP", fn_push=0)), [
                            Tree((6, rule("NP", (), fn_node="NP")), [])
                        ])
                    ])
                ])
            ])
        ]
    )
    assert AutoTree(eval_derivation(__extract_tree(t, "ROOT", set()))) == AutoTree("(ROOT (DU (PP 0 1) 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6)))) 7)")