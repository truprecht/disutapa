from sdcp.grammar.extract import rule, sdcp_clause
from sdcp.grammar.parser import parser, grammar
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree
from sdcp.grammar.sdcp import node_constructor

def test_corpus_extractor():
    c = corpus_extractor([(AutoTree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), "where the survey was carried out".split())])
    c.read()

    assert c.goldrules == [list(range(6))]
    assert c.goldtrees == [AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]
    assert c.sentences == [tuple("where the survey was carried out".split())]
    assert list(c.rules) == [
        rule("L-VP", sdcp_clause(None, 0, push_idx=0), ()),
        rule("ROOT", sdcp_clause("SBAR+S", 2, push_idx=1), ("VP", "NP")),
        rule("NP", sdcp_clause("NP", 0, push_idx=0), ()),
        rule("VP", sdcp_clause("VP", 1, push_idx=1), ("VP",)),
        rule("VP", sdcp_clause("VP", 2, push_idx=1), ("L-VP", "VP|<VBN,RP>")),
        rule("VP|<VBN,RP>", sdcp_clause(None, 0, push_idx=0), ()),
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
        (1, rule("SBAR+S", sdcp_clause("SBAR+S", 2, push_idx=1), ("VP", "NP"))), [
            Tree((3, rule("VP", sdcp_clause("VP", 1, push_idx=1), ("VP",))), [
                Tree((4, rule("VP", sdcp_clause("VP", 2, push_idx=1), ("L-VP", "VP|<>"))), [
                    Tree((0, rule("L-VP", sdcp_clause(None, 0, push_idx=0), ())), []),
                    Tree((5, rule("VP|<>", sdcp_clause(None, 0, push_idx=0), ())), [])
                ])
            ]),
            Tree((2, rule("NP", sdcp_clause("NP", 0, push_idx=0), ())), [])
        ]
    )
    assert AutoTree(eval_derivation(__extract_tree(t, "ROOT", set()))) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


    t = AutoTree("(ROOT (DU (PP 0 1) (DU|<> 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6))))) 7)")
    assert __extract_tree(t, "ROOT", set()) == Tree(
        (7,rule("ROOT", sdcp_clause("ROOT", 1, push_idx=1), ("DU",))), [
            Tree((2, rule("DU", sdcp_clause("DU", 2, push_idx=1), ("PP", "DU|<>"))), [
                Tree((1, rule("PP", sdcp_clause("PP", 1, push_idx=1), ("L-PP",))), [
                    Tree((0, rule("L-PP", sdcp_clause(None, 0, push_idx=0), ())), [])
                ]),
                Tree((3, rule("DU|<>", sdcp_clause(None, 1, push_idx=0), ("SMAIN",))), [
                    Tree((4, rule("SMAIN", sdcp_clause("SMAIN", 2, push_idx=1), ("NP", "PP"))), [
                        Tree((8, rule("NP", sdcp_clause("NP", 0, push_idx=0), ())), []),
                        Tree((5, rule("PP", sdcp_clause("PP", 1, push_idx=0), ("NP",))), [
                            Tree((6, rule("NP", sdcp_clause("NP", 0, push_idx=0), ())), [])
                        ])
                    ])
                ])
            ])
        ]
    )
    assert AutoTree(eval_derivation(__extract_tree(t, "ROOT", set()))) == AutoTree("(ROOT (DU (PP 0 1) 2 (SMAIN (NP 3 8) (PP 4 (NP 5 6)))) 7)")