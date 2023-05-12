from sdcp.grammar.extract import rule, extract, __extract_tree
from sdcp.grammar.parser.activeparser import ActiveParser, grammar
from sdcp.autotree import AutoTree, Tree

from sdcp.grammar.extract_head import headed_rule, Extractor, SortedSet, ordered_union_composition
from sdcp.headed_tree import HeadedTree, Tree, HEAD

from sortedcontainers import SortedSet

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}, ctype=ordered_union_composition) == \
        Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
    assert __extract_tree(tree[(0,0)], "VP", set(), ctype=ordered_union_composition) == \
        Tree((4, SortedSet([0,4,5]), rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition=ordered_union_composition("102", fanout=2))), [
            Tree((0, SortedSet([0]), rule("L-VP", ())), []),
            Tree((5, SortedSet([5]), rule("VP|<>", ())),[])
        ])
    
    assert extract(tree, ctype="dcp")[0] == [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S", composition=ordered_union_composition("102")),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", composition=ordered_union_composition("10", fanout=2)),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition=ordered_union_composition("102", fanout=2)),
        rule("VP|<>", ()),
    ]



def test_nonbin_extraction():
    e = Extractor(horzmarkov=0, rightmostunary=True, composition="dcp")
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)", composition=ordered_union_composition("102")),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)", composition=ordered_union_composition("01")),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)", composition=ordered_union_composition("01")),
        headed_rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)", composition=ordered_union_composition("102")),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("T|<>", "T|<>", "S|<>",), clause="(_|<> (T 1 0 2) 3)", composition=ordered_union_composition("1023")),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)", composition=ordered_union_composition("01")),
        headed_rule("S|<>", ()),
    ]

def test_active_parser():
    rules = [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), fn_node="SBAR+S", composition=ordered_union_composition("102")),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", composition=ordered_union_composition("10", fanout=2)),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP", composition=ordered_union_composition("102", fanout=2)),
        rule("VP|<>", ()),
    ]
    parse = ActiveParser(grammar(rules, "ROOT"))
    parse.init(*([(rid, 0)] for rid in range(6)))
    parse.fill_chart()
    assert parse.get_best() == [Tree("(SBAR (S (VP 3 (VP 0 4 5)) (NP 1 2)))")]


def test_pipeline():
    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    e = Extractor(composition="dcp")
    rules, _ = e(HeadedTree.convert(t))
    print(rules)
    parse = ActiveParser(grammar(rules))
    parse.init(*([(r, 0)] for r in range(6)))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")