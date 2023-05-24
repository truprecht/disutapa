from sdcp.grammar.extract import rule, extract, __extract_tree
from sdcp.grammar.parser.activeparser import ActiveParser, grammar
from sdcp.autotree import AutoTree, Tree

from sdcp.grammar.extract_head import Extractor, SortedSet, ordered_union_composition, sdcp_clause
from sdcp.autotree import AutoTree, Tree, HEAD

from sortedcontainers import SortedSet # type: ignore

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    
    assert __extract_tree(tree[(0,0,1)], "VP", {4}, ctype=ordered_union_composition) == \
        Tree((5, SortedSet([5]), rule("VP|<>", ())), [])
    assert __extract_tree(tree[(0,0)], "VP", set(), ctype=ordered_union_composition) == \
        Tree((4, SortedSet([0,4,5]), rule.from_guided("VP", ("L-VP", "VP|<>"), dnode="VP", scomp=ordered_union_composition("102", fanout=2))), [
            Tree((0, SortedSet([0]), rule("L-VP", ())), []),
            Tree((5, SortedSet([5]), rule("VP|<>", ())),[])
        ])
    
    assert extract(tree, ctype="dcp")[0] == [
        rule("L-VP", ()),
        rule.from_guided("ROOT", ("VP", "NP"), dnode="SBAR+S", scomp=ordered_union_composition("102")),
        rule.from_guided("NP", (), dnode="NP"),
        rule.from_guided("VP", ("VP",), dnode="VP", scomp=ordered_union_composition("10", fanout=2)),
        rule.from_guided("VP", ("L-VP", "VP|<>"), dnode="VP", scomp=ordered_union_composition("102", fanout=2)),
        rule("VP|<>", ()),
    ]



def test_nonbin_extraction():
    e = Extractor(horzmarkov=0, rightmostunary=True, composition="dcp")
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>", ()),
        rule("ROOT", ("S|<>", "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)"), scomp=ordered_union_composition("102")),
        rule("S|<>", ("S|<>",), dcp=sdcp_clause.default(1), scomp=ordered_union_composition("01")),
        rule("S|<>", ("S|<>",), dcp=sdcp_clause.default(1), scomp=ordered_union_composition("01")),
        rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>", ()),
        rule("ROOT", ("S|<>", "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)"), scomp=ordered_union_composition("102")),
        rule("T|<>", ()),
        rule("S|<>", ("T|<>", "T|<>", "S|<>",), dcp=sdcp_clause.spine("(T 1 0 2)", 3), scomp=ordered_union_composition("1023")),
        rule("T|<>", ()),
        rule("S|<>", ("S|<>",), dcp=sdcp_clause.default(1), scomp=ordered_union_composition("01")),
        rule("S|<>", ()),
    ]

def test_active_parser():
    rules = [
        rule("L-VP", ()),
        rule("ROOT", ("VP", "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=ordered_union_composition("102")),
        rule("NP", (), dcp=sdcp_clause.binary_node("NP")),
        rule("VP", ("VP",), dcp=sdcp_clause.binary_node("VP", 1), scomp=ordered_union_composition("10", fanout=2)),
        rule("VP", ("L-VP", "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=ordered_union_composition("102", fanout=2)),
        rule("VP|<>", ()),
    ]
    parse = ActiveParser(grammar(rules, "ROOT"))
    parse.init(*([(rid, 0)] for rid in range(6)))
    parse.fill_chart()
    assert parse.get_best() == [Tree("(SBAR (S (VP 3 (VP 0 5 4)) (NP 2 1)))")]


def test_pipeline():
    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    e = Extractor(composition="dcp")
    rules, _ = e(AutoTree.convert(t))
    print(rules)
    parse = ActiveParser(grammar(rules))
    parse.init(*([(r, 0)] for r in range(6)))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")