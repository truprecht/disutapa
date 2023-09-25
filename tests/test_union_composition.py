from sdcp.grammar.extraction.extract import rule, extract, __extract_tree, Guide
from sdcp.grammar.parser.activeparser import ActiveParser, grammar
from sdcp.autotree import AutoTree, Tree

from sdcp.grammar.extraction.extract_head import Extractor, SortedSet, ordered_union_composition, sdcp_clause
from sdcp.autotree import AutoTree, Tree, HEAD

from sortedcontainers import SortedSet # type: ignore
from sdcp.grammar.sdcp import integerize_rules
from sdcp.grammar.composition import union_from_positions

example_rules = [
    rule("L-VP"),
    rule("ROOT", ("VP", None, "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=ordered_union_composition(fanout=1)),
    rule("NP", dcp=sdcp_clause.binary_node("NP")),
    rule("VP", ("VP", None), dcp=sdcp_clause.binary_node("VP", 1), scomp=ordered_union_composition(fanout=2)),
    rule("VP", ("L-VP", None, "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=ordered_union_composition(fanout=2)),
    rule("VP|<>"),
]

def test_extract():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    guide = Guide.construct("strict", tree)
    
    assert __extract_tree(tree[(0,0,1)], guide, "VP", {4}, cconstructor=union_from_positions) == \
        Tree((5, SortedSet([5]), example_rules[5], SortedSet([4,5])), [])
    assert __extract_tree(tree[(0,0)], guide, "VP", set(), cconstructor=union_from_positions) == \
        Tree((4, SortedSet([0,4,5]), example_rules[4], SortedSet([0,4,5])), [
            Tree((0, SortedSet([0]), example_rules[0], SortedSet([0])), []),
            Tree((5, SortedSet([5]), example_rules[5], SortedSet([4,5])),[])
        ])
    
    assert extract(tree, ctype="dcp")[0] == example_rules



def test_nonbin_extraction():
    e = Extractor(horzmarkov=0, rightmostunary=True, composition="dcp")
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>"),
        rule("ROOT", ("S|<>", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)"), scomp=ordered_union_composition()),
        rule("S|<>", (None, "S|<>",), dcp=sdcp_clause.default(1), scomp=ordered_union_composition()),
        rule("S|<>", (None, "S|<>",), dcp=sdcp_clause.default(1), scomp=ordered_union_composition()),
        rule("S|<>"),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>"),
        rule("ROOT", ("S|<>", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)"), scomp=ordered_union_composition()),
        rule("T|<>"),
        rule("S|<>", ("T|<>", None, "T|<>", "S|<>",), dcp=sdcp_clause.spine("(T 1 0 2)", 3), scomp=ordered_union_composition()),
        rule("T|<>"),
        rule("S|<>", (None, "S|<>"), dcp=sdcp_clause.default(1), scomp=ordered_union_composition()),
        rule("S|<>"),
    ]

def test_active_parser():
    parse = ActiveParser(grammar(list(integerize_rules(example_rules)), 0))
    parse.init(6)
    for i in range(6):
        parse.add_rules_i(i, 1, (i,), (0,))
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
    parse = ActiveParser(grammar(list(integerize_rules(rules)), 0))
    parse.init(6)
    for i in range(6):
        parse.add_rules_i(i, 1, (i,), (0,))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")