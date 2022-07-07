from sdcp.grammar.sdcp import sdcp_clause, node_constructor, rule, grammar
from sdcp.grammar.parser import parser
from sdcp.grammar.extract import extract
from sdcp.autotree import AutoTree

def test_sdcp_fn():
    functions = [
        sdcp_clause(None, 0),
        sdcp_clause("SBAR+S", 2),
        sdcp_clause("NP", 0),
        sdcp_clause("VP", 1),
        sdcp_clause("VP", 2),
        sdcp_clause(None, 0)
    ]

    consts = [
        (node_constructor(None, 0), ()),
        (node_constructor("SBAR+S"), (None, 1)),
        (node_constructor("NP", 1, 2), ()),
        (node_constructor("VP", 3), (None,)),
        (node_constructor("VP"), (None, 4)),
        (node_constructor(None, 4, 5), ()),
    ]

    assert functions[0](0, None) == consts[0]
    assert functions[1](1, None) == consts[1]
    assert functions[2](2, 1) == consts[2]
    assert functions[3](3, None) == consts[3]
    assert functions[4](4, None) == consts[4]
    assert functions[5](5, 4) == consts[5]

    assert consts[0][0]() == "0"
    assert consts[5][0]() == "4 5"
    assert consts[4][0]("0", "4 5") == "(VP 0 4 5)"
    assert consts[3][0]("(VP 0 4 5)") == "(VP 3 (VP 0 4 5))"
    assert consts[2][0]() == "(NP 1 2)"
    assert consts[1][0]("(VP 3 (VP 0 4 5))", "(NP 1 2)") == "(SBAR (S (VP 3 (VP 0 4 5)) (NP 1 2)))"


def test_parser():
    rules = [
        rule("L-VP", sdcp_clause(None, 0), ()),
        rule("SBAR+S", sdcp_clause("SBAR+S", 2), ("VP", "NP")),
        rule("NP", sdcp_clause("NP", 0), ()),
        rule("VP", sdcp_clause("VP", 1), ("VP",)),
        rule("VP", sdcp_clause("VP", 2), ("L-VP", "VP|<>")),
        rule("VP|<>", sdcp_clause(None, 0), ()),
    ]
    parse = parser(grammar(rules, "SBAR+S"))
    parse.init(*([rid] for rid in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


def test_pipeline():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    rules = list(extract(tree))
    parse = parser(grammar(rules))
    parse.init(*([r] for r in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")
