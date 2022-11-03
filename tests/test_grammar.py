from sdcp.grammar.sdcp import sdcp_clause, node_constructor, rule

def test_str_rule():
    rules = [
        rule("L-VP", ()),
        rule("SBAR+S", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP"),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP"),
        rule("VP|<>", ()),
    ]
    assert [repr(r) for r in rules] ==  [
        "rule('L-VP', ())",
        "rule('SBAR+S', ('VP', 'NP'), fn_node='SBAR+S')",
        "rule('NP', (), fn_node='NP')",
        "rule('VP', ('VP',), fn_node='VP')",
        "rule('VP', ('L-VP', 'VP|<>'), fn_node='VP')",
        "rule('VP|<>', ())",
    ]



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
    assert consts[1][0]("(VP 3 (VP 0 4 5))", "(NP 1 2)") == "(SBAR+S (VP 3 (VP 0 4 5)) (NP 1 2))"