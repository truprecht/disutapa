from sdcp.grammar.sdcp import sdcp_clause, tree_constructor, rule, Tree, lcfrs_composition, ImmutableTree

example_rules = [
    rule("L-VP"),
    rule("ROOT", ("VP", None, "NP"), dcp=sdcp_clause.binary_node("SBAR+S", 2), scomp=lcfrs_composition("0120")),
    rule("NP", dcp=sdcp_clause.binary_node("NP")),
    rule("VP", ("VP", None), dcp=sdcp_clause.binary_node("VP", 1), scomp=lcfrs_composition("0,10")),
    rule("VP", ("L-VP", None, "VP|<>"), dcp=sdcp_clause.binary_node("VP", 2), scomp=lcfrs_composition("0,12")),
    rule("VP|<>"),
]

def test_str_rule():
    assert [repr(r) for r in example_rules] ==  [
        "rule('L-VP')",
        "rule('ROOT', ('VP', -1, 'NP'), Composition.lcfrs('0120'), sdcp_clause('(SBAR+S 2 3)', args=(1, 0)))",
        "rule('NP', dcp=sdcp_clause('(NP 0 1)'))",
        "rule('VP', ('VP', -1), Composition.lcfrs('0,10'), sdcp_clause('(VP 0 2)', args=(1,)))",
        "rule('VP', ('L-VP', -1, 'VP|<>'), Composition.lcfrs('0,12'), sdcp_clause('(VP 2 3)', args=(1, 0)))",
        "rule('VP|<>')",
    ]



def test_sdcp_fn():
    assert sdcp_clause.default(0) == sdcp_clause((0, 1)) == sdcp_clause.spine(0)
    assert sdcp_clause.default(1) == sdcp_clause((0, 1, 2)) != sdcp_clause.spine(0, 1)

    functions = [
        sdcp_clause.default(0),
        sdcp_clause.binary_node("SBAR+S", arity=2),
        sdcp_clause.binary_node("NP"),
        sdcp_clause.binary_node("VP", arity=1, transport_idx=None),
        sdcp_clause.binary_node("VP", arity=2),
        sdcp_clause.default(0)
    ]

    consts = [
        (tree_constructor((0, 1), [0, None]), ()),
        (tree_constructor((ImmutableTree("(SBAR+S 2 3)"),), [None, None]), (None, 1)),
        (tree_constructor((ImmutableTree("(NP 0 1)"),), [2, 1]), ()),
        (tree_constructor((ImmutableTree("(VP 0 2)"),), [3, None]), (None,)),
        (tree_constructor((ImmutableTree("(VP 2 3)"),), [None, None]), (None, 4)),
        (tree_constructor((0, 1), [5, 4]), ()),
    ]

    assert functions[0](0, None)[0] == consts[0][0]
    assert functions[1](1, None)[0] == consts[1][0]
    assert functions[2](2, 1)[0] == consts[2][0]
    assert functions[3](3, None)[0] == consts[3][0]
    assert functions[4](4, None)[0] == consts[4][0]
    assert functions[5](5, 4)[0] == consts[5][0]

    assert consts[0][0]() == [0]
    assert consts[5][0]() == [5, 4]
    assert consts[4][0]([0], [4,5]) == [Tree("VP", [0,4,5])]
    assert consts[3][0]([Tree("VP", [0,4,5])]) == [Tree("VP", [3, Tree("VP", [0,4,5])])]
    assert consts[2][0]() == [Tree("NP", [2,1])]
    assert consts[1][0]([Tree("VP", [Tree("VP", [0,4,5]), 3])], [Tree("NP", [1,2])]) == [Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]