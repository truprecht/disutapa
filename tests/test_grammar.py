from sdcp.grammar.sdcp import sdcp_clause, node_constructor, rule, grammar
from sdcp.grammar.parser import LeftCornerParser, BitSpan
from sdcp.grammar.extract import extract
from sdcp.autotree import AutoTree

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


def test_lc_parser():
    rules = [
        rule("L-VP", ()),
        rule("SBAR+S", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP"),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP"),
        rule("VP|<>", ()),
    ]
    parse = LeftCornerParser(grammar(rules, "SBAR+S"))
    parse.init(*([rid] for rid in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def test_pipeline():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    rules = list(extract(tree))
    parse = LeftCornerParser(grammar(rules))
    parse.init(*([r] for r in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def test_bitspans():
    assert BitSpan.fromit([1,2,3], len=10).union(BitSpan.fromit([], len=10)) == BitSpan.fromit([1,2,3], len=10)
    assert BitSpan.fromit([1,2,3], len=10).union(BitSpan.fromit([4,5,6], len=10)) == BitSpan.fromit(range(1,7), len=10)
    assert BitSpan.fromit([1,2,3], len=10).union(BitSpan.fromit([5,6], len=10)) == BitSpan.fromit([1,2,3,5,6], len=10)
    assert BitSpan.fromit([1,2,3,5,6], len=10).union(BitSpan.fromit([4], len=10)) == BitSpan.fromit(range(1,7), len=10)
    
    assert BitSpan.fromit([0,3,4,5], len=10).isdisjoint(BitSpan.fromit([1,2], len=10))
    assert BitSpan.fromit([0,3,4,5], len=10).union(BitSpan.fromit([1,2], len=10)) == BitSpan.fromit(range(6), len=10)

    assert BitSpan.fromit([1,2,3,5,6], len=10).leftmost == 1
    assert BitSpan.fromit([0,3,4,5], len=10).leftmost == 0

    assert BitSpan.fromit([1,2,3,4], len=10).numgaps() == 0
    assert BitSpan.fromit([1,2,3,6], len=10).numgaps() == 1

    from random import sample, randint
    for l in range(1, 91, 5):
        for i in range(randint(3, 10)):
            top = randint(l, l+100)
            pos1 = sample(range(top), l)
            pos2 = sample(range(top), l)
            spans1 = BitSpan.fromit(pos1, len=top)
            spans2 = BitSpan.fromit(pos2, len=top)
            set1 = frozenset(pos1)
            set2 = frozenset(pos2)

            assert list(spans1) == sorted(pos1)
            assert list(spans2) == sorted(pos2)

            assert spans1.isdisjoint(spans2) == set1.isdisjoint(set2)
            assert spans2.isdisjoint(spans1) == set2.isdisjoint(set1)

            if not spans1.isdisjoint(spans2):
                disp2 = BitSpan.fromit((p2 for p2 in pos2 if not p2 in pos1), len=top)
                sisp2 = frozenset(p2 for p2 in pos2 if not p2 in pos1)
                assert set(spans1.union(disp2)) == set1.union(sisp2)
            else:
                assert set(spans1.union(spans2)) == set1.union(set2)

            assert spans1.firstgap() == next(i for i in range(min(set1), max(set1)+2) if not i in set1)
            assert spans2.firstgap() == next(i for i in range(min(set2), max(set2)+2) if not i in set2)

            assert spans1.leftmost == min(pos1)
            assert spans2.leftmost == min(pos2)