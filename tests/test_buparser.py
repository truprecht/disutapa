from sdcp.grammar.sdcp import rule, grammar
from sdcp.grammar.buparser import BuParser, BitSpan
from sdcp.grammar.extract import extract
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree
from random import sample, randint, shuffle

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

            assert spans1.firstgap == next(i for i in range(min(set1), max(set1)+2) if not i in set1)
            assert spans2.firstgap == next(i for i in range(min(set2), max(set2)+2) if not i in set2)

            assert spans1.leftmost == min(pos1)
            assert spans2.leftmost == min(pos2)


def test_lc_parser():
    rules = [
        rule("L-VP", ()),
        rule("SBAR+S", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", fanout=2),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP", fanout=2),
        rule("VP|<>", ()),
    ]
    parse = BuParser(grammar(rules, "SBAR+S"))
    parse.init(*([(rid, 0)] for rid in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def rule_weight_vector(totallen: int, hot: int):
    vec = [(rid, totallen-abs(rid-hot)) for rid in range(totallen)]
    shuffle(vec)
    return vec


def test_weighted_lc_parser():
    rules = [
        rule("L-VP", ()),
        rule("SBAR+S", ("VP", "NP"), fn_node="SBAR+S"),
        rule("NP", (), fn_node="NP"),
        rule("VP", ("VP",), fn_node="VP", fanout=2),
        rule("VP", ("L-VP", "VP|<>"), fn_node="VP", fanout=2),
        rule("VP|<>", ()),
    ]
    parse = BuParser(grammar(rules, "SBAR+S"))
    parse.init(*(rule_weight_vector(6, position) for position in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def test_pipeline():
    tree = AutoTree("(SBAR+S (VP (VP 0 (VP|<> 4 5)) 3) (NP 1 2))")
    rules = list(extract(tree))
    parse = BuParser(grammar(rules))
    parse.init(*([(r, 0)] for r in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def test_sample():
    c = corpus_extractor("tests/sample.export", vertmarkov=1)
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = BuParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(*([(r, 0)] for r in rs))
        parse.fill_chart()
        assert AutoTree(parse.get_best()).tree(override_postags=pos) == gold


def test_weighted_sample():
    c = corpus_extractor("tests/sample.export", vertmarkov=1)
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = BuParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(*(rule_weight_vector(len(c.rules), r) for r in rs))
        parse.fill_chart()
        assert AutoTree(parse.get_best()).tree(override_postags=pos) == gold