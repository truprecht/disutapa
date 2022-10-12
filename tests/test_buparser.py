from sdcp.grammar.sdcp import rule, grammar
from sdcp.grammar.buparser import BuParser
from sdcp.grammar.extract import extract
from sdcp.corpus import corpus_extractor
from sdcp.autotree import AutoTree
from random import sample, randint, shuffle


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