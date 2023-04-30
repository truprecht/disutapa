from sdcp.grammar.sdcp import rule, grammar
from sdcp.grammar.extract_head import headed_rule, headed_clause, Extractor
from sdcp.grammar.activeparser import ActiveParser
from sdcp.corpus import corpus_extractor
from sdcp.autotree import with_pos
from random import sample, randint, shuffle
from sdcp.headed_tree import HeadedTree, Tree, HEAD

def test_active_parser():
    rules = [
        headed_rule("VP|<>", []),
        headed_rule("NP|<>", []),
        headed_rule("S|<>", ["NP|<>"], "(NP 1  0)"),
        headed_rule("ROOT", ["VP|<>", "S|<>"], "(SBAR (S (VP 1 0) 2))", lexidx=2),
        headed_rule("VP|<>", ["VP|<>", "VP|<>"], "(VP 1 0 2)", 2),
        headed_rule("VP|<>", []),
    ]
    parse = ActiveParser(grammar(rules, "ROOT"))
    parse.init(*([(rid, 0)] for rid in range(6)))
    parse.fill_chart()
    assert parse.get_best() == [Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]


def rule_weight_vector(totallen: int, hot: int):
    vec = [(rid, abs(rid-hot)) for rid in range(totallen)]
    shuffle(vec)
    return vec


def test_weighted_active_parser():
    rules = [
        headed_rule("VP|<>", [], headed_clause(0)),
        headed_rule("NP|<>", [], headed_clause(0)),
        headed_rule("S|<>", ["NP|<>"], headed_clause("(NP 1  0)")),
        headed_rule("ROOT", ["VP|<>", "S|<>"], headed_clause("(SBAR (S (VP 1 0) 2))"), lexidx=2),
        headed_rule("VP|<>", ["VP|<>", "VP|<>"], headed_clause("(VP 1 0 2)"), 2),
        headed_rule("VP|<>", [], headed_clause(0)),
    ]
    parse = ActiveParser(grammar(rules, "ROOT"))
    parse.init(*(rule_weight_vector(6, position) for position in range(6)))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


def test_pipeline():
    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    rules, _ = Extractor()(HeadedTree.convert(t))
    print(rules)
    parse = ActiveParser(grammar(rules))
    parse.init(*([(r, 0)] for r in range(6)))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")


def test_sample():
    c = corpus_extractor("tests/sample.export", headrules="../disco-dop/alpino.headrules")
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = ActiveParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(*([(r, 0)] for r in rs))
        parse.fill_chart()
        assert with_pos(parse.get_best()[0], pos) == gold


def test_weighted_sample():
    c = corpus_extractor("tests/sample.export", headrules="../disco-dop/alpino.headrules")
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = ActiveParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(*(rule_weight_vector(len(c.rules), r) for r in rs))
        parse.fill_chart()
        assert with_pos(parse.get_best()[0], pos) == gold