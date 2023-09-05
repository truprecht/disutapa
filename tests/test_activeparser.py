from sdcp.grammar.sdcp import rule, grammar, lcfrs_composition, sdcp_clause
from sdcp.grammar.extract_head import Extractor
from sdcp.grammar.parser.activeparser import ActiveParser
from sdcp.corpus import corpus_extractor
from sdcp.autotree import with_pos
from random import sample, randint, shuffle
from sdcp.autotree import AutoTree, Tree, HEAD

hrules = [
    rule("arg(V)[L]"),
    rule("arg(N)[L]"),
    rule("arg(S)[L]", ("arg(N)[L]", None,), dcp=sdcp_clause.spine("(NP 1 0)")),
    rule("ROOT", ("arg(V)[L]", "arg(S)[L]", None), dcp=sdcp_clause.spine("(SBAR (S (VP 1 0) 2))"), scomp=lcfrs_composition("0120")),
    rule("arg(V)[L]", ("arg(V)[L]", None, "arg(V)"), dcp=sdcp_clause.spine("(VP 1 0 2)"), scomp=lcfrs_composition("0,12")),
    rule("arg(V)"),
]

def test_active_parser():
    parse = ActiveParser(grammar(hrules, "ROOT"))
    parse.init(6)
    for i in range(6):
        parse.add_rules_i(i, 1, (i,), (0,))
    parse.fill_chart()
    assert parse.get_best() == [Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]


def rule_weight_vector(totallen: int, hot: int):
    vec = [(rid, abs(rid-hot)) for rid in range(totallen)]
    shuffle(vec)
    rids, ws = zip(*vec)
    return rids, ws


def test_weighted_active_parser():
    parse = ActiveParser(grammar(hrules, "ROOT"))
    parse.init(6)
    for position in range(6):
        parse.add_rules_i(position, 6, *rule_weight_vector(6, position))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


def test_pipeline():
    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    rules, _ = Extractor()(AutoTree.convert(t))
    parse = ActiveParser(grammar(rules, "ROOT"))
    parse.init(6)
    for r in range(6):
        parse.add_rules_i(r, 1, (r,), (0,))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


def test_sample():
    c = corpus_extractor("tests/sample.export", headrules="../disco-dop/alpino.headrules")
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = ActiveParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(len(rs))
        for position, r in enumerate(rs):
            parse.add_rules_i(position, 1, (r,), (0,))
        parse.fill_chart()
        assert with_pos(parse.get_best()[0], pos) == gold


def test_weighted_sample():
    c = corpus_extractor("tests/sample.export", headrules="../disco-dop/alpino.headrules")
    c.read()
    assert len(c.goldtrees) == len(c.goldrules) == len(c.sentences) == 3
    
    parse = ActiveParser(grammar(list(c.rules)))
    for rs, gold, pos in zip(c.goldrules, c.goldtrees, c.goldpos):
        parse.init(len(rs))
        # ActiveParser should not be used with such inputs to add_rules,
        # without early stopping
        for position, r in enumerate(rs):
            parse.add_rules_i(position, len(c.rules), *rule_weight_vector(len(c.rules), r))
        parse.fill_chart(stop_early=True)
        assert with_pos(parse.get_best()[0], pos) == gold