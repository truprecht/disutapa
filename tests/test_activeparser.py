from sdcp.grammar.sdcp import rule, grammar, lcfrs_composition, sdcp_clause, integerize_rules
from sdcp.grammar.composition import lcfrs_composition, ordered_union_composition, Composition
from sdcp.grammar.extraction.extract_head import Extractor
from sdcp.grammar.parser.activeparser import ActiveParser
from sdcp.grammar.extraction.corpus import corpus_extractor, ExtractionParameter
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
ihrules = list(integerize_rules(hrules))

def test_active_parser():
    parse = ActiveParser(grammar(ihrules, 0))
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
    parse = ActiveParser(grammar(ihrules, 0))
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
    parse = ActiveParser(grammar(list(integerize_rules(rules)), 0))
    parse.init(6)
    for r in range(6):
        parse.add_rules_i(r, 1, (r,), (0,))
    parse.fill_chart()
    assert parse.get_best()[0] == Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")


def read_corpus():
    from discodop.treebank import READERS, CorpusReader  # type: ignore
    ctrees = READERS["export"]("tests/sample.export", encoding="iso-8859-1", punct="move", headrules="../disco-dop/alpino.headrules")
    ex = corpus_extractor(ExtractionParameter())
    trees, rules, pos = [], [], []
    for ctree in ctrees.trees().values():
        trees.append(ctree)
        rs, ps = ex.read_tree(ctree)
        rules.append(rs)
        pos.append(ps)
    rtoi = {r: i for i,r in enumerate(ex.rules)}
    rs = [[rtoi[r] for r in srs] for srs in rules]
    return grammar(list(integerize_rules(eval(r) for r in ex.rules))), rs, pos, trees


def test_sample():
    gram, rules, pos, trees = read_corpus()

    parser = ActiveParser(gram)
    for rs, ps, gold in zip(rules, pos, trees):
        parser.init(len(rs))
        for position, r in enumerate(rs):
            parser.add_rules_i(position, 1, (r,), (0,))
        parser.fill_chart()
        assert AutoTree.convert(with_pos(parser.get_best()[0], ps)) == AutoTree.convert(gold)
        
        parser.init(len(rs))
        for position, r in enumerate(rs):
            parser.add_rules_i(position, len(gram.rules), *rule_weight_vector(len(gram.rules), r))
        parser.fill_chart(stop_early=True)
        assert AutoTree.convert(with_pos(parser.get_best()[0], ps)) == AutoTree.convert(gold)