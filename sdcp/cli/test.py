from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.activeparser import ActiveParser, headed_rule, lcfrs_composition
from sdcp.grammar.extract_head import headed_clause
from sdcp.autotree import AutoTree, with_pos, fix_rotation
from sdcp.tagging.parsing_scorer import CombinatorialParsingScorer, DummyScorer
from sdcp.tagging.data import DatasetWrapper
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree, ImmutableTree
from tqdm import tqdm

from datasets import DatasetDict
from random import sample, shuffle, random, seed
from math import exp, log

def rule_vector(total: int, k: int, hot: int):
    vec = sample(range(total), k-1)
    vec.append(hot)
    shuffle(vec)
    weights = [exp(((total-abs(hot-rid))/total)*random()) for rid in vec]
    denom = sum(weights)
    return [(rid, log(w/denom)) for rid, w in zip(vec, weights)]


def main(config: Namespace):
    seed(config.seed)
    evaluator = Evaluator(readparam(config.param))
    data = DatasetWrapper(DatasetDict.load_from_disk(config.corpus)["dev"])
    p = ActiveParser(grammar([eval(str_hr) for str_hr in data.labels()]))
    idtopos = data.labels("pos")
    snd_order_weights = CombinatorialParsingScorer(data, prior=config.snd_order_prior, separated=config.snd_order_separate) \
        if config.snd_order else DummyScorer()
    p.set_scoring(snd_order_weights)
    idtopos = data.labels("pos")
    datalen = len(data) if config.range is None else config.range[1]-config.range[0]
    data = enumerate(data)
    if not config.range is None:
        data = ((i,s) for i,s in data if i in range(*config.range))
    for i, sample in tqdm(data, total=datalen):
        p.init(
            *(rule_vector(len(p.grammar.rules), config.weighted, i) for i in sample.get_raw_labels("supertag")),
        )
        p.fill_chart()
        prediction = p.get_best()[0]
        prediction = with_pos(prediction, [idtopos[i] for i in sample.get_raw_labels("pos")])
        evaluator.add(i, ParentedTree(sample.get_raw_labels("tree")), list(sample.get_raw_labels("sentence")),
                ParentedTree.convert(prediction), list(sample.get_raw_labels("sentence")))
        if str(fix_rotation(prediction)[1]) != sample.get_raw_labels("tree"):
            print(sample.get_raw_labels("tree"))
            print(prediction)
    print(evaluator.summary())
    print(evaluator.breakdowns())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--param", help="evalb parameter file for score calculation", type=str, required=False, default="../disco-dop/proper.prm")
    sub.add_argument("--weighted", type=int, default=1)
    sub.add_argument("--range", type=int, nargs=2, default=None)
    sub.add_argument("--seed", type=int, default=None)
    sub.add_argument("--snd-order", action="store_true", default=False)
    sub.add_argument("--snd-order-prior", type=int, default=0)
    sub.add_argument("--snd-order-separate", action="store_true", default=False)
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)