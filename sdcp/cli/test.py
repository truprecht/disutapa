from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.parser.activeparser import ActiveParser
from sdcp.grammar.lcfrs import lcfrs_composition, ordered_union_composition
from sdcp.autotree import AutoTree, with_pos, fix_rotation
from sdcp.tagging.parsing_scorer import CombinatorialParsingScorer, DummyScorer
from sdcp.tagging.data import DatasetWrapper
from sdcp.tagging.ensemble_model import ParserAdapter
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree, ImmutableTree
from tqdm import tqdm

from datasets import DatasetDict
from random import sample, shuffle, random, seed
from math import exp, log

import torch
from itertools import islice
from sdcp.reranking.dop import Dop, Tree

def guess_weights(total, hot, k):
    weights = torch.zeros((len(hot), total))
    for i, h in enumerate(hot):
        start = max(0, h-k//2) if h+k < total else total-k
        end = start + k
        ws = (torch.arange(start, end)-h).abs()
        ws = (total-ws).pow(2)
        ws = (torch.randn(k) * .5 + 1).abs()
        weights[i, start:end] = ws
    return weights


def main(config: Namespace):
    seed(config.seed)
    evaluator = Evaluator(readparam(config.param))
    corpus = DatasetDict.load_from_disk(config.corpus)
    data = DatasetWrapper(corpus["dev"])
    p = ParserAdapter(grammar([eval(str_hr) for str_hr in data.labels()]), total_limit=config.weighted)
    nrules = len(p.parser.grammar.rules)
    idtopos = data.labels("pos")
    snd_order_weights = CombinatorialParsingScorer(data, prior=config.snd_order_prior, separated=config.snd_order_separate) \
        if config.snd_order else DummyScorer()
    p.set_scoring(snd_order_weights)
    idtopos = data.labels("pos")
    datalen = len(data)
    data = enumerate(data)

    if config.dop:
        dopgrammar = Dop((
            Tree(sentence.get_raw_labels("tree"))
                for sentence in DatasetWrapper(corpus["train"])),
            min_occurrences=1
        )

    for i, sample in tqdm(data, total=datalen):
        if not config.range is None and not i in range(*config.range):
            continue

        weights, indices = guess_weights(nrules, sample.get_raw_labels("supertag"), config.weighted).topk(config.weighted, sorted=True)
        weights = -weights.log_softmax(dim=-1)
        
        goldpostags = [idtopos[i] for i in sample.get_raw_labels("pos")]
        goldtree = ParentedTree(sample.get_raw_labels("tree"))

        p.init(len(sample), weights, indices)
        p.fill_chart()
        prediction = with_pos(p.get_best()[0], goldpostags)
        evaluator.add(i, goldtree, list(sample.get_raw_labels("sentence")),
                ParentedTree.convert(prediction), list(sample.get_raw_labels("sentence")))

        if str(fix_rotation(prediction)[1]) != sample.get_raw_labels("tree"):
            print("best tree is not gold")
            trees = islice(p.get_best_iter(), config.k)
            trees = [fix_rotation(with_pos(t[0], goldpostags))[1] for t, w in trees]
            if config.dop:
                trees.sort(key=dopgrammar.match, reverse=True)
            for i, prediction in enumerate(trees):
                # print(prediction)
                if str(prediction) == sample.get_raw_labels("tree"):
                    print("found match among k best at", i)
                    if config.dop:
                        print([dopgrammar.match(t) for t in trees])
            # print(sample.get_raw_labels("tree"))
            if len(trees) < config.k:
                print("found only", trees, "instances")
    print(evaluator.summary())
    print(evaluator.breakdowns())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--param", help="evalb parameter file for score calculation", type=str, required=False, default="../disco-dop/proper.prm")
    sub.add_argument("--weighted", type=int, default=1)
    sub.add_argument("--k", type=int, default=1)
    sub.add_argument("--range", type=int, nargs=2, default=None)
    sub.add_argument("--seed", type=int, default=None)
    sub.add_argument("--snd-order", action="store_true", default=False)
    sub.add_argument("--snd-order-prior", type=int, default=0)
    sub.add_argument("--snd-order-separate", action="store_true", default=False)
    sub.add_argument("--dop", action="store_true", default=False)
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)