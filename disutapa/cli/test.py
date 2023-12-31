from argparse import ArgumentParser, Namespace
from disutapa.autotree import with_pos, fix_rotation
from disutapa.tagging.data import DatasetWrapper
from disutapa.tagging.ensemble_model import ParserAdapter
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree
from tqdm import tqdm

from datasets import DatasetDict
from pickle import load

import torch
from itertools import islice

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
    torch.manual_seed(0)
    evaluator = Evaluator(readparam(config.param))
    corpus = DatasetDict.load_from_disk(config.corpus)
    data = DatasetWrapper(corpus["dev"])
    p = ParserAdapter(data.get_grammar(), total_limit=config.weighted)
    nrules = len(p.parser.grammar.rules)
    idtopos = data.labels("pos")
    datalen = len(data)
    data = enumerate(data)

    if not config.dop is None:
        dopgrammar = load(open(config.dop, "rb"))

    for i, sample in tqdm(data, total=datalen):
        if not config.range is None and not i in range(*config.range):
            continue

        weights, indices = guess_weights(nrules, sample.get_raw_labels("supertag"), config.weighted).topk(config.weighted, sorted=True)
        weights = -weights.log_softmax(dim=-1)
        
        goldpostags = [idtopos[i] for i in sample.get_raw_labels("pos")]
        goldtree = ParentedTree(sample.get_raw_labels("tree"))

        p.fill_chart(len(sample), weights.cpu().numpy(), indices.cpu().numpy())
        if config.k == 1:
            prediction = with_pos(p.get_best()[0], goldpostags)
            evaluator.add(i, goldtree, list(sample.get_raw_labels("sentence")),
                    ParentedTree.convert(prediction), list(sample.get_raw_labels("sentence")))
        else:
            trees = islice(p.get_best_iter(), config.k)
            trees = [fix_rotation(with_pos(t[0], goldpostags))[1] for t, w in trees]
            if not config.dop is None and len(trees) > 1:
                trees.sort(key=dopgrammar.match, reverse=False)
            evaluator.add(i, goldtree, list(sample.get_raw_labels("sentence")),
                ParentedTree.convert(trees[0]), list(sample.get_raw_labels("sentence")))
            if not str(trees[0]) == sample.get_raw_labels("tree"):
                # print("first candidate is not gold, but index", \
                #     next(i for i, prediction in enumerate(trees)
                #         if str(prediction) == sample.get_raw_labels("tree")))
                print(trees)
                print([dopgrammar.match(t) for t in trees])
    print(evaluator.summary())
    print(evaluator.breakdowns())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--param", help="evalb parameter file for score calculation", type=str, required=False, default="resources/disco-dop/proper.prm")
    sub.add_argument("--weighted", type=int, default=1)
    sub.add_argument("--k", type=int, default=1)
    sub.add_argument("--range", type=int, nargs=2, default=None)
    sub.add_argument("--seed", type=int, default=None)
    sub.add_argument("--dop", type=str, help="trained dop automaton")
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)