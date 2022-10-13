from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.buparser import BuParser
from sdcp.autotree import AutoTree
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree
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
    seed(config.seed )
    evaluator = Evaluator(readparam(config.param))
    data = DatasetDict.load_from_disk(config.corpus)["dev"]
    p = BuParser(grammar([eval(str_hr) for str_hr in data.features["supertag"].feature.names]))
    idtopos = data.features["pos"].feature.names
    datalen = len(data) if config.range is None else config.range[1]-config.range[0]
    data = enumerate(data)
    if not config.range is None:
        data = ((i,s) for i,s in data if i in range(*config.range))
    for i, sample in tqdm(data, total=datalen):
        p.init(
            *(rule_vector(len(p.grammar.rules), config.weighted, i) for i in sample["supertag"]),
        )
        p.fill_chart()
        prediction = p.get_best()
        prediction = AutoTree(prediction)
        evaluator.add(i, ParentedTree(sample["tree"]), list(sample["sentence"]),
                ParentedTree.convert(prediction.tree([idtopos[i] for i in sample["pos"]])), list(sample["sentence"]))
    print(evaluator.summary())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--param", help="evalb parameter file for score calculation", type=str, required=False, default="../disco-dop/proper.prm")
    sub.add_argument("--weighted", type=int, default=1)
    sub.add_argument("--range", type=int, nargs=2, default=None)
    sub.add_argument("--seed", type=int, default=None)
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)