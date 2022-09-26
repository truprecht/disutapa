from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.parser import LeftCornerParser
from sdcp.autotree import AutoTree
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree
from tqdm import tqdm

from datasets import DatasetDict
from random import sample, shuffle

def rule_vector(total: int, k: int, hot: int):
    vec = sample(range(total), k-1)
    vec.append(hot)
    shuffle(vec)
    return [(rid, -abs(hot-rid)) for rid in vec]


def main(config: Namespace):
    evaluator = Evaluator(readparam(config.param))
    data = DatasetDict.load_from_disk(config.corpus)["dev"]
    p = LeftCornerParser(grammar([eval(str_hr) for str_hr in data.features["supertag"].feature.names]))
    idtopos = data.features["pos"].feature.names
    for i, sample in tqdm(enumerate(data), total=len(data)):
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
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)