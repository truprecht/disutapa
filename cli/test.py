from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.parser import parser
from sdcp.autotree import AutoTree
from discodop.eval import Evaluator, readparam
from discodop.tree import ParentedTree

from datasets import DatasetDict


def main(config: Namespace):
    evaluator = Evaluator(readparam(config.param))
    data = DatasetDict.load_from_disk(config.corpus)["dev"]
    p = parser(grammar([eval(str_hr) for str_hr in data.features["supertag"].feature.names]))
    idtopos = data.features["pos"].feature.names
    for i, sample in enumerate(data):
        p.init(
            *([i] for i in sample["supertag"]),
        )
        p.fill_chart()
        prediction = AutoTree(p.get_best())
        print(prediction.tree([idtopos[i] for i in sample["pos"]]))
        evaluator.add(i, ParentedTree.convert(AutoTree(sample["tree"]).tree([idtopos[i] for i in sample["pos"]])), list(sample["sentence"]),
                ParentedTree.convert(prediction.tree([idtopos[i] for i in sample["pos"]])), list(sample["sentence"]))
    print(evaluator.summary())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--param", help="evalb parameter file for score calculation", type=str, required=False, default="../disco-dop/proper.prm")
    sub.set_defaults(func=lambda args: main(args))