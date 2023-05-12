from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule, sdcp_clause, grammar
from sdcp.grammar.parser.activeparser import ActiveParser, headed_rule
from sdcp.grammar.lcfrs import lcfrs_composition, ordered_union_composition
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

from dataclasses import dataclass
from pickle import load, dump
@dataclass
class supertags:
    stagandweight: tuple[tuple[rule, float], ...]
    postags: tuple[str]

    @classmethod
    def from_file(cls, filename) -> list["supertags"]:
        with open(filename, "rb") as file:
            return load(file)
        
    @classmethod
    def to_file(cls, stags: list["supertags"], filename: str):
        with open(filename, "wb") as file:
            return dump(stags, file)


def main(config: Namespace):
    stags = supertags.from_file(config.weights)
    tuletoid = dict()
    

    evaluator = Evaluator(readparam(config.param))
    data = DatasetWrapper(DatasetDict.load_from_disk(config.corpus)["train"])
    p = ActiveParser(grammar([eval(str_hr) for str_hr in data.labels()]))
    idtopos = data.labels("pos")

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
        # if str(fix_rotation(prediction)[1]) != sample.get_raw_labels("tree"):
        print(sample.get_raw_labels("tree"))
        print(prediction)
    print(evaluator.summary())
    print(evaluator.breakdowns())


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("weights", help="file containing pseudo-predicted supertags with confidence value", type=str)
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)