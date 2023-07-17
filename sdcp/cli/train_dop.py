from argparse import ArgumentParser, Namespace
from sdcp.tagging.data import DatasetWrapper
from tqdm import tqdm

from datasets import DatasetDict
from sdcp.reranking.dop import Dop, Tree
from pickle import dump


def main(config: Namespace):
    corpus = DatasetDict.load_from_disk(config.corpus)
    if config.hmarkov is None and config.vmarkov is None and config.bin is None:
        markov = None
    else:
        markov =(
            config.bin or "right",
            config.hmarkov if not config.hmarkov is None else 999,
            config.vmarkov if not config.vmarkov is None else 1
        )
    dopgrammar = Dop(tqdm(
        (Tree(sentence.get_raw_labels("tree"))
            for sentence in DatasetWrapper(corpus["train"])),
        total=len(corpus["train"]),
        desc="reading tree corpus"),
        markovization=markov
    )
    dump(dopgrammar, open(config.output, "wb"))


def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("output", help="grammar file", type=str, default="2dop.automaton", nargs="?")
    sub.add_argument("--hmarkov", help="horizontal markov window", type=int)
    sub.add_argument("--vmarkov", help="vertical markov window", type=int)
    sub.add_argument("--bin", help="binarization factor direction", type=str)
    sub.set_defaults(func=lambda args: main(args))


if __name__ == "__main__":
    args = ArgumentParser()
    subcommand(args)
    parsed_args = args.parse_args()
    parsed_args.func(parsed_args)