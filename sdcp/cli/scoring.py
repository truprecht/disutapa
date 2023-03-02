from argparse import ArgumentParser, Namespace
from sdcp.grammar.sdcp import rule
from sdcp.tagging.data import DatasetWrapper

from datasets import DatasetDict
from collections import Counter, defaultdict



def main(config: Namespace):
    corpus = DatasetWrapper(DatasetDict.load_from_disk(config.corpus)[config.portion])
    combinations: dict[tuple[int], int] = defaultdict(lambda: 0)
    denominator: dict[tuple[int], int] = defaultdict(lambda: 0)
    cnt_by_rhs = defaultdict(lambda: 0)

    for supertag in corpus.labels():
        sobj: rule = eval(supertag)
        if sobj.rhs:
            if config.separate and len(sobj.rhs) == 2:
                cnt_by_rhs[(sobj.rhs[0], None)] += 1
                cnt_by_rhs[(None, sobj.rhs[1])] += 1
            else:
                cnt_by_rhs[sobj.rhs] += 1

    for sentence in corpus:
        deriv = sentence.get_derivation()
        for node in deriv.subtrees():
            if not node.children:
                continue
            if config.separate and len(node) == 2:
                combinations[(node.label[0], node[0].label[0], None)] += 1
                combinations[(node.label[0], None, node[1].label[0])] += 1
                denominator[(node[0].label[0], None)] += 1
                denominator[(None, node[1].label[0])] += 1
            else:
                combinations[(node.label[0], *(c.label[0] for c in node))] += 1
                denominator[tuple(c.label[0] for c in node)] += 1

    tot = 0
    coms = Counter()
    for _, v in combinations.items():
        tot += v
        coms[v] += 1
    print("total:", tot, "occurrences of", len(combinations), "combinations")
    print(coms[1], "combinations occur only once")

def subcommand(sub: ArgumentParser):
    sub.add_argument("corpus", help="file containing gold tags", type=str)
    sub.add_argument("--portion", type=str, default="dev", choices=["train", "dev", "test"])
    sub.add_argument("--separate", action="store_true", default=False)
    sub.set_defaults(func=lambda args: main(args))