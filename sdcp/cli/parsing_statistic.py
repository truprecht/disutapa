from argparse import ArgumentParser
from pickle import load

import flair
import torch

from sdcp.tagging.ensemble_model import EnsembleModel, ParserAdapter, oracle_tree, float_or_zero
from sdcp.tagging.data import CorpusWrapper
from sdcp.autotree import with_pos, fix_rotation
from discodop.eval import Evaluator, Tree
from discodop.tree import ParentedTree

from pickle import dump
from tqdm import tqdm

def percentile(vs: torch.Tensor, p: float):
    values = sorted(set(vs))
    numoccs = p * len(vs)
    return next(i.item() for i in values if (vs <= i).sum() >= numoccs)


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    testset = corpus.dev
    model: EnsembleModel = EnsembleModel.load(config.model)
    model.eval()
    if config.maxks is None:
        config.maxks = [model.config.ktags]
    if config.steps is None:
        config.steps = [model.config.step]
    maximumk = max(config.maxks)

    parsers = [
        (s,k, ParserAdapter(model.__grammar__, step=s, total_limit=k))
        for s in config.steps
        for k in config.maxks
    ]

    wdiffs = {(s,k): [] for s in config.steps for k in config.maxks}
    chosen_idcs  = {(s,k): [] for s in config.steps for k in config.maxks}
    oracle_eval  = {(s,k): Evaluator(model.config.evalparam) for s in config.steps for k in config.maxks}
    first_eval  = {(s,k): Evaluator(model.config.evalparam) for s in config.steps for k in config.maxks}
    noparses  = {(s,k): [] for s in config.steps for k in config.maxks}
    oracle_score, first_score = {}, {}

    i = 0
    for sentence in tqdm(testset):
        with torch.no_grad():
            scores = next(t for f, t in model.forward(model._batch_to_embeddings([sentence], batch_first=True)) if f == "supertag")
            scores = dict((k,v[0]) for k,v in model.forward(model._batch_to_embeddings([sentence], batch_first=True)))
            topweights, toptags = scores["supertag"][:,1:].topk(maximumk, dim=-1, sorted=True)
            toptags += 1
            topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
            pos = [model.dictionaries["pos"].get_item_for_index(p) for p in scores["pos"].argmax(dim=-1)[:len(sentence)]]

            for (s,k,parser) in parsers:
                parser.fill_chart(len(sentence), topweights[:,:parser.total_limit].cpu().numpy(), (toptags[:,:parser.total_limit]-1).cpu().numpy())
                wd, chid = [], []
                leaves = list(str(i) for i in range(len(sentence)))
                noparses[(s,k)].append(parser.parser.numparses())
                if noparses[(s,k)][-1] > 0:
                    for lexi, rulei in sorted((t.label[1], t.label[0]) for t in parser.parser.get_best_derivation().subtrees()):
                        chid.append(next(i for i, ri in enumerate(toptags[lexi]) if ri-1 == rulei))
                        wd.append(topweights[lexi, chid[-1]] - topweights[lexi, 0])
                    wdiffs[(s,k)].append(wd)
                    chosen_idcs[(s,k)].append(chid)
                    predlist = [ParentedTree.convert(fix_rotation(with_pos(d[0], pos))[1]) for d, _ in parser.parser.get_best_iter()]
                    _, otree = oracle_tree(predlist, sentence.get_raw_labels("tree"), model.config.evalparam)
                else:
                    otree = ParentedTree("NOPARSE", [ParentedTree(p, [i]) for i,p in enumerate(pos)])
                    predlist = [ParentedTree.convert(otree)]
                oracle_eval[(s,k)].add(i, ParentedTree(sentence.get_raw_labels("tree")), list(leaves), ParentedTree.convert(otree), list(leaves))
                first_eval[(s,k)].add(i, ParentedTree(sentence.get_raw_labels("tree")), list(leaves), predlist[0], list(leaves))
    for s in config.steps:
        for k in config.maxks:
            oracle_score[(s,k)] = float_or_zero(oracle_eval[(s,k)].acc.scores()['lf'])
            first_score[(s,k)] = float_or_zero(first_eval[(s,k)].acc.scores()['lf'])

            print("-------------")
            print(f"k = {k}, s = {s}")
            print(f"score via first tree: {first_score[(s,k)]}, score via oracle in chart: {oracle_score[(s,k)]}")

            print("no. of noparse:", sum(1 if n == 0 else 0 for n in noparses[(s,k)]))
            maxidxs = torch.tensor([max(p for p in ps if not p is None) for ps in chosen_idcs])
            print("indices:", maxidxs.max().item(), "(max)")
            print("80% indices are below", percentile(maxidxs, 0.8)+1)
            print("90% indices are below", percentile(maxidxs, 0.9)+1)
            print("99% indices are below", percentile(maxidxs, 0.99)+1)
            print()
            maxcfd = torch.tensor([max(p for p in ps if not p is None) for ps in wdiffs])
            print("score distance:", "max", maxcfd.max().item(), "min", maxcfd.min().item())
            print("80% values are at or below", percentile(maxcfd, 0.8))
            print("90% values are at or below", percentile(maxcfd, 0.9))
            print("99% values are at or below", percentile(maxcfd, 0.99))
            print()
            parses = torch.tensor(noparses[(s,k)])
            print("maximum number of parses in chart", parses.max().item())
            print("in 80% cases there are less or equal to", percentile(parses, 0.8), "parses")
            print("in 90% cases there are less or equal to", percentile(parses, 0.9), "parses")
            print("in 99% cases there are less or equal to", percentile(parses, 0.99), "parses")
            print()
    

    if not config.output is None:
        with open(config.output, "wb") as outfile:
            obj = {"chosen_tag_weight": wdiffs, "chosen_tag_index": chosen_idcs, "noparses": noparses}
            dump(obj, outfile)



def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("output", type=str, nargs="?")
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--steps", nargs="+", type=int, required=False, default=None)
    sub.add_argument("--maxks", nargs="+", type=int, required=False, default=None)
    sub.set_defaults(func=lambda args: main(args))