from argparse import ArgumentParser
from pickle import load

import flair
import torch

from disutapa.tagging.ensemble_model import EnsembleModel, ParserAdapter, oracle_tree, float_or_zero
from disutapa.tagging.data import CorpusWrapper
from disutapa.autotree import with_pos, fix_rotation
from discodop.eval import Evaluator, Tree, TreePairResult
from discodop.tree import ParentedTree

from itertools import islice
from pickle import dump
from tqdm import tqdm

def oraclescore(gold, pred, leaves, param) -> float:
    res = TreePairResult(0, gold, list(leaves), pred, list(leaves), param)
    return float_or_zero(res.scores()["LF"])


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    testset = corpus.dev
    model: EnsembleModel = EnsembleModel.load(config.model)
    model.reranking = load(open(config.reranking, "rb"))
    print(model.reranking)
    model.eval()
    if config.maxk is None:
        config.maxk = model.config.ktags
    if config.step is None:
        config.step = model.config.step
    if config.maxn is None:
        config.maxn = model.config.ktrees

    assert config.maxn > 1

    parser = ParserAdapter(model.__grammar__, step=config.step, total_limit=config.maxk)
    first_eval = Evaluator(model.config.evalparam)
    oracle_eval = Evaluator(model.config.evalparam)
    dop_eval = Evaluator(model.config.evalparam)

    chose_better = 0
    chose_worse = 0
    chose_same = 0
    sentid = 0
    for sentence in tqdm(testset):
        with torch.no_grad():
            scores = dict((k,v[0]) for k,v in model.forward(model._batch_to_embeddings([sentence], batch_first=True)))
            topweights, toptags = scores["supertag"][:,1:].topk(config.maxk, dim=-1, sorted=True)
            toptags += 1
            topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
            pos = [model.dictionaries["pos"].get_item_for_index(p) for p in scores["pos"].argmax(dim=-1)[:len(sentence)]]
            gold = Tree(sentence.get_raw_labels("tree"))

            parser.fill_chart(len(sentence), topweights[:,:parser.total_limit].cpu().numpy(), (toptags[:,:parser.total_limit]-1).cpu().numpy())
            leaves = list(str(i) for i in range(len(sentence)))
            if parser.parser.numparses() <= 1:
                continue
            predlist1 = [(fix_rotation(with_pos(d[0], pos))[1], w) for d, w in islice(parser.parser.get_best_iter(), config.maxn)]
            predlist1, ws = zip(*predlist1)
            oracle_scores = [oraclescore(ParentedTree.convert(gold), ParentedTree.convert(p), leaves, model.config.evalparam) for p in predlist1]
            dop_scores = [model.reranking.match(p) for p in predlist1]
            oindex, otree = oracle_tree([ParentedTree.convert(p) for p in predlist1], sentence.get_raw_labels("tree"), model.config.evalparam)
            dindex, dtree = model.reranking.select((ParentedTree.convert(p), 0) for p in predlist1)
            print(oracle_scores, dop_scores, ws)
            print("chose", (dindex, oracle_scores[dindex], ws[dindex]), f"instead of {(oindex, oracle_scores[oindex], ws[oindex])}" if oindex != dindex else "")
            oracle_eval.add(sentid, ParentedTree.convert(gold), list(leaves), otree, list(leaves))
            dop_eval.add(sentid, ParentedTree.convert(gold), list(leaves), dtree, list(leaves))
            first_eval.add(sentid, ParentedTree.convert(gold), list(leaves), ParentedTree.convert(predlist1[0]), list(leaves))
            sentid += 1
            
            chose_same += int(dindex == 0)
            chose_better += int(oracle_scores[dindex] > oracle_scores[0])
            chose_worse += int(oracle_scores[dindex] < oracle_scores[0])
    print("chose first tree in", chose_same, "cases")
    print("chose better tree in", chose_better, "cases")
    print("chose worse tree in", chose_worse, "cases")
    print("via first:", float_or_zero(first_eval.acc.scores()['lf']))
    print("via dop:", float_or_zero(dop_eval.acc.scores()['lf']))
    print("via oracle:", float_or_zero(oracle_eval.acc.scores()['lf']))




def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("reranking", type=str)
    sub.add_argument("output", type=str, nargs="?")
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--step", type=float, default=None)
    sub.add_argument("--maxk", type=int, default=None)
    sub.add_argument("--maxn", type=int, default=None)
    sub.set_defaults(func=lambda args: main(args))