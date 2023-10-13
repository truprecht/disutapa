from argparse import ArgumentParser
from pickle import load

import flair
import torch

from sdcp.tagging.ensemble_model import EnsembleModel
from sdcp.tagging.data import CorpusWrapper

from pickle import dump


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    testset = corpus.dev
    model: EnsembleModel = EnsembleModel.load(config.model)

    gold_placement = []
    gold_distance = []
    after_gold_distance = []
    snd_distance = []

    for sentence in testset:
        scores = next(t for f, t in model.forward(model._batch_to_embeddings([sentence], batch_first=True)) if f == "supertag")
        idxlist = scores[0].argsort(descending=True)
        for scs, idxs, gold in zip(scores[0], idxlist, sentence.get_raw_labels("supertag")):
            gold += 1
            if gold >= len(scs):
                # tag only appears in dev set
                continue
            goldidx = (idxs == gold).nonzero().squeeze().item()
            gold_placement.append(goldidx)
            gold_distance.append((scs[idxs[0]]-scs[gold]).item())
            snd_distance.append((scs[idxs[0]]-scs[idxs[1]]).item())
            if goldidx < len(idxs):
                after_gold_distance.append((scs[idxs[0]]-scs[idxs[goldidx+1]]).item())
    print(gold_placement)
    print(gold_distance)
    print(after_gold_distance)

    with open(config.output, "wb") as outfile:
        obj = {"goldidx": gold_placement, "gold_score": gold_distance, "snd_score": snd_distance, "after_gold": after_gold_distance}
        dump(obj, outfile)



def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("output", type=str)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.set_defaults(func=lambda args: main(args))