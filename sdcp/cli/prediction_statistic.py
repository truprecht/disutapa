from argparse import ArgumentParser
from pickle import load

import flair
import torch

from sdcp.tagging.ensemble_model import EnsembleModel
from sdcp.tagging.data import CorpusWrapper

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

    gold_placement = []
    gold_distance = []
    after_gold_distance = []
    snd_distance = []

    for sentence in tqdm(testset):
        scores = next(t for f, t in model.forward(model._batch_to_embeddings([sentence], batch_first=True)) if f == "supertag")
        idxlist = scores[0].argsort(descending=True)
        plcmnts, dists, snd, aftgld = [], [], [], []
        for scs, idxs, gold in zip(scores[0], idxlist, sentence.get_raw_labels("supertag")):
            gold += 1
            snd.append((scs[idxs[0]]-scs[idxs[1]]).item())
            if gold >= len(scs):
                # tag only appears in dev set
                plcmnts.append(None)
                dists.append(None)
                aftgld.append(None)
                continue
            goldidx = (idxs == gold).nonzero().squeeze().item()
            plcmnts.append(goldidx)
            dists.append((scs[idxs[0]]-scs[gold]).item())
            aftgld.append((scs[idxs[0]]-scs[idxs[goldidx+1]]).item() if goldidx < len(idxs) else None)

        gold_placement.append(plcmnts)
        gold_distance.append(dists)
        snd_distance.append(snd)
        after_gold_distance.append(aftgld)

    maxidxs = torch.tensor([max(p for p in ps if not p is None) for ps in gold_placement])
    print("indices:", maxidxs.max(), "(max)")
    print("80% indices are below", percentile(maxidxs, 0.8)+1)
    print("90% indices are below", percentile(maxidxs, 0.9)+1)
    print("99% indices are below", percentile(maxidxs, 0.99)+1)
    print()
    maxcfd = torch.tensor([max(p for p in ps if not p is None) for ps in gold_distance])
    print("score distance:", "max", maxcfd.max(), "min", maxcfd.min())
    print("80% values are at or below", percentile(maxcfd, 0.8))
    print("90% values are at or below", percentile(maxcfd, 0.9))
    print("99% values are at or below", percentile(maxcfd, 0.99))
    print()
    dists = torch.tensor([d for ds in snd_distance for d in ds if not d is None])
    print("score difference to snd most confident prediction")
    print("max", dists.max().item(), "min", dists.min().item())
    print("mean", dists.sum().item() / len(dists), "stddev", dists.std().item())
    print()

    if not config.output is None:
        with open(config.output, "wb") as outfile:
            obj = {"goldidx": gold_placement, "gold_score": gold_distance, "snd_score": snd_distance, "after_gold": after_gold_distance}
            dump(obj, outfile)



def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("output", type=str, nargs = "?")
    sub.add_argument("--device", type=torch.device, default=None)
    sub.set_defaults(func=lambda args: main(args))