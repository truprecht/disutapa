from argparse import ArgumentParser

import flair
import torch

from sdcp.tagging.ensemble_model import EnsembleModel
from sdcp.tagging.data import CorpusWrapper
from sdcp.reranking.classifier import TreeRanker


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    model = EnsembleModel.load(config.model)
    model.set_config("ktrees", config.ktrees)

    ranker = TreeRanker(config.min_feature_occurrence)
    model.add_reranker(ranker, corpus.train, config.epochs, dev_set=corpus.dev)
    results = model.evaluate(corpus.dev, progressbar=True, kbest_oracle=config.ktrees)
    model.save(config.model + ".reranked")
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)


def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--ktags", type=int)
    sub.add_argument("--ktrees", type=int, default=50)
    sub.add_argument("--epochs", type=int, default=5)
    sub.add_argument("--min_feature_occurrence", type=int, default=1)
    sub.set_defaults(func=lambda args: main(args))