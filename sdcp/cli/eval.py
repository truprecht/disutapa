from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser
from pickle import load

import flair
import torch

from sdcp.grammar.sdcp import grammar, sdcp_clause, rule
from sdcp.tagging.ensemble_model import ModelParameters, EnsembleModel
from sdcp.tagging.data import CorpusWrapper


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    corpus = corpus.dev if config.dev else corpus.test
    model: EnsembleModel = EnsembleModel.load(config.model)
    for field in (f for f in ("ktags", "ktrees", "step") if not config.__dict__[f] is None):
        model.set_config(field, config.__dict__[field])
    if config.reranking:
        treeranker = load(open(config.reranking, "rb"))
        model.reranking = treeranker
    results = model.evaluate(corpus, progressbar=True, kbest_oracle=config.kbest_oracle)
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)


def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--dev", action="store_true", default=False)
    sub.add_argument("--ktags", type=int, default=None)
    sub.add_argument("--kbest-oracle", type=int, default=None)
    sub.add_argument("--reranking", type=str, default=None)
    sub.add_argument("--ktrees", type=int, default=None)
    sub.add_argument("--step", type=float, default=None)
    sub.set_defaults(func=lambda args: main(args))