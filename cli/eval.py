from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser

import flair
import torch

from sdcp.grammar.sdcp import grammar, sdcp_clause, rule
from tagging.model import ModelParameters, TaggerModel
from tagging.data import CorpusWrapper


def main(config):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    corpus = corpus.dev if config.dev else corpus.test
    model = TaggerModel.load(config.model)
    if config.ktags:
        model.__ktags__ = config.ktags
    results = model.evaluate(corpus)
    print(results.log_header)
    print(results.log_line)
    print(results.detailed_results)


def subcommand(sub: ArgumentParser):
    sub.add_argument("model", type=str)
    sub.add_argument("corpus", type=str)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--dev", action="store_true", default=False)
    sub.add_argument("--ktags", type=int, default=None)
    sub.set_defaults(func=lambda args: main(args))