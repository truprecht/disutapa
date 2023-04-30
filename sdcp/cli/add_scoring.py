from argparse import ArgumentParser
from dataclasses import dataclass, fields, MISSING, field
from tqdm import tqdm
from math import ceil

import flair
import torch

from sdcp.grammar.sdcp import grammar, sdcp_clause, rule
from sdcp.tagging.ensemble_model import ModelParameters, EnsembleModel
from sdcp.tagging.data import CorpusWrapper

@dataclass
class Parameter:
    model: str
    corpus: str
    epochs: int = 32
    lr: float = 1e-5
    batch: int = 16
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    output_dir: str = "/tmp/sdcp-training-scoring"
    scoring: str = None
    scoring_options: list[str] = field(default_factory=list)
    abort_nongold_prob: float = 0.9
    ktags: int = 1


def main(config: Parameter):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    model = EnsembleModel.load(config.model)
    model.abort_brass = config.abort_nongold_prob
    model.__ktags__ = config.ktags
    if config.cache:
        for j in tqdm(range(ceil(len(corpus.train)/config.batch)), desc="precomputing parses for each sentence"):
            start = j*config.batch
            end = min(len(corpus.train), (j+1)*config.batch)
            model.cache_scoring_items(corpus.train[start:end])
    if not config.scoring is None:
        model.set_scoring(config.scoring, corpus.train, config.scoring_options, abort_brass = config.abort_nongold_prob)
    elif not model.scoring.requires_training:
        raise Exception("Cannot train scoring object:", model.scoring)
    if config.fix_embedding:
        model.fix_tagging()

    trainer = flair.trainers.ModelTrainer(model, corpus)
    trainer.train(
        config.output_dir,
        learning_rate=config.lr,
        mini_batch_size=config.batch,
        max_epochs=config.epochs,
        weight_decay=config.weight_decay,
        optimizer=torch.optim.__dict__[config.optimizer],
        checkpoint=True,
        use_final_model_for_eval=True,
        patience=config.epochs
        #scheduler=torch.optim.lr_scheduler.OneCycleLR
    )


def subcommand(sub: ArgumentParser):
    for f in [f for f in fields(Parameter)]:
        optional = f.default is MISSING and f.default_factory is MISSING
        name = f.name if optional else f"--{f.name}"
        default = None if optional else f.default
        ftype = f.type
        nargs = None
        if f.type == list[str]:
            ftype = str
            nargs = "+"
            default = list()
        sub.add_argument(name, type=ftype, default=default, nargs=nargs)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.add_argument("--cache", action="store_true", default=False)
    sub.add_argument("--fix-embedding", action="store_true", default=False)
    sub.set_defaults(func=lambda args: main(args))