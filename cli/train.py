from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser

import flair
import torch

from sdcp.grammar.sdcp import grammar, sdcp_clause, rule
from tagging.model import ModelParameters, TaggerModel
from tagging.data import CorpusWrapper

@dataclass
class TrainingParameter:
    corpus: str
    epochs: int = 10
    lr: float = 0.1
    batch: int = 4
    micro_batch: int = None
    weight_decay: float = 0.0
    optimizer: str = "AdamW"
    output_dir: str = "/tmp/sdcp-training"


def main(config: TrainingParameter):
    if not config.device is None:
        flair.device = config.device
    corpus = CorpusWrapper(config.corpus)
    model = TaggerModel.from_corpus(
        corpus.train,
        grammar([eval(t) for t in corpus.train.labels()]),
        ModelParameters(embeddings=config.embeddings, ktags=config.ktags, dropout=config.dropout)
    )
    trainer = flair.trainers.ModelTrainer(model, corpus)
    train = trainer.fine_tune if any(em.fine_tune() for em in model.embedding_builder) else \
                trainer.train
    train(
        config.output_dir,
        learning_rate=config.lr,
        mini_batch_size=config.batch,
        mini_batch_chunk_size=config.micro_batch,
        max_epochs=config.epochs,
        weight_decay=config.weight_decay,
        optimizer=torch.optim.__dict__[config.optimizer],
        scheduler=torch.optim.lr_scheduler.OneCycleLR
    )


def subcommand(sub: ArgumentParser):
    for f in [f for f in fields(TrainingParameter) if not f.name == "model"] \
            + [f for f in fields(ModelParameters)]:
        optional = f.default is MISSING and f.default_factory is MISSING
        name = f.name if optional else f"--{f.name}"
        default = None if optional else f.default
        sub.add_argument(name, type=f.type, default=default)
    sub.add_argument("--device", type=torch.device, default=None)
    sub.set_defaults(func=lambda args: main(args))