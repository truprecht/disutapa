from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser

import flair
import torch

from sdcp.tagging.ensemble_model import ModelParameters, EnsembleModel
from sdcp.tagging.data import CorpusWrapper

@dataclass
class TrainingParameter:
    corpus: str
    epochs: int = 32
    lr: float = 5e-5
    batch: int = 32
    micro_batch: int = None
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    output_dir: str = "/tmp/sdcp-training"
    random_seed: int = 0



def main(config: TrainingParameter):
    if not config.device is None:
        flair.device = config.device
    torch.manual_seed(config.random_seed)
    corpus = CorpusWrapper(config.corpus)
    model = EnsembleModel.from_corpus(
        corpus.train,
        corpus.train.get_grammar(),
        config
    )
    trainer = flair.trainers.ModelTrainer(model, corpus)
    trainer.train(
        config.output_dir,
        learning_rate=config.lr,
        mini_batch_size=config.batch,
        mini_batch_chunk_size=config.micro_batch,
        max_epochs=config.epochs,
        weight_decay=config.weight_decay,
        optimizer=torch.optim.__dict__[config.optimizer],
        checkpoint=True,
        use_final_model_for_eval=True,
        patience=config.epochs
    )


def subcommand(sub: ArgumentParser):
    for f in [f for f in fields(TrainingParameter) if not f.name == "model"] \
            + [f for f in fields(ModelParameters)]:
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
    sub.set_defaults(func=lambda args: main(args))