from dataclasses import dataclass, fields, MISSING
from argparse import ArgumentParser

import flair
import torch
from tqdm import tqdm

from sdcp.grammar.sdcp import grammar, sdcp_clause, rule
from sdcp.grammar.lcfrs import lcfrs_composition, ordered_union_composition
from sdcp.tagging.ensemble_model import ModelParameters, EnsembleModel
from sdcp.tagging.data import CorpusWrapper
from sdcp.reranking.classifier import TreeRanker, Tree

from .train import ModelParameters, TrainingParameter
from os.path import exists
from pickle import dump, load


def get_trained_model(outdir, corpus, config):
    if exists(outdir + "/final-model.pt"):
        return EnsembleModel.load(outdir + "/final-model.pt")

    model = EnsembleModel.from_corpus(
        corpus.train,
        grammar([eval(t) for t in corpus.train.labels()]),
        ModelParameters(embedding=config.embedding, embedding_options=config.embedding_options, ktags=config.ktags, dropout=config.dropout, scoring=config.scoring, scoring_options=config.scoring_options)
    )
    trainer = flair.trainers.ModelTrainer(model, corpus)
    trainer.train(
        outdir,
        learning_rate=config.lr,
        mini_batch_size=config.batch,
        mini_batch_chunk_size=config.micro_batch,
        max_epochs=config.epochs,
        weight_decay=config.weight_decay,
        optimizer=torch.optim.__dict__[config.optimizer],
        patience=config.epochs,
        save_final_model=True
    )

    return model


def get_dev_set(config, corpus):
    if config.dev_model is None:
        return None
    devlist_with_multiple_trees  = []
    model = EnsembleModel.load(config.dev_model)
    iterator = tqdm(
        flair.datasets.DataLoader(corpus.dev, batch_size=config.batch, num_workers=1),
        desc=f"parsing sentences in preparation for dev")
    for batch in iterator:
        model.predict(batch, store_kbest=config.ktrees)
        for sentence in batch:
            if len(sentence.get_raw_prediction("kbest-trees")) <= 1:
                continue
            devlist_with_multiple_trees.append((
                sentence.get_raw_prediction("kbest-trees"),
                Tree(sentence.get_raw_labels("tree"))
            ))
    return devlist_with_multiple_trees


def main(config: TrainingParameter):
    vectorfile = config.output_dir + f"/vectors.tr"
    rankerfile = config.output_dir + f"/trained_ranker.tr"

    if not config.device is None:
        flair.device = config.device
    torch.manual_seed(config.random_seed)
    corpus = CorpusWrapper(config.corpus)

    if not exists(vectorfile):
        ranker = TreeRanker(config.min_feature_occurrence)
        for i, subcorpus in enumerate(corpus.train.get_fold_corpora(config.folds)):
            model = get_trained_model(config.output_dir + f"/fold-{i}", subcorpus, config)
            iterator = tqdm(
                flair.datasets.DataLoader(corpus.dev, batch_size=config.batch, num_workers=1),
                desc=f"parsing sentences ({i}) in preparation for training")
            for batch in iterator:
                model.predict(batch, store_kbest=config.ktrees)
                for sentence in batch:
                    ranker.add_tree(
                        Tree(sentence.get_raw_labels("tree")),
                        sentence.get_raw_prediction("kbest-trees")
                    )
        dump(ranker, open(vectorfile, "wb"))
    else:
        ranker = load(open(vectorfile, "rb"))

    ranker.fit(devset=get_dev_set(config, corpus))
    dump(ranker, open(rankerfile, "wb"))




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
    sub.add_argument("--min_feature_occurrence", type=int, default=1)
    sub.add_argument("--folds", type=int, default=10)
    sub.add_argument("--dev_model", type=str)
    sub.set_defaults(func=lambda args: main(args))