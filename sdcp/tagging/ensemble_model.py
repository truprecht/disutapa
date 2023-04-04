import torch
import flair
from tqdm import tqdm
from math import ceil

CPU = torch.device("cpu")

from typing import Tuple, Optional
from sdcp.grammar.sdcp import grammar
from sdcp.grammar.ensemble_parser import EnsembleParser
from sdcp.autotree import AutoTree

from discodop.eval import readparam

from .data import DatasetWrapper, SentenceWrapper
from .embeddings import TokenEmbeddingBuilder, EmbeddingPresets, PretrainedBuilder
from .parsing_scorer import ScoringBuilder


from dataclasses import dataclass, field

@dataclass
class ModelParameters:
    embedding: str = "Supervised"
    embedding_options: list[str] = field(default_factory=list)
    scoring: str = None
    scoring_options: list[str] = field(default_factory=list)
    abort_nongold_prob: float = 0.9
    ktags: int = 1
    dropout: float = 0.1
    evalparam: Optional[dict] = None

    def __post_init__(self):
        if self.embedding in EmbeddingPresets:
            self.embedding = EmbeddingPresets[self.embedding]
        else:
            options = eval(f"dict({','.join(self.embedding_options)})")
            self.embedding = [PretrainedBuilder(self.embedding, **options)]
        if self.evalparam is None:
            self.evalparam = readparam("../disco-dop/proper.prm")


class EnsembleModel(flair.nn.Model):
    @classmethod
    def from_corpus(cls, corpus: DatasetWrapper, grammar: grammar, parameters: ModelParameters):
        """ Construct an instance of the model using
            * supertags and pos tags from `grammar`, and
            * word embeddings (as specified in `parameters`) from `corpus`.
        """
        tag_dicts = { k: corpus.build_dictionary(k) for k in ("supertag", "pos") }

        embeddings = [
            embedding.build_vocab(corpus)
            for embedding in parameters.embedding
        ]
        parsing_scorer = ScoringBuilder(parameters.scoring, corpus, *parameters.scoring_options)
        return cls(embeddings, tag_dicts, grammar, parsing_scorer, parameters.dropout, parameters.ktags, parameters.evalparam, parameters.abort_nongold_prob)

    def __init__(self, embeddings, dictionaries, grammar, parsing_scorer, dropout: float = 0.1, ktags: int = 1, evalparam: dict = None, abort_brass: float = 0.9):
        super().__init__()
        self.embedding_builder = embeddings
        self.embedding = flair.embeddings.StackedEmbeddings([
            builder.produce() for builder in embeddings
        ])
        inputlen = self.embedding.embedding_length
        self.scoring_builder = parsing_scorer
        self.scoring = self.scoring_builder.produce(inputlen)
        self.abort_brass = abort_brass

        self.dropoutprob = dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.dictionaries = dictionaries
        self.scores = torch.nn.ModuleDict({
            field: torch.nn.Linear(inputlen, len(dict))
            for field, dict in self.dictionaries.items()
        })

        self.__grammar__ = grammar
        self.__evalparam__ = evalparam
        self.__ktags__ = ktags
        self.to(flair.device)

    def set_eval_param(self, ktags: int, evalparam: dict):
        self.__evalparam__ = evalparam
        self.__ktags__ = ktags

    @property
    def evalparam(self):
        return self.__evalparam__

    @property
    def ktags(self):
        return self.__ktags__

    def label_type(self):
        return "supertag"

    def _batch_to_embeddings(self, batch, batch_first: bool = False):
        if not type(batch) is list:
            batch = [batch]
        self.embedding.embed(batch)
        embedding_name = self.embedding.get_names()
        input = torch.nn.utils.rnn.pad_sequence([
            torch.stack([ word.get_embedding(embedding_name) for word in sentence ])
            for sentence in batch]).to(flair.device)
        if batch_first:
            input = input.transpose(0,1)
        return input


    def _parsing_loss(self, batch: list[SentenceWrapper], embeddings, feats):
        loss = torch.tensor(0.0, device=flair.device)
        npreds = 0

        brassitems = []
        parser = EnsembleParser(self.__grammar__, snd_order_weights=self.scoring.snd_order)
        topweights, toptags = feats.topk(self.ktags, dim=-1)
        topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
        for i, sent in enumerate(batch):
            derivation = sent.get_derivation()
            embeds = embeddings[:len(sent), i]
            predicted_tags = [
                [(tag-1, weight) for tag, weight in zip(ktags, kweights) if tag != 0]
                for ktags, kweights in zip(toptags[:len(sent), i], topweights[:len(sent), i])]
            parser.init(self.scoring, embeds, *predicted_tags)
            parser.add_nongold_filter(derivation, self.abort_brass)
            parser.fill_chart()

            loss += self.scoring.forward_loss(derivation)
            npreds += derivation.inner_nodes

            brassitems = [(parser.items[j], parser.backtraces[j]) for j in parser.brassitems if parser.backtraces[j].children]
            if brassitems:
                loss += self.scoring.norule_loss(brassitems, parser.backtraces)
                npreds += len(brassitems)

        return loss, npreds


    def _calculate_loss(self, feats, batch: list[SentenceWrapper], batch_first: bool = False, check_bounds: bool = False):
        loss = torch.tensor(0.0, device=flair.device)
        for field, logits in feats:
            gold = torch.nn.utils.rnn.pad_sequence([
                torch.tensor(sentence.get_raw_labels(field))+1 for sentence in batch],
                padding_value=-100
            ).to(flair.device)
            if batch_first:
                gold = gold.transpose(0,1)
            if check_bounds:
                gold[gold >= len(self.dictionaries[field])] = 0
            loss += torch.nn.functional.cross_entropy(
                logits.flatten(end_dim=1),
                gold.flatten(end_dim=1),
                reduction = "sum",
                ignore_index = -100
            )
        n_predictions = sum(len(sentence) for sentence in batch)
        return loss, n_predictions


    def forward_loss(self, batch):
        embeds = self._batch_to_embeddings(batch)
        feats = dict(self.forward(embeds))
        loss, predictions = self._calculate_loss(feats.items(), batch)
        if self.scoring.requires_training:
            lparse, nparse = self._parsing_loss(batch, embeds, feats["supertag"])
            loss += lparse
            predictions += nparse
        return loss, predictions


    def forward(self, embeddings):
        inputfeats = self.dropout(embeddings)
        for field, layer in self.scores.items():
            logits = layer(inputfeats)
            yield field, logits #(logits if not batch_first else logits.transpose(0,1))


    def predict(self,
            batch: list[SentenceWrapper],
            label_name: str = None,
            return_loss: bool = False,
            embedding_storage_mode: str = "none",
            supertag_storage_mode: str = "both",
            othertag_storage_mode: bool = True):
        """ Predicts pos tags and supertags for the given sentences and parses
            them.
            :param label_name: the predicted parse trees are stored in each
                sentence's `label_name` label, the predicted supertags are
                stored in `label_name`-tag of each single token.
            :param return_loss: if true, computes and returns the loss. Gold
                supertags and pos-tags are expected in the `supertag` and `pos`
                labels for each token.
            :param embedding_storage_mode: one of "none", "cpu", "store".
                "none" discards embedding predictions after each batch, "cpu"
                sends the tensors to cpu memory.
            :param supertag_storage_mode: one of "none", "kbest", "best" or
                "both". If "kbest" (or "best"), stores the `self.ktags` best
                (or the best) predicted tags per token. "both" stores the best
                as well as the `self.ktags` per token.
            :param supertag_storage_mode: if set to false, this will not store
                predicted pos tags in each token of the given sentences.
        """
        from flair.training_utils import store_embeddings
        from numpy import argpartition

        if not label_name:
            label_name = "predicted"
        get_label_name = lambda x: f"{label_name}-{x}"

        with torch.no_grad():
            parser = EnsembleParser(self.__grammar__, snd_order_weights=self.scoring.snd_order)
            embeds = self._batch_to_embeddings(batch, batch_first=True)
            scores = dict(self.forward(embeds))
            
            topweights, toptags = scores["supertag"].topk(self.ktags, dim=-1)
            topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
            postags = scores["pos"].argmax(dim=-1)
            totalbacktraces = 0
            totalqueuelen = 0

            for sentence, sembed, senttags, sentweights, postag in zip(batch, embeds, toptags, topweights, postags):
                # store tags in tokens
                sentence.store_raw_prediction("supertag-k", senttags[:len(sentence)])
                sentence.store_raw_prediction("supertag", senttags.gather(1, sentweights.argmin(dim=1, keepdim=True)).squeeze()[:len(sentence)])
                sentence.store_raw_prediction("pos", postag[:len(sentence)])

                # parse sentence and store parse tree in sentence
                predicted_tags = [
                    [(tag-1, weight) for tag, weight in zip(ktags, kweights) if tag != 0]
                    for ktags, kweights in zip(senttags[:len(sentence)], sentweights)]
                pos = [self.dictionaries["pos"].get_item_for_index(p) for p in postag[:len(sentence)]]

                parser.init(self.scoring, sembed[:len(sentence)], *predicted_tags)
                parser.fill_chart()
                sentence.set_label(label_name, str(AutoTree(parser.get_best()[0]).tree(pos)))
                totalbacktraces += len(parser.backtraces)
                totalqueuelen += parser.queue._qsize()
            print("finished parsing in prediction phase, saw", totalbacktraces, "backtraces")
            print(totalqueuelen, "items left in queue")

            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                l, n = self._calculate_loss(scores.items(), batch, batch_first=True, check_bounds=True)
                # return l + self._parsing_loss(batch), n
                return l, n

    def evaluate(self,
            dataset: DatasetWrapper,
            gold_label_type: str = "supertag",
            mini_batch_size: int = 32,
            num_workers: int = 1,
            embedding_storage_mode: str = "none",
            out_path = None,
            only_disc: str = "both",
            accuracy: str = "both",
            othertag_accuracy: bool = True,
            main_evaluation_metric = (),
            gold_label_dictionary = None,
            return_loss: bool = True,
            exclude_labels = [],
            progressbar: bool = False) -> Tuple[flair.training_utils.Result, float]:
        """ Predicts supertags, pos tags and parse trees, and reports the
            predictions scores for a set of sentences.
            :param sentences: a ``DataSet`` of sentences. For each sentence
                a gold parse tree is expected as value of the `tree` label, as
                provided by ``SupertagParseDataset``.
            :param only_disc: If set, overrides the setting `DISC_ONLY` in the
                evaluation parameter file ``self.evalparam``, i.e. only evaluates
                discontinuous constituents if True. Pass "both" to report both
                results.
            :param accuracy: either 'none', 'best', 'kbest' or 'both'.
                Determines if the accuracy is computed from the best, or k-best
                predicted tags.
            :param pos_accuracy: if set, reports acc. of predicted pos tags.
            :param return_loss: if set, nll loss wrt. gold tags is reported,
                otherwise the second component in the returned tuple is 0.
            :returns: tuple with evaluation ``Result``, where the main score
                is the f1-score (for all constituents, if only_disc == "both").
        """
        from flair.datasets import DataLoader
        from discodop.tree import ParentedTree, Tree
        from discodop.eval import Evaluator
        from timeit import default_timer
        from collections import Counter

        if self.__evalparam__ is None:
            raise Exception("Need to specify evaluator parameter file before evaluating")
        if only_disc == "both":
            evaluators = {
                "F1-all":  Evaluator({ **self.evalparam, "DISC_ONLY": False }),
                "F1-disc": Evaluator({ **self.evalparam, "DISC_ONLY": True  })}
        else:
            mode = self.evalparam["DISC_ONLY"] if only_disc == "param" else (only_disc=="true")
            strmode = "F1-disc" if mode else "F1-all"
            evaluators = {
                strmode: Evaluator({ **self.evalparam, "DISC_ONLY": mode })}
        
        data_loader = DataLoader(dataset, batch_size=mini_batch_size, num_workers=num_workers)
        iterator = data_loader if not progressbar else tqdm(data_loader)

        # predict supertags and parse trees
        eval_loss = 0
        n_predictions = 0
        start_time = default_timer()

        trees = []
        supertags = []
        postags = [] 
        for batch in iterator:
            loss = self.predict(
                batch,
                embedding_storage_mode=embedding_storage_mode,
                supertag_storage_mode=accuracy,
                othertag_storage_mode=othertag_accuracy,
                label_name='predicted',
                return_loss=return_loss
            )
            if return_loss:
                eval_loss += loss[0]
                n_predictions += loss[1]
            trees.extend(
                (len(sentence), sentence.get_raw_labels("tree"), sentence.get_labels("predicted")[0].value)
                for sentence in batch
            )
            supertags.extend(
                (sentence.get_raw_labels("supertag"), sentence.get_raw_prediction("supertag"))
                for sentence in batch
            )
            postags.extend(
                (sentence.get_raw_labels("pos"), sentence.get_raw_prediction("pos"))
                for sentence in batch
            )
        end_time = default_timer()
        if return_loss:
            eval_loss /= n_predictions

        i = 0
        noparses = 0
        for length, gold, prediction in trees:
            sent = [str(j) for j in range(length)]
            if "NOPARSE" in prediction:
                noparses += 1
            for evaluator in evaluators.values():
                evaluator.add(i, ParentedTree(gold), list(sent), ParentedTree(prediction), list(sent))
            i += 1
        scores = {
            strmode: float_or_zero(evaluator.acc.scores()['lf'])
            for strmode, evaluator in evaluators.items()}
        scores["supertag"] = 0
        scores["pos"] = 0
        for (gold, pred) in supertags:
            scores["supertag"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        for (gold, pred) in postags:
            scores["pos"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        scores["coverage"] = 1-(noparses/i)
        scores["time"] = end_time - start_time
        predictions = sum(len(s) for s, _ in supertags)
        scores["supertag"] = scores["supertag"].item() / predictions
        scores["pos"] = scores["pos"].item() / predictions

        result_args = dict(
            main_score=scores['supertag'],
            log_header="\t".join(f"{mode}" for mode in scores),
            log_line="\t".join(f"{s}" for s in scores.values()),
            detailed_results='\n\n'.join(evaluator.summary() for evaluator in evaluators.values()))
        
        return flair.training_utils.Result(**result_args, loss=eval_loss, classification_report=None)

    def _get_state_dict(self):
        return {
            "state_dict": self.state_dict(),
            "embedding_builder": self.embedding_builder,
            "dropout": self.dropoutprob,
            "tags": self.dictionaries,
            "ktags": self.ktags,
            "evalparam": self.evalparam,
            "grammar": (self.__grammar__.rules, self.__grammar__.root),
            "scoring_builder": self.scoring_builder
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embedding_builder"],
            state["tags"],
            grammar(*state["grammar"]),
            state["scoring_builder"],
            dropout = state["dropout"],
            ktags = state["ktags"],
            evalparam = state["evalparam"]
        )
        model.load_state_dict(state["state_dict"])
        return model

def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0