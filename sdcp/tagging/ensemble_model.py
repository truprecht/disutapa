import torch
import flair
from tqdm import tqdm
from typing import Any

CPU = torch.device("cpu")

from typing import Tuple, Optional
from sdcp.grammar.sdcp import grammar
from sdcp.grammar.parser.activeparser import ActiveParser
from sdcp.autotree import AutoTree, with_pos

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
    step: float = 2.0
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
        return cls(embeddings, tag_dicts, grammar, parsing_scorer, parameters)

    def set_scoring(self, scoring_str, corpus, option_strs, abort_brass: float = 0.9):
        self.scoring_builder = ScoringBuilder(scoring_str, corpus, *option_strs)
        self.scoring = self.scoring_builder.produce(self.embedding.embedding_length)
        self.abort_brass = abort_brass

    def fix_tagging(self):
        self.__fix_tagging__ = True
        for e in self.embedding.embeddings:
            if "fine_tune" in e.__dict__:
                e.fine_tune = False

    def __init__(self, embeddings, dictionaries, grammar, parsing_scorer, config: ModelParameters):
        super().__init__()
        self.embedding_builder = embeddings
        self.embedding = flair.embeddings.StackedEmbeddings([
            builder.produce() for builder in embeddings
        ])
        inputlen = self.embedding.embedding_length
        self.scoring_builder = parsing_scorer
        self.scoring = self.scoring_builder.produce(inputlen)
        self.config = config

        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.dictionaries = dictionaries
        self.scores = torch.nn.ModuleDict({
            field: torch.nn.Linear(self.embedding.embedding_length, len(dict))
            for field, dict in self.dictionaries.items()
        })

        self.__grammar__ = grammar
        self.__fix_tagging__ = False
        self.to(flair.device)

    def set_config(self, field: str, value: Any):
        assert field in ("ktags", "step", "evalparam")
        self.config.__dict__[field] = value

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


    # def _parsing_loss(self, batch: list[SentenceWrapper], embeddings: torch.Tensor, feats: torch.Tensor):
    #     loss = torch.tensor(0.0, device=flair.device)
    #     npreds = 0
    #     if self.__fix_tagging__:
    #         feats = feats.detach()
    #         embeddings = embeddings.detach()

    #     for i, sent in enumerate(batch):
    #         embeds = embeddings[:len(sent), i]
    #         self.scoring.init_embeddings(embeds)
    #         deriv = sent.get_derivation()
    #         loss += self.scoring.forward_loss(deriv)
    #         npreds += deriv.inner_nodes

    #         brassitems = None
    #         if not (cache := sent.cache("brassitems")) is None:
    #             brassitems, backtraces = cache
    #         elif self.abort_brass <= 1:
    #             parser = ActiveParser(self.__grammar__)
    #             parser.set_scoring(self.scoring)
    #             weights, indices = feats[:len(sent), i].sort(descending=True)
    #             weights = -torch.nn.functional.log_softmax(weights, dim=-1)
    #             start = torch.zeros(len(sent))
    #             end = torch.zeros(len(sent))
                
    #             parser.init(len(sent))
    #             while parser.rootid is None:
    #                 for i in range(len(sent)):
    #                     end[i] = next(
    #                         j for j in range(start[i]+1, len(self.__grammar__.rules))
    #                         if weights[j] > weights[start[i]] - self.supertag_weight_range)
    #                 tags = [
    #                     [(indices[tag]-1, weights[tag]) for tag in range(s,e) if indices[tag] != 0]
    #                     for s,e in zip(start,end)]
                    
    #                 parser.add_rules(*tags)
    #                 parser.add_nongold_filter(deriv, self.abort_brass)
    #                 parser.fill_chart()

    #             brassitems = [(parser.items[j], parser.backtraces[j]) for j in parser.brassitems if parser.backtraces[j].children]
    #             backtraces = parser.backtraces
    #         if brassitems:
    #             loss += self.scoring.norule_loss(brassitems, backtraces)
    #             npreds += len(brassitems)

    #     return loss, npreds


    # def cache_scoring_items(self, batch: list[SentenceWrapper]):
    #     if self.abort_brass > 1:
    #         return
    #     with torch.no_grad():
    #         embeds = self._batch_to_embeddings(batch)
    #         scores = dict(self.forward(embeds))
    #         parser = ActiveParser(self.__grammar__)
    #         parser.set_scoring(self.scoring)
    #         for i, sentence in enumerate(batch):
    #             self.scoring.init_embeddings(embeds[:, i])

    #             weights, indices = scores["supertag"][:len(sentence), i].sort(descending=True)
    #             weights = -torch.nn.functional.log_softmax(weights, dim=-1)
    #             start = torch.zeros(len(sentence))
    #             end = torch.zeros(len(sentence))
    #             parser.init(len(sentence))
    #             maxweight = sum(scores["supertag"][j, i, s] for j, s in enumerate(sentence.get_raw_labels("supertag")))
    #             parser.set_gold_item_filter(sentence.get_derivation(), nongold_stopping_prob=self.abort_brass, early_stopping=maxweight)
    #             while parser.rootid is None:
    #                 for i in range(len(sentence)):
    #                     end[i] = next(
    #                         j for j in range(start[i]+1, len(self.__grammar__.rules))
    #                         if weights[j] > weights[start[i]] - self.supertag_weight_range)
    #                 tags = [
    #                     [(indices[tag]-1, weights[tag]) for tag in range(s,e) if indices[tag] != 0]
    #                     for s,e in zip(start,end)]
                    
    #                 parser.add_rules(*tags)
    #                 parser.fill_chart()

    #             brassitems = [(parser.items[j], parser.backtraces[j]) for j in parser.brassitems if parser.backtraces[j].children]
    #             sentence.cache("brassitems", (brassitems, parser.backtraces))


    def _tagging_loss(self, feats, batch: list[SentenceWrapper], batch_first: bool = False, check_bounds: bool = False):
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
        loss, predictions = torch.tensor(0.0, device=flair.device), 0
        if not self.__fix_tagging__:
            pl, pp = self._tagging_loss(feats.items(), batch)
            loss += pl
            predictions += pp
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
            parser = ParserAdapter(self.__grammar__, step=self.config.step, total_limit=self.config.ktags)
            parser.set_scoring(self.scoring)
            embeds = self._batch_to_embeddings(batch, batch_first=True)
            scores = dict(self.forward(embeds))
            topweights, toptags = scores["supertag"].topk(parser.total_limit, dim=-1, sorted=True)
            topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
            postags = scores["pos"].argmax(dim=-1)
            totalbacktraces = 0
            totalqueuelen = 0

            for sentence, sembed, sentweights, senttags, postag in zip(batch, embeds, topweights, toptags, postags):
                # store tags in tokens
                sentence.store_raw_prediction("supertag-k", senttags[:len(sentence), :self.config.ktags])
                sentence.store_raw_prediction("supertag", senttags[:len(sentence), 0])
                sentence.store_raw_prediction("pos", postag[:len(sentence)])

                # parse sentence and store parse tree in sentence
                pos = [self.dictionaries["pos"].get_item_for_index(p) for p in postag[:len(sentence)]]
                parser.init(len(sentence), sentweights, senttags-1)
                parser.fill_chart()

                # todo: move into evaluate
                chosen_tag_stats = []
                if not parser.parser.rootid is None:
                    for lexi, rulei in sorted((t.label[1], t.label[0]) for t in parser.parser.get_best_derivation().subtrees()):
                        idx = next(i for i, ri in enumerate(senttags[lexi]) if ri-1 == rulei)
                        weightdiff = sentweights[lexi, idx] - sentweights[lexi, 0]
                        chosen_tag_stats.append((idx, weightdiff))
                sentence.store_raw_prediction("chosen-supertag", chosen_tag_stats)
                
                sentence.set_label(label_name, str(with_pos(parser.get_best()[0], pos)))
                totalbacktraces += len(parser.parser.backtraces)
                totalqueuelen += len(parser.parser.queue)

            store_embeddings(batch, storage_mode=embedding_storage_mode)
            if return_loss:
                l, n = self._tagging_loss(scores.items(), batch, batch_first=True, check_bounds=True)
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

        if self.config.evalparam is None:
            raise Exception("Need to specify evaluator parameter file before evaluating")
        if only_disc == "both":
            evaluators = {
                "F1-all":  Evaluator({ **self.config.evalparam, "DISC_ONLY": False }),
                "F1-disc": Evaluator({ **self.config.evalparam, "DISC_ONLY": True  })}
        else:
            mode = self.config.evalparam["DISC_ONLY"] if only_disc == "param" else (only_disc=="true")
            strmode = "F1-disc" if mode else "F1-all"
            evaluators = {
                strmode: Evaluator({ **self.config.evalparam, "DISC_ONLY": mode })}
        
        data_loader = DataLoader(dataset, batch_size=mini_batch_size, num_workers=num_workers)
        iterator = data_loader if not progressbar else tqdm(data_loader)

        # predict supertags and parse trees
        eval_loss = 0
        n_predictions = 0
        start_time = default_timer()

        trees = []
        supertags = []
        postags = []
        chosen_supertags = []
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
            chosen_supertags.extend(
                idxwdiff
                for sentence in batch
                for idxwdiff in sentence.get_raw_prediction("chosen-supertag")
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
        
        scores["chose-first"] = 0
        if chosen_supertags:
            choices = sorted(idx for idx, _ in chosen_supertags if not idx == 0)
            wdiffs = sorted(wdiff for idx, wdiff in chosen_supertags if not idx == 0)
            scores["median-choice"] = choices[len(choices)//2]
            scores["worst-choice"] = choices[-1]
            scores["90-choice"] = choices[int(len(choices)*0.9)]
            scores["median-wdiff"] = wdiffs[len(choices)//2]
            scores["worst-wdiff"] = wdiffs[-1]
            scores["90-wdiff"] = wdiffs[int(len(choices)*0.9)]

        for (gold, pred) in supertags:
            scores["supertag"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        for (gold, pred) in postags:
            scores["pos"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        for (sidx, wdiff) in chosen_supertags:
            scores["chose-first"] += sidx == 0
        scores["coverage"] = 1-(noparses/i)
        scores["time"] = end_time - start_time
        predictions = sum(len(s) for s, _ in supertags)
        scores["chose-first"] /= predictions
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
            "tags": self.dictionaries,
            "grammar": (self.__grammar__.rules, self.__grammar__.root),
            "scoring_builder": self.scoring_builder,
            "config": self.config
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embedding_builder"],
            state["tags"],
            grammar(*state["grammar"]),
            state["scoring_builder"],
            state["config"]
        )
        model.load_state_dict(state["state_dict"])
        return model

def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0


class ParserAdapter:
    def __init__(self, grammar, step: float = 2, total_limit = 10):
        self.parser = ActiveParser(grammar)
        self.step = step
        self.total_limit = total_limit

    def init(self, length, weights, tags):
        self.parser.init(length)
        self.weights = weights[:self.parser.len]
        self.tags = tags[:self.parser.len]

    def fill_chart(self):
        start = torch.zeros(self.parser.len, dtype=int)
        end = torch.zeros(self.parser.len, dtype=int)
        threshs = self.weights[:, 0].clone().detach().unsqueeze(1)
        iteration = 1
        while self.parser.rootid is None:
            threshs += self.step
            end = (self.weights < threshs).sum(dim=1)
            tags = [
                [(self.tags[pos, tag], self.weights[pos, tag]) for tag in range(s,e) if tag >= 0]
                for pos, (s,e) in enumerate(zip(start,end))]
            
            self.parser.add_rules(*tags)
            self.parser.fill_chart()
            start = end
            if all(s == self.total_limit for s in start):
                print("abort after", iteration, "iterations")
                break
            iteration += 1
        print("done after", iteration, "iterations")

    def get_best(self):
        return self.parser.get_best()
    
    def set_scoring(self, *args):
        self.parser.set_scoring(*args)