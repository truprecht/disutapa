import torch
import flair
from tqdm import tqdm
from typing import Any
from itertools import islice

CPU = torch.device("cpu")

from typing import Tuple, Optional
from disutapa.grammar.sdcp import grammar
from disutapa.autotree import AutoTree, with_pos, fix_rotation

from discodop.eval import readparam, TreePairResult
from discodop.tree import ParentedTree, Tree

from .data import DatasetWrapper, SentenceWrapper, CorpusWrapper
from .embeddings import TokenEmbeddingBuilder, EmbeddingPresets, PretrainedBuilder
from .parser_adapter import ParserAdapter

from dataclasses import dataclass, field, fields
from argparse import Namespace

@dataclass
class ModelParameters:
    embedding: str = "Supervised"
    embedding_options: list[str] = field(default_factory=list)
    ktags: int = 1
    ktrees: int = 1
    step: float = 2.0
    dropout: float = 0.1
    evalparam: Optional[dict] = None
    lstm_layers: int = 0
    lstm_size: int = 512

    def __post_init__(self):
        if self.embedding in EmbeddingPresets:
            self.embedding = EmbeddingPresets[self.embedding]
        else:
            options = eval(f"dict({','.join(self.embedding_options)})")
            self.embedding = [PretrainedBuilder(self.embedding, **options)]
        
        if self.evalparam is None:
            self.evalparam = readparam("resources/disco-dop/proper.prm")

    @classmethod
    def from_namespace(cls, config: Namespace) -> "ModelParameters":
        kwargs = {
            class_field.name: config.__dict__[class_field.name]
            for class_field in fields(cls)
        }
        return cls(**kwargs)


class EnsembleModel(flair.nn.Model, flair.nn.model.ReduceTransformerVocabMixin):
    def get_used_tokens(self, corpus: CorpusWrapper):
        for sentence in corpus.get_all_sentences():
            yield [t.text for t in sentence]

    @classmethod
    def from_corpus(cls, corpus: DatasetWrapper, grammar: grammar, parameters: Namespace):
        """ Construct an instance of the model using
            * supertags and pos tags from `grammar`, and
            * word embeddings (as specified in `parameters`) from `corpus`.
        """
        tag_dicts = { k: corpus.build_dictionary(k) for k in ("supertag", "pos") }
        parameters = ModelParameters.from_namespace(parameters)
        embeddings = [
            embedding.build_vocab(corpus)
            for embedding in parameters.embedding
        ]
        return cls(embeddings, tag_dicts, grammar, parameters)


    def supertag_layer(self, n_inputs, n_outputs, hidden=None):
        if any(e.fine_tune() for e in self.embedding_builder):
            return torch.nn.Linear(n_inputs, n_outputs)
        if hidden is None:
            hidden = n_inputs*2
        return torch.nn.Sequential(
            torch.nn.Linear(n_inputs, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_outputs))


    def __init__(self, embeddings, dictionaries, grammar, config: ModelParameters):
        super().__init__()
        self.embedding_builder = embeddings
        self.embeddings = flair.embeddings.StackedEmbeddings([
            builder.produce() for builder in embeddings
        ])
        self.config = config
        embedding_len = self.embeddings.embedding_length

        if self.config.lstm_layers > 0:
            self.lstm = torch.nn.LSTM(embedding_len, self.config.lstm_size, self.config.lstm_layers, bidirectional=True)
            self.dropout = torch.nn.Dropout(self.config.dropout)
            embedding_len = self.config.lstm_size * 2
        self.dictionaries = dictionaries
        self.scores = torch.nn.ModuleDict({
            "supertag": self.supertag_layer(embedding_len, len(self.dictionaries["supertag"])),
            "pos": torch.nn.Linear(embedding_len, len(self.dictionaries["pos"]))
        })

        self.reranking: TreeRanker = None

        self.__grammar__ = grammar
        self.to(flair.device)

    def set_config(self, field: str, value: Any):
        assert field in ("ktags", "ktrees", "step", "evalparam")
        self.config.__dict__[field] = value

    def label_type(self):
        return "supertag"

    def _batch_to_embeddings(self, batch):
        if not type(batch) is list:
            batch = [batch]
        self.embeddings.embed(batch)
        embedding_name = self.embeddings.get_names()
        input = torch.nn.utils.rnn.pad_sequence([
            torch.stack([ word.get_embedding(embedding_name) for word in sentence ])
            for sentence in batch]).to(flair.device)
        return input


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
        return self._tagging_loss(feats.items(), batch)


    def forward(self, embeddings, batch_first=False):
        if self.config.lstm_layers > 0:
            embeddings = self.dropout(embeddings)
            embeddings, _ = self.lstm(embeddings)
            embeddings = self.dropout(embeddings)
        for field, layer in self.scores.items():
            logits = layer(embeddings)
            if batch_first:
                logits=logits.transpose(0,1)
            yield field, logits


    def predict(self,
            batch: list[SentenceWrapper],
            label_name: str = None,
            return_loss: bool = False,
            embedding_storage_mode: str = "none",
            supertag_storage_mode: str = "both",
            othertag_storage_mode: bool = True,
            store_kbest: int = None):
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

        if not label_name:
            label_name = "predicted"
        get_label_name = lambda x: f"{label_name}-{x}"

        with torch.no_grad():
            parser = ParserAdapter(self.__grammar__, step=self.config.step, total_limit=self.config.ktags)
            scores = dict(self.forward(self._batch_to_embeddings(batch), batch_first=True))
            topweights, toptags = scores["supertag"][:,:,1:].topk(parser.total_limit, dim=-1, sorted=True)
            toptags += 1
            topweights = -torch.nn.functional.log_softmax(topweights, dim=-1)
            postags = scores["pos"].argmax(dim=-1)

            for sentence, sentweights, senttags, postag in zip(batch, topweights, toptags, postags):
                # store tags in tokens
                sentence.store_raw_prediction("supertag-k", senttags[:len(sentence), :self.config.ktags])
                sentence.store_raw_prediction("supertag", senttags[:len(sentence), 0])
                sentence.store_raw_prediction("pos", postag[:len(sentence)])

                # parse sentence and store parse tree in sentence
                pos = [self.dictionaries["pos"].get_item_for_index(p) for p in postag[:len(sentence)]]
                parser.fill_chart(len(sentence), sentweights.cpu().numpy(), (senttags-1).cpu().numpy())

                if self.config.ktrees > 1:
                    derivs = islice(parser.get_best_iter(), self.config.ktrees)
                    derivs = [(fix_rotation(with_pos(d[0], pos))[1], w) for d, w in derivs]
                    sentence.store_raw_prediction("kbest-trees", derivs)

                if not self.reranking is None and self.config.ktrees > 1:
                    _, tree = self.reranking.select(derivs)
                else:
                    tree = fix_rotation(with_pos(parser.get_best()[0], pos))[1]
                
                sentence.set_label(label_name, str(tree))

            if return_loss:
                return self._tagging_loss(scores.items(), batch, batch_first=True, check_bounds=True)

    def evaluate(self,
            dataset: DatasetWrapper,
            mini_batch_size: int = 32,
            embedding_storage_mode: str = "none",
            only_disc: str = "both",
            accuracy: str = "both",
            othertag_accuracy: bool = True,
            return_loss: bool = True,
            progressbar: bool = False,
            oracle_scores: bool = False,
            **kwargs
        ) -> Tuple[flair.training_utils.Result, float]:
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
        from discodop.eval import Evaluator
        from timeit import default_timer

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
        
        data_loader = DataLoader(dataset, batch_size=mini_batch_size)
        iterator = data_loader if not progressbar else tqdm(data_loader)

        # predict supertags and parse trees
        eval_loss = 0
        n_predictions = 0
        start_time = default_timer()

        trees = []
        supertags = []
        postags = []
        chosen_supertags = []
        oracle_trees = []
        for batch in iterator:
            for sentence in batch:
                sentence.clear_embeddings()
            loss = self.predict(
                batch,
                embedding_storage_mode=embedding_storage_mode,
                supertag_storage_mode=accuracy,
                othertag_storage_mode=othertag_accuracy,
                label_name='predicted',
                return_loss=return_loss,
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
            if oracle_scores:
                oracle_trees.extend(
                    (len(sentence),
                        *oracle_tree(
                            sentence.get_raw_prediction("kbest-trees"),
                            sentence.get_raw_labels("tree"),
                            params=self.config.evalparam),
                        sentence.get_raw_labels("tree"))
                    for sentence in batch)
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

        if oracle_scores:
            ev = Evaluator(self.config.evalparam)
            indices = []
            for i, (sl, otidx, ot, gt) in enumerate(oracle_trees):
                sent = [str(i) for i in range(sl)]
                ev.add(i, ParentedTree(gt), list(sent), ParentedTree.convert(ot), sent)
                if otidx > 0:
                    indices.append(otidx)
            scores["oracle-f1"] = float_or_zero(ev.acc.scores()["lf"])
            indices.sort()
            scores["num-kbest"] = len(indices)
            scores["median-kbest"] = indices[len(indices)//2] if indices else 0
            scores["90-kbest"] = indices[int(len(indices)*0.9)] if indices else 0

        for (gold, pred) in supertags:
            scores["supertag"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        for (gold, pred) in postags:
            scores["pos"] += sum(torch.tensor(gold)+1 == pred.to(CPU))
        for (sidx, wdiff) in chosen_supertags:
            scores["chose-first"] += sidx == 0
        scores["coverage"] = 1-(noparses/i)
        scores["time"] = end_time - start_time
        predictions = sum(len(s) for s, _ in supertags)
        # scores["chose-first"] /= predictions
        scores["supertag"] = scores["supertag"].item() / predictions
        scores["pos"] = scores["pos"].item() / predictions
        scores["loss"] = eval_loss

        results = flair.training_utils.Result(
            main_score=scores['F1-all'],
            scores=scores,
            detailed_results='\n\n'.join(evaluator.summary() for evaluator in evaluators.values()))
        
        return results

    def _get_state_dict(self):
        return {
            "state_dict": self.state_dict(),
            "embedding_builder": self.embedding_builder,
            "tags": self.dictionaries,
            "grammar": (self.__grammar__.rules, self.__grammar__.root),
            "config": self.config,
        }

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls(
            state["embedding_builder"],
            state["tags"],
            grammar(*state["grammar"]),
            state["config"],
        )
        # compatibility with old version, embeddings field was renamed to embedding
        for key in list(state["state_dict"].keys()):
            if key.startswith("embedding."):
                newkey = "embeddings" + key[9:]
                state["state_dict"][newkey] = state["state_dict"].pop(key)
        model.load_state_dict(state["state_dict"])
        return model


def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0


# TODO: move into TreeRanker
def oracle_tree(kbestlist, gold, params):
    besttree, bestscore, bestindex = None, None, None
    gold = ParentedTree(gold)
    sent = [str(i) for i in range(len(gold.leaves()))]
    for i, candidate in enumerate(kbestlist):
        scores = TreePairResult(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(candidate), list(sent), params)
        lf1 = scores.scores()["LF"]
        if bestscore is None or lf1 > bestscore:
            besttree = candidate
            bestscore = lf1
            bestindex = i
    return bestindex, besttree
