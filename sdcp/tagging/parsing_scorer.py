import flair as f
import torch as t

from collections import Counter
from math import log

from ..grammar.sdcp import rule
from ..grammar.derivation import Derivation
from ..grammar.buparser import BitSpan, PassiveItem, backtrace
from .data import DatasetWrapper


class ScoringBuilder:
    def __init__(self, typestr: str, trainingset: DatasetWrapper, *options: str|int):
        self.poptions = ()
        self.kwoptions = dict()
        if typestr is None or typestr.lower() == "none":
            self.constructor = DummyScorer
        elif typestr.lower().startswith("snd"):
            options = eval(f"dict({', '.join(options)})")
            self.constructor = CombinatorialParsingScorer(trainingset, **options)
        elif typestr.lower().startswith("neu"):
            self.constructor = NeuralCombinatorialScorer
            self.poptions = (len(trainingset.labels()),)
            self.kwoptions = eval(f"dict({', '.join(options)})")
        elif typestr.lower().startswith("span"):
            self.constructor = SpanScorer
            self.poptions = (len(trainingset.labels()),)
            self.kwoptions: dict = eval(f"dict({', '.join(options)})")
        else:
            raise ValueError(f"did not recognize scoring identifier: {typestr}")
            
    def produce(self, encoding_len: None|int = None):
        if self.constructor in (SpanScorer, NeuralCombinatorialScorer) and len(self.poptions) < 2:
            assert not encoding_len is None
            self.poptions += (encoding_len,)
        return self.constructor(*self.poptions, **self.kwoptions)


class DummyScorer:
    def __init__(self):
        pass

    @property
    def snd_order(self):
        return False

    @property
    def requires_training(self):
        return False

    def score(*args):
        return 0.0


class CombinatorialParsingScorer:
    def __init__(self, corpus: DatasetWrapper, prior: int = 1, separated: bool = True):
        combinations: dict[tuple[int], int] = Counter()
        self.ntags = len(corpus.labels())
        self.denominator: dict[tuple[int], int] = Counter()
        self.cnt_by_rhs = Counter()
        self.lhs = []
        self.prior = prior
        self.cnt_separated = separated

        for supertag in corpus.labels():
            sobj: rule = eval(supertag)
            if sobj.rhs:
                if self.cnt_separated and len(sobj.rhs) == 2:
                    self.cnt_by_rhs[(sobj.rhs[0], None)] += 1
                    self.cnt_by_rhs[(None, sobj.rhs[1])] += 1
                else:
                    self.cnt_by_rhs[sobj.rhs] += 1
            self.lhs.append(sobj.lhs)
        self.lhs = tuple(self.lhs)

        for sentence in corpus:
            deriv = sentence.get_derivation()
            for node in deriv.subderivs():
                if not node.children:
                    continue
                if self.cnt_separated and len(node.children) == 2:
                    combinations[(node.rule, node.children[0].rule, None)] += 1
                    combinations[(node.rule, None, node.children[1].rule)] += 1
                    for node_ in deriv.subderivs():
                        if not len(node_.children) == 2: continue
                        if node.children[0].rule == node_.children[0].rule:
                            self.denominator[(node.rule, node.children[0].rule, None)] += 1
                        if node.children[1].rule == node_.children[1].rule:
                            self.denominator[(node.rule, None, node.children[1].rule)] += 1
                else:
                    combinations[(node.rule, *(c.rule for c in node.children))] += 1
                    rhs = tuple(c.rule for c in node.children)
                    for node_ in deriv.subderivs():
                        if rhs == tuple(c.rule for c in node_.children):
                            self.denominator[(node.rule, *rhs)] += 1

        self.probs = {
            comb: -log((combinations[comb] + prior) / self.denom(comb[0], comb[1:]))
            for comb in combinations
        }

    def init_embeddings(self, *args):
        pass

    def denom(self, rule, children):
        rhs = tuple(self.lhs[c] if not c is None else None for c in children)
        s1 = self.denominator[(rule, *children)]
        s2 = self.cnt_by_rhs[rhs]
        return s1 + self.prior * s2

    def score(self, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace]):
        for item, bt in items:
            root = bt.rid
            children = tuple(bts[c].rid for c in bt.children)
            yield self._single_score(root, children)

    def _single_score(self, root, children, *_unused_args):
        if self.cnt_separated and len(children) == 2 and not any(c is None for c in children):
            return self._single_score(root, (children[0], None), *_unused_args) + self._single_score(root, (None, children[1]), *_unused_args)
        if not (v := self.probs.get((root, *children))) is None:
            return v
        return -log(self.prior / self.denom(root, children)) if self.prior else float("inf")

    def __call__(self):
        # pretend a class object is a constructor
        return self

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
        return False
    
    def get_probab_distribution(self, children, *_unused_args):
        return t.tensor([
            self._single_score(n, children)
            for n in range(self.ntags)
        ], device=f.device)


class NeuralCombinatorialScorer(t.nn.Module):
    def __init__(self, nrules: int, sentence_embedding_dim: int, rule_embedding_dim: int = 32, sentence_compression_dim: int = 32):
        super().__init__()
        self.n_rule_features = rule_embedding_dim
        self.input_dim = sentence_embedding_dim
        self.n_word_features = sentence_compression_dim
        self.nrules = nrules
        self.word_compression = t.nn.Linear(self.input_dim, self.n_word_features)
        self.rule_embedding = t.nn.Embedding(nrules+1, self.n_rule_features)
        nfeatures = self.n_rule_features+self.n_word_features
        self.binary = t.nn.Bilinear(nfeatures, nfeatures, self.n_rule_features)
        self.unary = t.nn.Linear(nfeatures, self.n_rule_features, bias=False) # share bias with binary
        self.to(f.device)

    def init_embeddings(self, word_embeddings: t.Tensor):
        self.word_embeddings = word_embeddings        

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
        return True
    
    def get_probab_distribution(self, children, *_unused_args):
        return -t.nn.functional.log_softmax(self.forward(*children), dim=-1)

    def forward(self, *children: t.Tensor):
        mid, tot = children[0].size(1), children[0].size(1)
        if len(children) > 1:
            tot *= 2
        feats = t.empty((tot, self.n_rule_features + self.n_word_features), device=f.device)
        feats[:mid, :self.n_rule_features] = self.rule_embedding(children[0][0])
        feats[:mid, self.n_rule_features:] = self.word_compression(self.word_embeddings[children[0][1]])
        if len(children) == 1:
            feats = t.nn.functional.relu(feats)
            feats = self.unary(feats) + self.binary.bias
        elif len(children) == 2:
            feats[mid:, :self.n_rule_features] = self.rule_embedding(children[1][0])
            feats[mid:, self.n_rule_features:] = self.word_compression(self.word_embeddings[children[1][1]])
            feats = t.nn.functional.relu(feats)
            feats = self.binary(feats[:mid], feats[mid:])
        feats = t.nn.functional.relu(feats) @ self.rule_embedding.weight.transpose(0, 1)
        return feats
 
    def forward_loss(self, batch: list[Derivation]):
        loss = t.tensor(0.0, device=f.device)
        for j, deriv in enumerate(batch):
            combinations = t.empty((5, deriv.inner_nodes), dtype=t.long, device=f.device)
            unary_mask = t.empty((deriv.inner_nodes,), dtype=t.bool, device=f.device)
            for i, subderiv in enumerate(n for n in deriv.subderivs() if n.children):
                combinations[0, i] = subderiv.rule
                combinations[1, i] = subderiv.children[0].rule
                combinations[2, i] = subderiv.children[0].leaf
                if len(subderiv.children) == 2:
                    combinations[3, i] = subderiv.children[1].rule
                    combinations[4, i] = subderiv.children[1].leaf
                unary_mask[i] = len(subderiv.children) == 1
            loss += t.nn.functional.cross_entropy(self.forward(combinations[1:3, unary_mask]), combinations[0, unary_mask], reduction="sum")
            loss += t.nn.functional.cross_entropy(self.forward(combinations[1:3, ~unary_mask], combinations[3:5, ~unary_mask]), combinations[0, ~unary_mask], reduction="sum")
        return loss

    @classmethod
    def _backtrace_to_tensor(cls, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace]):
        combinations = t.empty((5, len(items)), dtype=t.long, device=f.device)
        unary_mask = t.empty((len(items),), dtype=t.bool, device=f.device)
        for i, (_, bt) in enumerate(items):
            if len(bt.children) == 0: continue
            combinations[0, i] = bt.rid
            combinations[1, i] = bts[bt.children[0]].rid
            combinations[2, i] = bts[bt.children[0]].leaf
            if len(bt.children) == 2:
                combinations[3, i] = bts[bt.children[1]].rid
                combinations[4, i] = bts[bt.children[1]].leaf
            unary_mask[i] = len(bt.children) == 1
        return combinations[:3, unary_mask], combinations[:, ~unary_mask], unary_mask

    def norule_loss(self, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace]):
        unaries, binaries, _ = self._backtrace_to_tensor(items, bts)
        unaries[0,:], binaries[0, :] = self.nrules, self.nrules
        return t.nn.functional.cross_entropy(self.forward(unaries[1:3]), unaries[0], reduction="sum") \
            + t.nn.functional.cross_entropy(self.forward(binaries[1:3], binaries[3:5]), binaries[0], reduction="sum")
    
    def score(self, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace]):
        unaries, binaries, mask = self._backtrace_to_tensor(items, bts)
        unary_scores = -t.nn.functional.log_softmax(self.forward(unaries[1:3]), dim=1).gather(1, unaries[0].unsqueeze(1)).squeeze()
        binary_scores = -t.nn.functional.log_softmax(self.forward(binaries[1:3], binaries[3:5]), dim=1).gather(1, binaries[0].unsqueeze(1)).squeeze()
        scores = t.empty_like(mask, dtype=t.float, device=f.device)
        scores[mask] = unary_scores
        scores[~mask] = binary_scores
        return scores


class SpanScorer(t.nn.Module):
    def __init__(self, nrules: int, encoding_dim: int, fencepost_dim: int = 32, lstm_layers: int = 2):
        super().__init__()
        self.vecdim = encoding_dim+2*fencepost_dim
        self.nrules = nrules
        self.bos_eos = t.nn.Embedding(2, encoding_dim)
        self.fencepost_dim = fencepost_dim
        self.fencepost_lstm = t.nn.LSTM(encoding_dim, fencepost_dim, lstm_layers, bidirectional=True)
        self.classifier = t.nn.Linear(self.vecdim, nrules+1)
        self.to(f.device)


    def init_embeddings(self, word_embeddings: t.Tensor):
        self.word_embeddings = word_embeddings
        self.fenceposts = self.fencepost_lstm(
            t.cat((
                self.bos_eos(t.tensor([0], device=f.device)),
                self.word_embeddings,
                self.bos_eos(t.tensor([1], device=f.device)),
            ))
        )[0]


    def to_vec(self, span: BitSpan):
        mask = t.zeros((self.word_embeddings.size(0),), dtype=t.bool, device=f.device)
        for j in span:
            mask[j] = True
        lefts, rights = zip(*span.fences())
        lefts, rights = t.tensor(lefts, device=f.device), t.tensor(rights, device=f.device)
        fenceposts = t.cat([
            self.fenceposts[rights, :self.fencepost_dim]-self.fenceposts[lefts, :self.fencepost_dim],
            self.fenceposts[rights+1, self.fencepost_dim:]-self.fenceposts[lefts+1, self.fencepost_dim:]
        ], dim = 1)
        return  t.cat((
            self.word_embeddings[mask].max(dim=0)[0],
            fenceposts.max(dim=0)[0]
        ))
        

    def forward(self, spans: list[BitSpan], heads: list[int]):
        feats = t.empty((len(spans), self.vecdim), device=f.device)
        for i, s in enumerate(spans):
            feats[i, :] = self.to_vec(s)
        return self.classifier(feats)


    def forward_loss(self, batch: list[Derivation]):
        loss = t.tensor(0.0, device=f.device)
        for i, deriv in enumerate(batch):
            spans = [n.yd for n in deriv.subderivs() if n.children]
            heads = [n.leaf for n in deriv.subderivs() if n.children]
            gold = t.tensor([n.rule for n in deriv.subderivs() if n.children], dtype=t.long, device=f.device)
            loss += t.nn.functional.cross_entropy(self.forward(spans, heads), gold, reduction="sum")
        return loss


    def norule_loss(self, items: list[tuple[PassiveItem, backtrace]], _bts: list[backtrace]):
        spans = [n.leaves for n, _ in items]
        heads = [bt.leaf for _, bt in items]
        gold = t.empty((len(items),), dtype=t.long, device=f.device)
        gold[:] = self.nrules
        return t.nn.functional.cross_entropy(self.forward(spans, heads), gold, reduction="sum")

  
    def score(self, items: list[tuple[PassiveItem, backtrace]], _bts: list[backtrace]):
        spans = [n.leaves for n, _ in items]
        heads = [bt.leaf for _, bt in items]
        gold = t.tensor([bt.rid for _, bt in items], dtype=t.long, device=f.device)
        return -t.nn.functional.log_softmax(self.forward(spans, heads), dim=-1).gather(1, gold.unsqueeze(1)).squeeze(dim=1)


    @property
    def snd_order(self):
        return False


    @property
    def requires_training(self):
        return True