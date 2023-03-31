from .data import DatasetWrapper
from collections import Counter
import torch as t
import flair as f
from math import log, sqrt
from ..grammar.sdcp import rule
from ..grammar.derivation import Derivation
from ..grammar.buparser import BitSpan, PassiveItem, backtrace, qelement
from typing import Iterable
from discodop.tree import Tree


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
            
    def produce(self, encoding_len: None|int = None):
        if self.constructor is SpanScorer and len(self.poptions) < 2:
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
            for node in deriv.subtrees():
                if not node.children:
                    continue
                if self.cnt_separated and len(node) == 2:
                    combinations[(node.label[0], node[0].label[0], None)] += 1
                    combinations[(node.label[0], None, node[1].label[0])] += 1
                    for node_ in deriv.subtrees():
                        if not len(node_) == 2: continue
                        if node[0].label[0] == node_[0].label[0]:
                            self.denominator[(node.label[0], node[0].label[0], None)] += 1
                        if node[1].label[0] == node_[1].label[0]:
                            self.denominator[(node.label[0], None, node[1].label[0])] += 1
                else:
                    combinations[(node.label[0], *(c.label[0] for c in node))] += 1
                    rhs = tuple(c.label[0] for c in node)
                    for node_ in deriv.subtrees():
                        if rhs == tuple(c.label[0] for c in node_):
                            self.denominator[(node.label[0], *rhs)] += 1

        self.probs = {
            comb: -log((combinations[comb] + prior) / self.denom(comb[0], comb[1:]))
            for comb in combinations
        }

    def denom(self, rule, children):
        rhs = tuple(self.lhs[c] if not c is None else None for c in children)
        s1 = self.denominator[(rule, *children)]
        s2 = self.cnt_by_rhs[rhs]
        return s1 + self.prior * s2

    def score(self, root, children, *_unused_args):
        if self.cnt_separated and len(children) == 2 and not any(c is None for c in children):
            return self.score(root, (children[0], None), *_unused_args) + self.score(root, (None, children[1]), *_unused_args)
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
            self.score(n, children)
            for n in range(self.ntags)
        ])


def singleton(num: int):
    if isinstance(num, t.Tensor):
        return num.to(f.device)
    return t.tensor(num, dtype=t.long, device=f.device)


class NeuralCombinatorialScorer(t.nn.Module):
    def __init__(self, n: int, embedding_dim: int = 32):
        super().__init__()
        self.nfeatures = embedding_dim
        self.nrules = n
        self.embedding = t.nn.Embedding(n+1, self.nfeatures)
        self.binary = t.nn.Bilinear(self.nfeatures, self.nfeatures, self.nfeatures)
        self.unary = t.nn.Linear(self.nfeatures, self.nfeatures, bias=False) # share bias with binary
        self.to(f.device)

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
        return True
    
    def get_probab_distribution(self, children, *_unused_args):
        return -t.nn.functional.log_softmax(self.forward(*children), dim=-1)

    def forward(self, *children):
        if len(children) == 1:
            feats = self.unary(self.embedding(children[0])) + self.binary.bias
        elif len(children) == 2:
            feats = self.binary(self.embedding(children[0]), self.embedding(children[1]))
        return feats @ self.embedding.weight.transpose(0, 1)
 
    def forward_loss(self, batch: list[Derivation], _embeddings: t.Tensor):
        size = sum(d.inner_nodes for d in batch)
        combinations = t.empty((3, size), dtype=t.long)
        unary_mask = t.empty((size,), dtype=t.bool)
        i = 0
        for deriv in batch:
            for subderiv in deriv.subderivs():
                if len(subderiv.children) == 0: continue
                combinations[0, i] = subderiv.rule
                combinations[1, i] = subderiv.children[0].rule
                if len(subderiv.children) == 2:
                    combinations[2, i] = subderiv.children[1].rule
                unary_mask[i] = len(subderiv.children) == 1
                i += 1
        loss = t.nn.functional.cross_entropy(self.forward(combinations[1, unary_mask]), combinations[0, unary_mask], reduction="sum")
        loss += t.nn.functional.cross_entropy(self.forward(combinations[1, ~unary_mask], combinations[2, ~unary_mask]), combinations[0, ~unary_mask], reduction="sum")
        return loss

    @classmethod
    def _backtrace_to_tensor(cls, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace]):
        combinations = t.empty((3, len(items)), dtype=t.long)
        unary_mask = t.empty((len(items),), dtype=t.bool)
        for i, (_, bt) in enumerate(items):
            if len(bt.children) == 0: continue
            combinations[0, i] = bt.rid
            combinations[1, i] = bts[bt.children[0]].rid
            if len(bt.children) == 2:
                combinations[2, i] = bts[bt.children[1]].rid
            unary_mask[i] = len(bt.children) == 1
        return combinations[:2, unary_mask], combinations[:, ~unary_mask], unary_mask

    def norule_loss(self, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace], _embeddings: t.Tensor):
        unaries, binaries, _ = self._backtrace_to_tensor(items, bts)
        unaries[0,:], binaries[0, :] = self.nrules, self.nrules
        return t.nn.functional.cross_entropy(self.forward(unaries[1]), unaries[0], reduction="sum") \
            + t.nn.functional.cross_entropy(self.forward(binaries[1], binaries[2]), binaries[0], reduction="sum")
    
    def score(self, items: list[tuple[PassiveItem, backtrace]], bts: list[backtrace], _embeddings: t.Tensor):
        unaries, binaries, mask = self._backtrace_to_tensor(items, bts)
        unary_scores = -t.nn.functional.log_softmax(self.forward(unaries[1]), dim=1).gather(1, unaries[0].unsqueeze(1)).squeeze()
        binary_scores = -t.nn.functional.log_softmax(self.forward(binaries[1], binaries[2]), dim=1).gather(1, binaries[0].unsqueeze(1)).squeeze()
        scores = t.empty_like(mask, dtype=t.float)
        scores[mask] = unary_scores
        scores[~mask] = binary_scores
        return scores


class SpanScorer(t.nn.Module):
    def __init__(self, nrules: int, encoding_dim: int, embedding_dim: int = 32, score_head: bool = False):
        super().__init__()
        self.vecdim = encoding_dim
        self.embedding_dim = embedding_dim
        self.nrules = nrules
        self.encoding_to_embdding = t.nn.Sequential(
            t.nn.Linear(encoding_dim, embedding_dim),
            t.nn.ReLU(),
        )
        if score_head:
            self.combinator = t.nn.Bilinear(self.embedding_dim, self.embedding_dim, self.embedding_dim)
        else:
            self.combinator = None
        self.decompression = t.nn.Linear(embedding_dim, nrules+1)
        self.to(f.device)

    def forward(self, encoding: t.Tensor, spans: list[BitSpan], heads: list[int]):
        feats = t.empty((len(spans), self.vecdim))
        for i, s in enumerate(spans):
            mask = t.zeros((encoding.size(0),), dtype=t.bool)
            for j in s: mask[j] = True
            feats[i, :] = encoding[mask].max(dim=0)[0]
        feats = self.encoding_to_embdding(feats)
        if not self.combinator is None:
            headenc = self.encoding_to_embdding(encoding[t.tensor(heads)])
            feats = t.nn.functional.relu(self.combinator(self.combinator(feats, headenc)))
        return self.decompression(feats)
 
    def forward_loss(self, batch: list[Derivation], embeddings: t.Tensor):
        loss = t.tensor(0.0)
        for i, deriv in enumerate(batch):
            spans = [n.yd for n in deriv.subderivs() if n.children]
            heads = [n.leaf for n in deriv.subderivs() if n.children]
            gold = t.tensor([n.rule for n in deriv.subderivs() if n.children], dtype=t.long)
            loss += t.nn.functional.cross_entropy(self.forward(embeddings[:,i], spans, heads), gold, reduction="sum")
        return loss

    def norule_loss(self, items: list[tuple[PassiveItem, backtrace]], _bts: list[backtrace], embeddings: t.Tensor):
        spans = [n.leaves for n, _ in items]
        heads = [bt.leaf for _, bt in items]
        gold = t.empty((len(items),), dtype=t.long)
        gold[:] = self.nrules
        return t.nn.functional.cross_entropy(self.forward(embeddings, spans, heads), gold, reduction="sum")
    
    def score(self, items: list[tuple[PassiveItem, backtrace]], _bts: list[backtrace], embeddings: t.Tensor):
        spans = [n.leaves for n, _ in items]
        heads = [bt.leaf for _, bt in items]
        gold = t.tensor([bt.rid for _, bt in items], dtype=t.long)
        return -t.nn.functional.log_softmax(self.forward(embeddings, spans, heads), dim=-1).gather(1, gold.unsqueeze(1)).squeeze(dim=1)

    @property
    def snd_order(self):
        return False

    @property
    def requires_training(self):
        return True