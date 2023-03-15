from .data import DatasetWrapper
from collections import Counter
import torch as t
import flair as f
from math import log, sqrt
from ..grammar.sdcp import rule
from ..grammar.buparser import BitSpan, PassiveItem, backtrace
from typing import Iterable


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
        if self.constructor is SpanScorer:
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


def singleton(num: int):
    if isinstance(num, t.Tensor):
        return num.to(f.device)
    return t.tensor(num, dtype=t.long, device=f.device)


class NeuralCombinatorialScorer(t.nn.Module):
    def __init__(self, n: int, embedding_dim: int = 32):
        super().__init__()
        self.nfeatures = embedding_dim
        self.embedding = t.nn.Embedding(n, self.nfeatures)
        self.bias = t.nn.Parameter(t.empty((self.nfeatures,)))
        self.unary = t.nn.Parameter(t.empty((self.nfeatures,)*2))
        self.binary = t.nn.Parameter(t.empty((self.nfeatures,)*3))
        self.reset_parameters()
        self.to(f.device)

    def reset_parameters(self):
        bound = 1 / sqrt(self.nfeatures)
        t.nn.init.uniform_(self.bias, -bound, bound)
        t.nn.init.uniform_(self.unary, -bound, bound)
        t.nn.init.uniform_(self.binary, -bound, bound)
        self.embedding.reset_parameters()

    def forward(self, root: int, *children: tuple[int]):
        if len(children) == 1:
            em = self.embedding(singleton(children[0]))
            feats = (self.unary * em).sum(-1)
        elif len(children) == 2:
            feats = ((self.binary * self.embedding(singleton(children[0]))).sum(-1) * self.embedding(singleton(children[1]))).sum(-1)
        feats += self.bias
        return (feats * self.embedding.weight).sum(-1)
    
    def forward_loss(self, root, children, head, span, sentence_encoding):
        if len(children) == 0: return t.tensor(0.0, device=f.device)
        feats = self.forward(root, *children)
        loss = t.nn.functional.cross_entropy(feats, singleton(root), reduction="sum")
        return loss

    def score(self, root, children, head, span, sentence_encoding):
        if len(children) == 0: return t.tensor(0.0, device=f.device)
        return -t.nn.functional.log_softmax(self.forward(root, *children), dim=-1)[root]

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
        return True


class SpanScorer(t.nn.Module):
    @classmethod
    def spanvec(cls, sent_encoding: t.Tensor, span: Iterable[int]):
        return sent_encoding[t.tensor(list(span))].max(dim=0)[0]
        
    def __init__(self, nrules: int, encoding_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.vecdim = encoding_dim
        self.embedding_dim = embedding_dim
        self.rule_embedding = t.nn.Embedding(nrules, embedding_dim)
        self.encoding_to_embdding = t.nn.Sequential(
            t.nn.Linear(encoding_dim, encoding_dim),
            t.nn.ReLU(),
            t.nn.Linear(encoding_dim, embedding_dim)
        )
        self.combinator = t.nn.Parameter(t.empty((self.embedding_dim,)*3))
        self.to(f.device)

    def forward(self, encoding: t.Tensor, span: Iterable[int], head: int):
        spanenc = self.encoding_to_embdding(self.__class__.spanvec(encoding, span))
        headenc = self.encoding_to_embdding(encoding[head])
        return (((self.combinator * spanenc).sum(-1) * headenc).sum(-1) * self.rule_embedding.weight).sum(-1)

    def score(self, root, children, head, span, encoding):
        return -t.nn.functional.log_softmax(self.forward(encoding, span, head), dim=-1)[root]
    
    def forward_loss(self, root, children, head, span, encoding):
        feats = self.forward(encoding, span, head)
        return t.nn.functional.cross_entropy(feats, singleton(root))

    @property
    def snd_order(self):
        return False

    @property
    def requires_training(self):
        return True