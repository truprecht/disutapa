from .data import DatasetWrapper
from collections import defaultdict
import torch as t
from math import log
from ..grammar.sdcp import rule

class CombinatorialParsingScorer:
    def __init__(self, corpus: DatasetWrapper, prior: int = 1):
        combinations: dict[tuple[int], int] = defaultdict(lambda: 0)
        self.denominator: dict[tuple[int], int] = defaultdict(lambda: 0)
        self.cnt_by_rhs = defaultdict(lambda: 0)
        self.rhs = []
        self.prior = prior

        for i, supertag in enumerate(corpus.labels()):
            sobj: rule = eval(supertag)
            if sobj.rhs:
                self.cnt_by_rhs[sobj.rhs] += 1
            self.rhs.append(sobj.rhs)
        self.rhs = tuple(self.rhs)

        for sentence in corpus:
            deriv = sentence.get_derivation()
            for node in deriv.subtrees():
                if not node.children:
                    continue
                combinations[(node.label[0], *(c.label[0] for c in node))] += 1
                self.denominator[tuple(c.label[0] for c in node)] += 1
        
        self.probs = {
            comb: -log((combinations[comb] + prior) / self.denom(comb[0], comb[1:]))
            for comb in combinations
        }

    def denom(self, rule, children):
        s1 = self.denominator[children]
        rhs = self.rhs[rule]
        s2 = self.cnt_by_rhs[rhs]
        return s1 + self.prior * s2

    def produce(self):
        return self

    def score(self, root: int, *children: tuple[int]):
        if not (v := self.probs.get((root, *children))) is None:
            return v
        return -log(self.prior / self.denom(root, children)) if self.prior else float("inf")


class AffineLayer(t.nn.Module):
    def __init__(self, rules: int, embedding_dim: int):
        self.embeddings = t.nn.Embedding(rules, embedding_dim)
        self.unary = t.nn.Bilinear(embedding_dim, embedding_dim, 1)
        self.binaries = [
            t.nn.Bilinear(embedding_dim, embedding_dim, 1)
            for _ in range(0,3)
        ]

    def forward(self, *rules):
        rules = tuple(map(self.embeddings, rules))
        match rules:
            case (top, bot):
                return self.unary(top, bot)
            case (top, left, right):
                return self.binaries[0](top, left) \
                        + self.binaries[1](top, right) \
                        + self.binaries[2](left, right)


class SpanParsingScorer(t.nn.Module):
    def __init__(self, corpus: DatasetWrapper, embedding_len: int):
        super().__init__()
        nrules = len(corpus.labels())
        embedding_len *= 2
        self.w = t.nn.Sequence(
            t.nn.Linear((embedding_len, embedding_len)),
            t.nn.Relu(),
            t.nn.Linear((embedding_len, nrules))
        )

    def forward(self, embedded_span: t.tensor):
        return self.w(embedded_span)
