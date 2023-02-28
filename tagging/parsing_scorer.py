from .data import DatasetWrapper
from collections import defaultdict
import torch as t
from math import log

class CombinatorialParsingScorer:
    def __init__(self, corpus: DatasetWrapper):
        combinations: dict[tuple[int], int] = defaultdict(lambda: 0)
        denominator: dict[tuple[int], int] = defaultdict(lambda: 0)
        for sentence in corpus.train:
            deriv = sentence.get_derivation()
            for node in deriv.subtrees():
                if not node.children:
                    continue
                combinations[(node.label, *(c.label for c in node))] += 1
                denominator[tuple(c.label for c in node)] += 1
        self.probs = {
            comb: log(combinations[comb] / denominator[comb[1:]])
            for comb in combinations
        }

    def build(self):
        return self

    def score(self, root: int, *children: tuple[int]):
        if len(children) == 0: return 1
        return self.probs.get((root, *children), 0)


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
