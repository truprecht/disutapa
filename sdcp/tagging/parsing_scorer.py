from .data import DatasetWrapper
from collections import Counter
import torch as t
from math import log
from ..grammar.sdcp import rule


class ScoringBuilder:
    def __init__(self, typestr: str, trainingset: DatasetWrapper, *optionstrs: str):
        if typestr is None or typestr.lower() == "none":
            self.constructor = DummyScorer
            self.options = ()
        elif typestr.lower().startswith("c"):
            options = eval(f"dict({', '.join(optionstrs)})")
            self.constructor = CombinatorialParsingScorer(trainingset, **options)
            self.options = ()
    
    def produce(self):
        return self.constructor(*self.options)
    

class DummyScorer:
    def __init__(self):
        pass

    @property
    def snd_order(self):
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

    def score(self, root: int, *children: tuple[int]):
        if self.cnt_separated and len(children) == 2 and not any(c is None for c in children):
            return self.score(root, children[0], None) + self.score(root, None, children[1])
        if not (v := self.probs.get((root, *children))) is None:
            return v
        return -log(self.prior / self.denom(root, children)) if self.prior else float("inf")

    def __call__(self):
        # pretend a class object is a constructor
        return self

    @property
    def snd_order(self):
        return True


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
