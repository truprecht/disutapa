from .data import DatasetWrapper
from collections import Counter
import torch as t
from math import log, sqrt
from ..grammar.sdcp import rule


class ScoringBuilder:
    def __init__(self, typestr: str, trainingset: DatasetWrapper, *optionstrs: str):
        self.poptions = ()
        self.kwoptions = dict()
        if typestr is None or typestr.lower() == "none":
            self.constructor = DummyScorer
        elif typestr.lower().startswith("c"):
            options = eval(f"dict({', '.join(optionstrs)})")
            self.constructor = CombinatorialParsingScorer(trainingset, **options)
        elif typestr.lower().startswith("neu"):
            self.constructor = NeuralCombinatorialScorer
            self.poptions = (len(trainingset.labels()),)
            self.kwoptions = eval(f"dict({', '.join(optionstrs)})")
    
    def produce(self):
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

    @property
    def requires_training(self):
        return False


def singleton(num: int):
    if not isinstance(num, int):
        return num
    return t.tensor(num, dtype=t.long)


class NeuralCombinatorialScorer(t.nn.Module):
    def __init__(self, n: int, embedding_dim: int = 32):
        super().__init__()
        self.nfeatures = embedding_dim
        self.embedding = t.nn.Embedding(n, self.nfeatures)
        self.bias = t.nn.Parameter(t.empty((self.nfeatures,)))
        self.unary = t.nn.Parameter(t.empty((self.nfeatures,)*2))
        self.binary = t.nn.Parameter(t.empty((self.nfeatures,)*3))
        self.reset_parameters()

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
    
    def forward_loss(self, root, *children, check_bounds: bool = False):
        if len(children) == 0: return t.tensor(0.0)
        feats = self.forward(root, *children)
        loss = t.nn.functional.cross_entropy(feats, singleton(root), reduction="sum")
        return loss

    def score(self, root, *children):
        if len(children) == 0: return t.tensor(0.0)
        return -t.nn.functional.log_softmax(self.forward(root, *children), dim=-1)[root]

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
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
