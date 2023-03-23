from .data import DatasetWrapper
from collections import Counter
import torch as t
import flair as f
from math import log, sqrt
from ..grammar.sdcp import rule
from ..grammar.buparser import BitSpan, PassiveItem, backtrace
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
        self.embedding = t.nn.Embedding(n+1, self.nfeatures)
        self.binary = t.nn.Bilinear(self.nfeatures, self.nfeatures, self.nfeatures)
        self.unary = t.nn.Linear(self.nfeatures, self.nfeatures, bias=False) # share bias with binary
        self.to(f.device)

    def forward(self, *children: tuple[int]):
        if len(children) == 1:
            feats = self.unary(self.embedding(singleton(children[0]))) + self.binary.bias
        elif len(children) == 2:
            feats = self.binary(self.embedding(singleton(children[0])), self.embedding(singleton(children[1])))
        return (feats * self.embedding.weight[:-1]).sum(-1)
    
    def forward_loss(self, _embedding, root, *children):
        if len(children) == 0: return t.tensor(0.0, device=f.device)
        feats = self.forward(*children)
        return t.nn.functional.cross_entropy(feats, singleton(root), reduction="sum")
    
    def norule_loss(self, _embedding, *children):
        if len(children) == 0: return t.tensor(0.0, device=f.device)
        feats = self.forward(*children)
        return t.nn.functional.cross_entropy(feats, singleton(self.nfeatures), reduction="sum")

    def score(self, root, children, head, span, sentence_encoding):
        if len(children) == 0: return t.tensor(0.0, device=f.device)
        return -t.nn.functional.log_softmax(self.forward(*children), dim=-1)[root]

    @property
    def snd_order(self):
        return True

    @property
    def requires_training(self):
        return True
    
    def get_probab_distribution(self, children, *_unused_args):
        return -t.nn.functional.log_softmax(self.forward(*children), dim=-1)
        

    def get_key_from_chart_item(self, item: PassiveItem, bts: list[backtrace]):
        tup = tuple(bts[i].rid for i in bts[-1].children)
        return tup, tup
    

    def get_key_from_gold_node(self, gold: Tree, sentlen: int):
        tup = tuple(n.label[0] for n in gold.children)
        return tup, tup


class SpanScorer(t.nn.Module):
    @classmethod
    def spanvec(cls, sent_encoding: t.Tensor, span: Iterable[int]):
        return sent_encoding[t.tensor(list(span))].max(dim=0)[0]
        
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

    def forward(self, encoding: t.Tensor, span: Iterable[int], head: int):
        feats = self.encoding_to_embdding(self.__class__.spanvec(encoding, span))
        if not self.combinator is None:
            headenc = self.encoding_to_embdding(encoding[head])
            feats = t.nn.functional.relu(self.combinator(self.combinator(feats, headenc)))
        return self.decompression(feats)
    
    def forward_loss(self, encoding, root, span, head):
        feats = self.forward(encoding, span, head)
        return t.nn.functional.cross_entropy(feats, singleton(root), reduction="sum")
    
    def norule_loss(self, encoding, span, head):
        feats = self.forward(encoding, span, head)
        return t.nn.functional.cross_entropy(feats, singleton(self.nrules), reduction="sum")

    def score(self, root, children, head, span, encoding):
        return -t.nn.functional.log_softmax(self.forward(encoding, span, head), dim=-1)[root]

    @property
    def snd_order(self):
        return False

    @property
    def requires_training(self):
        return True
    
    def get_probab_distribution(self, children, head, span, encoding):
        return -t.nn.functional.log_softmax(self.forward(encoding, span, head), dim=-1)


    def get_key_from_chart_item(self, item: PassiveItem, bts: list[backtrace]):
        head = bts[-1].leaf if not self.combinator is None else None
        span = item.leaves
        return (span, head), (span.freeze(), head)
    

    def get_key_from_gold_node(self, gold: Tree, sentlen: int):
        span = BitSpan.fromit((n.label[1] for n in gold.subtrees()), sentlen)
        head = gold.label[1] if not self.combinator is None else None
        return (span, head), (span.freeze(), head)