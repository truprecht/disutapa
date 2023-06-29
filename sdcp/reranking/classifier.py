from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
import torch
from typing import Iterable
from tqdm import tqdm
from math import log

from .features import FeatureExtractor


def perceptron_loss(scores, goldidx):
    prediction = scores.argmax()
    if prediction == goldidx:
        return torch.tensor(0.0, requires_grad=True)
    return scores[prediction]-scores[goldidx]


def max_margin_loss(scores, goldidx):
    nongoldscores = torch.cat((scores[:goldidx], scores[goldidx+1:]))
    return torch.nn.functional.relu(nongoldscores-scores[goldidx]+1).sum()


def softmax_loss(scores, goldidx):
    return torch.nn.functional.cross_entropy(scores, torch.tensor(goldidx))


class TreeRanker:
    def __init__(self, min_feature_count: int = 5, evalparam: dict | None = None):
        self.features = FeatureExtractor()
        self.featoccs = min_feature_count
        self.oracle_trees = list()
        self.kbest_trees = list()
        self.evalparam = evalparam if not evalparam is None else \
            readparam("../disco-dop/proper.prm")

    
    def add_tree(self, gold: Tree, kbest: Iterable[tuple[Tree, float]]):
        sent = [str(i) for i in range(len(gold.leaves()))]
        kts = list()
        best, bestscore = None, None
        for sidx, (silver, weight) in enumerate(kbest):
            kts.append(self.features.extract(silver).add("parsing_score", sidx+1))
            evaluator = TreePairResult(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(silver), sent, self.evalparam)
            score = evaluator.scores()["LF"]
            if bestscore is None or bestscore < score:
                best, bestscore = sidx, score
        self.kbest_trees.append(kts)
        self.oracle_trees.append(best)


    def fit(self, epochs: int = 100, lossfunction = softmax_loss, lr: float = 1.0, weight_decay: float = 0.0, devset = None):
        self.features.truncate(self.featoccs)
        self.features.objects["parsing_score"] = {"": 0}
        self.features.counts[("parsing_score", 0)] = self.featoccs+1

        self.weights = torch.zeros(len(self.features), requires_grad=True, dtype=float)
        optim = torch.optim.SGD((self.weights,), lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            optim.zero_grad()
            loss = torch.tensor(0.0)
            iterator = tqdm(zip(self.oracle_trees, self.kbest_trees), total=len(self.oracle_trees), desc=f"training reranking in epoch {epoch}")
            for goldidx, trees in iterator:
                mat = torch.stack([
                    torch.tensor(t.tup(self.features), dtype=float)
                    for t in trees
                ])
                l = lossfunction(mat @ self.weights, goldidx)
                l.backward()
                loss += l
            optim.step()
            print(f"finished epoch {epoch}, loss: {loss.item()}")
            if not devset is None:
                self.evaluate(devset)


    def evaluate(self, devset):
        evaluator = Evaluator(self.evalparam)
        correct, overshot, early = 0, 0, 0
        for (goldidx, kbest) in devset:
            bestidx, best = self.select(kbest)
            gold, _ = kbest[goldidx]
            sent = [str(i) for i in range(len(kbest[0][0].leaves()))]
            evaluator.add(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(best), sent)
            if bestidx > goldidx:
                overshot += 1
            elif goldidx > bestidx:
                early += 1
            else:
                correct += 1
        print("monitoring dev set, f-score:", evaluator.acc.scores()["lf"])
        print("correct:", correct, "overshot:", overshot, "too early:", early)


    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        mat = torch.stack([
            torch.tensor(self.features.extract(tree).add("parsing_score", sidx+1).tup(self.features), dtype=float)
            for sidx, (tree, weight) in enumerate(kbest)
        ])
        scores = (mat @ self.weights)
        idx = scores.argmax()
        return idx, kbest[idx][0]