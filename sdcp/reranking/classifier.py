from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam
import torch
from typing import Iterable
from tqdm import tqdm

from .features import FeatureExtractor

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
            kts.append(self.features.extract(silver).add("parsing_score", weight))
            evaluator = TreePairResult(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(silver), sent, self.evalparam)
            score = evaluator.scores()["LF"]
            if bestscore is None or bestscore < score:
                best, bestscore = sidx, score
        self.kbest_trees.append(kts)
        self.oracle_trees.append(best)

        
    def fit(self, epochs: int = 100):
        self.features.truncate(self.featoccs)
        self.features.objects["parsing_score"] = {"": 0}
        self.features.counts[("parsing_score", 0)] = self.featoccs+1

        weights = torch.zeros(len(self.features))
        for epoch in range(epochs):
            iterator = tqdm(zip(self.oracle_trees, self.kbest_trees), total=len(self.oracle_trees), desc=f"training reranking in epoch {epoch}")
            for goldidx, trees in iterator:
                mat = torch.stack([
                    torch.tensor(t.tup(self.features))
                    for t in trees
                ])
                bestidx = (mat @ weights).argmax()
                if bestidx != goldidx:
                    weights += mat[goldidx] - mat[bestidx]
        self.weights = weights


    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        mat = torch.stack([
            torch.tensor(self.features.extract(tree).add("parsing_score", weight).tup(self.features))
            for tree, weight in kbest
        ])
        idx = (mat @ self.weights).argmax()
        return idx, kbest[idx][0]
