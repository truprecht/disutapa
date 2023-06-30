from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from pickle import dump, load
import torch
from typing import Iterable
from tqdm import tqdm

from .features import FeatureExtractor



def perceptron_loss(scores, goldidx):
    prediction = scores.argmax()
    if prediction == goldidx:
        return torch.tensor(0.0, requires_grad=True)
    return scores[prediction]-scores[goldidx]


class TreeRanker:
    def __init__(self, min_feature_count: int = 5, evalparam: dict | None = None):
        self.features = FeatureExtractor()
        self.features.objects["parsing_score"] = {"": 0}
        self.features.counts[("parsing_score", 0)] = min_feature_count+1
        self.featoccs = min_feature_count
        self.oracle_trees = list()
        self.kbest_trees = list()
        self.evalparam = evalparam if not evalparam is None else \
            readparam("../disco-dop/proper.prm")


    @classmethod
    def oracle(cls, gold: Tree, kbest: list[tuple[Tree, float]], evalparam: dict):
        sent = [str(i) for i in range(len(gold.leaves()))]
        best, bestscore = None, None
        for sidx, (silver, weight) in enumerate(kbest):
            evaluator = TreePairResult(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(silver), sent, evalparam)
            score = evaluator.scores()["LF"]
            if bestscore is None or bestscore < score:
                best, bestscore = sidx, score
        return best, kbest[best]

    
    def add_tree(self, gold: Tree, kbest: list[tuple[Tree, float]]):
        # extract vectors even when len(kbest)==1, as it counts features
        vectors = [self.features.extract(t).add("parsing_score", w) for t, w in kbest]
        if len(kbest) > 1:
            oracle_index, _ = self.__class__.oracle(gold, kbest, self.evalparam)
            self.kbest_trees.append(vectors)
            self.oracle_trees.append(oracle_index)


    def fit(self, epochs: int = 10, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        self.weights = torch.zeros(len(self.features), dtype=float)

        if not self.oracle_trees:
            print("there are no trees for training")
            return
        
        self.features.truncate(self.featoccs)

        for epoch in range(epochs):
            iterator = tqdm(zip(self.oracle_trees, self.kbest_trees), total=len(self.oracle_trees), desc=f"training reranking in epoch {epoch}")
            accuracies = 0
            for goldidx, trees in iterator:
                mat = torch.stack([
                    torch.tensor(t.tup(self.features), dtype=float)
                    for t in trees
                ])
                selection = (mat @ self.weights).argmax()
                self.weights += mat[goldidx] - mat[selection]
                accuracies += int(selection == goldidx)
            print(f"finished epoch {epoch}, accuracy: {accuracies/len(self.oracle_trees)}")
            if not devset is None:
                self.evaluate(devset)


    def evaluate(self, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        evaluator = Evaluator(self.evalparam)
        correct, overshot, early = 0, 0, 0
        for (kbest, goldtree) in devset:
            prediction_idx, preciction_tree = self.select(kbest)
            oracle_idx, _ = self.__class__.oracle(goldtree, kbest, self.evalparam)
            
            sent = [str(i) for i in range(len(kbest[0][0].leaves()))]
            evaluator.add(0, ParentedTree.convert(goldtree), list(sent), ParentedTree.convert(preciction_tree), sent)
            if prediction_idx > oracle_idx:
                overshot += 1
            elif oracle_idx > prediction_idx:
                early += 1
            else:
                correct += 1
        print("monitoring dev set, f-score:", evaluator.acc.scores()["lf"])
        print("correct:", correct, "overshot:", overshot, "too early:", early)


    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        mat = torch.stack([
            torch.tensor(self.features.extract(tree).add("parsing_score", weight).tup(self.features), dtype=float)
            for sidx, (tree, weight) in enumerate(kbest)
        ])
        scores = (mat @ self.weights)
        idx = scores.argmax()
        return idx, kbest[idx][0]