from sklearn import svm, pipeline, preprocessing, linear_model
import numpy as np

from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from typing import Iterable

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
            kts.append(self.features.extract(silver).add("parsing_score", sidx+1))
            # kts.append(self.features.extract(silver))
            evaluator = TreePairResult(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(silver), sent, self.evalparam)
            score = evaluator.scores()["LF"]
            if bestscore is None or bestscore < score:
                best, bestscore = sidx, score
        self.kbest_trees.append(kts)
        self.oracle_trees.append(best)


    def fit(self, devset = None):
        self.features.truncate(self.featoccs)
        self.features.objects["parsing_score"] = {"": 0}
        self.features.counts[("parsing_score", 0)] = self.featoccs+1

        X = np.array([
            t.tup(self.features)
            for trees in self.kbest_trees
            for t in trees
        ])
        y = np.array([
            1 if i == goldidx else -1
            for trees, goldidx in zip(self.kbest_trees, self.oracle_trees)
            for i in range(len(trees))
        ])

        print(X)
        self.classifier = linear_model.Perceptron()
        self.classifier.fit(X, y)
        print(self.classifier.coef_)


    # def evaluate(self, devset):
    #     evaluator = Evaluator(self.evalparam)
    #     correct, overshot, early = 0, 0, 0
    #     for (goldidx, kbest) in devset:
    #         bestidx, best = self.select(kbest)
    #         gold, _ = kbest[goldidx]
    #         sent = [str(i) for i in range(len(kbest[0][0].leaves()))]
    #         evaluator.add(0, ParentedTree.convert(gold), list(sent), ParentedTree.convert(best), sent)
    #         if bestidx > goldidx:
    #             overshot += 1
    #         elif goldidx > bestidx:
    #             early += 1
    #         else:
    #             correct += 1
    #     print("monitoring dev set, f-score:", evaluator.acc.scores()["lf"])
    #     print("correct:", correct, "overshot:", overshot, "too early:", early)


    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        X = np.array([
            self.features.extract(tree).add("parsing_score", sidx+1).tup(self.features)
            for sidx, (tree, weight) in enumerate(kbest)
        ])
    
        scores = self.classifier.predict(X)
        bestidx = scores.argmax()
        print(scores, bestidx)

        return bestidx, kbest[bestidx][0]