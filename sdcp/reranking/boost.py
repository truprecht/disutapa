from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from pickle import dump, load
import torch
from typing import Iterable
from tqdm import tqdm
from collections import defaultdict

from .features import FeatureExtractor
from .classifier import TreeRanker, get_float


class BoostTreeRanker(TreeRanker):
    def __init__(self, min_feature_count: int = 5):
        super().__init__(min_feature_count=min_feature_count)
        self.scores = list()

    def add_tree(self, gold: Tree, kbest: list[tuple[Tree, float]]):
        # extract vectors even when len(kbest)==1, as it counts features
        vectors = [self.features.extract(t).add("parsing_score", w) for t, w in kbest]
        if len(kbest) > 1:
            scores = []
            sentence = [str(i) for i in range(len(gold.leaves()))]
            for candidate, _ in kbest:
                result = TreePairResult(0, ParentedTree.convert(gold), list(sentence), ParentedTree.convert(candidate), list(sentence), self.evalparam)
                scores.append(get_float(result.scores()["LF"]))
            oracleidx, _ = self.__class__.oracle(gold, kbest, self.evalparam)
            self.kbest_trees.append(vectors)
            self.scores.append(torch.tensor([scores[oracleidx]-s for s in scores]))
            self.oracle_trees.append(oracleidx)

    def fit(self, epochs: int = 1000, smoothing: float = 2e-4, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        self.features.truncate(self.featoccs)
        self.weights = torch.zeros(len(self.features), dtype=float)
        self.kbest_trees = [
            torch.tensor([
                vector.tup(self.features)
                for vector in vectors
            ])
            for vectors in self.kbest_trees
        ]

        if not self.kbest_trees:
            print("there are no trees for training")
            return
        
        alphas = torch.arange(-10, 10, 1e-3)
        losses = torch.zeros_like(alphas)
        for trees, scores, oracle in zip(self.kbest_trees, self.scores, self.oracle_trees):
            losses += torch.tensor([
                [-(trees[oracle][0]-vector[0])*alpha for vector in trees]
                for alpha in alphas
            ]).exp() @ scores
        self.weights[0] = alphas[losses.argmin()]
        print("using weight coefficient", self.weights[0])
        
        feature_diffs = torch.zeros((len(self.kbest_trees), max(len(kbest) for kbest in self.kbest_trees), len(self.features)))
        score_diffs = torch.zeros((len(self.kbest_trees), max(len(kbest) for kbest in self.kbest_trees)), dtype=torch.double)
        for i, (vectors, oracleidx) in enumerate(zip(self.kbest_trees, self.oracle_trees)):
            feature_diffs[i, :len(vectors), :] = vectors[oracleidx]-vectors
            score_diffs[i, :len(vectors)] = self.scores[i]

        for epoch in range(epochs):
            w = (-feature_diffs*self.weights).exp()
            w = torch.tensordot(score_diffs, w, dims=2)
            z = w.sum()
            wplus = (w * (feature_diffs > 0)).sum(dim=[0, 1])
            wminus = (w * (feature_diffs < 0)).sum(dim=[0, 1])
            k = (wplus.sqrt() - wminus.sqrt()).argmax()
            delta = ((wplus[k] + z*smoothing) / (wminus[k] + z*smoothing)).log() / 2
            self.weights[k] += delta

            print("select feature", self.features.backward[k], delta)
            
            if not devset is None:
                self.evaluate(devset)