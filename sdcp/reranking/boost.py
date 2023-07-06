from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from pickle import dump, load
import torch
from typing import Iterable
from tqdm import tqdm
from collections import defaultdict

from .features import FeatureExtractor, redistribute
from .classifier import TreeRanker, get_float


class BoostTreeRanker(TreeRanker):
    def __init__(self, min_feature_count: int = 5):
        super().__init__(min_feature_count=min_feature_count)
        self.scores = list()

    def add_tree(self, gold: Tree, kbest: list[tuple[Tree, float]]):
        if len(kbest) <= 1:
            return
        vectors = self.features.extract(t for t, _ in kbest)
        vectors = [v.add("parsing_score", w) for v, w in zip(vectors, redistribute([w for _, w in kbest]))]
        scores = []
        sentence = [str(i) for i in range(len(gold.leaves()))]
        for candidate, _ in kbest:
            result = TreePairResult(0, ParentedTree.convert(gold), list(sentence), ParentedTree.convert(candidate), list(sentence), self.evalparam)
            candidate_len = sum(1 for node in candidate.subtrees())
            scores.append(get_float(result.scores()["LF"]) * candidate_len / (100 * (len(kbest)-1)))
        oracleidx, _ = self.__class__.oracle(gold, kbest, self.evalparam)
        self.kbest_trees.append(vectors)
        self.scores.append(torch.tensor([scores[oracleidx]-s for s in scores]))
        self.oracle_trees.append(oracleidx)

    def fit(self, epochs: int = int(1e4), smoothing: float = 1e-7, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        self.features.truncate(self.featoccs)
        self.weights = torch.zeros(len(self.features))
        self.kbest_trees = [
            torch.stack([
                vector.expand(self.features)
                for vector in vectors
            ])
            for vectors in self.kbest_trees
        ]

        if not self.kbest_trees:
            print("there are no trees for training")
            return
        
        max_k_per_sentence = max(len(kbest) for kbest in self.kbest_trees)
        feature_diffs = torch.zeros((len(self.kbest_trees), max_k_per_sentence, len(self.features)))
        score_diffs = torch.zeros((len(self.kbest_trees), max_k_per_sentence))
        for i, (vectors, oracleidx) in enumerate(zip(self.kbest_trees, self.oracle_trees)):
            feature_diffs[i, :len(vectors), :] = vectors[oracleidx]-vectors
            score_diffs[i, :len(vectors)] = self.scores[i]

        alphas = torch.arange(-10, 10, 1e-3)
        single_losses = torch.tensordot(-feature_diffs[:,:,0].unsqueeze(-1), alphas.unsqueeze(-1), dims=([2], [1])).exp().permute(2,0,1)
        mean_loss = torch.tensordot(single_losses, score_diffs, dims=([1,2], [0,1]))
        self.weights[0] = alphas[mean_loss.argmin()]
        print("using weight coefficient", self.weights[0])

        aplus = (feature_diffs.permute(2,0,1) > 0).to(dtype=torch.float)
        aminus = (feature_diffs.permute(2,0,1) < 0).to(dtype=torch.float)
        for epoch in range(epochs):
            print("accuracy", sum(int((vectors @ self.weights).argmax() == oracleidx) for vectors, oracleidx in zip(self.kbest_trees, self.oracle_trees)))

            w = ((-feature_diffs*self.weights).exp().permute(2,0,1) * score_diffs)
            z = w.sum()
            wplus = (w * aplus).sum(dim=[1,2])
            wminus = (w * aminus).sum(dim=[1,2])
            k = (wplus.sqrt() - wminus.sqrt()).abs().argmax()
            delta = ((wplus[k] + z*smoothing) / (wminus[k] + z*smoothing)).log() / 2
            self.weights[k] += delta

            print("select feature", self.features.backward[k], delta)
            
            if not devset is None:
                self.evaluate(devset)