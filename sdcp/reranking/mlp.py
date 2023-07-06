from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from pickle import dump, load
import torch
from typing import Iterable
from tqdm import tqdm
from collections import defaultdict

from .features import FeatureExtractor, redistribute
from .classifier import TreeRanker, get_float


class MlpTreeRanker(TreeRanker):
    def __init__(self, min_feature_count: int = 5, hidden_dim: int = int(1e4)):
        super().__init__(min_feature_count=min_feature_count)
        self.scores = list()
        self.hidden_dim = hidden_dim

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

    def fit(self, epochs: int = 10, lr: float = 1e-3, batchsize=32, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        self.features.truncate(self.featoccs)
        self.kbest_trees = [
            torch.stack([
                vector.expand(self.features)
                for vector in vectors
            ])
            for vectors in self.kbest_trees
        ]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(len(self.features), self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )


        if not self.kbest_trees:
            print("there are no trees for training")
            return
        
        adam = torch.optim.Adam(self.mlp.parameters(True), lr=lr)
        for epoch in range(epochs):
            accuracy = 0
            accloss = 0
            steps = 0
            for example, oracleidx, scorediffs in zip(self.kbest_trees, self.oracle_trees, self.scores):
                scores = self.mlp(example).squeeze()
                loss = torch.nn.functional.relu((1-scores[oracleidx]+scores)*scorediffs).sum()
                loss.backward()
                accloss += loss.item()
                steps += 1
                accuracy += scores.argmax().item() == oracleidx
                if steps % batchsize == 0:
                    adam.step()
                    adam.zero_grad() 
                    print("loss =", accloss/steps)
                    print("accuracy =", accuracy/steps)
            if steps % batchsize != 0:
                adam.step()
                adam.zero_grad() 
                print("finished epoch", epoch)
                print("loss =", accloss/steps)
                print("accuracy =", accuracy/steps)
            if not devset is None:
                self.evaluate(devset)



    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        vectors = self.features.extract(t for t, _ in kbest)
        vectors = [v.add("parsing_score", w).expand(self.features) for v, w in zip(vectors, redistribute([w for _, w in kbest]))]
        mat = torch.stack(vectors)

        scores = self.mlp(mat)
        idx = scores.argmax()
        return idx, kbest[idx][0]        