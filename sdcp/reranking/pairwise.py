from discodop.tree import Tree, ParentedTree
from discodop.eval import TreePairResult, readparam, Evaluator
from pickle import dump, load
import torch
from typing import Iterable
from tqdm import tqdm

from .features import FeatureExtractor
from .classifier import TreeRanker, get_float


class PairwiseTreeRanker(TreeRanker):
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
            self.kbest_trees.append(vectors)
            self.scores.append(scores)

    def fit(self, epochs: int = 10, swap_trees_prob: float = 0.2, devset: Iterable[tuple[list[tuple[Tree, float]], Tree]] = None):
        self.features.truncate(self.featoccs)
        self.weights = torch.zeros(len(self.features), dtype=float)
        
        if not self.kbest_trees:
            print("there are no trees for training")
            return
        
        for epoch in range(epochs):
            iterator = tqdm(zip(self.kbest_trees, self.scores), total=len(self.scores), desc=f"training reranking in epoch {epoch}")
            accuracies = 0
            
            for trees, scores in iterator:
                swap_actions = torch.rand(len(trees)-1) < swap_trees_prob
                trees_and_scores = iter(zip(trees,scores))
                current_tree, current_score = next(trees_and_scores)
                for (next_tree, next_score), swap_action in zip(trees_and_scores, swap_actions):
                    gold_action = 1 if current_score >= next_score else -1
                    v = torch.tensor(current_tree.tup(self.features), dtype=float) \
                      - torch.tensor(next_tree.tup(self.features), dtype=float)
                    action = v @ self.weights
                    accuracies += (action * gold_action) > 0

                    if gold_action * action < 1:
                        self.weights += gold_action * v
                    
                    if (gold_action == -1 and not swap_action) or (gold_action == 1 and swap_action):
                        current_tree, current_score = next_tree, next_score
            print(f"finished epoch {epoch}, accuracy: {accuracies/sum(len(scs)-1 for scs in self.scores)}")
            if not devset is None:
                self.evaluate(devset)


    def select(self, kbest: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        iterator = iter(kbest)
        current_idx = 0
        current_tree, current_weight = next(iterator)
        current_vector = torch.tensor(self.features.extract(current_tree).add("parsing_score", current_weight).tup(self.features), dtype=float)
        for idx, (next_tree, next_weight) in enumerate(iterator):
            next_vector = torch.tensor(self.features.extract(next_tree).add("parsing_score", next_weight).tup(self.features), dtype=float)
            action = (current_vector - next_vector) @ self.weights
            if action < 0:
                current_tree, current_vector, current_weight = next_tree, next_vector, next_weight
                current_idx = idx+1
        return current_idx, current_tree