from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from discodop.tree import Tree
from math import log

@dataclass(frozen=True)
class DopRule:
    root: str
    children: tuple[str, ...]
    grandchildren: tuple[int, tuple[str, ...]] | None = None

    @classmethod
    def at_root(cls, tree: Tree):
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return
        root = tree.label
        children = tuple(n.label for n in tree)
        yield cls(root, children)
        for i, child in enumerate(tree):
            if len(child) == 1 and not isinstance(child[0], Tree):
                continue
            grandchildren = tuple(n.label for n in child)
            yield cls(root, children, (i, grandchildren))


class Dop:
    def __init__(self, training_set: Iterable[Tree], min_occurrences: int = 1):
        rules = Counter()
        rules_per_sentence = Counter()
        for tree in training_set:
            seen_rules = set()
            for subtree in tree.subtrees():
                for rule in DopRule.at_root(subtree):
                    rules[rule] += 1
                    if not rule in seen_rules:
                        rules_per_sentence[rule] += 1
                        seen_rules.add(rule)
        self.logocc = {
            rule: 1+log(occurrences)
            for rule, occurrences in rules.items()
            if rules_per_sentence[rule] >= min_occurrences
        }
        
    def match(self, tree: Tree) -> float:
        reward = 0
        for subtree in tree.subtrees():
            for rule in DopRule.at_root(subtree):
                reward += self.logocc.get(rule, 0)
        return reward
    
    def select(self, trees: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        bestidx, besttree, bestweight = None, None, None
        for i, (tree, _) in enumerate(trees):
            weight = self.match(tree)
            if bestweight is None or weight > bestweight:
                bestidx, besttree, bestweight = i, tree, weight
        return bestidx, besttree