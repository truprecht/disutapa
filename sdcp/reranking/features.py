from collections import defaultdict, Counter
from dataclasses import dataclass
from discodop.tree import Tree
from itertools import chain, pairwise
from typing import Any


def bigrams(tree: Tree):
    for node in tree.subtrees(lambda n: len(n) > 1 or isinstance(n[0], Tree)):
        root = node.label
        children = (c.label for c in node)
        for bigram in pairwise(chain(children, ("END",))):
            yield (root, *bigram)

def rules(tree: Tree):
    for node in tree.subtrees(lambda n: len(n) > 1 or isinstance(n[0], Tree)):
        root = node.label
        children = (c.label for c in node)
        yield (root, *children)

def parent_rules(tree: Tree|int):
    queue = [(tree, "TOP")]
    while queue:
        node, gp = queue.pop()
        if not isinstance(node, Tree) or \
                len(node) == 1 and not isinstance(node[0], Tree):
            continue
        root = node.label
        children = (c.label for c in node)
        yield (gp, root, *children)
        queue.extend((child, root) for child in node)


PUNCT = {"$.", "$(", "$,", ",", "."}

def _count_rightmost(tree: Tree|int) -> int:
    if not isinstance(tree, Tree):
        return 0
    
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        return int(not tree.label in PUNCT)
    
    for child in reversed(tree):
        if tree.label in PUNCT:
            continue
        return 1+_count_rightmost(child)

    raise ValueError(f"Oops. did not find a valid path without punctuation in tree {tree}.")


def rightbranch(tree: Tree|int):
    allnodes = sum(1 for t in tree.subtrees())
    rightmostpath = _count_rightmost(tree)
    for _ in range(rightmostpath):
        yield "RightBranch"
    for _ in range(allnodes-rightmostpath):
        yield "NonRightBranch"
    

def parent_bigrams(tree: Tree|int):
    queue = [(tree, "TOP")]
    while queue:
        node, gp = queue.pop()
        if not isinstance(node, Tree) or \
                len(node) == 1 and not isinstance(node[0], Tree):
            continue
        root = node.label
        children = (c.label for c in node)
        for bigram in pairwise(chain(children, ("END",))):
            yield (gp, root, *bigram)
        queue.extend((child, root) for child in node)


@dataclass
class FeatureVector:
    # store features as sparse map during extraction
    features: dict[tuple[str, int], float]

    def add(self, name: str, value: float) -> "FeatureVector":
        self.features[(name, 0)] = value
        return self

    def tup(self, extractor: "FeatureExtractor"):
        return tuple(
            self.features[idx]
            for idx in extractor.iterate_features()
        )


class FeatureExtractor:
    def __init__(self):
        self.objects: dict[str, dict[Any, int]] = dict()
        self.counts: Counter[tuple[str, int]] = Counter()
        self.fixed: bool = False


    def iterate_features(self):
        return (
            (fname, fidx)
            for fname, fobjects in self.objects.items()
            for fidx in fobjects.values()
        )


    def extract(self, tree: Tree|int) -> FeatureVector:
        sparsevec = defaultdict(lambda: 0)

        # counted objects features
        for featname in ("rules", "parent_rules", "bigrams", "parent_bigrams", "rightbranch"):
            if self.fixed and not featname in self.objects:
                continue
            feature_iterator = globals()[featname](tree)
            feature_dict = self.objects.setdefault(featname, {})
            for obj in feature_iterator:
                if self.fixed and not obj in feature_dict:
                    continue
                idx = feature_dict.setdefault(obj, len(feature_dict))
                sparsevec[(featname, idx)] += 1
                if not self.fixed:
                    self.counts[(featname, idx)] += 1

        return FeatureVector(sparsevec)
    
    
    def truncate(self, mincount: int = 5):
        self.objects = {
            featname: {
                k: idx
                for k, idx in featobjects.items()
                if self.counts[(featname, idx)] >= mincount
            }
            for featname, featobjects in self.objects.items()
        }
        self.counts = {
            k: v
            for k, v in self.counts.items()
            if v >= mincount
        }
        self.fixed = True

    
    def __len__(self):
        return len(self.counts)