from collections import defaultdict, Counter
from dataclasses import dataclass
from discodop.tree import Tree
from itertools import chain, pairwise
from typing import Any
import torch
from typing import Iterable
from sortedcontainers import SortedSet


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


def ranks(tree: Tree):
    for node in tree.subtrees(lambda n: len(n) > 1 or isinstance(n[0], Tree)):
        root = node.label
        yield (root, len(node.children))


def wordedges(tree: Tree|int, sent: list[str]):
    for node in tree.subtrees(lambda n: len(n) > 1 or isinstance(n[0], Tree)):
        root = node.label
        before, after = min(tree.leaves())-1, max(tree.leaves())+1
        before = sent[before] if before >= 0 else "BOS"
        after = sent[after] if after < len(sent) else "EOS"
        yield (root, len(node.leaves()), before, after)


def branching_direction(tree: Tree):
    for node in tree.subtrees(lambda n: len(n) > 1 or isinstance(n[0], Tree)):
        root = node.label
        is_leaf_node: list[bool] = [len(c) == 1 and not isinstance(c[0], Tree) for c in node]
        num_branches = sum(int(not b) for b in is_leaf_node)

        if num_branches == 1 and not is_leaf_node[-1]:
            yield (root, "Right")
        elif num_branches == 1 and not is_leaf_node[0]:
            yield (root, "Left")
        elif num_branches == 1:
            yield (root, "Center")
        elif num_branches > 1:
            yield (root, "Multi", num_branches)
        else:
            yield (root, "None")


def redistribute(ws: Iterable[float]) -> list[int]:
    buckets = SortedSet(ws)
    return [buckets.bisect_left(weight) for weight in ws]



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

    def expand(self, extractor: "FeatureExtractor") -> torch.Tensor:
        t = torch.zeros(len(extractor))
        for key, value in self.features.items():
            if not (idx := extractor.key_to_idx.get(key)) is None:
                t[idx] = value
        return t


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


    def _extract(self, tree: Tree|int) -> FeatureVector:
        sparsevec = Counter()

        # counted objects features
        for featname in ("rules", "parent_rules", "bigrams", "parent_bigrams", "rightbranch", "branching_direction", "ranks"):
            if self.fixed and not featname in self.objects:
                continue
            feature_iterator = globals()[featname](tree)
            feature_dict = self.objects.setdefault(featname, {})
            for obj in feature_iterator:
                if self.fixed and not obj in feature_dict:
                    continue
                idx = feature_dict.setdefault(obj, len(feature_dict))
                sparsevec[(featname, idx)] += 1

        return FeatureVector(sparsevec)
    

    def extract(self, trees: list[Tree]):
        features = set()
        for tree in trees:
            vec = self._extract(tree)
            if not self.fixed:
                for f in vec.features:
                    if f in features: continue
                    self.counts[f] += 1
                features.update(vec.features)
            yield vec

    
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
        self.backward = [
            (fname, fidx)
            for fname, fobjects in self.objects.items()
            for fidx in fobjects.keys()
        ]
        self.key_to_idx = {
            (fname, idx): i
            for i, (fname, idx) in enumerate(self.iterate_features())
        }

    
    def __len__(self):
        return len(self.counts)