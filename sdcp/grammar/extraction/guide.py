from abc import ABC, abstractmethod
from ...autotree import AutoTree

Guide_Classes = dict()   

class Guide(ABC):
    def __init__(self, tree: AutoTree):
        pass

    @abstractmethod
    def __call__(self, tree: AutoTree) -> int:
        raise NotImplementedError()

    @staticmethod
    def construct(type: str, tree: AutoTree) -> "Guide":
        Guide_t = Guide_Classes.get(type, StrictGuide)
        return Guide_t(tree)


class StrictGuide(Guide):
    def __call__(self, tree: AutoTree) -> int:
        return min(tree[1].leaves()) if isinstance(tree[1], AutoTree) else tree[1]

Guide_Classes["strict"] = StrictGuide 


class DependentGuide(Guide):
    def __call__(self, tree: AutoTree):
        if tree.headidx == -1:
            raise Exception(f"Could not find the head in the constiuent tree:\n{tree}\nDid you pass the headrules file?")
        depidx = 1 - tree.headidx
        return tree[depidx] if not isinstance(tree[depidx], AutoTree) \
            else tree[depidx].headterm

Guide_Classes["dependent"] = DependentGuide


class VanillaGuide(Guide):
    def _construct_guide(self, tree, moved_leaves):
        if not isinstance(tree, AutoTree):
            return
        try:
            leaf = next(
                c for c in reversed(tree.children) 
                if not isinstance(c, AutoTree) 
                if not c in moved_leaves
            )
        except StopIteration:
            leaf = min(tree[1].leaves())
            moved_leaves.add(leaf)
        self.assignment[id(tree)] = leaf
        for child in tree:
            self._construct_guide(child, moved_leaves)

    def __init__(self, tree: AutoTree):
        self.assignment: dict[AutoTree, int] = dict()
        self._construct_guide(tree, set())

    def __call__(self, tree: AutoTree) -> int:
        return self.assignment[id(tree)]

Guide_Classes["vanilla"] = VanillaGuide


class LeastGuide(Guide):
    @staticmethod
    def bfsleaf(tree: AutoTree, exclude: set[int], exclude_subtrees: bool = False) -> int:
        queue = [tree]
        while queue:
            t = queue.pop(0)
            if not isinstance(t, AutoTree) and not t in exclude:
                return t
            if isinstance(t, AutoTree):
                successors = (
                    c for c in t.children
                    if not exclude_subtrees or not isinstance(c, AutoTree) or \
                        not c.leaves().intersection(exclude)
                )
                queue.extend(successors)
        raise Exception(f"Could not find any admissible leaf in {tree} excluding {exclude}")

    def _construct_guide(self, tree):
        if not isinstance(tree, AutoTree):
            return
        for child in tree:
            self._construct_guide(child)
        leaf = self.__class__.bfsleaf(tree, self.assignment.values())
        self.assignment[id(tree)] = leaf

    def __call__(self, tree: AutoTree) -> int:
        return self.assignment[id(tree)]

    def __init__(self, tree: AutoTree):
        self.assignment: dict[AutoTree, int] = dict()
        self._construct_guide(tree)

class NearGuide(LeastGuide):
    def _construct_guide(self, tree):
        if not isinstance(tree, AutoTree):
            return
        leaf = self.__class__.bfsleaf(tree, self.assignment.values(), exclude_subtrees=True)
        self.assignment[id(tree)] = leaf
        for child in tree:
            self._construct_guide(child)

Guide_Classes["least"] = LeastGuide
Guide_Classes["near"] = NearGuide