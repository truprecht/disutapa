from discodop.tree import Tree, ImmutableTree, ParentedTree, HEAD  # type: ignore
from sortedcontainers import SortedSet # type: ignore
from typing import Iterable

def unmerge(tree_or_label: Tree|str, children: Iterable[Tree] | None = None) -> Tree:
    if not children is None:
        tree_or_label = Tree(tree_or_label, children)
    else:    
        assert isinstance(tree_or_label, Tree)
    labels = tree_or_label.label.split("+")
    children = tree_or_label.children
    for l in reversed(labels[1:]):
        children = [Tree(l, children)]
    tree_or_label.label = labels[0]
    tree_or_label.children = children
    return tree_or_label


class AutoTree(Tree):
    def __init__(self, label_or_str: str, children: Iterable["AutoTree"] | None = None):
        if children is None:
            return
        super().__init__(label_or_str, children)
        self.children.sort(
            key=lambda node: node._leaves[0] if isinstance(node, AutoTree) else node)
        self.headidx: int = -1
        self.headterm: int = -1
        self.postags: dict[int, str] = {}
        self._leaves: SortedSet = SortedSet()
        for i in range(len(self.children)):
            if not isinstance(self[i], Tree): # just a pos node, child is a sent position
                self.headidx = i
                self.headterm = self[i]
                self._leaves.add(self[i])
                continue
            self._leaves |= self[i]._leaves
            if self[i].type == HEAD:
                self.headidx = i
                self.headterm = self[i].headterm
            if len(self[i]) == 1 and not isinstance(self[(i,0)], Tree) and not self[(i, 0)] in self[i].postags.keys():
                self.postags[self[(i,0)]] = self[i].label
                self[i] = self[(i,0)]
            else:
                self.postags.update(self[i].postags)
        
    def leaves(self) -> SortedSet:
        return self._leaves

    def tree(self, override_postags: list[str] | dict[int,str] = []) -> Tree:
        if not override_postags:
            override_postags = self.postags
        children = (
            c.tree(override_postags) if isinstance(c, AutoTree) else 
                unmerge(override_postags[c], [c])
            for c in self.children
        )
        return unmerge(self.label, children)

    def immutable(self, *args):
        return ImmutableTree.convert(self.tree(*args))

    def parented(self, *args):
        return ParentedTree.convert(self.tree(*args))


def fix_rotation(tree: Tree):
    if not isinstance(tree, Tree):
        return tree, tree
    leftmosts, children = zip(*sorted(fix_rotation(c) for c in tree))
    return leftmosts[0], Tree(tree.label, children)


def with_pos(tree: Tree, pos: tuple[str, ...]):
    if not isinstance(tree, Tree):
        cs = [tree]
        labels = pos[tree].split("+")
        for l in reversed(labels[1:]):
            cs = [Tree(l, cs)]
        return Tree(labels[0], cs)
    return Tree(tree.label, [with_pos(t, pos) for t in tree])