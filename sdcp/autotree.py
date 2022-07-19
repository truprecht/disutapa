from calendar import c
from discodop.tree import Tree, ImmutableTree, ImmutableParentedTree


def unmerge(tree_or_label: Tree|str, children: list = None):
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
    def __init__(self, label_or_str, children=None):
        if children is None:
            return
        super().__init__(label_or_str, children)
        self.children.sort(
            key=lambda node: node._minleaf if isinstance(node, AutoTree) else node)
        self.postags = {}
        for i in range(len(self.children)):
            if not isinstance(self[i], Tree):
                continue
            if len(self[i]) == 1 and not isinstance(self[(i,0)], Tree) and not self[(i, 0)] in self[i].postags.keys():
                self.postags[self[(i,0)]] = self[i].label
                self[i] = self[(i,0)]
            else:
                self.postags.update(self[i].postags)
        self._minleaf = next(
            (c._minleaf if isinstance(c, AutoTree) else c) for c in self.children)

    def tree(self, override_postags: list[str]|dict[int,str] = None):
        if not override_postags:
            override_postags = self.postags
        children = (
            c.tree(override_postags) if isinstance(c, AutoTree) else 
                unmerge(override_postags[c], [c]) # if c in self.postags else c
            for c in self.children
        )
        return unmerge(self.label, children)

    def immutable(self):
        return ImmutableTree(self.tree())

    def parented(self):
        return ImmutableParentedTree(self.tree())