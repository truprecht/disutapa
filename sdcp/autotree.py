from calendar import c
from discodop.tree import Tree, ImmutableTree, ImmutableParentedTree


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
            if len(self[i]) == 1 and not isinstance(self[(i,0)], Tree):
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
                Tree(override_postags[c], [c]) # if c in self.postags else c
            for c in self.children
        )
        return Tree(self.label, children)

    def immutable(self):
        return ImmutableTree(self.tree())

    def parented(self):
        return ImmutableParentedTree(self.tree())