from discodop.tree import Tree, ImmutableTree, ImmutableParentedTree, HEAD


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

# TODO merge with AutoTree
class HeadedTree(Tree):
    def __init__(self, label_or_str, children=None):
        if children is None:
            return
        super().__init__(label_or_str, children)
        self.postags = {}
        self.headidx = None
        self.headterm = None
        self._minleaf = None
        for i in range(len(self.children)):
            if not isinstance(self[i], Tree): # just a pos node, child is a sent position
                self.headidx = i
                self.headterm = self[i]
                continue
            if self[i].type == HEAD:
                self.headidx = i
                self.headterm = self[i].headterm
            if len(self[i]) == 1 and not isinstance(self[(i,0)], Tree) and not self[(i, 0)] in self[i].postags.keys():
                self.postags[self[(i,0)]] = self[i].label
                self[i] = self[(i,0)]
            else:
                self.postags.update(self[i].postags)
        self._minleaf = min(c._minleaf if isinstance(c, HeadedTree) else c for c in self)
        if self.headidx is None:
            raise Exception(f"{self.__class__} should be converted from Tree class with initialized head markings")

    def tree(self, override_postags: list[str]|dict[int,str] = None):
        if not override_postags:
            override_postags = self.postags
        children = (
            c.tree(override_postags) if isinstance(c, Tree) else 
                unmerge(override_postags[c], [c]) # if c in self.postags else c
            for c in self.children
        )
        return unmerge(self.label, children)

    def immutable(self):
        return ImmutableTree(self.tree())

    def parented(self):
        return ImmutableParentedTree(self.tree())