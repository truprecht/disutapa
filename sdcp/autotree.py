from discodop.tree import Tree

class AutoTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        self.children.sort(
            key=lambda node: node._minleaf if isinstance(node, AutoTree) else node)
        self._minleaf = next(
            (c._minleaf if isinstance(c, AutoTree) else c) for c in self.children)


def test_tree():
    assert AutoTree("(S 0 1 2)") == AutoTree("(S 2 0 1)")
    assert AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))") == AutoTree("(SBAR+S (NP 1 2) (VP 3 (VP 0 4 5)))")