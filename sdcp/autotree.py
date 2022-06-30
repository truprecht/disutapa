from discodop.tree import Tree

class AutoTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        q = [self]
        while q:
            node = q.pop()
            node.children.sort(key=lambda node: min(node.leaves()) if isinstance(node, Tree) else node)
            q.extend(n for n in node.children if isinstance(n, Tree))


def test_tree():
    assert AutoTree("(S 0 1 2)") == AutoTree("(S 2 0 1)")
    assert AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))") == AutoTree("(SBAR+S (NP 1 2) (VP 3 (VP 0 4 5)))")