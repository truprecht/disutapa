from discodop.tree import Tree
from dataclasses import dataclass, field
from .lcfrs import disco_span

@dataclass(init=False)
class Derivation:
    rule: int
    leaf: int
    children: tuple["Derivation"]
    yd: disco_span
    size: int

    def __init__(self, rule, leaf, len: int, children: tuple["Derivation"] = ()):
        self.rule = rule
        self.leaf = leaf
        self.children = children
        self.yd = disco_span.singleton(self.leaf)
        for child in children:
            self.yd = self.yd.exclusive_union(child.yd)
        self.size = 1 + sum(c.size for c in self.children)
        self.inner_nodes = (1 + sum(c.inner_nodes for c in self.children)) \
            if self.children else 0

    def subderivs(self):
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    @classmethod
    def from_tree(cls, leaftree, ruleseq):
        leaf = leaftree.label
        return Derivation(ruleseq[leaf], leaf, len(ruleseq), [cls.from_tree(c, ruleseq) for c in leaftree])

    @classmethod
    def from_str(cls, leaftree: str, ruleseq: list[int]):
        deriv = Tree(0, []) if leaftree == "0" else \
            Tree.parse(leaftree, parse_label=int, parse_leaf=lambda x: Tree(int(x), []))
        return cls.from_tree(deriv, ruleseq)
        
