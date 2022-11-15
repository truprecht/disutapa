from discodop.tree import Tree
from dataclasses import dataclass, field

from ..headed_tree import HeadedTree, HEAD


@dataclass
class headed_clause:
    spine: Tree

    def subst(self, lex: int, *args: Tree):
        def _subst(tree, lex, *args):
            if tree == 0:
                return lex
            if not isinstance(tree, Tree):
                return args[tree-1]
            return Tree(tree.label, (_subst(c, lex, *args) for c in tree))
        return _subst(self.spine, lex, *args)

    def __call__(self, lex: int):
        return (lambda *args: self.subst(lex, *args))


@dataclass(frozen=True, init=False)
class headed_rule:
    lhs: str
    rhs: tuple[str]
    clause: str
    fanout: int

    def __init__(self, lhs, rhs, clause, fanout):
        if isinstance(clause, headed_clause):
            clause = str(clause.spine)
        self.__dict__["lhs"], self.__dict__["rhs"], self.__dict__["clause"], self.__dict__["fanout"] = lhs, rhs, clause, fanout

    @property
    def hclause(self):
        return headed_clause(Tree(self.clause))


def extract_node(tree: HeadedTree, overridelhs: str = None):
    if not isinstance(tree, HeadedTree):
        # TODO: use pos symbol?
        lhs = overridelhs if not overridelhs is None else "ARG"
        return Tree((tree, headed_rule(lhs, (), headed_clause(0), 0)), [])
    lex = tree.headterm
    topmost = tree.label.split("+")[0]
    children = []
    rhs_nts = []
    c, succs, _ = read_spine(tree)
    for succ in succs:
        children.append(extract_node(succ))
        rhs_nts.append(children[-1].label[1].lhs)
    lhs = overridelhs if not overridelhs is None else topmost
    return Tree((lex, headed_rule(lhs, tuple(rhs_nts), headed_clause(c), 0)), children)


def read_spine(tree: HeadedTree, firstvar: int = 1):
    if not isinstance(tree, HeadedTree):
        return 0, [], firstvar
    children = []
    successors = []
    for i, child in enumerate(tree):
        if tree.headidx == i:
            child, successors_, firstvar = read_spine(child, firstvar)
            successors.extend(successors_)
            children.append(child)
        else:
            children.append(firstvar)
            successors.append(child)
            firstvar += 1
    return Tree(tree.label, children), successors, firstvar



def extract_head(tree: Tree, override_root: str = "ROOT"):
    derivation = extract_node(tree, override_root)
    return (r for _, r in sorted(node.label for node in derivation.subtrees()))