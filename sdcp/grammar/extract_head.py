from discodop.tree import Tree, ImmutableTree
from dataclasses import dataclass, field

from ..headed_tree import HeadedTree, HEAD
from .extract import fanout


RMLABEL = "_|<>"

@dataclass(init=False)
class headed_clause:
    spine: Tree

    def __init__(self, spine: Tree|str) -> None:
        self.spine = ImmutableTree(spine) if isinstance(spine, str) else spine

    def subst(self, lex: int, *args: list[Tree]) -> list[Tree]:
        def _subst(tree, lex, *args):
            if tree == 0:
                return [lex]
            if not isinstance(tree, Tree):
                return args[tree-1]
            children = list(restree for c in tree for restree in _subst(c, lex, *args))
            if tree.label == RMLABEL:
                return children
            return [Tree(tree.label, children)]
        return _subst(self.spine, lex, *args)

    def __call__(self, lex: int):
        return (lambda *args: self.subst(lex, *args))


@dataclass(frozen=True, init=False)
class headed_rule:
    lhs: str
    rhs: tuple[str]
    _clause: ImmutableTree
    fanout: int

    def __init__(self, lhs, rhs, clause: str | headed_clause = 0, fanout = 1):
        if isinstance(clause, str):
            clause = ImmutableTree(clause)
        if isinstance(clause, headed_clause):
            clause = ImmutableTree.convert(clause.spine)
        self.__dict__["lhs"], self.__dict__["rhs"], self.__dict__["_clause"], self.__dict__["fanout"] = lhs, tuple(rhs), clause, fanout

    @property
    def clause(self):
        return headed_clause(self._clause)

    def fn(self, lex, _):
        return self.clause(lex), [None]*len(self.rhs)


def extract_node(tree: HeadedTree, overridelhs: str = None, hmarkov: int = 999, markendpoint: bool = True):
    if not isinstance(tree, HeadedTree):
        # TODO: use pos symbol?
        lhs = overridelhs if not overridelhs is None else "ARG"
        return Tree((tree, tree, headed_rule(lhs, (), headed_clause(0), 1)), [])
    lex = tree.headterm
    children = []
    rhs_nts = []
    c, succs, _ = read_spine(tree)
    for node, succ in succs:
        children.append(extract_nodes(succ, node, hmarkov, markendpoint))
        rhs_nts.append(children[-1].label[2].lhs)
    lhs = overridelhs if not overridelhs is None else tree.label
    leftmost = min(lex, *(c.label[1] for c in children)) if children else lex
    return Tree((lex, leftmost, headed_rule(lhs, tuple(rhs_nts), headed_clause(c), fanout(sorted(tree.leaves())))), children)


def fuse_modrule(mod_deriv: Tree, successor_mods: Tree, all_leaves):
    # TODO: reorder vars according to leftmost position
    toprule = mod_deriv.label[2]
    botrule = successor_mods.label[2]
    newrule = headed_rule(
        toprule.lhs,
        toprule.rhs+(botrule.lhs,),
        clause=ImmutableTree(RMLABEL, [toprule._clause, len(toprule.rhs)+1]),
        fanout=fanout(sorted(all_leaves)))
    return Tree((*mod_deriv.label[:2], newrule), [*mod_deriv.children, successor_mods])
 

def extract_nodes(trees: list[HeadedTree], parent: str, hmarkov: int = 999, markendpoint: bool = True):
    markovnts = [trees[-1].label if isinstance(trees[-1], Tree) else "POS"]
    lhsnt = lambda: f"{parent}|<{','.join(markovnts[:hmarkov])}>"
    deriv = extract_node(trees[-1], lhsnt() if markendpoint else None, hmarkov, markendpoint)
    yd = trees[-1].leaves() if isinstance(trees[-1], Tree) else [trees[-1]]
    for tree in trees[-2::-1]:
        markovnts.append(tree.label if isinstance(tree, Tree) else "POS")
        yd += tree.leaves() if isinstance(tree, Tree) else [tree]
        child = extract_node(tree, lhsnt(), hmarkov, markendpoint)
        deriv = fuse_modrule(child, deriv, yd)
    return deriv


def read_spine(tree: HeadedTree, firstvar: int = 1):
    # TODO: reorder vars according to leftmost position
    if not isinstance(tree, HeadedTree):
        return 0, [], firstvar
    children = []
    successors = []
    if tree.headidx > 0:
        successors.append((tree.label, tree[:tree.headidx]))
        children.append(firstvar)
        firstvar+=1
    child, successors_, firstvar = read_spine(tree[tree.headidx], firstvar)
    successors.extend(successors_)
    children.append(child)
    if tree.headidx < len(tree)-1:
        successors.append((tree.label, tree[tree.headidx+1:]))
        children.append(firstvar)
        firstvar+=1
    return Tree(tree.label, children), successors, firstvar


def extract_head(tree: Tree, override_root: str = "ROOT", hmarkov: int = 999, markendpoint: bool = True):
    derivation = extract_node(tree, override_root, hmarkov, markendpoint)
    return (r for _, _, r in sorted(node.label for node in derivation.subtrees()))