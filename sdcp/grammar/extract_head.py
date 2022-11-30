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


def subvars(tree: ImmutableTree, newvars: dict[int, int]):
    if tree == 0:
        return tree
    if not isinstance(tree, Tree):
        return newvars[tree]
    return ImmutableTree(tree.label, (subvars(c, newvars) for c in tree))


@dataclass(frozen=True, init=False)
class headed_rule:
    lhs: str
    rhs: tuple[str]
    clause: ImmutableTree
    fanout: int
    lexidx: int

    def __init__(self, lhs, rhs, clause: str | headed_clause = 0, fanout: int = 1, lexidx: int = None):
        if isinstance(clause, str):
            clause = ImmutableTree(clause)
        if isinstance(clause, headed_clause):
            clause = ImmutableTree.convert(clause.spine)
        self.__dict__["lhs"], self.__dict__["rhs"], self.__dict__["clause"], self.__dict__["fanout"] = lhs, tuple(rhs), clause, fanout
        if lexidx is None:
            lexidx = 0
            while clause != 0:
                for c in clause:
                    if isinstance(c, Tree) or c == 0:
                        clause = c
                        break
                    lexidx = max(lexidx, c)
        self.__dict__["lexidx"] = lexidx

    def fn(self, lex, _):
        return headed_clause(self.clause)(lex), [None]*len(self.rhs)

    def reorder(self, leftmosts: tuple[int]):
        srti = sorted(range(len(leftmosts)), key=leftmosts.__getitem__)
        rordi = {o: i+1 for i, o in enumerate(i for i in srti if not i == 0)}
        lxidx = next(i for i, o in enumerate(srti) if o == 0)
        clause = subvars(self.clause, rordi)
        rhs = (self.rhs[i-1] for i in srti if not i == 0)
        return self.__class__(self.lhs, rhs, clause, self.fanout, lxidx)


@dataclass(init=False)
class Extractor:
    hmarkov: int = 999
    vmarkov: int = 1
    rightmostunary: bool = True
    markrepeats: bool = True
    root: str = "ROOT"

    def __init__(self, horzmarkov=999, vertmarkov=1, rightmostunary=True, markrepeats=True, root="ROOT"):
        if vertmarkov < 1:
            raise ValueError("vertical markovization should be ≥ 1")
        self.hmarkov = horzmarkov
        self.vmarkov = vertmarkov
        self.rightmostunary = rightmostunary
        self.markrepeats = markrepeats
        self.root = root

    @classmethod
    def read_spine(cls, tree: HeadedTree, parents: tuple[str, ...], firstvar: int = 1):
        if not isinstance(tree, HeadedTree):
            return 0, [], firstvar
        children = []
        successors = []
        parents += (tree.label,)
        if tree.headidx > 0:
            successors.append((parents, tree[:tree.headidx]))
            children.append(firstvar)
            firstvar+=1
        child, successors_, firstvar = cls.read_spine(tree[tree.headidx], parents, firstvar)
        successors.extend(successors_)
        children.append(child)
        if tree.headidx < len(tree)-1:
            successors.append((parents, tree[tree.headidx+1:]))
            children.append(firstvar)
            firstvar+=1
        return Tree(tree.label, children), successors, firstvar


    def extract_node(self, tree: HeadedTree, overridelhs: str = None, parents: tuple[str, ...] = ()):
        if not isinstance(tree, HeadedTree):
            # TODO: use pos symbol?
            lhs = overridelhs if not overridelhs is None else "ARG"
            return Tree((tree, tree, headed_rule(lhs, (), headed_clause(0), 1)), [])
        lex = tree.headterm
        children = []
        rhs_nts = []
        c, succs, _ = self.__class__.read_spine(tree, parents)
        for nparents, succ in succs:
            children.append(self.extract_nodes(succ, nparents))
            rhs_nts.append(children[-1].label[2].lhs)
        lhs = overridelhs if not overridelhs is None else \
                (";".join((parents+(tree.label,))[-self.vmarkov:]) if parents else tree.label)
        leftmost = min(lex, *(c.label[1] for c in children)) if children else lex
        rule = headed_rule(lhs, tuple(rhs_nts), headed_clause(c), fanout(sorted(tree.leaves()))).reorder((lex,) + tuple(c.label[1] for c in children))
        return Tree((lex, leftmost, rule), children)


    def _fuse_modrule(_self, mod_deriv: Tree, successor_mods: Tree, all_leaves):
        toprule = mod_deriv.label[2]
        botrule = successor_mods.label[2]
        lex = mod_deriv.label[0]
        children = [*mod_deriv.children, successor_mods]
        newrule = headed_rule(
            toprule.lhs,
            toprule.rhs+(botrule.lhs,),
            clause=ImmutableTree(RMLABEL, [toprule.clause, len(toprule.rhs)+1]),
            fanout=fanout(sorted(all_leaves))).reorder((lex,) + tuple(c.label[1] for c in children))
        return Tree((*mod_deriv.label[:2], newrule), children)
 

    def extract_nodes(self, trees: list[HeadedTree], parents: tuple[str, ...]):
        markovnts = [trees[-1].label if isinstance(trees[-1], Tree) else "POS"]
        parentstr = ";".join(parents[-self.vmarkov:])
        if self.markrepeats and len(parents) >= 2 and parents[-1] == parents[-2]:
            parentstr += "+"
        lhsnt = lambda: f"{parentstr}|<{','.join(markovnts[:self.hmarkov])}>"
        deriv = self.extract_node(trees[-1], lhsnt() if self.rightmostunary else None, parents)
        yd = trees[-1].leaves() if isinstance(trees[-1], Tree) else [trees[-1]]
        for tree in trees[-2::-1]:
            markovnts.append(tree.label if isinstance(tree, Tree) else "POS")
            yd += tree.leaves() if isinstance(tree, Tree) else [tree]
            child = self.extract_node(tree, lhsnt(), parents)
            deriv = self._fuse_modrule(child, deriv, yd)
        return deriv


    def __call__(self, tree):
        derivation = self.extract_node(tree, self.root)
        return (r for _, _, r in sorted(node.label for node in derivation.subtrees()))