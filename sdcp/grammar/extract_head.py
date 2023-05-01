from discodop.tree import Tree, ImmutableTree
from dataclasses import dataclass, field

from ..headed_tree import HeadedTree, HEAD
from .extract import fanout


def read_clusters(filename):
    label_to_clusterid = {}
    with open(filename, "r") as cfile:
        for line in cfile:
            array = line.strip().split()
            clusterid = array[0]
            for label in array:
                assert not label in label_to_clusterid, f"label {label} appears multiple times in {filename}"
                label_to_clusterid[label] = clusterid
    return label_to_clusterid


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

    def guess_lex_position(self) -> int:
        lexidx, clause = 0, self.spine
        while clause != 0:
            for c in clause:
                if isinstance(c, Tree) or c == 0:
                    clause = c
                    break
                lexidx = max(lexidx, c)
        return lexidx

    def __call__(self, lex: int):
        return (lambda *args: self.subst(lex, *args))


def subvars(tree: ImmutableTree, newvars: dict[int, int]):
    if tree == 0:
        return tree
    if not isinstance(tree, Tree):
        return newvars[tree]
    return ImmutableTree(tree.label, (subvars(c, newvars) for c in tree))


@dataclass(frozen=True, init=False, repr=False)
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
        self.__dict__["lexidx"] = lexidx if not lexidx is None else headed_clause(clause).guess_lex_position()

    def fn(self, lex, _):
        return headed_clause(self.clause)(lex), [None]*len(self.rhs)

    def reorder(self, leftmosts: tuple[int]):
        srti = sorted(range(len(leftmosts)), key=leftmosts.__getitem__)
        rordi = {o: i+1 for i, o in enumerate(i for i in srti if not i == 0)}
        lxidx = next(i for i, o in enumerate(srti) if o == 0)
        clause = subvars(self.clause, rordi)
        rhs = (self.rhs[i-1] for i in srti if not i == 0)
        return self.__class__(self.lhs, rhs, clause, self.fanout, lxidx)

    def __repr__(self) -> str:
        clausestr = f", '{self.clause}'" if self.clause != 0 else ""
        fostr = f", fanout={self.fanout}" if self.fanout > 1 else ""
        lexposstr = f", lexidx={self.lexidx}" if self.lexidx != headed_clause(self.clause).guess_lex_position() else ""
        return f"{self.__class__.__name__}({repr(self.lhs)}, {repr(self.rhs)}{clausestr}{fostr}{lexposstr})"
    
    def with_lhs(self, lhs):
        return self.__class__(lhs, self.rhs, self.clause, self.fanout, self.lexidx)


@dataclass
class Nonterminal:
    horzmarkov: int = 999
    vertmarkov: int = 1
    rightmostunary: bool = False
    markrepeats: bool = False
    coarselabels: dict[str, str] = None
    bindirection: bool = False

    def __post_init__(self):
        if self.horzmarkov < 0 or self.vertmarkov < 1:
            raise ValueError("illegal markov. parameters: h =", self.horzmarkov, "and v =", self.vertmarkov)
        if self.coarselabels:
            self.coarselabels = read_clusters(self.coarselabels)

    def get_label(self, node: HeadedTree) -> str:
        if not isinstance(node, Tree): return "ARG"
        label = node.label
        if not self.coarselabels is None:
            if not (nlabel := self.coarselabels.get(label, None)) is None:
                label = nlabel
            else:
                print(f"Warning: in {self.__class__}.get_label: no coarse label found for", label)
        return label

    def vert(self, parents: tuple[str, ...], siblings: tuple[str, ...]) -> str:
        return self(parents) + f"|<{','.join(siblings[:self.horzmarkov])}>"

    def __call__(self, parents: tuple[str, ...]) -> str:
        lab = ";".join(parents[-self.vertmarkov:])
        if self.markrepeats and len(parents) >= 2 and parents[-1] == parents[-2]:
            lab += "+"
        return lab



@dataclass(init=False)
class Extractor:
    root: str = "ROOT"

    def __init__(self, root: str = "ROOT", **ntargs):
        self.nonterminals = Nonterminal(**ntargs)
        self.root = root

    def read_spine(self, tree: HeadedTree, parents: tuple[str, ...], firstvar: int = 1):
        if not isinstance(tree, HeadedTree):
            return 0, [], firstvar
        children = []
        successors = []
        parents += (self.nonterminals.get_label(tree),)
        if tree.headidx > 0:
            successors.append((parents, tree[:tree.headidx], -1))
            children.append(firstvar)
            firstvar+=1
        child, successors_, firstvar = self.read_spine(tree[tree.headidx], parents, firstvar)
        successors.extend(successors_)
        children.append(child)
        if tree.headidx < len(tree)-1:
            successors.append((parents, tree[tree.headidx+1:], +1))
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
        c, succs, _ = self.read_spine(tree, parents)
        for nparents, succ, direction in succs:
            children.append(self.extract_nodes(succ, nparents, direction=direction))
            rhs_nts.append(children[-1].label[2].lhs)
        lhs = overridelhs if not overridelhs is None else \
                self.nonterminals(parents + (self.nonterminals.get_label(tree),))
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
 

    def extract_nodes(self, trees: list[HeadedTree], parents: tuple[str, ...], direction: int = 0):
        markovnts = [trees[-1].label if isinstance(trees[-1], Tree) else "POS"]
        lowestnt = None
        if self.nonterminals.rightmostunary:
            lowestnt = self.nonterminals.vert(parents, markovnts)
        deriv = self.extract_node(trees[-1], lowestnt, parents)
        yd = trees[-1].leaves() if isinstance(trees[-1], Tree) else [trees[-1]]
        for tree in trees[-2::-1]:
            markovnts.append(tree.label if isinstance(tree, Tree) else "POS")
            yd += tree.leaves() if isinstance(tree, Tree) else [tree]
            child = self.extract_node(tree, self.nonterminals.vert(parents, markovnts), parents)
            deriv = self._fuse_modrule(child, deriv, yd)
        direction = {-1: "[L]", +1: ""}[direction]
        deriv.label = (*deriv.label[:2], deriv.label[2].with_lhs(deriv.label[2].lhs+direction))
        return deriv


    def __call__(self, tree):
        derivation = self.extract_node(tree, self.root)
        rules = [r for _, _, r in sorted(node.label for node in derivation.subtrees())]
        for node in derivation.subtrees():
            node.label = node.label[0]
        return rules, derivation