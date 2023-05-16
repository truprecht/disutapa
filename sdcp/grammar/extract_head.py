from discodop.tree import Tree, ImmutableTree
from dataclasses import dataclass, field
from sortedcontainers import SortedSet
from collections import namedtuple

from ..headed_tree import HeadedTree, HEAD
from .lcfrs import lcfrs_composition, ordered_union_composition


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
    composition: lcfrs_composition | ordered_union_composition = None

    def __init__(self, lhs, rhs, clause: str | headed_clause = 0, composition: lcfrs_composition | str | None = None):
        if isinstance(clause, str):
            clause = ImmutableTree(clause)
        if isinstance(clause, headed_clause):
            clause = ImmutableTree.convert(clause.spine)
        self.__dict__["lhs"], self.__dict__["rhs"], self.__dict__["clause"] = lhs, tuple(rhs), clause
        if composition is None:
            composition = lcfrs_composition(range(len(rhs)+1))
        if isinstance(composition, str):
            composition = lcfrs_composition(composition)
        self.__dict__["composition"] = composition

    @property
    def fanout(self):
        return self.composition.fanout        

    def fn(self, lex, _):
        return headed_clause(self.clause)(lex), [None]*len(self.rhs)

    def __repr__(self) -> str:
        clausestr = f", '{self.clause}'" if self.clause != 0 else ""
        comp = f", composition={repr(self.composition)}" if self.composition != lcfrs_composition(range(len(self.rhs)+1)) else ""
        return f"{self.__class__.__name__}({repr(self.lhs)}, {repr(self.rhs)}{clausestr}{comp})"
    
    def with_lhs(self, lhs):
        return self.__class__(lhs, self.rhs, self.clause, self.composition)


@dataclass
class Nonterminal:
    horzmarkov: int = 999
    vertmarkov: int = 1 
    rightmostunary: bool = False
    markrepeats: bool = False
    coarselabels: dict[str, str] = None
    bindirection: bool = False
    mark: str = "plain"

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
    
    def fo(self, fanout: int):
        if self.mark.startswith("f") and fanout > 1: return f"/{fanout}"
        if self.mark.startswith("d") and fanout > 1: return "/D"
        return ""


extraction_result = namedtuple("extracion_result", ["lex", "leaves", "rule"])
@dataclass(init=False)
class Extractor:
    nonterminals: Nonterminal
    root: str = "ROOT"
    ctype: str = "lcfrs"

    def __init__(self, root: str = "ROOT", composition: str = "lcfrs", **ntargs):
        self.nonterminals = Nonterminal(**ntargs)
        self.root = root
        self.__binarize = self.nonterminals.horzmarkov < 999
        self.ctype = {"lcfrs": lcfrs_composition, "dcp": ordered_union_composition}.get(composition, composition)

    def read_spine(self, tree: HeadedTree, parents: tuple[str, ...], firstvar: int = 1):
        if not isinstance(tree, HeadedTree):
            return 0, [], firstvar
        children = []
        successors = []
        parents += (self.nonterminals.get_label(tree),)
        if tree.headidx > 0:
            if not self.__binarize:
                ts = [[tree[i]] for i in range(tree.headidx)]
            else:
                ts = [tree[:tree.headidx]]
            for t in ts:
                successors.append((parents, t, -1))
                children.append(firstvar)
                firstvar+=1
        child, successors_, firstvar = self.read_spine(tree[tree.headidx], parents, firstvar)
        successors.extend(successors_)
        children.append(child)
        if tree.headidx < len(tree)-1:
            if not self.__binarize:
                ts = [[t] for t in tree[tree.headidx+1:]]
            else:
                ts = [tree[tree.headidx+1:]]
            for t in ts:
                successors.append((parents, t, +1))
                children.append(firstvar)
                firstvar+=1
        return Tree(tree.label, children), successors, firstvar


    def extract_node(self, tree: HeadedTree, overridelhs: str = None, parents: tuple[str, ...] = ()):
        if not isinstance(tree, HeadedTree):
            # TODO: use pos symbol?
            lhs = overridelhs if not overridelhs is None else f"ARG({parents[-1]})"
            resultnode = extraction_result(tree, SortedSet([tree]), headed_rule(lhs, ()))
            return Tree(resultnode, [])
        lex = tree.headterm
        children = []
        rhs_nts = []
        c, succs, _ = self.read_spine(tree, parents)
        for nparents, succ, direction in succs:
            children.append(self.extract_nodes(succ, nparents, direction_marker=direction))
            rhs_nts.append(children[-1].label.rule.lhs)
        lhs = overridelhs if not overridelhs is None else \
                self.nonterminals(parents + (self.nonterminals.get_label(tree),))
        
        leaves = SortedSet([lex])
        for child in children:
            leaves |= child.label.leaves
        lcfrs = self.ctype.from_positions(leaves, [c.label.leaves for c in children])
        lhs += self.nonterminals.fo(lcfrs.fanout)

        rule = headed_rule(lhs, tuple(rhs_nts), headed_clause(c), composition=lcfrs)
        # rule = rule.reorder((lex,) + tuple(c.label.leaves[0] for c in children))
        return Tree(extraction_result(lex, leaves, rule), children)


    def _fuse_modrule(self, mod_deriv: Tree, successor_mods: Tree, all_leaves):
        toprule = mod_deriv.label.rule
        botrule = successor_mods.label.rule

        children = [*mod_deriv.children, successor_mods]
        lcfrs = self.ctype.from_positions(all_leaves, [c.label.leaves for c in children])

        newrule = headed_rule(
            toprule.lhs,
            toprule.rhs+(botrule.lhs,),
            clause=ImmutableTree(RMLABEL, [toprule.clause, len(toprule.rhs)+1]),
            composition=lcfrs
        )
        positions = mod_deriv.label.leaves | successor_mods.label.leaves
        # newrule = newrule.reorder((lex,) + tuple(c.label.leaves[0] for c in children))
        return Tree(
            extraction_result(mod_deriv.label.lex, positions, newrule), children)
 

    def extract_nodes(self, trees: list[HeadedTree], parents: tuple[str, ...], direction_marker: int = 0):
        markovnts = [trees[-1].label if isinstance(trees[-1], Tree) else "POS"]
        lowestnt = None
        if self.nonterminals.rightmostunary:
            lowestnt = self.nonterminals.vert(parents, markovnts)
        deriv = self.extract_node(trees[-1], lowestnt, parents)
        yd = trees[-1].leaves() if isinstance(trees[-1], Tree) else SortedSet([trees[-1]])
        for tree in trees[-2::-1]:
            markovnts.append(tree.label if isinstance(tree, Tree) else "POS")
            yd = yd.union(tree.leaves() if isinstance(tree, Tree) else SortedSet([tree]))
            child = self.extract_node(tree, self.nonterminals.vert(parents, markovnts), parents)
            deriv = self._fuse_modrule(child, deriv, yd)
        if self.nonterminals.bindirection:
            direction_marker = {-1: "[L]", +1: ""}[direction_marker]
            deriv.label = extraction_result(
                deriv.label.lex,
                deriv.label.leaves,
                deriv.label.rule.with_lhs(deriv.label.rule.lhs+direction_marker)
            )
        return deriv


    def __call__(self, tree):
        derivation = self.extract_node(tree, self.root)
        rules = [r for _, _, r in sorted(node.label for node in derivation.subtrees())]
        for node in derivation.subtrees():
            node.label = node.label[0]
        return rules, derivation