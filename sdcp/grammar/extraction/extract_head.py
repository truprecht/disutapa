from discodop.tree import Tree, ImmutableTree  # type: ignore
from dataclasses import dataclass, field
from sortedcontainers import SortedSet  # type: ignore
from collections import namedtuple
from typing import Any

from ...autotree import AutoTree
from ..composition import lcfrs_from_positions, union_from_positions, lcfrs_composition, ordered_union_composition, fanout
from ..sdcp import rule, sdcp_clause, swap_vars


def read_clusters(filename: str):
    label_to_clusterid = {}
    with open(filename, "r") as cfile:
        for line in cfile:
            array = line.strip().split()
            clusterid = array[0]
            for label in array:
                assert not label in label_to_clusterid, f"label {label} appears multiple times in {filename}"
                label_to_clusterid[label] = clusterid
    return label_to_clusterid


@dataclass
class Nonterminal:
    type: str = "classic"
    hmarkov: int = 999
    vmarkov: int = 1
    coarselabels: dict[str, str] | None = None
    rightmostunary: bool = field(init=False, default=True)
    bindirection: bool = field(init=False, default=True)
    decoration: str = field(init=False, default="")
    
    def __post_init__(self):
        if self.hmarkov < 0 or self.vmarkov < 1:
            raise ValueError("illegal markov. parameters: h =", self.hmarkov, "and v =", self.vmarkov)
        if self.coarselabels:
            self.coarselabels = read_clusters(self.coarselabels)
        if self.hmarkov == 999:
            self.bindirection = False
            self.rightmostunary = False
        if "-" in self.type:
            self.type, self.decoration = self.type.split("-")

    def get_label(self, node: AutoTree, postags: dict[int, str] | None = None) -> str:
        label = node.label if isinstance(node, Tree) else postags[node]
        if self.type == "coarse":
            label = self.coarselabels.get(label, label) if not self.coarselabels is None else label[0]
        return label

    def vert(self, parents: tuple[str, ...], siblings: list[str]) -> str:
        if self.hmarkov == 999:
            return self(parents)
        return self(parents) + f"|<{','.join(siblings[:self.hmarkov])}>"

    def __call__(self, parents: tuple[str, ...]) -> str:
        return ";".join(parents[-self.vmarkov:])
    
    def fo(self, fanout: int):
        match self.decoration:
            case "nof":
                return ""
            case "disc":
                return "/D" if fanout > 1 else ""
            case "":
                return f"/{fanout}"


extraction_result = namedtuple("extraction_result", ["lex", "leaves", "rule"])
@dataclass(init=False)
class Extractor:
    nonterminals: Nonterminal
    root: str
    ctype: Any

    def __init__(self, root: str = "ROOT", composition: str = "lcfrs", ntype: str = "classic", **ntargs):
        self.nonterminals = Nonterminal(ntype, **ntargs)
        self.root = root
        self.__binarize = self.nonterminals.hmarkov < 999
        self.cconstructor = {"lcfrs": lcfrs_from_positions, "dcp": union_from_positions}.get(composition, lcfrs_from_positions)

    def read_spine(self, tree: AutoTree, parents: tuple[str, ...], firstvar: int = 1):
        if not isinstance(tree, AutoTree):
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


    def extract_node(self, tree: AutoTree, overridelhs: str | None = None, parents: tuple[str, ...] = ()):
        if not isinstance(tree, AutoTree):
            # TODO: use pos symbol?
            lhs = overridelhs if not overridelhs is None else f"ARG({';'.join(parents[-self.nonterminals.vmarkov:])})"
            resultnode = extraction_result(tree, SortedSet([tree]), rule(lhs))
            return Tree(resultnode, [])
        lex = tree.headterm
        children = []
        rhs_nts = []
        c, succs, _ = self.read_spine(tree, parents)
        for nparents, succ, direction in succs:
            children.append(self.extract_nodes(succ, nparents, direction=direction))
            rhs_nts.append(children[-1].label.rule.lhs)
        lhs = overridelhs if not overridelhs is None else \
                self.nonterminals(parents + (self.nonterminals.get_label(tree),))
        children.sort(key=lambda t: t.label.leaves[0])
        
        leaves = SortedSet([lex])
        for child in children:
            leaves |= child.label.leaves
        lcfrs, rhs_order = self.cconstructor(leaves, [c.label.leaves for c in children])
        if overridelhs is None:
            lhs += self.nonterminals.fo(lcfrs.fanout)
        rhs = tuple((None, *rhs_nts)[o] for o in rhs_order)

        r = rule(lhs, rhs, dcp=sdcp_clause.spine(c), scomp=lcfrs)
        # rule = rule.reorder((lex,) + tuple(c.label.leaves[0] for c in children))
        return Tree(extraction_result(lex, leaves, r), children)


    def _fuse_modrule(self, mod_deriv: Tree, successor_mods: Tree, all_leaves):
        toprule = mod_deriv.label.rule

        children = [*mod_deriv.children, successor_mods]
        lcfrs, rhs_order = self.cconstructor(all_leaves, [c.label.leaves for c in children])
        oldrhs = (None, *(child.label.rule.lhs for child in children))
        rhs = tuple(oldrhs[o] for o in rhs_order)
        old_context = (*toprule.dcp.tree, len(toprule.rhs)+1)
        reorder = dict((oldvar+1, i+2) for i, oldvar in enumerate(i for i in rhs_order if i > 0))
        dcp = sdcp_clause(tuple(swap_vars(old_tree, reorder) for old_tree in old_context))

        newrule = rule(toprule.lhs, rhs, dcp=dcp, scomp=lcfrs)
        positions = mod_deriv.label.leaves | successor_mods.label.leaves
        return Tree(
            extraction_result(mod_deriv.label.lex, positions, newrule), children)
 

    def extract_nodes(self, trees: list[AutoTree], parents: tuple[str, ...], direction: int = 0):
        markovnts = [self.nonterminals.get_label(trees[-1], self.postags)]
        lowestnt = None
        if self.nonterminals.rightmostunary:
            lowestnt = self.nonterminals.vert(parents, markovnts)
            if self.nonterminals.bindirection:
                lowestnt += "[l]" if direction < 0 and len(trees) == 1 else "[r]"
            lowestnt += self.nonterminals.fo(fanout(trees[-1].leaves()) if isinstance(trees[-1], Tree) else 1)
        deriv = self.extract_node(trees[-1], lowestnt, parents)
        yd = trees[-1].leaves() if isinstance(trees[-1], Tree) else SortedSet([trees[-1]])
        for tree in trees[-2::-1]:
            markovnts.append(self.nonterminals.get_label(tree, self.postags))
            yd = yd.union(tree.leaves() if isinstance(tree, Tree) else SortedSet([tree]))
            label = self.nonterminals.vert(parents, markovnts)
            if self.nonterminals.bindirection:
                label += "[l]" if direction < 0 and tree is trees[0] else "[r]"
            label += self.nonterminals.fo(fanout(yd) if isinstance(tree, Tree) else 1)
            child = self.extract_node(tree, label, parents)
            deriv = self._fuse_modrule(child, deriv, yd)
        return deriv


    def __call__(self, tree: AutoTree):
        self.postags = tree.postags
        derivation = self.extract_node(tree, self.root)
        rules = [r for _, _, r in sorted(node.label for node in derivation.subtrees())]
        for node in derivation.subtrees():
            node.label = node.label[0]
        return rules, derivation