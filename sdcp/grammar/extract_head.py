from discodop.tree import Tree
from dataclasses import dataclass, field

from ..headed_tree import HeadedTree


@dataclass
class headed_clause:
    spine: list[str]
    vars: list[tuple[int, ...]]
   
    # def subst(self, lex: int, *args: Tree):
    #     def _subst(t: Tree, lex, *args):
    #         if not isinstance(t, Tree):
    #             return lex if t == -1 else args[t]
    #         return Tree(t.label, [_subst(c, lex, *args) for c in t])

    def subst(self, lex: int, *args: Tree):
        def _subst(spine, vars, lex, *args):
            if len(spine) == len(vars):
                vars = [()]+vars
            siblings = [t for id in vars[0] for t in args[id]]
            if not spine:
                return siblings + [lex]
            return siblings + [Tree(spine[0], _subst(spine[1:], vars[1:], lex, *args))]
        return _subst(self.spine, self.vars, lex, *args)

    def __call__(self, lex: int):
        return (lambda *args: self.subst(lex, *args))

concat_clause = lambda _: (lambda *args: args)


@dataclass
class headed_rule:
    lhs: str
    rhs: tuple[str]
    clause: headed_clause
    fanout: int


def minleaf(tree):
    return tree._minleaf if isinstance(tree, HeadedTree) else tree


def extract_node(tree: HeadedTree, overridelhs: str = None):
    if not isinstance(tree, HeadedTree):
        # TODO: use pos symbol?
        lhs = overridelhs if not overridelhs is None else "ARG"
        return Tree((tree, headed_rule(lhs, [], headed_clause([], []), 0)), [])
    lex = tree.headterm
    topmost = tree.label.split("+")[0]
    children = []
    rhs_nts = []
    c, succs = read_spine(tree)
    for modnt, succ in succs:
        children.append(extract_mods(succ, modnt))
        rhs_nts.append(children[-1].label[1].lhs)
    lhs = overridelhs if not overridelhs is None else topmost
    return Tree((lex, headed_rule(lhs, rhs_nts, c, 0)), children)

def read_spine(tree: HeadedTree, spine = None, succs = None):
    if spine is None or succs is None:
        spine, succs = [], []
    succs.append((tree[:tree.headidx], tree[tree.headidx+1:]))
    spine.append(tree.label)
    if isinstance(tree[tree.headidx], HeadedTree):
        return read_spine(tree[tree.headidx], spine, succs)
    sortedsuccs = sorted(
        ((i,j) for i in range(len(succs)) for j in (0,1) if succs[i][j]),
        key=lambda idx: minleaf(succs[idx[0]][idx[1]][0]))
    succssorted = {t: i for i,t in enumerate(sortedsuccs)}
    vars = [tuple(succssorted[(i,j)] for j in (0,1) if (i,j) in succssorted) for i in range(len(spine))]
    successors = list(
        (f"{spine[i]}"+("->" if j == 0 else "<-"), succs[i][j])
        for i,j in sortedsuccs
    )
    return headed_clause(spine, vars), successors


def extract_mods(trees: list[HeadedTree], modnt: str):
    if len(trees) == 1:
        return extract_node(trees[0])
    deriv = []
    for tree in trees:
        deriv_ = extract_node(tree, overridelhs=modnt)
        deriv = [Tree(deriv_.label, deriv_.children + deriv)]
    return deriv
