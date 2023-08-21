from discodop.tree import Tree   # type: ignore
from itertools import chain
from sortedcontainers import SortedSet   # type: ignore
from .lcfrs import lcfrs_composition, ordered_union_composition
from .sdcp import rule, sdcp_clause


def singleton(tree: Tree, nonterminal: str = "ROOT") -> tuple[tuple[rule, ...], tuple[str, ...]]:
    label, pos = "+".join(tree.label.split("+")[:-1]), tree.label.split("+")[-1]
    return (rule(nonterminal, dcp=sdcp_clause.binary_node(label or None)),), (pos,)


def getnt(type: str, base: str, fanout: int) -> str:
    if type.startswith("d"):
        return base + ("/D" if fanout > 1 else "")
    if type.startswith("f"):
        return f"{base}/{fanout}"
    return base


def __extract_tree(tree: Tree, parent: str, exclude: set, override_lhs: str | None = None, ctype = lcfrs_composition, ntype = "plain") -> Tree:
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if not override_lhs is None else \
            "L-" + parent.split("+")[0]
        return Tree((tree, SortedSet([tree]), rule(lhs), SortedSet([tree])), [])
    lex: int = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    yd = SortedSet([lex])
    exclude.add(lex)
    rules = []
    for c in tree:
        crules = __extract_tree(c, parent=tree.label, exclude=exclude, ctype=ctype, ntype=ntype)
        if not crules is None:
            rules.append(crules)
            yd |= crules.label[1]

    # sort successors via least leaf in yield,
    # b/c lexicalization removes some leaves from subtrees
    rules.sort(key=lambda t: t.label[1][0])
    push_idx = next(
        i for i, t in chain(enumerate(rules), ((None, None),))
        if t is None or lex in t.label[3])

    # drop constituents that were introduced during binarization
    nodestr = None if "|<" in tree.label else tree.label.split("^")[0]
    lhs = tree.label
    composition, rhs_order = ctype.from_positions(yd, [c.label[1] for c in rules]) if rules else (None, [0])
    origrhs = (None, *(t.label[2].lhs for t in rules))
    rhs = tuple(origrhs[o] for o in rhs_order)
    if not override_lhs is None:
        lhs = override_lhs
    else:
        if "+" in tree.label:
            lhs = tree.label.split("+")[0] + ("|<>" if "|<" in tree.label else "")
        lhs = getnt(ntype, lhs, composition.fanout if composition else 1)
    dcp = sdcp_clause.binary_node(nodestr, len(rules), push_idx)
    return Tree((lex, yd, rule(lhs, rhs, dcp=dcp, scomp=composition), tree.leaves()), rules)


def extract(tree: Tree, override_root: str = "ROOT", ctype = "lcfrs", ntype = "plain"):
    ctype = {"lcfrs": lcfrs_composition, "dcp": ordered_union_composition}.get(ctype, ctype)
    derivation = __extract_tree(tree, "ROOT", set(), override_lhs=override_root, ctype=ctype, ntype=ntype)
    rules = [r for _, _, r, _ in sorted(node.label for node in derivation.subtrees())]
    for node in derivation.subtrees():
        node.label = node.label[0]
    return rules, derivation