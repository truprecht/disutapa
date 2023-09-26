from discodop.tree import Tree   # type: ignore
from itertools import chain
from sortedcontainers import SortedSet   # type: ignore
from ..composition import lcfrs_from_positions, union_from_positions, lcfrs_composition, ordered_union_composition
from ..sdcp import rule, sdcp_clause
from .guide import Guide
from .nonterminal import NtConstructor


def singleton(tree: Tree, nonterminal: str = "ROOT") -> tuple[tuple[rule, ...], tuple[str, ...]]:
    label, pos = "+".join(tree.label.split("+")[:-1]), tree.label.split("+")[-1]
    return (rule(nonterminal, dcp=sdcp_clause.binary_node(label or None)),), (pos,)


def __extract_tree(tree: Tree, guide: Guide, ntype: NtConstructor, parent: str, exclude: set, override_lhs: str | None = None, cconstructor = lcfrs_from_positions) -> Tree:
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if not override_lhs is None else \
            "L-" + parent.split("+")[0]
        return Tree((tree, SortedSet([tree]), rule(lhs), SortedSet([tree])), [])
    lex: int = guide(tree)
    yd = SortedSet([lex])
    exclude.add(lex)
    rules = []
    for c in tree:
        crules = __extract_tree(c, guide, ntype, parent=tree.label, exclude=exclude, cconstructor=cconstructor)
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
    composition, rhs_order = cconstructor(yd, [c.label[1] for c in rules]) if rules else (None, [0])
    origrhs = (None, *(t.label[2].lhs for t in rules))
    rhs = tuple(origrhs[o] for o in rhs_order)
    lhs = override_lhs or ntype(tree, yd)
    dcp = sdcp_clause.binary_node(nodestr, len(rules), push_idx)
    return Tree((lex, yd, rule(lhs, rhs, dcp=dcp, scomp=composition), tree.leaves()), rules)


def extract(tree: Tree, override_root: str = "ROOT", ctype = "lcfrs", ntype = "classic", nt_tab = None, gtype = "strict"):
    ctype = {"lcfrs": lcfrs_from_positions, "dcp": union_from_positions}.get(ctype, ctype)
    ntype = NtConstructor(ntype, nt_tab)
    guide = Guide.construct(gtype, tree)
    derivation = __extract_tree(tree, guide, ntype, "ROOT", set(), override_lhs=override_root, cconstructor=ctype)
    rules = [r for _, _, r, _ in sorted(node.label for node in derivation.subtrees())]
    for node in derivation.subtrees():
        node.label = node.label[0]
    return rules, derivation