from discodop.tree import Tree
from sortedcontainers import SortedSet
from .sdcp import rule, sdcp_clause
from .lcfrs import lcfrs_composition, ordered_union_composition


def singleton(tree: Tree, nonterminal: str = "ROOT"):
    label, pos = "+".join(tree.label.split("+")[:-1]), tree.label.split("+")[-1]
    return (rule(nonterminal, (), fn_node=label),), (pos,)


def __extract_tree(tree: Tree, parent: str, exclude: set, override_lhs: str = None, ctype = lcfrs_composition) -> Tree:
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if not override_lhs is None else \
            "L-" + parent.split("+")[0]
        return Tree((tree, SortedSet([tree]), rule(lhs, ())), [])
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    yd = SortedSet([lex])
    exclude.add(lex)
    rules = []
    rhs = []
    for c in tree:
        crules = __extract_tree(c, parent=tree.label, exclude=exclude, ctype=ctype)
        if not crules is None:
            rhs.append(crules.label[2].lhs)
            rules.append(crules)
            yd |= crules.label[1]

    push_idx = 1 if len(rules) == 2 else (-1 if not isinstance(tree[1], Tree) else 0)

    nodestr = None if "|<" in tree.label else tree.label.split("^")[0]
    lhs = tree.label
    if not override_lhs is None:
        lhs = override_lhs
    elif "+" in tree.label:
        lhs = tree.label.split("+")[0] + ("|<>" if "|<" in tree.label else "")
    # if fo > 1:
    #     lhs = f"D-{lhs}"
    composition = ctype.from_positions(yd, [c.label[1] for c in rules]) \
        if rules else None
    return Tree((lex, yd, rule(lhs, tuple(rhs), fn_node=nodestr, fn_push=push_idx, composition=composition)), rules)


def extract(tree: Tree, override_root: str = "ROOT", ctype = "lcfrs"):
    ctype = {"lcfrs": lcfrs_composition, "dcp": ordered_union_composition}.get(ctype, ctype)
    derivation = __extract_tree(tree, "ROOT", set(), override_lhs=override_root, ctype = ctype)
    rules = [r for _, _, r in sorted(node.label for node in derivation.subtrees())]
    for node in derivation.subtrees():
        node.label = node.label[0]
    return rules, derivation