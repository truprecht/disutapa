from discodop.tree import Tree
from .sdcp import rule, sdcp_clause


def singleton(tree: Tree, nonterminal: str = "ROOT"):
    label, pos = "+".join(tree.label.split("+")[:-1]), tree.label.split("+")[-1]
    return (rule(nonterminal, (), fn_node=label),), (pos,)


def fanout(leaves: set[int]) -> int:
    # ol = sorted(leaves)
    return 1+sum(1 for x,y in zip(leaves[:-1], leaves[1:]) if x+1 != y)


def __extract_tree(tree: Tree, parent: str, exclude: set, override_lhs: str = None):
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if not override_lhs is None else \
            "L-" + parent.split("+")[0]
        return Tree((tree, tree, rule(lhs, (), fanout=1)), [])
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    yd = sorted(l for l in tree.leaves() if not l in exclude)
    exclude.add(lex)
    rules = []
    rhs = []
    for c in tree:
        crules = __extract_tree(c, parent=tree.label, exclude=exclude)
        if not crules is None:
            rhs.append(crules.label[2].lhs)
            rules.append(crules)

    push_idx = 1 if len(rules) == 2 else (-1 if not isinstance(tree[1], Tree) else 0)
    if len(rules) == 2 and rules[0].label[1] > rules[1].label[1]:
        # TODO if rules[1] is a binarization node, this should not be swapped
        rules = rules[::-1]
        rhs = rhs[::-1]
        push_idx = 0

    nodestr = None if "|<" in tree.label else tree.label.split("^")[0]
    lhs = tree.label
    if not override_lhs is None:
        lhs = override_lhs
    elif "+" in tree.label:
        lhs = tree.label.split("+")[0] + ("|<>" if "|<" in tree.label else "")
    
    fo = fanout(yd)
    if fo > 1:
        lhs = f"D-{lhs}"
    
    return Tree((lex, yd[0], rule(lhs, tuple(rhs), fn_node=nodestr, fn_push=push_idx, fanout=fo)), rules)


def extract(tree: Tree, override_root: str = "ROOT"):
    derivation = __extract_tree(tree, "ROOT", set(), override_lhs=override_root)
    rules = [r for _, _, r in sorted(node.label for node in derivation.subtrees())]
    for node in derivation.subtrees():
        node.label = node.label[0]
    return rules, derivation