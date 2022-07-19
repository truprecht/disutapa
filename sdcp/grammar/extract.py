from discodop.tree import Tree
from .sdcp import rule, sdcp_clause


def singleton(tree: Tree, nonterminal: str = "ROOT"):
    label, pos = "+".join(tree.label.split("+")[:-1]), tree.label.split("+")[-1]
    return (rule(nonterminal, (), fn_node=label),), (pos,)


def __extract_tree(tree: Tree, parent: str, exclude: set, override_lhs: str = None):
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if override_lhs else f"L-{parent}"
        return Tree((tree, rule(lhs, ())), [])
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    exclude.add(lex)
    rules = []
    rhs = []
    for c in tree:
        crules = __extract_tree(c, parent=tree.label, exclude=exclude)
        if not crules is None:
            rhs.append(crules.label[1].lhs)
            rules.append(crules)
    nodestr = None if "|<" in tree.label else tree.label.split("^")[0]
    lhs = override_lhs if override_lhs else tree.label
    push_idx = 1 if len(rules) == 2 else (-1 if not isinstance(tree[1], Tree) else 0)
    return Tree((lex, rule(lhs, tuple(rhs), fn_node=nodestr, fn_push=push_idx)), rules)


def extract(tree: Tree, override_root: str = "ROOT"):
    derivation = __extract_tree(tree, "ROOT", set(), override_lhs=override_root)
    return (r for _, r in sorted(node.label for node in derivation.subtrees()))