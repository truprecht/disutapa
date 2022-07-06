from asyncio import transports
from cProfile import label
from discodop.tree import Tree
from .sdcp import rule, sdcp_clause


def __extract(tree: Tree, parent: str, exclude: set, override_lhs: str = None):
    if not isinstance(tree, Tree):
        if tree in exclude:
            return []
        lhs = override_lhs if override_lhs else f"L-{parent}"
        return [(tree, rule(lhs, sdcp_clause(None, 0, 0), ()))]
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    push_idx = 0 if not isinstance(tree[0], Tree) and tree[0] in exclude else 1 # push index becomes 0 as the first successor vanishes
    exclude.add(lex)
    rules = []
    rhs = []
    for c in tree:
        crules = __extract(c, parent=tree.label, exclude=exclude)
        if crules:
            rhs.append(crules[0][1].lhs)
            rules.extend(crules)
    nodestr = None if "|<" in tree.label else tree.label.split("^")[0]
    lhs = override_lhs if override_lhs else tree.label
    return [(lex, rule(lhs, sdcp_clause(nodestr, len(rhs), push_idx=push_idx), tuple(rhs)))] + rules

def __extract_tree(tree: Tree, parent: str, exclude: set, override_lhs: str = None):
    if not isinstance(tree, Tree):
        if tree in exclude:
            return None
        lhs = override_lhs if override_lhs else f"L-{parent}"
        return Tree((tree, rule(lhs, sdcp_clause(None, 0, 0), ())), [])
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    push_idx = 0 if not isinstance(tree[0], Tree) and tree[0] in exclude else 1 # push index becomes 0 as the first successor vanishes
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
    return Tree((lex, rule(lhs, sdcp_clause(nodestr, len(rhs), push_idx=push_idx), tuple(rhs))), rules)


def extract(tree: Tree, override_root: str = "ROOT"):
    rulelist = __extract(tree, "ROOT", set(), override_lhs=override_root)
    rulelist.sort()
    return (r for _, r in rulelist)