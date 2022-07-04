from asyncio import transports
from cProfile import label
from discodop.tree import Tree
from .sdcp import rule, sdcp_clause


def __extract(tree: Tree, parent: str, exclude: set, override_lhs: str = None):
    if not isinstance(tree, Tree):
        if tree in exclude:
            return []
        lhs = override_lhs if override_lhs else f"L-{parent}"
        return [(tree, rule(lhs, sdcp_clause(None, 0), ()))]
    lex = min(tree[1].leaves()) if isinstance(tree[1], Tree) else tree[1]
    exclude.add(lex)
    rules = []
    rhs = []
    for c in tree:
        crules = __extract(c, parent=tree.label, exclude=exclude)
        if crules:
            rhs.append(crules[0][1].lhs)
            rules.extend(crules)
    nodestr = None if "|<" in tree.label else tree.label
    lhs = override_lhs if override_lhs else tree.label
    return [(lex, rule(lhs, sdcp_clause(nodestr, len(rhs), push_idx=1), tuple(rhs)))] + rules


def extract(tree: Tree, override_root: str = "ROOT"):
    rulelist = __extract(tree, "ROOT", set(), override_lhs=override_root)
    rulelist.sort()
    return (r for _, r in rulelist)