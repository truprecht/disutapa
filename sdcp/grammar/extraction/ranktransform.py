from dataclasses import dataclass, MISSING, field
from discodop.tree import Tree, HEAD

@dataclass
class Binarizer:
    hmarkov: int = 999
    vmarkov: int = 1
    factor: str = "right"
    mark_direction: bool = field(init=False, default=True)

    def __post_init__(self):
        if self.factor != "headoutward":
            self.mark_direction = False

    def __call__(self, tree: Tree, vlist=[]) -> Tree:
        if not isinstance(tree, Tree):
            return tree

        if len(tree) == 1 and isinstance(tree[0], Tree):
            collapsenode = Tree(f"{tree.label}+{tree[0].label}", tree[0].children)
            collapsenode.type = tree.type
            return self(collapsenode , vlist)

        if len(tree) <= 2:
            constructree = Tree(tree.label, [self(c, [tree.label]+vlist) for c in tree])
            constructree.type = tree.type
            return constructree

        rootlabel = tree.label
        if self.vmarkov > 1:
            rootlabel = f"{tree.label}^<{','.join(vlist[:self.vmarkov-1])}>"
        constructree = Tree(rootlabel, [])
        constructree.type = tree.type
        rootnode = constructree

        vlist = [tree.label] + vlist
        head_outward = self.factor == "headoutward"
        currentfactor = self.factor if not head_outward else "right"
        siblings = list(tree.children)
        for i in range(len(tree)-1):
            if head_outward and tree[i].type == HEAD:
                currentfactor = "left"
            
            factornode = siblings.pop(0 if currentfactor == "right" else -1)
            hmarkovslice = slice(0, self.hmarkov) if currentfactor == "right" else \
                    slice(len(siblings)-self.hmarkov, len(siblings))
            siblinglabels = (s.label for s in siblings[hmarkovslice])
            nodelabel = f"{rootlabel}|<{','.join(siblinglabels)}>"
            if self.mark_direction:
                nodelabel += f"[{currentfactor[0]}]"

            constructree.children.append(self(factornode, vlist))
            if i < len(tree)-2:
                constructree.children.append(Tree(nodelabel, []))
                if head_outward:
                    constructree[1].type = HEAD
            else:
                assert len(siblings) == 1
                constructree.children.append(self(siblings[0], vlist))
            if currentfactor == "left":
                constructree.children = constructree.children[::-1]
            constructree = constructree[1 if currentfactor == "right" else 0]

        return rootnode