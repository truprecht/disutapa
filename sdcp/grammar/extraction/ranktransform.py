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
        # abort at pos nodes
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return tree

        # unary node, merge with successor
        if len(tree) == 1 and isinstance(tree[0], Tree):
            collapsenode = Tree(f"{tree.label}+{tree[0].label}", tree[0].children)
            collapsenode.type = tree.type
            return self(collapsenode , vlist)

        rootlabel = tree.label
        if self.vmarkov > 1:
            rootlabel = f"{tree.label}^<{','.join(vlist[:self.vmarkov-1])}>"
        vlist = list(reversed(tree.label.split("+"))) + vlist

        # binary node
        if len(tree) == 2:
            constructree = Tree(rootlabel, [self(c, vlist) for c in tree])
            constructree.type = tree.type
            return constructree

        basehorzlabel = vlist[0]
        if self.vmarkov > 1:
            basehorzlabel += f"^<{','.join(vlist[1:self.vmarkov])}>"
        constructree = Tree(rootlabel, [])
        constructree.type = tree.type
        rootnode = constructree

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
            nodelabel = f"{basehorzlabel}|<{','.join(siblinglabels)}>"
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


@dataclass
class HeadInward:
    hmarkov: int = 999
    vmarkov: int = 1
    mark_direction: bool = field(init=False, default=True)
    # trailing_node: bool = field(init=False, default=True)

    def __call__(self, tree: Tree, vlist=[]) -> Tree:
        # abort at pos nodes
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            node = Tree(tree.label, tree.children)
            node.type = HEAD
            return node

        rootlabel = tree.label
        if self.vmarkov > 1:
            rootlabel = f"{tree.label}^<{','.join(vlist[:self.vmarkov-1])}>"
        vlist = [tree.label] + vlist

        headidx = next(i for i, c in enumerate(tree.children) if c.type == HEAD)
        print(headidx, tree)

        rootnode = Tree(rootlabel, [])
        for siblinglist, direction in ((tree.children[:headidx], "l"), (tree.children[headidx], "m"), (tree.children[headidx+1:], "r")):
            if direction == "m":
                rootnode.children.append(self(siblinglist, vlist))
                continue
            processedchildren = (self(s) for s in siblinglist)
            origlabels = list(c.label for c in siblinglist)
            print(origlabels)
            constructnodes = rootnode.children
            for s in processedchildren:
                label = f"{rootlabel}|<{','.join(origlabels[:self.hmarkov])}>"
                if self.mark_direction:
                    label += f"[{direction}]"
                constructnodes.append(Tree(label, []))
                constructnodes = constructnodes[-1].children
                constructnodes.append(s)
                origlabels.pop(0)
                direction = "r"

        rootnode.type = HEAD
        return rootnode