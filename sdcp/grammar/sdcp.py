from dataclasses import dataclass
from typing import Tuple, Iterable
from itertools import chain, repeat
from discodop.tree import Tree, ImmutableTree # type: ignore
from sortedcontainers import SortedSet # type: ignore

from .composition import Composition, default_lcfrs, lcfrs_composition, ordered_union_composition
    

@dataclass
class tree_constructor:
    context: tuple[ImmutableTree|int, ...]
    arguments: list[int|None]

    def subst(self, tree, *args):
        if not isinstance(tree, Tree):
            return args[tree]
        children = list(restree for c in tree for restree in self.subst(c, *args))
        for lab in reversed(tree.label.split("+")):
            children = [Tree(lab, children)]
        return children

    def __call__(self, *children: list[Tree]) -> list[Tree]:
        args = tuple([i] if not i is None else [] for i in self.arguments) + children
        return [t for c in self.context for t in self.subst(c, *args)]


def _increase_vars(context: Tree|int) -> Tree|int:
    if not isinstance(context, Tree):
        return 0 if context == 0 else context+1
    for i, c in enumerate(context):
        context[i] = _increase_vars(c)
    return context


def swap_vars(context: ImmutableTree|int, old_to_new: dict[int, int]) -> ImmutableTree|int:
    if not isinstance(context, Tree):
        return context if context <= 1 else old_to_new[context]
    return ImmutableTree(context.label, [swap_vars(c, old_to_new) for c in context])


@dataclass(frozen=True, init=False)
class sdcp_clause:
    tree: tuple[ImmutableTree|int, ...]
    arguments: tuple[None|int, ...]

    def __init__(self, context: tuple[ImmutableTree, ...], args: tuple[int|None, ...] | None = None):
        if not isinstance(context, tuple):
            context = (ImmutableTree(context),)
        if args is None:
            args = ()
        object.__setattr__(self, "tree", context)
        object.__setattr__(self, "arguments", args)

    def __repr__(self) -> str:
        args = []
        args.append(
            repr(self.tree) if len(self.tree) > 1 or isinstance(self.tree[0], int) else \
                repr(str(self.tree[0])))
        if self.arguments:
            args.append("args="+repr(self.arguments))
        return f"{self.__class__.__name__}({', '.join(args)})"
        

    @classmethod
    def default(cls, arity: int):
        return cls(tuple(range(arity+2)))

    @classmethod
    def binary_node(cls, node: str|None = None, arity: int = 0, transport_idx: int|None = None):
        args: list[int|None] = [None]*arity
        children = [0, 1]
        if arity == 2:
            if transport_idx is None:
                transport_idx = 1
            args[transport_idx] = 0
            args[1-transport_idx] = 1
            children = [2, 3]
        if arity == 1:
            args[0] = 1 if transport_idx is None else 0
            children = [0 if transport_idx is None else 1, 2]
        context = (ImmutableTree(node, children),) if not node is None else tuple(children)
        return cls(context, tuple(args))

    @classmethod
    def spine(cls, *context: Tree|int|str):
        if context == (0,):
            return cls.default(0)
        context_with_snd_order_var = (
            _increase_vars(Tree(t) if isinstance(t, str) else t)
            for t in context
        )
        context = tuple(
            ImmutableTree.convert(t) if isinstance(t, Tree) else t
            for t in context_with_snd_order_var
        )
        return cls(context)

    def __call__(self, lex: int, arg: int|None = None) -> tuple[tree_constructor, Iterable[int|None]]:
        largs = [lex, arg]
        for i in (0,1):
            if i in self.arguments:
                largs[i] = None
        args = (
            lex if sarg == 0 else (arg if sarg == 1 else None)
            for sarg in chain(self.arguments, repeat(None))
        )
        return tree_constructor(self.tree, largs), args


@dataclass(frozen=True, init=False)
class rule:
    lhs: int
    rhs: tuple[int, ...]
    scomp: Composition
    dcp: sdcp_clause

    def __init__(self, lhs: int, rhs: tuple[int, ...] = (-1,), scomp = None, dcp = None):
        rhs = tuple((-1 if nt is None else nt) for nt in rhs)
        assert any(r == -1 for r in rhs) and len(rhs) >= 1
        if scomp is None:
            scomp = default_lcfrs(len(rhs))
        if dcp is None:
            dcp = sdcp_clause.default(len(rhs)-1)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "scomp", scomp)
        object.__setattr__(self, "dcp", dcp)


    @classmethod
    def from_guided(
            cls,
            lhs: str,
            rhs: tuple[str, ...],
            dnode: str | None = None,
            dtrans: int | None = None,
            scomp: str | Composition | None = None
            ) -> "rule":
        dcp = sdcp_clause.binary_node(dnode, len(rhs), dtrans)
        if scomp is None and dtrans == 0:
            scomp = default_lcfrs(len(rhs)+1)
        if isinstance(scomp, str):
            scomp = Composition.lcfrs(scomp)
        return cls(lhs, rhs, dcp=dcp, scomp=scomp)
        
        
    @classmethod
    def from_spine(
            cls,
            lhs: str,
            rhs: tuple[str, ...],
            spine: tuple[str|int|Tree] | str | int | Tree,
            scomp: str | Composition | None = None
            ) -> "rule":
        if not isinstance(spine, tuple):
            spine = (spine,)
        dcp = sdcp_clause.spine(spine)
        if isinstance(scomp, str):
            scomp = Composition.lcfrs(scomp)
        return cls(lhs, rhs, dcp=dcp, scomp=scomp)
    

    def __repr__(self):
        args = [repr(self.lhs)]
        if self.rhs != (-1,) and self.rhs != (None,):
            args.append(repr(tuple(r for r in self.rhs)))
        if self.scomp != default_lcfrs(len(self.rhs)):
            kw = "scomp=" if len(args) < 2 else ""
            args.append(f"{kw}{repr(self.scomp)}")
        if self.dcp != sdcp_clause.default(len(self.rhs)-1):
            kw = "dcp=" if len(args) < 3 else ""
            args.append(f"{kw}{repr(self.dcp)}")
        return f"{self.__class__.__name__}({', '.join(args)})"


def integerize_rules(rules):
    rtoi = dict(ROOT=0)
    i = lambda n: rtoi.setdefault(n, len(rtoi))
    for r in rules:
        yield rule(i(r.lhs), tuple(-1 if n == -1 else i(n) for n in r.rhs), r.scomp, r.dcp)

@dataclass
class grammar:
    rules: list[rule]
    root: int = 0