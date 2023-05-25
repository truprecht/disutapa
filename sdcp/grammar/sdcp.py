from dataclasses import dataclass
from typing import Tuple, Iterable
from itertools import chain, repeat
from discodop.tree import Tree, ImmutableTree # type: ignore
from sortedcontainers import SortedSet # type: ignore

from .lcfrs import lcfrs_composition, ordered_union_composition, NtOrLeaf
    

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

@dataclass(frozen=True, init=False)
class sdcp_clause:
    tree: tuple[ImmutableTree|int, ...]
    arguments: tuple[None|int, ...]
    # arity: int

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
    lhs: str
    rhs: tuple[NtOrLeaf, ...]
    scomp: lcfrs_composition | ordered_union_composition
    dcp: sdcp_clause

    def __init__(self, lhs, rhs: tuple[str|None, ...] = (None,), scomp = None, dcp = None):
        irhs = tuple(
            NtOrLeaf.nt(r) if not r is None else NtOrLeaf.leaf()
            for r in rhs)
        assert any(r.is_leaf() for r in irhs) and len(irhs) >= 1
        if scomp is None:
            scomp = lcfrs_composition.default(len(rhs))
        if dcp is None:
            dcp = sdcp_clause.default(len(rhs)-1)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", irhs)
        object.__setattr__(self, "scomp", scomp)
        object.__setattr__(self, "dcp", dcp)


    @classmethod
    def from_guided(
            cls,
            lhs: str,
            rhs: tuple[str, ...],
            dnode: str | None = None,
            dtrans: int | None = None,
            scomp: str | lcfrs_composition | ordered_union_composition | None = None
            ) -> "rule":
        dcp = sdcp_clause.binary_node(dnode, len(rhs), dtrans)
        if scomp is None and dtrans == 0:
            scomp = lcfrs_composition(range(len(rhs)+1))
        if isinstance(scomp, str):
            scomp = lcfrs_composition(scomp)
        return cls(lhs, rhs, dcp=dcp, scomp=scomp)
        
    @classmethod
    def from_spine(
            cls,
            lhs: str,
            rhs: tuple[str, ...],
            spine: tuple[str|int|Tree] | str | int | Tree,
            scomp: str | lcfrs_composition | ordered_union_composition | None = None
            ) -> "rule":
        if not isinstance(spine, tuple):
            spine = (spine,)
        dcp = sdcp_clause.spine(spine)
        if isinstance(scomp, str):
            scomp = lcfrs_composition(scomp)
        return cls(lhs, rhs, dcp=dcp, scomp=scomp)


    # def normalize_order(self, lexical: int, child_spans: list[SortedSet]) -> "rule":
    #     original_order: tuple[NtOrLeaf, ...] = (NtOrLeaf(lexical, is_leaf=True), *(NtOrLeaf(nt, is_leaf=False) for nt in self.rhs))
    #     reordered_rhs = tuple(original_order[i] for i in self.order_and_fanout[:-1])
    #     occs = sorted(range(len(original_order)), key=lambda x: next(i for i,v in enumerate(self.inner) if v ==x))
    #     revoccs = {oldpos: newpos for newpos, oldpos in enumerate(occs)}
    #     revoccs[255] = 255
    #     comp = self.__class__(revoccs[v] for v in self.inner)
    #     return comp, tuple(original_order[i] for i in occs)
    #     return canon_composition, reordered_rhs


    def __repr__(self):
        args = [repr(self.lhs)]
        if self.rhs != (NtOrLeaf.leaf(),):
            args.append(repr(tuple(r.payload for r in self.rhs)))
        if self.scomp != lcfrs_composition.default(len(self.rhs)):
            kw = "scomp=" if len(args) < 2 else ""
            args.append(f"{repr(self.scomp)}")
        if self.dcp != sdcp_clause.default(len(self.rhs)-1):
            kw = "dcp=" if len(args) < 3 else ""
            args.append(f"{kw}{repr(self.dcp)}")
        return f"{self.__class__.__name__}({', '.join(args)})"


@dataclass
class grammar:
    rules: list[rule]
    root: str = "ROOT"