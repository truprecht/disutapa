from dataclasses import dataclass
from typing import Tuple
from itertools import chain
from discodop.tree import Tree

from .lcfrs import lcfrs_composition, ordered_union_composition

class node_constructor:
    def __init__(self, label: str, *fixed_children):
        self.label = label
        self.fixed_children = fixed_children

    def __call__(self, *children) -> list[Tree]:
        children = list(chain(
            (t for t in self.fixed_children if not t is None),
            (t for ts in children for t in ts)
        ))
        if self.label is None: return children
        trees = children
        for l in reversed(self.label.split("+")):
            trees = [Tree(l, trees)]
        return trees

    def __str__(self):
        return f"~{self()}"

    def __repr__(self):
        return str(self)

    def __eq__(self, o) -> bool:
        return self.label == o.label and [l for l in self.fixed_children if not l is None] == [l for l in o.fixed_children if not l is None]


@dataclass(frozen=True, init=False)
class sdcp_clause:
    label: str
    arity: int
    push_idx: int

    def __init__(self, label: str, arity: int, push_idx: int = None):
        if push_idx is None:
            push_idx = 1 if arity == 2 else -1
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "push_idx",push_idx)

    def __call__(self, lex: int, pushed: int) -> Tuple[node_constructor, Tuple[int, ...]]:
        if self.push_idx == 0:
            pushed, lex = lex, pushed
        match self.arity:
            case 2:
                return node_constructor(self.label), (pushed, lex)
            case 1:
                return node_constructor(self.label, lex), (pushed,)
            case 0:
                return node_constructor(self.label, pushed, lex), ()
        return NotImplementedError()


@dataclass(frozen=True, init=False)
class rule:
    lhs: str
    fn: sdcp_clause
    rhs: tuple[str] = ()
    composition: lcfrs_composition | ordered_union_composition = None


    def __init__(self, lhs: str, rhs: Tuple[str,...], fn_node: str = None, fn_push: int = None, composition: lcfrs_composition | str | None = None):
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "fn", sdcp_clause(fn_node, len(self.rhs), fn_push))
        if composition is None:
            composition = lcfrs_composition(range(len(rhs)+1))
        if isinstance(composition, str):
            composition = lcfrs_composition(composition)
        object.__setattr__(self, "composition", composition)


    def __repr__(self):
        fn = f", fn_node={repr(self.fn.label)}" if not self.fn.label is None else ""
        fp = "" if (self.fn.arity < 2 and self.fn.push_idx == -1) or (self.fn.arity==2 and self.fn.push_idx == 1) else f", fn_push={self.fn.push_idx}"
        comp = f", composition={repr(self.composition)}" if self.composition != lcfrs_composition(range(len(self.rhs)+1)) else ""
        return f"rule({repr(self.lhs)}, {repr(self.rhs)}{fn}{fp}{comp})"


@dataclass
class grammar:
    rules: list[rule]
    root: str = "ROOT"