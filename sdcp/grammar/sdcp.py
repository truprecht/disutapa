from dataclasses import dataclass
from typing import Tuple
from itertools import chain

class node_constructor:
    def __init__(self, label, *fixed_children):
        self.label = label
        self.fixed_children = fixed_children

    def __call__(self, *children) -> str:
        childstrs = chain((str(i) for i in self.fixed_children if not i is None), children)
        childstrs = ' '.join(childstrs)
        if self.label:
            childstrs = f"({self.label} {childstrs})"
        return childstrs

    def __str__(self):
        return f"~{self()}"

    def __repr__(self):
        return str(self)

    def __eq__(self, o) -> bool:
        return self.label == o.label and [l for l in self.fixed_children if not l is None] == [l for l in o.fixed_children if not l is None]


@dataclass
class sdcp_clause:
    label: str
    arity: int
    push_idx: int = 1

    def __call__(self, lex: int, pushed: int) -> Tuple[node_constructor, Tuple[int, ...]]:
        if self.push_idx == 0:
            lex, pushed = pushed, lex
        match self.arity:
            case 2:
                return node_constructor(self.label), (pushed, lex)
            case 1:
                return node_constructor(self.label, lex), (pushed,)
            case 0:
                return node_constructor(self.label, pushed, lex), ()
        return NotImplementedError()


@dataclass
class rule:
    lhs: str
    fn: sdcp_clause
    rhs: tuple[str]

    def as_tuple(self):
        return self.lhs, self.rhs


@dataclass
class grammar:
    rules: list[rule]
    root: str = "ROOT"