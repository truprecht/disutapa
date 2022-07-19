from dataclasses import dataclass
from typing import Tuple
from itertools import chain

class node_constructor:
    def __init__(self, label: str, *fixed_children):
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


@dataclass(frozen=True)
class sdcp_clause:
    label: str
    arity: int
    push_idx: int = -1

    def __post_init__(self):
        # set default value of push_idx to -1 (keep) for arities 0,1
        # and to 1 (push to right successor) for arity 2
        if self.arity == 2 and self.push_idx == -1:
            object.__setattr__(self, "push_idx", 1)


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


    def __init__(self, lhs: str, rhs: Tuple[str,...], fn_node: str = None, fn_push: int = -1):
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "fn", sdcp_clause(fn_node, len(self.rhs), fn_push))


    def as_tuple(self):
        return self.lhs, self.rhs


    def __repr__(self):
        fn = f", fn_node={repr(self.fn.label)}" if not self.fn.label is None else ""
        fp = f", fn_push={repr(self.fn.push_idx)}" if not self.fn.push_idx == -1 and not (self.fn.arity==2 and self.fn.push_idx == 1) else ""
        return f"rule({repr(self.lhs)}, {repr(self.rhs)}{fn}{fp})"


@dataclass
class grammar:
    rules: list[rule]
    root: str = "ROOT"