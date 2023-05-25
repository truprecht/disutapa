from dataclasses import dataclass
from typing import cast
from ..lcfrs import disco_span, lcfrs_composition, NtOrLeaf, ordered_union_composition

@dataclass
class backtrace:
    rid: int
    leaf: int
    children: tuple[int, ...]

    def as_tuple(self):
        return self.rid, self.children


@dataclass(frozen=True)
class PassiveItem:
    lhs: str
    leaves: disco_span

    def __gt__(self, other: "PassiveItem") -> bool:
        if isinstance(other, ActiveItem): return False
        return other.leaves > self.leaves


@dataclass(frozen=True)
class ActiveItem:
    lhs: str
    leaves: disco_span
    remaining_function: lcfrs_composition | ordered_union_composition
    remaining: tuple[NtOrLeaf, ...]
    leaf: int | None

    def __gt__(self, other: "ActiveItem") -> bool:
        if isinstance(other, PassiveItem): return True
        return (other.leaves, len(self.remaining)) > (self.leaves, len(other.remaining))
    
    def is_compatible(self, span: disco_span) -> bool:
        # print(self.leaves, span, self.remaining, self.leaves < span and all(span < n.get_leaf() for n in self.remaining if n.is_leaf))
        return self.leaves < span and (self.leaf is None or span < self.leaf)


@dataclass(eq=False, order=False)
class qelement:
    item: ActiveItem | PassiveItem
    bt: backtrace
    weight: float

    # priority queue retrieves lowest weighted elements first
    # with nll values, lower weights are better
    def __gt__(self, other):
        return self.weight > other.weight
        # return (self.weight, self.item) > (other.weight,  other.item)

    def tup(self):
        return self.item, self.bt, self.weight


def item(
        lhs: str,
        leaves: disco_span,
        remaining_function: lcfrs_composition | ordered_union_composition,
        remaining_rhs: tuple[NtOrLeaf, ...],
        leaf: int | None
        ) -> ActiveItem | PassiveItem:
    if remaining_rhs and remaining_rhs[0].is_leaf():
        nleaves, nfunction = remaining_function.partial(leaves, disco_span.singleton(leaf))
        # if nleaves is None or nfunction is None:
        #     # should not happen, because the spans are checked beforehand
        #     raise ValueError("tried to construct item with spans", leaves, "and leaf", remaining_rhs[0].get_leaf())
        leaves, remaining_function = nleaves, nfunction
        remaining_rhs = remaining_rhs[1:]
        leaf = None
    if not remaining_rhs:
        return PassiveItem(lhs, leaves)
    return ActiveItem(lhs, leaves, remaining_function, remaining_rhs, leaf)