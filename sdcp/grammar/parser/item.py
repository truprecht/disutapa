from dataclasses import dataclass
from typing import cast
from ..lcfrs import disco_span, lcfrs_composition, NtOrLeaf, ordered_union_composition

@dataclass(frozen=True)
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
        return (not self.leaves or self.leaves > span) and (self.leaf is None or span > self.leaf)


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
    if remaining_rhs and remaining_rhs[-1].is_leaf():
        leaves, remaining_function = remaining_function.partial(disco_span.singleton(leaf), leaves)
        remaining_rhs = remaining_rhs[:-1]
        leaf = None
    if not remaining_rhs:
        return PassiveItem(lhs, remaining_function.finalize(leaves) if remaining_function else None)
    return ActiveItem(lhs, leaves, remaining_function, remaining_rhs, leaf)