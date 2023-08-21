from dataclasses import dataclass
from ..lcfrs import disco_span, lcfrs_composition, ordered_union_composition
import cython


@dataclass(frozen=True)
class backtrace:
    rid: int
    leaf: int
    children: tuple[int, ...]

    def as_tuple(self):
        return self.rid, self.children


@dataclass(frozen=True)
class PassiveItem:
    lhs: int
    leaves: disco_span

    def __gt__(self, other: "PassiveItem") -> bool:
        if isinstance(other, ActiveItem): return False
        return other.leaves > self.leaves


@dataclass(frozen=True)
class ActiveItem:
    lhs: int
    leaves: disco_span
    remaining_function: lcfrs_composition | ordered_union_composition
    remaining: tuple[int, ...]
    leaf: int | None

    def __gt__(self, other: "ActiveItem") -> bool:
        if isinstance(other, PassiveItem): return True
        return (other.leaves, len(self.remaining)) > (self.leaves, len(other.remaining))
    
    def is_compatible(self, span: disco_span) -> bool:
        return (not self.leaves or self.leaves > span) and (self.leaf is None or span > self.leaf)


@dataclass(eq=False, order=False)
@cython.cclass
class qelement:
    item: ActiveItem | PassiveItem
    bt: backtrace
    weight: cython.float

    def __gt__(self, other):
        return self.weight > other.weight


def item(
        lhs: int,
        leaves: disco_span,
        remaining_function: lcfrs_composition | ordered_union_composition,
        remaining_rhs: tuple[int, ...],
        leaf: int | None
        ) -> ActiveItem | PassiveItem:
    if remaining_rhs and (remaining_rhs[-1] == -1 or remaining_rhs[-1] is None):
        leaves, remaining_function = remaining_function.partial(disco_span.singleton(leaf), leaves)
        remaining_rhs = remaining_rhs[:-1]
        leaf = None
    if not remaining_rhs:
        return PassiveItem(lhs, remaining_function.finalize(leaves) if remaining_function else None)
    return ActiveItem(lhs, leaves, remaining_function, remaining_rhs, leaf)