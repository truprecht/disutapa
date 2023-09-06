# cython: profile=True
# cython: linetrace=True
from dataclasses import dataclass
import cython

from ..lcfrs import lcfrs_composition, ordered_union_composition
from .span cimport Discospan

@cython.cclass
class backtrace:
    def __init__(self, rid: cython.int, leaf: cython.int, children: tuple[cython.int, ...]):
        self.rid = rid
        self.leaf = leaf
        self.children = children

cdef class ParseItem:
    def __init__(self, lhs: cython.int, leaves: Discospan, remaining_function: lcfrs_composition | ordered_union_composition, remaining: tuple[int, ...], leaf: cython.int):
        self.lhs = lhs
        self.leaves = leaves
        self.remaining_function = remaining_function
        self.remaining = remaining
        self.leaf = leaf

    cdef cython.bint is_passive(self) noexcept:
        return not self.remaining

    def complete(self, other_span: Discospan) -> ParseItem:
        newpos: Discospan
        newcomp: lcfrs_composition | ordered_union_composition
        if (self.leaves and not self.leaves > other_span) or (self.leaf != -1 and not other_span.gt_leaf(self.leaf)):
            return None
        newpos, newcomp = self.remaining_function.partial(other_span, self.leaves)
        assert newpos is None or newpos, str(newpos)
        if newpos is None:
            return None
        return item(self.lhs, newpos, newcomp, self.remaining[:-1], self.leaf)

    cdef cython.int next_nt(self) noexcept:
        return self.remaining[-1]

    def __repr__(self) -> str:
        return f"ParseItem({self.lhs}, {self.leaves}, {self.remaining_function}, {self.remaining}, {self.leaf})"

    def __eq__(self, other: ParseItem) -> bool:
        return self.lhs == other.lhs and self.leaf == other.leaf and self.leaves == other.leaves and self.remaining_function == other.remaining_function and self.remaining == other.remaining

    def __hash__(self) -> int:
        return hash((self.lhs, self.leaf, self.leaves, self.remaining_function, self.remaining))


@cython.ccall
@cython.exceptval(check=False)
def item(
        lhs: cython.int,
        leaves: Discospan,
        remaining_function: lcfrs_composition | ordered_union_composition,
        remaining_rhs: tuple[int, ...],
        leaf: cython.int
        ) -> ParseItem:
    if remaining_rhs and remaining_rhs[-1] == -1:
        leaves, remaining_function = remaining_function.partial(Discospan.singleton(leaf), leaves)
        if leaves is None:
            return None
        remaining_rhs = remaining_rhs[:-1]
        leaf = -1
    if not remaining_rhs:
        leaves = remaining_function.finalize(leaves)
    if leaves is None:
        return None
    assert leaves or leaf != -1, str(leaves)
    return ParseItem(lhs, leaves, remaining_function, remaining_rhs, leaf)