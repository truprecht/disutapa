import cython

from .span cimport Discospan, singleton_span

cdef class backtrace:
    def __init__(self, int rid, int leaf, tuple children):
        self.rid = rid
        self.leaf = leaf
        self.children = children

@cython.dataclasses.dataclass(init=False, eq=True, frozen=True)
cdef class ParseItem:
    def __init__(self, int lhs, Discospan leaves, tuple remaining, CompositionView remaining_function, int leaf):
        self.lhs = lhs
        self.leaves = leaves
        self.remaining = remaining
        self.remaining_function = remaining_function
        self.leaf = leaf

    cdef cython.bint is_passive(self) noexcept:
        return not self.remaining

    cdef ParseItem complete(self, other_span: Discospan) noexcept:
        newpos: Discospan
        if (self.leaves and not self.leaves > other_span) or (self.leaf != -1 and not other_span.gt_leaf(self.leaf)):
            return None
        newpos = self.remaining_function.partial(other_span, self.leaves)
        if newpos is None:
            return None
        return item(self.lhs, newpos, self.remaining_function.next(), self.remaining[:-1], self.leaf)

    cdef cython.int next_nt(self) noexcept:
        return self.remaining[-1]

cdef ParseItem item(
        lhs: cython.int,
        leaves: Discospan,
        remaining_function: CompositionView,
        remaining_rhs: tuple[int, ...],
        leaf: cython.int
        ) noexcept:
    if remaining_rhs and remaining_rhs[-1] == -1:
        leaves = remaining_function.partial(singleton_span(leaf), leaves)
        if leaves is None:
            return None
        remaining_function = remaining_function.next()
        remaining_rhs = remaining_rhs[:-1]
        leaf = -1
    if not remaining_rhs:
        leaves = remaining_function.finalize(leaves)
        if leaves is None:
            return None
    return ParseItem(lhs, leaves, remaining_rhs, remaining_function, leaf)