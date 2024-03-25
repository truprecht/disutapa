from .span cimport Discospan
from ..composition cimport CompositionView

cdef class backtrace:
    cdef public int rid
    cdef public int leaf
    cdef public tuple children


cdef class ParseItem:
    cdef public int lhs
    cdef public Discospan leaves
    cdef tuple remaining
    cdef CompositionView remaining_function
    cdef int leaf

    cdef bint is_passive(self) noexcept
    cdef int next_nt(self) noexcept
    cdef ParseItem complete(self, Discospan other_span) noexcept

cdef ParseItem item(int lhs, Discospan leaves, CompositionView remaining_function, tuple remaining_rhs, int leaf) noexcept