import cython
from .span cimport Discospan

cdef class backtrace:
    cdef public cython.int rid
    cdef public cython.int leaf
    cdef public tuple children


cdef class ParseItem:
    cdef public cython.int lhs
    cdef public Discospan leaves
    cdef tuple remaining
    cdef object remaining_function
    cdef cython.int leaf

    cdef cython.bint is_passive(self) noexcept
    cdef cython.int next_nt(self) noexcept