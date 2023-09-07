cdef class Discospan:
    cdef tuple borders
    cdef int length

    cdef bint gt_leaf(self, int other) noexcept

cdef Discospan singleton_span(int idx) noexcept
cdef Discospan empty_span() noexcept