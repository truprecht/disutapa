from .parser.span cimport Discospan

cdef class Composition:
    cdef public int fanout
    cdef public int arity
    cdef public bytes variables

    cpdef CompositionView view(self, int arg=*) noexcept

cdef class CompositionView(Composition):
    cdef int next_arg
    cdef CompositionView next(self) noexcept
    cpdef Discospan partial(self, Discospan arg, Discospan acc) noexcept
    cdef Discospan finalize(self, Discospan acc) noexcept