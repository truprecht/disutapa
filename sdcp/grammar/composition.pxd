from .parser.span import Discospan
import cython

cdef class Composition:
    cdef public cython.int fanout
    cdef public cython.int arity
    cdef public bytes variables

cdef class CompositionView(Composition):
    cdef public cython.int next_arg