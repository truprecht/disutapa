import cython

cdef class Discospan:
    cdef tuple borders
    cdef cython.int length