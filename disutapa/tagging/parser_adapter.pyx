# cython: profile=True
# cython: linetrace=True

from ..grammar.parser.activeparser import ActiveParser
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef class ParserAdapter:
    cdef public object parser
    cdef public float step
    cdef public int total_limit
    cdef public float timeout

    def __init__(self, grammar, step: float = 2, total_limit = 10, timeout = 0.01):
        self.parser = ActiveParser(grammar)
        self.step = step
        self.total_limit = total_limit
        self.timeout = timeout

    def fill_chart(self, int length, cnp.ndarray[float,ndim=2] weights, cnp.ndarray[long,ndim=2] tags):
        cdef cnp.ndarray[long] starts = np.zeros(length, dtype=long)
        cdef cnp.ndarray[float] threshs = weights[:, 0].copy()
        cdef int tidx
        cdef int s
        cdef int e
        cdef bint all_at_end

        self.parser.init(length)
        found_root_node = False
        while not found_root_node:
            threshs += self.step
            all_at_end = True
            for tidx in range(length):
                e = starts[tidx]+1
                for e in range(starts[tidx], self.total_limit+1):
                    if e < self.total_limit and weights[tidx,e] > threshs[tidx]:
                        break
                s = starts[tidx]
                self.parser.add_rules_i(tidx, e-s, tags[tidx,s:e], weights[tidx,s:e])
                starts[tidx] = e
                all_at_end = all_at_end and e == self.total_limit
            found_root_node = self.parser.fill_chart(timeout=self.timeout)
            if all_at_end:
                break

    def get_best(self):
        return self.parser.get_best()
    
    def get_best_iter(self):
        return self.parser.get_best_iter()