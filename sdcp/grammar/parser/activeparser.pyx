from _heapq import heapify, heappush, heappop

from ..sdcp import grammar, rule, Tree, Composition
from .kbestchart import KbestChart
from .item cimport ParseItem, backtrace, item
from .span cimport Discospan, empty_span

cimport cython
cimport numpy as cnp
cnp.import_array()


@cython.cclass
class Qelement:
    item: ParseItem
    bt: backtrace
    weight: cython.float

    def __init__(self, item: ParseItem, bt: backtrace, weight: cython.float):
        self.item=item
        self.bt=bt
        self.weight=weight

    def __gt__(self, other: Qelement):
        return self.weight > other.weight


@cython.cclass
class ActiveParser:
    _grammar: grammar
    len: cython.int
    rootid: cython.int
    queue: list[Qelement]
    actives: dict[int, list[tuple[ParseItem, backtrace, float]]]
    expanded: dict[ParseItem, int]
    from_lhs: dict[int, list[tuple[ParseItem, backtrace, float]]]
    backtraces: list[list[backtrace]]
    items: list[ParseItem]
    ruleweights: dict[tuple[cython.int, cython.int], cython.float]
    itemweights: list[cython.float]
    kbestchart: KbestChart
    
    @property
    def grammar(self) -> grammar:
        return self._grammar

    def __init__(self, grammar: grammar):
        self._grammar = grammar

    def init(self, sentlen: cython.int):
        self.len = sentlen
        self.rootid = -1
        self.queue = []
        self.actives = {}

        self.expanded = dict()
        self.from_lhs = {}
        self.backtraces = []  
        self.items = []
        
        self.ruleweights = dict()
        self.itemweights = list()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_rules_i(self, i: cython.int, arlen: cython.int, rules: cnp.ndarray[cython.int], weights: cnp.ndarray[cython.float]):
        idx: cython.int
        rid: cython.int
        it: ParseItem
        r: rule
        comp: Composition
        weight: cython.float
        emptybt: tuple = ()
        for idx in range(arlen):
            rid = rules[idx]
            weight = weights[idx]
            r = self._grammar.rules[rid]
            comp = r.scomp
            it = item(r.lhs, empty_span(), comp.view(), r.rhs, i)
            self.queue.append(Qelement(
                it,
                backtrace(rid, i, emptybt),
                weight
            ))
            self.ruleweights[(rid, i)] = weight


    def fill_chart(self, stop_early: bool = False) -> bool:
        qi: Qelement
        backtrace_id: cython.int
        rootspan: Discospan = Discospan((0, self.len))
        rootnt: cython.int = self._grammar.root
        active: ParseItem
        abt: backtrace
        _weight: cython.float
        newitem: ParseItem
        span: Discospan
        pbt: cython.int
        pweight: cython.float

        heapify(self.queue)     
        while self.queue:
            qi = heappop(self.queue)
            if qi.item in self.expanded:
                if qi.item.is_passive():
                    self.backtraces[self.expanded[qi.item]].append(qi.bt)
                continue
            self.expanded[qi.item] = len(self.backtraces)

            if qi.item.is_passive():
                backtrace_id = len(self.backtraces)
                self.backtraces.append([qi.bt])
                self.items.append(qi.item)
                self.itemweights.append(qi.weight)

                # check if this is the root item and stop the parsing, if early stopping is activated
                if qi.item.lhs == rootnt and qi.item.leaves == rootspan:
                    if self.rootid == -1:
                        self.rootid = backtrace_id
                    if stop_early:
                        return True

                self.from_lhs.setdefault(qi.item.lhs, []).append((qi.item.leaves, backtrace_id, qi.weight))
                
                for active, abt, _weight in self.actives.get(qi.item.lhs, []):
                    newitem = active.complete(qi.item.leaves)
                    if newitem is None or newitem.leaves is None: continue
                    heappush(self.queue, Qelement(
                        newitem,
                        backtrace(abt.rid, abt.leaf, (backtrace_id, *abt.children)),
                        qi.weight+_weight,
                    ))
                continue

            assert not qi.item.is_passive()
            self.actives.setdefault(qi.item.next_nt(), []).append((qi.item, qi.bt, qi.weight))
            for (span, pbt, pweight) in self.from_lhs.get(qi.item.next_nt(), []):
                newitem = qi.item.complete(span)
                if newitem is None or newitem.leaves is None: continue
                heappush(self.queue, Qelement(
                    newitem,
                    backtrace(qi.bt.rid, qi.bt.leaf, (pbt, *qi.bt.children)),
                    qi.weight+pweight
                ))
        return self.rootid != -1

    def get_best_iter(self):
        if self.rootid == -1:
            yield [Tree("NOPARSE", list(range(self.len)))], 0.0
            return
      
        self.kbestchart = KbestChart(
            self.backtraces,
            self.ruleweights,
            self._grammar.rules,
            self.itemweights,
            self.rootid
        )

        for t in iter(self.kbestchart):
            yield t
    
    def get_best(self):
        return next(self.get_best_iter())[0]

    def get_best_derivation(self, item=None):
        if item is None:
            item = self.rootid
            if self.rootid == -1:
                return None, None
        bt: backtrace = self.backtraces[item][0]
        return Tree((bt.rid, bt.leaf), [self.get_best_derivation(i) for i in bt.children])

    def numparses(self, item: int = None):
        if item is None:
            item = self.rootid
            if self.rootid == -1:
                return 0

        nodesum = 0
        for bt in self.backtraces[item]:
            edgeprod = 1
            for child in bt.children:
                edgeprod *= self.numparses(child)
            nodesum += edgeprod
        return nodesum