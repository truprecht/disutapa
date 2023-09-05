# cython: profile=True
# cython: linetrace=True
from _heapq import heapify, heappush, heappop
from typing import Iterable

from ..extract_head import Tree
from ..sdcp import grammar, rule
from .kbestchart import KbestChart
from .item import item, ActiveItem, PassiveItem, backtrace, disco_span

cimport cython
import numpy as np
cimport numpy as cnp
cnp.import_array()


@cython.cclass
class Qelement:
    item: ActiveItem|PassiveItem
    bt: backtrace
    weight: cython.float

    def __init__(self, item: ActiveItem|PassiveItem, bt: backtrace, weight: cython.float):
        self.item=item
        self.bt=bt
        self.weight=weight

    def __gt__(self, other: "Qelement"):
        if not isinstance(other, Qelement):
            return NotImplemented
        return self.weight > other.weight


@cython.cclass
class ActiveParser:
    _grammar: grammar
    len: cython.int
    rootid: cython.int
    queue: list[Qelement]
    actives: dict[int, list[tuple[ActiveItem, backtrace, float]]]
    expanded: dict[PassiveItem, int]
    from_lhs: dict[int, list[tuple[ActiveItem, backtrace, float]]]
    backtraces: list[list[backtrace]]
    items: list[PassiveItem]
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
        weight: cython.float
        for idx in range(arlen):
            rid = rules[idx]
            weight = weights[idx]
            r: rule = self._grammar.rules[rid]
            it = item(r.lhs, disco_span(), r.scomp, r.rhs, i)
            self.queue.append(Qelement(
                it,
                backtrace(rid, i, ()),
                weight
            ))
            self.ruleweights[(rid, i)] = weight


    def fill_chart(self, stop_early: bool = False) -> bool:   
        heapify(self.queue)     
        while self.queue:
            qi: Qelement = heappop(self.queue)
            if qi.item in self.expanded:
                if isinstance(qi.item, PassiveItem):
                    self.backtraces[self.expanded[qi.item]].append(qi.bt)
                continue
            self.expanded[qi.item] = len(self.backtraces)

            if isinstance(qi.item, PassiveItem):
                backtrace_id = len(self.backtraces)
                self.backtraces.append([qi.bt])
                self.items.append(qi.item)
                self.itemweights.append(qi.weight)

                # check if this is the root item and stop the parsing, if early stopping is activated
                if qi.item.lhs == self._grammar.root and qi.item.leaves == disco_span((0, self.len)):
                    if self.rootid == -1:
                        self.rootid = backtrace_id
                    if stop_early:
                        return True

                qi.bt = backtrace_id
                self.from_lhs.setdefault(qi.item.lhs, []).append((qi.item.leaves, qi.bt, qi.weight))
                
                for active, abt, _weight in self.actives.get(qi.item.lhs, []):
                    if not active.is_compatible(qi.item.leaves): continue
                    newpos, newcomp = active.remaining_function.partial(qi.item.leaves, active.leaves)
                    if newpos is None: continue
                    newitem = item(active.lhs, newpos, newcomp, active.remaining[:-1], active.leaf)
                    if newitem.leaves is None: continue
                    heappush(self.queue, Qelement(
                        newitem,
                        backtrace(abt.rid, abt.leaf, (backtrace_id, *abt.children)),
                        qi.weight+_weight,
                    ))
                continue

            assert isinstance(qi.item, ActiveItem)
            self.actives.setdefault(qi.item.remaining[-1], []).append((qi.item, qi.bt, qi.weight))
            for (span, pbt, pweight) in self.from_lhs.get(qi.item.remaining[-1], []):
                if not qi.item.is_compatible(span): continue
                newpos, newfunc = qi.item.remaining_function.partial(span, qi.item.leaves)
                if newpos is None: continue
                newitem = item(qi.item.lhs, newpos, newfunc, qi.item.remaining[:-1], qi.item.leaf)
                if newitem.leaves is None: continue
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