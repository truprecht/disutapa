from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple


from .extract_head import headed_rule, headed_clause, Tree
from .buparser import BitSpan, backtrace, qelement
from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray, bitarray
from bitarray.util import count_and
from sortedcontainers import SortedList
from collections import defaultdict


@dataclass
class PassiveItem:
    lhs: str
    leaves: BitSpan

    def freeze(self):
        return (self.lhs, self.leaves.freeze())

    def __gt__(self, other: "PassiveItem") -> bool:
        if isinstance(other, ActiveItem): return False
        return (self.lhs, self.leaves) > (other.lhs, other.leaves)


@dataclass
class ActiveItem:
    lhs: str
    leaf: int
    leaves: BitSpan
    maxfo: int
    remaining: tuple[str]

    def freeze(self):
        return (self.lhs, self.leaf, self.leaves.freeze(), self.maxfo, self.remaining)

    def __gt__(self, other: "ActiveItem") -> bool:
        if isinstance(other, PassiveItem): return True
        return (self.lhs, self.leaves, self.maxfo, self.remaining) > (other.lhs, other.leaves, other.maxfo, other.remaining)


class ActiveParser:
    def __init__(self, grammar: grammar, gamma: float = 0.1):
        self.grammar = grammar
        self.discount = gamma


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.chart = {}
        self.weight = {}
        self.queue = PriorityQueue()
        for i, rules in enumerate(rules_per_position):
            maxweight = max(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = maxweight - weight
                rule: headed_rule = self.grammar.rules[rid]
                self.queue.put_nowait(qelement(
                        ActiveItem(rule.lhs, i, BitSpan.fromit((), self.len), rule.fanout, rule.rhs),
                        backtrace(rid, i, ()),
                        weight, 0
                ))


    def fill_chart(self):
        expanded = set()
        self.from_lhs: dict[str, list[tuple[BitSpan, int, float, float]]] = defaultdict(SortedList) 
        self.actives: dict[str, list[tuple[ActiveItem, backtrace, float, float]]] = {}
        self.backtraces = []
        while not self.queue.empty():
            qi: qelement = self.queue.get_nowait()
            # todo leaf should be a part of activeitem
            fritem = qi.item.freeze()
            print(qi)
            if fritem in expanded:
                print("was already expanded")
                continue
            expanded.add(fritem)

            if isinstance(qi.item, PassiveItem):
                backtrace_id = len(self.backtraces)
                self.backtraces.append(qi.bt)
                if qi.item.lhs == self.grammar.root and qi.item.leaves.leaves.all():
                    self.rootid = -1
                    return

                qi.bt = backtrace_id
                self.from_lhs[qi.item.lhs].add((qi.item.leaves, qi.bt, qi.weight, qi.gapscore))
                
                for active, abt, _weight, _gapscore in self.actives.get(qi.item.lhs, []):
                    if not active.leaves.leftmost < qi.item.leaves.leftmost or \
                            abt.leaf in qi.item.leaves or \
                            not active.leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = active.leaves.union(qi.item.leaves)
                    self.queue.put_nowait(qelement(
                        ActiveItem(active.lhs, active.leaf, newpos, active.maxfo, active.remaining[1:]),
                        backtrace(abt.rid, abt.leaf, abt.children+(backtrace_id,)),
                        qi.weight+_weight,
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
                continue

            assert isinstance(qi.item, ActiveItem)
            # print(qi.item.remaining)
            if not qi.item.remaining:
                leaves = qi.item.leaves.with_leaf(qi.item.leaf)
                # print(leaves, leaves.numgaps(), qi.item.maxfo)
                if not leaves.numgaps() < qi.item.maxfo: continue
                self.queue.put_nowait(qelement(
                    PassiveItem(qi.item.lhs, leaves),
                    qi.bt,
                    qi.weight,
                    qi.gapscore
                ))
                continue

            self.actives.setdefault(qi.item.remaining[0], []).append((qi.item, qi.bt, qi.weight, qi.gapscore))
            for (span, pbt, pweight, pgaps) in self.from_lhs.get(qi.item.remaining[0], []):
                if not qi.item.leaves.leftmost < span.leftmost or \
                        qi.bt.leaf in span or \
                        not qi.item.leaves.isdisjoint(span):
                    continue
                newpos = qi.item.leaves.union(span)
                self.queue.put_nowait(qelement(
                    ActiveItem(qi.item.lhs, qi.item.leaf, newpos, qi.item.maxfo, qi.item.remaining[1:]),
                    backtrace(qi.bt.rid, qi.bt.leaf, qi.bt.children+(pbt,)),
                    qi.weight+pweight,
                    newpos.gaps + self.discount*(qi.gapscore+pgaps)
                ))


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return f"(NOPARSE {' '.join(str(p) for p in range(self.len))})"
        bt: backtrace = self.backtraces[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        # match bt.as_tuple():
        #     case (rid, ()):
        #         return fn()
        #     case (rid, (pos,)):
        #         return fn(self.get_best(bt.children[0], push[0]))
        #     case (rid, (pos1,pos2)):
        #         return fn(self.get_best(bt.children[0], push[0]), self.get_best(bt.children[1], push[1]))
        return fn(*(self.get_best(c, p) for c, p in zip(bt.children, push)))

    def get_best_deriv(self, item = None):
        if item is None:
            item = self.rootid
        bt: backtrace = self.backtraces[item]
        return Tree((bt.leaf, self.grammar.rules[bt.rid]), (self.get_best_deriv(c) for c in bt.children))
