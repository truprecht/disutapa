from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray, bitarray
from bitarray.util import count_and
from sortedcontainers import SortedList
from collections import defaultdict


a01 = frozenbitarray('01')
a1 = frozenbitarray('1')
a0 = frozenbitarray('0')
@dataclass
class BitSpan:
    leaves: bitarray
    leftmost: int = field(init=False, repr=False, hash=False, compare=False)
    gaps: int = field(init=False, repr=False, hash=False, compare=False)
    firstgap: int = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        self.__dict__["leftmost"] = self._leftmost()
        self.__dict__["gaps"] = self.numgaps()
        self.__dict__["firstgap"] = self._firstgap()

    def isdisjoint(self, other: "BitSpan") -> bool:
        return count_and(self.leaves, other.leaves) == 0

    def union(self, other: "BitSpan", and_leaf: int = None) -> "BitSpan":
        v = self.leaves | other.leaves
        if not and_leaf is None:
            v[and_leaf] = 1
        return BitSpan(v)

    def _leftmost(self) -> int:
        return self.leaves.find(a1)

    def _firstgap(self) -> int:
        if (m := self.leaves[self.leftmost:].find(a0)) != -1:
            return self.leftmost+m
        return len(self.leaves)

    @classmethod
    def fromit(cls, ps: Iterable[int], len: int):
        bv = [False] * len
        for p in ps:
            bv[p] = True
        return cls(bitarray(bv))

    def numgaps(self):
        return sum(1 for _ in self.leaves[self.leftmost:].itersearch(a01))

    def __bool__(self):
        return self.leaves.any()

    def __iter__(self):
        return (i for i,b in enumerate(self.leaves) if b)

    def __str__(self):
        return str(self.leaves)

    def __repr__(self) -> str:
        return str(self.leaves)

    def numleaves(self):
        return self.leaves.count(1)

    def __lt__(self, other: "BitSpan"):
        return self.leaves < other.leaves

    def __contains__(self, i: int) -> bool:
        return self.leaves[i] == 1

    def with_leaf(self, i: int) -> "BitSpan":
        v = self.leaves.copy()
        v[i] = 1
        return BitSpan(v)

    def freeze(self) -> bytes:
        return self.leaves.tobytes()

    def __len__(self) -> int:
        return self.leaves.count(1)

@dataclass
class PassiveItem:
    lhs: str
    leaves: BitSpan
    maxfo: int

    def freeze(self):
        return (self.lhs, self.leaves.freeze(), self.maxfo)

    def __gt__(self, other: "PassiveItem") -> bool:
        return (self.lhs, self.leaves, self.maxfo) > (other.lhs, other.leaves, other.maxfo)

@dataclass
class backtrace:
    rid: int
    leaf: int
    children: tuple[int]

    def as_tuple(self):
        return self.rid, self.children


@dataclass(eq=False, order=False)
class qelement:
    item: PassiveItem
    bt: backtrace
    weight: float
    wheuristic: float
    gapscore: float

    def __gt__(self, other):
        return (self.gapscore, self.wheuristic, self.item) > (other.gapscore, other.wheuristic, other.item)

    def tup(self):
        return self.item, self.bt, self.weight, self.gapscore


class BuParser:
    def __init__(self, grammar: grammar, gamma: float = 0.1):
        self.grammar = grammar
        self.discount = gamma


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.chart = {}
        self.weight = {}
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        self.queue = PriorityQueue()
        for i, rules in enumerate(rules_per_position):
            maxweight = max(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = maxweight - weight
                rule = self.grammar.rules[rid]
                match rule.as_tuple():
                    case (lhs, ()):
                        self.queue.put(qelement(
                            PassiveItem(lhs, BitSpan.fromit((i,), self.len), rule.fanout_hint),
                            backtrace(rid, i, ()),
                            weight,
                            weight,
                            0   
                        ))
                    case (lhs, (r1,)):
                        self.unaries.setdefault(r1, []).append((rid, i, weight))
                    case (lhs, (r1, r2)):
                        self.from_left.setdefault(r1, []).append((rid, i, weight))
                        self.from_right.setdefault(r2, []).append((rid, i, weight))


    def fill_chart(self):
        expanded = set()
        self.from_lhs: dict[str, list[tuple[BitSpan, int, int, float, float]]] = defaultdict(SortedList)
        self.backtraces = []
        iterations = 0
        maxq = self.queue._qsize()
        while not self.queue.empty():
            iterations += 1
            maxq = max(maxq, self.queue._qsize())
            qi: qelement = self.queue.get_nowait()
            fritem = qi.item.freeze()
            if fritem in expanded:
                continue
            expanded.add(fritem)
            backtrace_id = len(self.backtraces)
            self.backtraces.append(qi.bt)
            qi.bt = backtrace_id
            self.from_lhs[qi.item.lhs].add((qi.item.leaves, qi.item.maxfo, qi.bt, qi.weight, qi.gapscore))

            if qi.item.lhs == self.grammar.root and qi.item.leaves.leaves.all():
                self.rootid = -1
                return

            for rid, i, weight in self.unaries.get(qi.item.lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                # TODO: check gaps first?
                newpos = qi.item.leaves.with_leaf(i)
                if newpos.gaps >= rule.fanout_hint:
                    continue
                self.queue.put_nowait(qelement(
                    PassiveItem(rule.lhs, newpos, rule.fanout_hint),
                    backtrace(rid, i, (backtrace_id,)),
                    qi.weight+weight,
                    qi.weight+weight,
                    newpos.gaps + self.discount*qi.gapscore
                ))
            for rid, i, weight in self.from_left.get(qi.item.lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _maxfo, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[1]]:
                    if rule.fanout_hint == 1 and not ((qi.item.leaves.firstgap == i or qi.item.leaves.firstgap == _leaves.leftmost) \
                            or qi.item.leaves.gaps+1 < qi.item.maxfo and (_leaves.firstgap == i == qi.item.leaves.leftmost-1 or _leaves.firstgap == qi.item.leaves.leftmost)):
                        continue
                    # TODO push leafs before checking leftmost 
                    if i in _leaves \
                            or (qi.item.leaves.gaps+1 == qi.item.maxfo) and not qi.item.leaves.leftmost < _leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps >= rule.fanout_hint:
                        continue
                    self.queue.put_nowait(qelement(
                        PassiveItem(rule.lhs, newpos, rule.fanout_hint),
                        backtrace(rid, i, (backtrace_id, _bt)),
                        qi.weight+_weight+weight,
                        qi.weight+_weight+weight,
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
            for rid, i, weight in self.from_right.get(qi.item.lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _maxfo, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[0]]:
                    if rule.fanout_hint == 1 and not ((_leaves.firstgap == i or _leaves.firstgap == qi.item.leaves.leftmost) \
                            or _leaves.gaps+1 < _maxfo and (qi.item.leaves.firstgap == i == _leaves.leftmost-1 or qi.item.leaves.firstgap == _leaves.leftmost)):
                        continue
                    # TODO push leafs before checking leftmost 
                    if i in _leaves \
                            or (_leaves.gaps+1 == _maxfo) and not _leaves.leftmost < qi.item.leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps >= rule.fanout_hint:
                        continue
                    self.queue.put_nowait(qelement(
                        PassiveItem(rule.lhs, newpos, rule.fanout_hint),
                        backtrace(rid, i, (_bt, backtrace_id)),
                        qi.weight+_weight+weight,
                        qi.weight+_weight+weight,
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return f"(NOPARSE {' '.join(str(p) for p in range(self.len))})"
        bt: backtrace = self.backtraces[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn()
            case (rid, (pos,)):
                return fn(self.get_best(bt.children[0], push[0]))
            case (rid, (pos1,pos2)):
                return fn(self.get_best(bt.children[0], push[0]), self.get_best(bt.children[1], push[1]))
