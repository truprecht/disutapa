from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray
from bitarray.util import count_and
from collections import defaultdict

@dataclass
class backtrace:
    rid: int
    leaf: int
    children: tuple[int]

    def as_tuple(self):
        return self.rid, self.children


a01 = frozenbitarray('01')
a1 = frozenbitarray('1')
a0 = frozenbitarray('0')
@dataclass(frozen=True)
class BitSpan:
    leaves: frozenbitarray
    leftmost: int = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        self.__dict__["leftmost"] = self._leftmost()

    def isdisjoint(self, other: "BitSpan") -> bool:
        return count_and(self.leaves, other.leaves) == 0

    def union(self, other: "BitSpan") -> "BitSpan":
        return BitSpan(self.leaves | other.leaves)

    def _leftmost(self) -> int:
        return self.leaves.find(a1)

    def firstgap(self) -> int:
        if (m := self.leaves[self.leftmost:].find(a0)) != -1:
            return self.leftmost+m
        return len(self.leaves)

    @classmethod
    def fromit(cls, ps: Iterable[int], len: int):
        bv = [False] * len
        for p in ps:
            bv[p] = True
        return cls(frozenbitarray(bv))

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


@dataclass(frozen=True)
class ActiveItem:
    lhs: str
    successors: Tuple[Tuple[str, Optional[int]]]
    leaves: BitSpan
    lex: int
    pushed: Optional[int]
    leftmost: int


@dataclass(frozen=True)
class PassiveItem:
    lhs: str
    leaves: BitSpan
    pushed: Optional[int]


@dataclass(eq=False, order=False)
class qelement:
    item: ActiveItem | PassiveItem
    bt: backtrace
    weight: float

    def __gt__(self, other):
        return self.weight > other.weight

    def tup(self):
        return self.item, self.bt, self.weight


class LeftCornerParser:
    def __init__(self, grammar: grammar, gamma: float = 0.5):
        self.grammar = grammar
        self.gamma = gamma


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.from_top = {}
        for i, rules in enumerate(rules_per_position):
            for rid in rules:
                self.from_top.setdefault(self.grammar.rules[rid].lhs, []).append((rid, i))
        self._rootitem = PassiveItem(self.grammar.root, BitSpan(frozenbitarray([1]*self.len)), None)


    def _initial_items(self, lhs: str, push: Optional[int], leftmost: int):
        for (nrid, nlex) in self.from_top.get(lhs, []):
            if push == nlex or (not push is None and push < leftmost) or nlex < leftmost: continue
            rule = self.grammar.rules[nrid]
            _, pushes = rule.fn(nlex, push)
            nleaves = BitSpan.fromit((l for l in (nlex, push) if not l is None and not l in pushes), self.len)
            nleftmost = leftmost+1 if nleaves and nleaves.leftmost == leftmost else leftmost
            if not rule.rhs:
                yield qelement(PassiveItem(lhs, nleaves, push), backtrace(nrid, nlex, ()), nleaves.numgaps())
            else:
                yield qelement(ActiveItem(lhs, tuple(zip(rule.rhs, pushes)), nleaves, nlex, push, nleftmost), (nrid,), 0)


    def _active_step(self, actives, pitems):
        for aitem, abt, aw1 in actives:
            for pleaves, pitemid, pw1 in pitems:
                if not pleaves.isdisjoint(aitem.leaves) or \
                        pleaves.leftmost < aitem.leftmost:
                    continue
                nleaves = pleaves.union(aitem.leaves)
                nbacktrace = abt + (pitemid,)
                if len(aitem.successors) > 1:
                    nleftmost = nleaves.firstgap()
                    yield qelement(ActiveItem(aitem.lhs, aitem.successors[1:], nleaves, aitem.lex, aitem.pushed, nleftmost), nbacktrace, pw1)
                else:
                    rid, *nlvs = nbacktrace
                    yield qelement(PassiveItem(aitem.lhs, nleaves, aitem.pushed), backtrace(rid, aitem.lex, nlvs), self.gamma*(aw1+pw1) + nleaves.numgaps())


    def fill_chart(self):
        actives = {}
        passives = {}
        backtraces = {}
        initialized = {(self.grammar.root, None): 0}
        seen = defaultdict(lambda: len(seen))
        queue = PriorityQueue()
        for i in self._initial_items(self.grammar.root, None, 0):
            queue.put(i)
        while not queue.empty():
            item, backtrace, w = queue.get_nowait().tup()
            newitemid = len(seen)
            if (itemid := seen[item]) != newitemid:
                continue
            match item:
                case PassiveItem(lhs, leaves, push):
                    backtraces[itemid] = backtrace
                    if item == self._rootitem:
                        self.rootid = itemid
                        break

                    passives.setdefault((lhs, push), []).append((leaves, itemid, w))
                    for it in self._active_step(actives.get((lhs, push), []), [(leaves, itemid, w)]):
                        if not it.item in seen:
                            queue.put(it)

                case ActiveItem(lhs, successors, leaves, lex, pushed, lm):
                    if initialized.get(successors[0], self.len) > lm:
                        initialized[successors[0]] = lm
                        for i in self._initial_items(*successors[0], lm):
                            queue.put(i)
                    for it in self._active_step([(item, backtrace, w)], passives.get(successors[0], [])):
                        if not it.item in seen:
                            queue.put(it)
                    actives.setdefault(successors[0], []).append((item, backtrace, w))
        self.chart = backtraces


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
        bt = self.chart[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn()
            case (rid, (pos,)):
                return fn(self.get_best(pos, push[0]))
            case (rid, (pos1,pos2)):
                return fn(self.get_best(pos1, push[0]), self.get_best(pos2, push[1]))
