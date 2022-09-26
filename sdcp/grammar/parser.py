from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray
from bitarray.util import count_and

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

    def __lt__(self, other: "BitSpan"):
        return self.leaves < other.leaves


@dataclass(frozen=True, order=False)
class ActiveItem:
    lhs: str
    successors: Tuple[Tuple[str, Optional[int]]]
    leaves: BitSpan
    lex: int
    pushed: Optional[int]
    leftmost: int = field(compare=False, hash=False, repr=False)
    maxfo: int

    def __gt__(self, other: "ActiveItem"):
        if isinstance(other, PassiveItem): return True
        return (self.leaves, self.maxfo, self.lhs, self.pushed or 0, self.lex, self.successors) > (other.leaves, other.maxfo, other.lhs, other.pushed or 0, other.lex, other.successors)


@dataclass(frozen=True, order=False)
class PassiveItem:
    lhs: str
    leaves: BitSpan
    pushed: Optional[int]

    def __gt__(self, other: "PassiveItem"):
        if isinstance(other, ActiveItem): return False
        return (self.leaves, self.lhs, self.pushed or 0) > (other.leaves, other.lhs, other.pushed or 0)


@dataclass(eq=False, order=False)
class qelement:
    item: ActiveItem | PassiveItem
    bt: backtrace
    weight: float
    wheuristic: float
    gapscore: float

    def __gt__(self, other):
        return (other.wheuristic, self.gapscore, self.item) > (self.wheuristic, other.gapscore, other.item)

    def tup(self):
        return self.item, self.bt, self.weight, self.gapscore


class LeftCornerParser:
    def __init__(self, grammar: grammar, fanout_discount: float = 0.5):
        self.grammar = grammar
        self.gamma = fanout_discount


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.bestweights = []
        self.from_top = {}
        for i, rules in enumerate(rules_per_position):
            for rid, w in rules:
                ruleobj = self.grammar.rules[rid]
                self.from_top.setdefault(ruleobj.lhs, []).append((rid, i, ruleobj.fanout_hint, w))
            self.bestweights.append(max(w for _, w in rules))
        self._rootitem = PassiveItem(self.grammar.root, BitSpan(frozenbitarray([1]*self.len)), None)


    def _heuristic(self, span: BitSpan):
        return sum(self.bestweights[i] for i in range(self.len) if span.leaves[i] == 0)


    def _initial_items(self, lhs: str, push: Optional[int], leftmost: int):
        for (nrid, nlex, maxfo, w) in self.from_top.get(lhs, []):
            if push == nlex or (not push is None and push < leftmost) or nlex < leftmost: continue
            rule = self.grammar.rules[nrid]
            _, pushes = rule.fn(nlex, push)
            nleaves = BitSpan.fromit((l for l in (nlex, push) if not l is None and not l in pushes), self.len)
            if (gapscore := nleaves.numgaps()) >= maxfo:
                continue
            nleftmost = leftmost+1 if nleaves and nleaves.leftmost == leftmost else leftmost
            wh = w+self._heuristic(nleaves)
            if not rule.rhs:
                yield qelement(PassiveItem(lhs, nleaves, push), backtrace(nrid, nlex, ()), w, wh, gapscore)
            else:
                yield qelement(ActiveItem(lhs, tuple(zip(rule.rhs, pushes)), nleaves, nlex, push, nleftmost, maxfo), (nrid,), w, wh, gapscore)


    def _active_step(self, actives, pitems):
        for aitem, abt, aw, ags in actives:
            for pleaves, pitemid, pw, pgs in pitems:
                if not pleaves.isdisjoint(aitem.leaves) or \
                        pleaves.leftmost < aitem.leftmost:
                    continue
                nleaves = pleaves.union(aitem.leaves)
                nbacktrace = abt + (pitemid,)
                if len(aitem.successors) > 1:
                    nleftmost = nleaves.firstgap()
                    wh = aw+pw+self._heuristic(nleaves)
                    yield qelement(ActiveItem(aitem.lhs, aitem.successors[1:], nleaves, aitem.lex, aitem.pushed, nleftmost, aitem.maxfo), nbacktrace, aw+pw, wh, pgs)
                else:
                    if (gaps := nleaves.numgaps()) >= aitem.maxfo:
                        continue
                    gapscore = self.gamma*(ags+pgs) + gaps
                    rid, *nlvs = nbacktrace
                    wh = aw+pw+self._heuristic(nleaves)
                    yield qelement(PassiveItem(aitem.lhs, nleaves, aitem.pushed), backtrace(rid, aitem.lex, nlvs), aw+pw, wh, gapscore)


    def fill_chart(self):
        actives = {}
        passives = {}
        backtraces = []
        initialized = {(self.grammar.root, None): 0}
        seen = set()
        queue = PriorityQueue()
        for i in self._initial_items(self.grammar.root, None, 0):
            queue.put(i)
        while not queue.empty():
            item, backtrace, w, gs = queue.get_nowait().tup()
            if item in seen:
                continue
            seen.add(item)
            match item:
                case PassiveItem(lhs, leaves, push):
                    itemid = len(backtraces)
                    backtraces.append(backtrace)
                    if item == self._rootitem:
                        self.rootid = itemid
                        break

                    passives.setdefault((lhs, push), []).append((leaves, itemid, w, gs))
                    for it in self._active_step(actives.get((lhs, push), []), [(leaves, itemid, w, gs)]):
                        if not it.item in seen:
                            queue.put(it)

                case ActiveItem(lhs, successors, leaves, lex, pushed, lm):
                    if initialized.get(successors[0], self.len) > lm:
                        initialized[successors[0]] = lm
                        for i in self._initial_items(*successors[0], lm):
                            queue.put(i)
                    for it in self._active_step([(item, backtrace, w, gs)], passives.get(successors[0], [])):
                        if not it.item in seen:
                            queue.put(it)
                    actives.setdefault(successors[0], []).append((item, backtrace, w, gs))
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
