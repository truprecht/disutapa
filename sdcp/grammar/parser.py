from ctypes import Union
from dataclasses import dataclass, field
from optparse import Option
from typing import Callable, Iterable, Optional, Tuple

from responses import Call
from .sdcp import grammar, rule, sdcp_clause
from queue import PriorityQueue
from itertools import product
from discodop.tree import Tree
from bitarray import frozenbitarray
from bitarray.util import count_and
from collections import defaultdict

@dataclass
class backtrace:
    rid: int
    leaf: int
    child_leafs: tuple[frozenset]

    def as_tuple(self):
        return self.rid, self.child_leafs

@dataclass(eq=False, order=False)
class qitem:
    lhs: str
    leaves: set()
    weight: float

    def __lt__(self, other):
        return self.weight < other.weight

def spans(positions):
    if not positions: return
    ps = sorted(positions)
    start, end = ps[0], ps[0]
    for p in ps[1:]:
        if p == end+1:
            end = p
        else:
            yield start, end
            start, end = p, p
    yield start, end

def gaplens(positions):
    ps = sorted(positions)
    for (last, next) in zip(ps[:-1], ps[1:]):
        if last+1 != next:
            yield next-last-1

def gapscore(positions, totallen: int):
    last_included = False
    score = 0
    for i in range(totallen):
        if not i in positions:
            score += 1 if not last_included else 2
            last_included = False
        else:
            last_included = True
    return score

def numgaps(positions, totallen: int = None):
    return sum(1 for _ in gaplens(positions))

def relative_gaps(positions, totallen: int):
    total_span = max(positions)-min(positions)+1
    total_gaps = sum(gaplens(positions))
    return total_gaps/total_span


class parser:
    def __init__(self, grammar: grammar, score: Callable = numgaps):
        self.grammar = grammar
        self.score = score


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.chart = {}
        self.weight = {}
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        for i, rules in enumerate(rules_per_position):
            nullary_entries = []
            for rid in rules:
                match self.grammar.rules[rid].as_tuple():
                    case (lhs, ()):
                        nullary_entries.append((lhs, backtrace(rid, i, ())))
                    case (lhs, (r1,)):
                        self.unaries.setdefault(r1, []).append((rid, i))
                    case (lhs, (r1, r2)):
                        self.from_left.setdefault(r1, []).append((rid, i))
                        self.from_right.setdefault(r2, []).append((rid, i))
            for (lhs, bt) in nullary_entries:
                self.chart[(lhs, frozenset([i]))] = bt
                self.weight[(lhs, frozenset([i]))] = self.score({i}, self.len)


    def save_backtrace(self, item, backtrace, weight):
        if not item in self.weight or self.weight[item] > weight:
            self.weight[item] = weight
            self.chart[item] = backtrace
            return True
        return False


    def fill_chart(self):
        queue = PriorityQueue()
        for item in self.chart:
            queue.put(qitem(*item, 0))
        iterations = 0
        qmax = 0
        expanded = set()
        while not queue.empty():
            iterations += 1
            qi = queue.get_nowait()
            qmax = max(qmax, queue.qsize())
            lhs, positions = qi.lhs, qi.leaves
            if (lhs, positions) in expanded:
                continue
            expanded.add((lhs, positions))
            new_elements = []
            for rid, i in self.unaries.get(lhs, []):
                if i in positions:
                    continue
                newpos = positions.union({i})
                newlhs = self.grammar.rules[rid].lhs
                weight = max(self.weight[(lhs, positions)], self.score(newpos, self.len))
                new_elements.append(((newlhs, newpos), backtrace(rid, i, (positions,)), weight))
            for rid, i in self.from_left.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]
                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[1] and not i in positions2 \
                            and not positions2.intersection(positions) \
                            and min(positions) < min(positions2):
                        newpos = positions.union({i}).union(positions2)
                        weight = max(self.weight[(lhs, positions)], self.weight[(rhs2, positions2)], self.score(newpos, self.len))
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions, positions2)), weight))
            for rid, i in self.from_right.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]
                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[0] and not i in positions2 \
                            and not positions2.intersection(positions) \
                            and min(positions2) < min(positions):
                        newpos = positions.union({i}).union(positions2)
                        weight = max(self.weight[(lhs, positions)], self.weight[(rhs2, positions2)], self.score(newpos, self.len))
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions2, positions)), weight))
            for item, bt, w in new_elements:
                if self.save_backtrace(item, bt, w):
                    queue.put(qitem(*item, w))
            if lhs == self.grammar.root and positions == set(range(self.len)):
                return


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.grammar.root, frozenset(range(0,self.len))
        bt = self.chart[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn()
            case (rid, (pos,)):
                childitem = self.grammar.rules[rid].rhs[0], pos
                return fn(self.get_best(childitem, push[0]))
            case (rid, (pos1,pos2)):
                childitem1 = self.grammar.rules[rid].rhs[0], pos1
                childitem2 = self.grammar.rules[rid].rhs[1], pos2
                return fn(self.get_best(childitem1, push[0]), self.get_best(childitem2, push[1]))



class TopdownParser:
    def __init__(self, grammar: grammar, score: Callable = numgaps):
        self.grammar = grammar
        self.score = score

    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.from_top = {}
        for i, rules in enumerate(rules_per_position):
            for rid in rules:
                self.from_top.setdefault(self.grammar.rules[rid].lhs, []).append((rid, i))

    def fill_chart(self):
        pass

    def enumerate(self, nt: str, usedidx: set, pushed: int = None, rightmost = False):
        for rid, i in self.from_top.get(nt, []):
            if i in usedidx: continue
            usedidx.add(i)
            _, rhs = self.grammar.rules[rid].as_tuple()
            fn, push = self.grammar.rules[rid].fn(i, pushed)
            match rhs:
                case ():
                    if not rightmost or usedidx == set(range(self.len)):
                        yield fn(), usedidx
                case (rhs1,):
                    for c, u in self.enumerate(rhs1, set(usedidx), push[0], rightmost):
                        if not rightmost or u == set(range(self.len)):
                            yield fn(c), u
                case (rhs1, rhs2):
                    for c1, u1 in self.enumerate(rhs1, set(usedidx), push[0]):
                        for c2, u2 in self.enumerate(rhs2, set(u1), push[1], rightmost):
                            if not rightmost or u2 == set(range(self.len)):
                                yield fn(c1, c2), u2
            
    def get_best(self):
        tree, _ = next(iter(self.enumerate(self.grammar.root, set(), rightmost=True)))
        return tree


@dataclass(frozen=True)
class Spans:
    tups: Tuple[Tuple[int, int]]

    # def __contains__(self, position: int) -> bool:
    #     return any(l <= position <= r for l,r in self.tups)

    def isdisjoint(self, other: Optional["Spans"]) -> bool:
        if other is None:
            return True
        ospans = iter(other.tups)
        ole, ori = next(ospans)
        for l, r in self.tups:
            while ori < l:
                if not (m := next(ospans, None)) is None:
                    ole, ori = m
                else:
                    return True
            if r >= ole: return False
        return True

    @classmethod
    def _addspan(cls, l: list[int], s: Tuple[int, int]):
        if not l or l[-1][1]+1 < s[0]:
            l.append(s)
            return
        l[-1] = (l[-1][0], s[1])

    def union(self, other: Optional["Spans"]) -> bool:
        if other is None:
            return self
        spans = []
        ospans = iter(other.tups)
        sspans = iter(self.tups)
        ole, ori = next(ospans)
        for l, r in sspans:
            while ole < l:
                self.__class__._addspan(spans, (ole, ori))
                if not (m := next(ospans, None)) is None:
                    ole, ori = m
                else:
                    self.__class__._addspan(spans, (l,r))
                    spans.extend(sspans)
                    return Spans(tuple(spans))
            self.__class__._addspan(spans, (l,r))
        self.__class__._addspan(spans, (ole, ori))
        spans.extend(ospans)
        return Spans(tuple(spans))

    def leftmost(self) -> int:
        return self.tups[0][0]

    def firstgap(self) -> int:
        return self.tups[0][1]+1

    @classmethod
    def fromit(cls, ps: Iterable[int]):
        obj = None
        for p in ps:
            news = cls(((p,p),))
            if obj is None:
                obj = news
                continue
            obj = obj.union(news)
        return obj

    def numgaps(self):
        return len(self.tups)-1

    def __iter__(self):
        return (i for r in self.tups for i in range(r[0], r[1]+1))

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



@dataclass(frozen=True, init=False)
class SetSpans:
    positions: frozenset[int]

    def __init__(self, s):
        if isinstance(s, SetSpans):
            s = s.positions
        self.__dict__["positions"] = s

    def isdisjoint(self, other: Optional["SetSpans"]) -> bool:
        return self.positions.isdisjoint(other.positions)

    def union(self, other: Optional["SetSpans"]) -> bool:
        return self.__class__(self.positions.union(other.positions))

    def leftmost(self) -> int:
        return min(self.positions)

    def firstgap(self) -> int:
        return next(spans(self.positions))[1]+1

    @classmethod
    def fromit(cls, ps: Iterable[int]):
        return cls(frozenset(ps))

    def numgaps(self):
        return numgaps(self.positions)

    def __bool__(self):
        return bool(self.positions)

    def __iter__(self):
        return iter(sorted(self.positions))




@dataclass(frozen=True)
class ActiveItem:
    lhs: str
    successors: Tuple[Tuple[str, Optional[int]]]
    leaves: Spans
    lex: int
    pushed: Optional[int]
    leftmost: int


@dataclass(frozen=True)
class PassiveItem:
    lhs: str
    leaves: Spans
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


    def _active_step(self, actives, pitems, ppush):
        for aitem, abt, aw in actives:
            for pleaves, pitemid, pw in pitems:
                if not pleaves.isdisjoint(aitem.leaves) or \
                        pleaves.leftmost < aitem.leftmost:
                    continue
                nleaves = pleaves.union(aitem.leaves)
                nbacktrace = abt + (pitemid,)
                if len(aitem.successors) > 1:
                    nleftmost = nleaves.firstgap()
                    yield qelement(ActiveItem(aitem.lhs, aitem.successors[1:], nleaves, aitem.lex, aitem.pushed, nleftmost), nbacktrace, pw)
                else:
                    rid, *nlvs = nbacktrace
                    yield qelement(PassiveItem(aitem.lhs, nleaves, aitem.pushed), backtrace(rid, aitem.lex, nlvs), self.gamma*(aw+pw) + nleaves.numgaps())


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
                    for it in self._active_step(actives.get((lhs, push), []), [(leaves, itemid, w)], push):
                        if not it.item in seen:
                            queue.put(it)

                case ActiveItem(lhs, successors, leaves, lex, pushed, lm):
                    if initialized.get(successors[0], self.len) > lm:
                        initialized[successors[0]] = lm
                        for i in self._initial_items(*successors[0], lm):
                            queue.put(i)
                    for it in self._active_step([(item, backtrace, w)], passives.get(successors[0], []), successors[0][1]):
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
