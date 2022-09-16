from dataclasses import dataclass, field
from optparse import Option
from typing import Callable, Optional, Tuple

from responses import Call
from .sdcp import grammar, rule, sdcp_clause
from queue import PriorityQueue
from itertools import product
from discodop.tree import Tree

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


# @dataclass(init=False, frozen=True)
# class ItemFilter:
#     hard: bool
#     leftmost: int
#     rightmost: Optional[int]

#     def __init__(self, leftmost: int, soft: bool = False, rightmost: Optional[int] = None):
#         self.hard = soft
#         self.leftmost = leftmost
#         self.rightmost = rightmost

#     def compatible(self, positions: set, last: bool):
#         mip, map = min(positions), max(positions)
#         mifits = mip == self.leftmost or \
#             (not self.hard and mip > self.leftmost)
#         mafits = not last or self.rightmost is None or map == self.rightmost
#         return mafits and mifits

#     def next(self, positions: set):
#         if not positions: return self
#         spanit = iter(spans(positions))
#         start, end = next(spanit)
#         leftmost = end+1 if start == self.leftmost else self.leftmost
#         soft = soft or (start == leftmost and self.rightmost is None)
#         if not self.rightmost is None:
#             for start, end in spanit:
#                 continue
#             rightmost = start-1 if end == self.rightmost else self.rightmost
#         else:
#             rightmost = None
#         return ItemFilter(leftmost, soft, rightmost)


@dataclass(frozen=True)
class ActiveItem:
    lhs: str
    successors: Tuple[Tuple[str, Optional[int]]]
    leaves: set
    lex: int
    pushed: Optional[int]
    # successorfilter: ItemFilter
    leftmost: int

    def weight(self):
        return numgaps(self.leaves)


@dataclass(frozen=True)
class PassiveItem:
    lhs: str
    leaves: set
    pushed: Optional[int]

    def weight(self):
        return numgaps(self.leaves)


@dataclass(eq=False, order=False, init=False)
class qelement:
    item: ActiveItem | PassiveItem
    bt: backtrace
    weight: float

    def __init__(self, item, bt):
        self.item = item
        self.bt = bt
        self.weight = item.weight()

    def __gt__(self, other):
        return self.weight > other.weight

    def tup(self):
        return self.item, self.bt


class LeftCornerParser:
    def __init__(self, grammar: grammar, score: Callable = numgaps):
        self.grammar = grammar
        self.score = score


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.from_top = {}
        for i, rules in enumerate(rules_per_position):
            for rid in rules:
                self.from_top.setdefault(self.grammar.rules[rid].lhs, []).append((rid, i))


    def _initial_items(self, lhs: str, push: Optional[int], leftmost: int):
        for (nrid, nlex) in self.from_top.get(lhs, []):
            if push == nlex or (not push is None and push < leftmost) or nlex < leftmost: continue
            rule = self.grammar.rules[nrid]
            _, pushes = rule.fn(nlex, push)
            nleaves = frozenset(l for l in (nlex, push) if not l is None and not l in pushes)
            nleftmost = leftmost+1 if nleaves and min(nleaves) == leftmost else leftmost
            if not rule.rhs:
                yield qelement(PassiveItem(lhs, nleaves, push), backtrace(nrid, nlex, ()))
            else:
                yield qelement(ActiveItem(lhs, tuple(zip(rule.rhs, pushes)), nleaves, nlex, push, leftmost), (nrid,))


    def _active_step(self, actives, passives):
        for aitem, abt in actives:
            for pitem in passives:
                if not aitem.leaves.isdisjoint(pitem.leaves) or \
                        min(pitem.leaves) < aitem.leftmost:
                    continue
                nleaves = aitem.leaves.union(pitem.leaves)
                nbacktrace = abt + ((pitem.leaves, pitem.pushed),)
                if len(aitem.successors) > 1:
                    nleftmost = next(spans(nleaves))[1]+1
                    yield qelement(ActiveItem(aitem.lhs, aitem.successors[1:], nleaves, aitem.lex, aitem.pushed, nleftmost), nbacktrace)
                else:
                    rid, *nlvs = nbacktrace
                    yield qelement(PassiveItem(aitem.lhs, nleaves, aitem.pushed), backtrace(rid, aitem.lex, nlvs))


    def fill_chart(self):
        actives = {}
        passives = {}
        backtraces = {}
        initialized = {(self.grammar.root, None): 0}
        seen = set()
        queue = PriorityQueue()
        for i in self._initial_items(self.grammar.root, None, 0):
            queue.put(i)
        while not queue.empty():
            item, backtrace = queue.get_nowait().tup()
            if item in seen:
                continue
            seen.add(item)
            match item:
                case PassiveItem(lhs, leaves, push):
                    backtraces[(lhs, leaves, push)] = backtrace
                    if lhs == self.grammar.root and \
                            leaves == frozenset(range(self.len)) and \
                            push is None:
                        break

                    passives.setdefault((lhs, push), []).append(item)
                    for it in self._active_step(actives.get((lhs, push), []), [item]):
                        if not it.item in seen:
                            queue.put(it)

                case ActiveItem(lhs, successors, leaves, lex, pushed, lm):
                    if initialized.get(successors[0], self.len) > lm:
                        initialized[successors[0]] = lm
                        for i in self._initial_items(*successors[0], lm):
                            queue.put(i)
                    for it in self._active_step([(item, backtrace)], passives.get(successors[0], [])):
                        if not it.item in seen:
                            queue.put(it)
                    actives.setdefault(successors[0], []).append((item, backtrace))
        self.chart = backtraces


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.grammar.root, frozenset(range(0,self.len)), None
        bt = self.chart[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn()
            case (rid, (pos,)):
                childitem = self.grammar.rules[rid].rhs[0], *pos
                return fn(self.get_best(childitem, push[0]))
            case (rid, (pos1,pos2)):
                childitem1 = self.grammar.rules[rid].rhs[0], *pos1
                childitem2 = self.grammar.rules[rid].rhs[1], *pos2
                return fn(self.get_best(childitem1, push[0]), self.get_best(childitem2, push[1]))
