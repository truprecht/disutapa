from dataclasses import dataclass, field
from typing import Callable

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


def gaps(positions):
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

def numgaps(positions, totallen: int):
    return sum(1 for _ in gaps(positions))

def relative_gaps(positions, totallen: int):
    total_span = max(positions)-min(positions)+1
    total_gaps = sum(gaps(positions))
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
                print("done after", iterations, "iterations with qmax =", qmax)
                return
        print("failed parsing after", iterations, "iterations with qmax =", qmax)


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
        print(self.from_top)

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
                        print(fn)
                        yield fn(), usedidx
                case (rhs1,):
                    for c, u in self.enumerate(rhs1, set(usedidx), push[0], rightmost):
                        if not rightmost or u == set(range(self.len)):
                            print(fn(c))
                            yield fn(c), u
                case (rhs1, rhs2):
                    for c1, u1 in self.enumerate(rhs1, set(usedidx), push[0]):
                        for c2, u2 in self.enumerate(rhs2, set(u1), push[1], rightmost):
                            if not rightmost or u2 == set(range(self.len)):
                                print(fn(c1, c2))
                                yield fn(c1, c2), u2
            
    def get_best(self):
        tree, _ = next(iter(self.enumerate(self.grammar.root, set(), rightmost=True)))
        print(tree)
        return tree


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

    def _active_step(self, actives, passives):
        for (rid, lex, leaves, push, i), obacktrace in actives:
            rule = self.grammar.rules[rid]
            for (_, nleaves, _, npush) in passives:
                if not leaves.isdisjoint(nleaves): continue
                nnleaves = leaves.union(nleaves)
                nbacktrace = obacktrace + ((nleaves, npush),)
                if i+1 == len(rule.as_tuple()[1]):
                    yield (rule.lhs, nnleaves, lex, push), backtrace(rid, lex, nbacktrace)
                else:
                    yield (rid, lex, nnleaves, push, i+1), nbacktrace


    def _initial_items(self, lhs, push):
        for (nrid, nlex) in self.from_top.get(lhs, []):
            _, pushes = self.grammar.rules[nrid].fn(nlex, push)
            nleaves = frozenset(l for l in (nlex, push) if not l is None and not l in pushes)
            if len(self.grammar.rules[nrid].rhs) == 0:
                yield (lhs, nleaves, nlex, push), backtrace(nrid, nlex, ())
            else:
                yield (nrid, nlex, nleaves, push, 0), ()


    def fill_chart(self):
        actives = {}
        passives = {}
        backtraces = {}
        initialized = {(self.grammar.root, None)}
        queue = list(self._initial_items(self.grammar.root, None))
        while queue:
            item, backtrace = queue.pop()
            print("item:", item, backtrace)
            match item:
                case (lhs, leaves, lex, push):
                    backtraces.setdefault((lhs, leaves, push), []).append(backtrace)
                    passives.setdefault((lhs, push), []).append(item)
                    print("completing atives:", actives.get((lhs, push), []))
                    queue.extend(self._active_step(
                        actives.get((lhs, push), []),
                        [item],
                    ))

                case (rid, lex, leaves, push, i):
                    rule = self.grammar.rules[rid]
                    _, ps = rule.fn(lex, push)
                    if not (rule.rhs[i], ps[i]) in initialized:
                        initialized.add((rule.rhs[i], ps[i]))
                        queue.extend(self._initial_items(rule.rhs[i], ps[i]))
                    else:
                        print("completing active with", passives.get((rule.rhs[i], ps[i]), []))
                        queue.extend(self._active_step(
                            [(item, backtrace)],
                            passives.get((rule.rhs[i], ps[i]), [])
                        ))
                    actives.setdefault((rule.rhs[i], ps[i]), []).append((item, backtrace))
            print("queue:", queue)
            print("actives:", actives)
            print("passives:", passives)
        self.chart = backtraces


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.grammar.root, frozenset(range(0,self.len)), None
        bt = self.chart[item][0]
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
