from dataclasses import dataclass
from typing import Callable
from .sdcp import grammar, rule, sdcp_clause

@dataclass
class backtrace:
    rid: int
    leaf: int
    child_leafs: tuple[frozenset]

    def as_tuple(self):
        return self.rid, self.child_leafs


def gaps(positions):
    ps = sorted(positions)
    for (last, next) in zip(ps[:-1], ps[1:]):
        if last+1 != next:
            yield next-last-1


class parser:
    def __init__(self, grammar: grammar, gap_panelty: Callable = lambda gaps: sum(g+1 for g in gaps)):
        self.grammar = grammar
        self.gap_panelty = gap_panelty


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
                self.weight[(lhs, frozenset([i]))] = self.gap_panelty([])


    def save_backtrace(self, item, backtrace):
        weight = self.gap_panelty(gaps(item[1]))
        if not item in self.weight or self.weight[item] > weight:
            self.weight[item] = weight
            self.chart[item] = backtrace
            return True
        return False


    def fill_chart(self):
        queue = list(self.chart)
        while queue:
            lhs, positions = queue.pop()
            new_elements = []
            for rid, i in self.unaries.get(lhs, []):
                if i in positions:
                    continue
                newpos = positions.union({i})
                newlhs = self.grammar.rules[rid].lhs
                new_elements.append(((newlhs, newpos), backtrace(rid, i, (positions,))))
            for rid, i in self.from_left.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]
                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[1] and not i in positions2 and not positions2.intersection(positions):
                        newpos = positions.union({i}).union(positions2)
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions, positions2))))
            for rid, i in self.from_right.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]

                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[0] and not i in positions2 and not positions2.intersection(positions):
                        newpos = positions.union({i}).union(positions2)
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions2, positions))))
            for item, bt in new_elements:
                if self.save_backtrace(item, bt):
                    queue.append(item)


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