from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray, bitarray
from bitarray.util import count_and
from sortedcontainers import SortedList
from collections import defaultdict
from .buparser import BitSpan, PassiveItem, backtrace, qelement
from discodop.tree import Tree


class EnsembleParser:
    def __init__(self, grammar: grammar, gamma: float = 0.1):
        self.grammar = grammar
        self.discount = gamma


    def init(self, rule_combination_scores, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.chart = {}
        self.weight = {}
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        self.queue = PriorityQueue()
        self.rule_scores = rule_combination_scores
        for i, rules in enumerate(rules_per_position):
            maxweight = max(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = maxweight - weight
                rule = self.grammar.rules[rid]
                match rule.as_tuple():
                    case (lhs, ()):
                        self.queue.put(qelement(
                            PassiveItem(rid, BitSpan.fromit((i,), self.len), rule.fanout_hint),
                            backtrace(rid, i, ()),
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
        while not self.queue.empty():
            qi: qelement = self.queue.get_nowait()
            fritem = qi.item.freeze()
            if fritem in expanded:
                continue
            expanded.add(fritem)
            backtrace_id = len(self.backtraces)
            self.backtraces.append(qi.bt)
            self.weight.append(qi.weight)
            qi.bt, backtrace_id = backtrace_id, qi.bt
            self.from_lhs[qi.item.lhs].add((qi.item.leaves, backtrace_id.rid, qi.bt, qi.weight, qi.gapscore))

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
                    backtrace(rid, i, (qi.bt,)),
                    qi.weight+weight+self.rule_scores(rid, backtrace_id.rid),
                    newpos.gaps + self.discount*qi.gapscore
                ))
            for rid, i, weight in self.from_left.get(qi.item.lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[1]]:
                    if rule.fanout_hint == 1 and not (qi.item.leaves.firstgap == i or qi.item.leaves.firstgap == _leaves.leftmost):
                        continue
                    if i in _leaves \
                            or not qi.item.leaves.leftmost < _leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps >= rule.fanout_hint:
                        continue
                    self.queue.put_nowait(qelement(
                        PassiveItem(rid, newpos, rule.fanout_hint),
                        backtrace(rid, i, (qi.bt, _bt)),
                        qi.weight+_weight+weight+self.rule_scores(rid, backtrace_id.rid, self.backtraces[_bt].rid),
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
            for rid, i, weight in self.from_right.get(qi.item.lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[0]]:
                    if rule.fanout_hint == 1 and not (_leaves.firstgap == i or _leaves.firstgap == qi.item.leaves.leftmost):
                        continue
                    if i in _leaves \
                            or not _leaves.leftmost < qi.item.leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps >= rule.fanout_hint:
                        continue
                    self.queue.put_nowait(qelement(
                        PassiveItem(rid, newpos, rule.fanout_hint),
                        backtrace(rid, i, (_bt, qi.bt)),
                        qi.weight+_weight+weight+self.rule_scores(rid, self.backtraces[_bt].rid, backtrace_id.rid),
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))

    
    def get_best_derivation(self, item=None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return None, None
        bt: backtrace = self.backtraces[item]
        w: float = self.weight[item]
        return Tree((bt.rid, bt.leaf), [self.get_best_derivation(i) for i in bt.children])


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return f"(NOPARSE {' '.join(str(p) for p in range(self.len))})", None
        bt: backtrace = self.backtraces[item]
        w: float = self.weight[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn(), w
            case (rid, (pos,)):
                return fn(self.get_best(bt.children[0], push[0])[0]), w
            case (rid, (pos1,pos2)):
                return fn(self.get_best(bt.children[0], push[0])[0], self.get_best(bt.children[1], push[1])[0]), w
