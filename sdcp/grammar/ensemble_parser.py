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
    def __init__(self, grammar: grammar, gamma: float = 0.1, snd_order_weights: bool = True):
        self.grammar = grammar
        self.discount = gamma
        self.sow = snd_order_weights


    def init(self, parsing_scorer, sentence_embedding, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.chart = {}
        self.weight = []
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        self.queue = PriorityQueue()
        self.backtraces = []
        self.rule_scores = lambda it, bt: parsing_scorer(
            bt.rid,
            tuple(self.backtraces[b].rid for b in bt.children),
            bt.leaf,
            it.leaves,
            sentence_embedding)
        for i, rules in enumerate(rules_per_position):
            minweight = min(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = weight - minweight
                rule = self.grammar.rules[rid]
                passive_item_lhs = rid if self.sow else rule.lhs
                match rule.as_tuple():
                    case (lhs, ()):
                        self.queue.put(qelement(
                            PassiveItem(passive_item_lhs, BitSpan.fromit((i,), self.len), rule.fanout_hint),
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
        self.from_lhs: dict[str, list[tuple[BitSpan, int, float, float]]] = defaultdict(SortedList)
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
            lhs = qi.item.lhs if not self.sow else self.grammar.rules[qi.item.lhs].lhs
            self.from_lhs[lhs].add((qi.item.leaves, qi.bt, qi.weight, qi.gapscore))

            if lhs == self.grammar.root and qi.item.leaves.leaves.all():
                self.rootid = -1
                return

            for rid, i, weight in self.unaries.get(lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                # TODO: check gaps first?
                newpos = qi.item.leaves.with_leaf(i)
                passive_item_lhs = rid if self.sow else rule.lhs
                if newpos.gaps >= rule.fanout_hint:
                    continue
                passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout_hint)
                backt = backtrace(rid, i, (qi.bt,))
                self.queue.put_nowait(qelement(
                    passive_item,
                    backt,
                    qi.weight+weight+self.rule_scores(passive_item, backt),
                    newpos.gaps + self.discount*qi.gapscore
                ))
            for rid, i, weight in self.from_left.get(lhs, []):
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
                    passive_item_lhs = rid if self.sow else rule.lhs
                    passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout_hint)
                    backt = backtrace(rid, i, (qi.bt, _bt))
                    self.queue.put_nowait(qelement(
                        passive_item,
                        backt,
                        qi.weight+_weight+weight+self.rule_scores(passive_item, backt),
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
            for rid, i, weight in self.from_right.get(lhs, []):
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
                    passive_item_lhs = rid if self.sow else rule.lhs
                    passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout_hint)
                    backt = backtrace(rid, i, (_bt, qi.bt))
                    self.queue.put_nowait(qelement(
                        passive_item,
                        backt,
                        qi.weight+_weight+weight+self.rule_scores(passive_item, backt),
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
