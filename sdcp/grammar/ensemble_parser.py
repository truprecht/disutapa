from collections import defaultdict
from discodop.tree import Tree
from queue import PriorityQueue
from random import random
from sortedcontainers import SortedList

from .derivation import Derivation
from .buparser import BitSpan, PassiveItem, backtrace, qelement
from .sdcp import grammar


class EnsembleParser:
    def __init__(self, grammar: grammar, gamma: float = 0.1, snd_order_weights: bool = True):
        self.grammar = grammar
        self.discount = gamma
        self.sow = snd_order_weights


    def init(self, parsing_scorer, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.weight = []
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        self.queue = PriorityQueue()
        self.backtraces = []
        self.items = []
        self.rule_scorer = parsing_scorer
        for i, rules in enumerate(rules_per_position):
            minweight = min(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = weight - minweight
                rule = self.grammar.rules[rid]
                passive_item_lhs = rid if self.sow else rule.lhs
                match rule.as_tuple():
                    case (lhs, ()):
                        self.queue.put(qelement(
                            PassiveItem(passive_item_lhs, BitSpan.fromit((i,), self.len), rule.fanout),
                            backtrace(rid, i, ()),
                            weight,
                            0   
                        ))
                    case (lhs, (r1,)):
                        self.unaries.setdefault(r1, []).append((rid, i, weight))
                    case (lhs, (r1, r2)):
                        self.from_left.setdefault(r1, []).append((rid, i, weight))
                        self.from_right.setdefault(r2, []).append((rid, i, weight))
        self.golditems = None
        self.stop_early = True
     
     
    def add_nongold_filter(self, gold_tree: Derivation, nongold_stopping_prob: float = 0.9, early_stopping: float = None):
        self.golditems = set()
        self.brassitems = list()
        for node in gold_tree.subderivs():
            lhs = node.rule if self.sow else self.grammar.rules[node.rule].lhs
            self.golditems.add((lhs, node.yd.freeze()))
        self.nongold_stop_prob = nongold_stopping_prob
        self.stop_early = early_stopping


    def check_nongold_filter(self, item, bt):
        # return true if this item shall not be further explored
        # this is used during training for faster parsing
        # but will only explore wrong items found in limited depth
        return not self.golditems is None and bt.children and \
                not item[:2] in self.golditems and \
                random() < self.nongold_stop_prob


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
            self.items.append(qi.item)

            if self.check_nongold_filter(fritem, qi.bt):
                self.brassitems.append(len(self.items)-1)
                continue

            qi.bt, backtrace_id = backtrace_id, qi.bt
            lhs = qi.item.lhs if not self.sow else self.grammar.rules[qi.item.lhs].lhs
            self.from_lhs[lhs].add((qi.item.leaves, qi.bt, qi.weight, qi.gapscore))

            if lhs == self.grammar.root and qi.item.leaves.leaves.all():
                if self.rootid is None:
                    self.rootid = len(self.backtraces)-1
                if self.stop_early is None:
                    return
            if not self.rootid is None and not self.stop_early is None and qi.weight > self.stop_early:
                return

            self.new_items: list[qelement] = []
            for rid, i, weight in self.unaries.get(lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                # TODO: check gaps first?
                newpos = qi.item.leaves.with_leaf(i)
                passive_item_lhs = rid if self.sow else rule.lhs
                if newpos.gaps != rule.fanout-1:
                    continue
                passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout)
                backt = backtrace(rid, i, (qi.bt,))
                self.new_items.append(qelement(
                    passive_item,
                    backt,
                    qi.weight+weight,
                    newpos.gaps + self.discount*qi.gapscore
                ))
            for rid, i, weight in self.from_left.get(lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[1]]:
                    if rule.fanout == 1 and not (qi.item.leaves.firstgap == i or qi.item.leaves.firstgap == _leaves.leftmost):
                        continue
                    if i in _leaves \
                            or not qi.item.leaves.leftmost < _leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps != rule.fanout-1:
                        continue
                    passive_item_lhs = rid if self.sow else rule.lhs
                    passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout)
                    backt = backtrace(rid, i, (qi.bt, _bt))
                    self.new_items.append(qelement(
                        passive_item,
                        backt,
                        qi.weight+_weight+weight,
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
            for rid, i, weight in self.from_right.get(lhs, []):
                if i in qi.item.leaves:
                    continue
                rule = self.grammar.rules[rid]
                for (_leaves, _bt, _weight, _gapscore) in self.from_lhs[rule.rhs[0]]:
                    if rule.fanout == 1 and not (_leaves.firstgap == i or _leaves.firstgap == qi.item.leaves.leftmost):
                        continue
                    if i in _leaves \
                            or not _leaves.leftmost < qi.item.leaves.leftmost \
                            or not _leaves.isdisjoint(qi.item.leaves):
                        continue
                    newpos = qi.item.leaves.union(_leaves, and_leaf=i)
                    if newpos.gaps != rule.fanout-1:
                        continue
                    passive_item_lhs = rid if self.sow else rule.lhs
                    passive_item = PassiveItem(passive_item_lhs, newpos, rule.fanout)
                    backt = backtrace(rid, i, (_bt, qi.bt))
                    self.new_items.append(qelement(
                        passive_item,
                        backt,
                        qi.weight+_weight+weight,
                        newpos.gaps + self.discount*(qi.gapscore+_gapscore)
                    ))
            if self.new_items:
                weights = self.rule_scorer.score([(i.item, i.bt) for i in self.new_items], self.backtraces)
                for weight, qe in zip(weights, self.new_items):
                    qe.weight += weight
                    self.queue.put(qe)

    
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