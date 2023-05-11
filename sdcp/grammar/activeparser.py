from dataclasses import dataclass
from random import random


from .extract_head import headed_rule, Tree
from .buparser import backtrace, qelement
from .sdcp import grammar
from queue import PriorityQueue
from sortedcontainers import SortedList
from collections import defaultdict
from ..tagging.parsing_scorer import DummyScorer
from .derivation import Derivation
from .lcfrs import disco_span, lcfrs_composition


@dataclass
class PassiveItem:
    lhs: str
    leaves: disco_span

    def freeze(self):
        return (self.lhs, self.leaves)

    def __gt__(self, other: "PassiveItem") -> bool:
        if isinstance(other, ActiveItem): return False
        return (self.lhs, self.leaves) > (other.lhs, other.leaves)


@dataclass
class ActiveItem:
    lhs: str
    leaf: int
    leaves: disco_span
    remaining_function: lcfrs_composition
    remaining: tuple[str]

    def freeze(self):
        return (self.lhs, self.leaf, self.leaves, self.remaining, self.remaining_function)

    def __gt__(self, other: "ActiveItem") -> bool:
        if isinstance(other, PassiveItem): return True
        return (self.lhs, self.leaves, len(self.remaining)) > (other.lhs, other.leaves, len(other.remaining))


class ActiveParser:
    def __init__(self, grammar: grammar):
        self.grammar = grammar
        self.scoring = DummyScorer()


    def set_scoring(self, scorer: DummyScorer):
        self.scoring = scorer
        if scorer.snd_order:
            raise ValueError("this parser does not support second order scores at the moment")


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.rootid = None
        self.queue = PriorityQueue()
        for i, rules in enumerate(rules_per_position):
            minweight = min(w for _, w in (rules or [(0,0)]))
            for rid, weight in rules:
                weight = weight - minweight
                rule: headed_rule = self.grammar.rules[rid]
                self.queue.put_nowait(qelement(
                        ActiveItem(rule.lhs, i, disco_span(), *rule.composition.reorder_rhs(rule.rhs)),
                        backtrace(rid, i, ()),
                        weight
                ))
        self.golditems = None
        self.stop_early = True

    
    def set_gold_item_filter(self, gold_tree: Derivation, nongold_stopping_prob: float = 0.9, early_stopping: float | bool = True):
        self.golditems = set()
        self.brassitems = list()
        for node in gold_tree.subderivs():
            lhs = node.rule if self.sow else self.grammar.rules[node.rule].lhs
            self.golditems.add((lhs, node.yd.freeze()))
        self.nongold_stop_prob = nongold_stopping_prob
        self.stop_early = early_stopping


    def fill_chart(self):
        expanded = set()
        self.from_lhs: dict[str, list[tuple[disco_span, int, float]]] = defaultdict(SortedList) 
        self.actives: dict[str, list[tuple[ActiveItem, backtrace, float]]] = {}
        self.backtraces = []  
        self.items = []

        self.new_item_batch = []
        self.new_item_batch_minweight = None
        def register_passive_item(qele: qelement):
            self.new_item_batch.append(qele)
            if self.new_item_batch_minweight is None or self.new_item_batch_minweight > qele.weight:
                self.new_item_batch_minweight = qele.weight
        def flush_items():
            if self.new_item_batch and (self.queue.empty() or self.new_item_batch_minweight < self.queue.queue[0].weight):
                weights = self.scoring.score([(i.item, i.bt) for i in self.new_item_batch], self.backtraces)
                for weight, qe in zip(weights, self.new_item_batch):
                    qe.weight += weight
                    self.queue.put_nowait(qe)
                self.new_item_batch.clear()
                self.new_item_batch_minweight = None
        
        while not self.queue.empty() or self.new_item_batch:
            flush_items()
            qi: qelement = self.queue.get_nowait()
            print(qi.item)
            fritem = qi.item.freeze()
            if fritem in expanded:
                continue
            expanded.add(fritem)

            if isinstance(qi.item, PassiveItem):
                backtrace_id = len(self.backtraces)
                qi.bt.children = self.grammar.rules[qi.bt.rid].composition.undo_reorder(qi.bt.children)
                self.backtraces.append(qi.bt)
                self.items.append(qi.item)

                # check if a gold item filter was added and if the item is gold or brass
                # if it is brass, then probabilistically ignore it
                if not self.golditems is None and not qi.item in self.golditems:
                    self.brassitems.append(backtrace_id)
                    if qi.bt.children and random() < self.nongold_stop_prob:
                        continue

                # check if this is the root item and stop the parsing, if early stopping is activated
                if qi.item.lhs == self.grammar.root and qi.item.leaves == disco_span((0, self.len)):
                    if self.rootid is None:
                        self.rootid = backtrace_id
                    if self.stop_early is True:
                        return
                # if a root item was already found and the current weight is more than a threshold, then exit
                if not self.rootid is None and qi.weight > self.stop_early:
                    return

                qi.bt = backtrace_id
                self.from_lhs[qi.item.lhs].add((qi.item.leaves, qi.bt, qi.weight))
                
                for active, abt, _weight in self.actives.get(qi.item.lhs, []):
                    # if active.lexidx > 0 and not qi.item.leaves.leftmost < active.leaf or \
                    #         not active.leaves.leftmost < qi.item.leaves.leftmost or \
                    #         abt.leaf in qi.item.leaves or \
                    #         not active.leaves.isdisjoint(qi.item.leaves):
                    #     continue
                    newpos, newcomp = active.remaining_function.partial(active.leaves, qi.item.leaves)
                    if newpos is None: continue
                    self.queue.put_nowait(qelement(
                        ActiveItem(active.lhs, active.leaf, newpos, newcomp, active.remaining[1:]),
                        backtrace(abt.rid, abt.leaf, abt.children+(backtrace_id,)),
                        qi.weight+_weight,
                    ))
                continue

            # todo: skip this
            assert isinstance(qi.item, ActiveItem)
            if qi.item.remaining and qi.item.remaining[0] is None:
                leaves, remaining = qi.item.remaining_function.partial(qi.item.leaves, disco_span.singleton(qi.item.leaf))
                if leaves is None: continue
                self.queue.put_nowait(qelement(
                    ActiveItem(qi.item.lhs, qi.item.leaf, leaves, remaining, qi.item.remaining[1:]),
                    qi.bt,
                    qi.weight,
                ))
                continue

            if not qi.item.remaining:
                if len(qi.item.leaves) != qi.item.remaining_function.fanout: continue
                register_passive_item(qelement(
                    PassiveItem(qi.item.lhs, qi.item.leaves),
                    qi.bt,
                    qi.weight,
                ))
                continue

            self.actives.setdefault(qi.item.remaining[0], []).append((qi.item, qi.bt, qi.weight))
            for (span, pbt, pweight) in self.from_lhs.get(qi.item.remaining[0], []):
                # if qi.item.lexidx > 0 and not span.leftmost < qi.item.leaf or \
                #         not qi.item.leaves.leftmost < span.leftmost or \
                #         qi.bt.leaf in span or \
                #         not qi.item.leaves.isdisjoint(span):
                #     continue
                newpos, newfunc = qi.item.remaining_function.partial(qi.item.leaves, span)
                if newpos is None: continue
                self.queue.put_nowait(qelement(
                    ActiveItem(qi.item.lhs, qi.item.leaf, newpos, newfunc, qi.item.remaining[1:]),
                    backtrace(qi.bt.rid, qi.bt.leaf, qi.bt.children+(pbt,)),
                    qi.weight+pweight
                ))


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return [Tree("NOPARSE", list(range(self.len)))]
        bt: backtrace = self.backtraces[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        return fn(*(self.get_best(c, p) for c, p in zip(bt.children, push)))

    # todo: merge with function below
    def get_best_deriv(self, item = None):
        if item is None:
            item = self.rootid
        bt: backtrace = self.backtraces[item]
        return Tree((bt.leaf, self.grammar.rules[bt.rid]), (self.get_best_deriv(c) for c in bt.children))

    def get_best_derivation(self, item=None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return None, None
        bt: backtrace = self.backtraces[item]
        w: float = self.weight[item]
        return Tree((bt.rid, bt.leaf), [self.get_best_derivation(i) for i in bt.children])