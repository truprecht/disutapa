from collections import defaultdict
from heapq import heapify, heappush, heappop
from random import random
from typing import Iterable, cast


from ..extract_head import Tree
from ..sdcp import grammar, rule
from ...tagging.parsing_scorer import DummyScorer
from ..derivation import Derivation
from ..lcfrs import disco_span
from .item import qelement, PassiveItem, ActiveItem, backtrace, item


class ActiveParser:
    def __init__(self, grammar: grammar):
        self.grammar = grammar
        self.scoring = DummyScorer()


    def set_scoring(self, scorer: DummyScorer):
        self.scoring = scorer
        if scorer.snd_order:
            raise ValueError("this parser does not support second order scores at the moment")


    def init(self, *rules_per_position: Iterable[tuple[int, float]]):
        self.len = len(rules_per_position)
        self.rootid: int | None = None
        self.queue: list[qelement] = []
        self.actives: dict[str, list[tuple[ActiveItem, backtrace, float]]] = {}
        for i, rules in enumerate(rules_per_position):
            minweight = min(w for _, w in (rules or [(0,0.0)]))
            for rid, weight in rules:
                weight = weight - minweight
                r: rule = self.grammar.rules[rid]
                it = item(r.lhs, disco_span(), r.scomp, r.rhs, i)
                self.queue.append(qelement(
                    it,
                    backtrace(rid, i, ()),
                    weight
                ))
        heapify(self.queue)
        self.filtering = False
        self.stop_early: float | bool = True

    
    def set_gold_item_filter(self, gold_tree: Derivation, nongold_stopping_prob: float = 0.9, early_stopping: float | bool = True):
        self.filtering = True
        self.golditems: set[PassiveItem] = set()
        self.brassitems: list[PassiveItem] = list()
        for node in gold_tree.subderivs():
            # lhs = node.rule if self.sow else self.grammar.rules[node.rule].lhs
            lhs = self.grammar.rules[node.rule].lhs
            self.golditems.add(PassiveItem(lhs, node.yd))
        self.nongold_stop_prob = nongold_stopping_prob
        self.stop_early = early_stopping


    def fill_chart(self) -> None:
        expanded: set[ActiveItem|PassiveItem] = set()
        self.from_lhs: dict[str, list[tuple[disco_span, int, float]]] = defaultdict(list)
        self.backtraces: list[backtrace] = []  
        self.items: list[PassiveItem] = []

        self.new_item_batch: list[PassiveItem] = []
        self.new_item_batch_minweight: float | None = None
        def register_item(qele: qelement):
            if qele.item.leaves is None: return
            if isinstance(qele, PassiveItem):
                self.new_item_batch.append(qele)
                if self.new_item_batch_minweight is None or self.new_item_batch_minweight > qele.weight:
                    self.new_item_batch_minweight = qele.weight
            else:
                heappush(self.queue, qele)
        def flush_items():
            if self.new_item_batch and (not self.queue or self.new_item_batch_minweight < self.queue.queue[0].weight):
                weights = self.scoring.score([(i.item, i.bt) for i in self.new_item_batch], self.backtraces)
                for weight, qe in zip(weights, self.new_item_batch):
                    qe.weight += weight
                    heappush(self.queue, qe)
                self.new_item_batch.clear()
                self.new_item_batch_minweight = None
        
        while self.queue or self.new_item_batch:
            flush_items()
            qi: qelement = heappop(self.queue)
            if qi.item in expanded:
                continue
            expanded.add(qi.item)

            if isinstance(qi.item, PassiveItem):
                backtrace_id = len(self.backtraces)
                self.backtraces.append(qi.bt)
                self.items.append(qi.item)

                # check if a gold item filter was added and if the item is gold or brass
                # if it is brass, then probabilistically ignore it
                if self.filtering and not qi.item in self.golditems:
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
                self.from_lhs[qi.item.lhs].append((qi.item.leaves, qi.bt, qi.weight))
                
                for active, abt, _weight in self.actives.get(qi.item.lhs, []):
                    if not active.is_compatible(qi.item.leaves): continue
                    newpos, newcomp = active.remaining_function.partial(qi.item.leaves, active.leaves)
                    if newpos is None: continue
                    register_item(qelement(
                        item(active.lhs, newpos, newcomp, active.remaining[:-1], active.leaf),
                        backtrace(abt.rid, abt.leaf, (backtrace_id, *abt.children)),
                        qi.weight+_weight,
                    ))
                continue

            assert isinstance(qi.item, ActiveItem)
            self.actives.setdefault(qi.item.remaining[-1].get_nt(), []).append((qi.item, qi.bt, qi.weight))
            for (span, pbt, pweight) in self.from_lhs.get(qi.item.remaining[-1].get_nt(), []):
                if not qi.item.is_compatible(span): continue
                newpos, newfunc = qi.item.remaining_function.partial(span, qi.item.leaves)
                if newpos is None: continue
                register_item(qelement(
                    item(qi.item.lhs, newpos, newfunc, qi.item.remaining[:-1], qi.item.leaf),
                    backtrace(qi.bt.rid, qi.bt.leaf, (pbt, *qi.bt.children)),
                    qi.weight+pweight
                ))


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return [Tree("NOPARSE", list(range(self.len)))]
        bt: backtrace = self.backtraces[item]
        fn, push = self.grammar.rules[bt.rid].dcp(bt.leaf, pushed)
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