from collections import defaultdict
from heapq import heapify, heappush, heappop
from random import random
from typing import Iterable


from ..extract_head import Tree
from ..sdcp import grammar, rule
from ...tagging.parsing_scorer import DummyScorer
from ..derivation import Derivation
from ..lcfrs import disco_span
from .item import qelement, PassiveItem, ActiveItem, backtrace, item
from .kbestchart import KbestChart


class ActiveParser:
    def __init__(self, grammar: grammar):
        self.grammar = grammar
        self.scoring = DummyScorer()


    def set_scoring(self, scorer: DummyScorer):
        self.scoring = scorer
        if scorer.snd_order:
            raise ValueError("this parser does not support second order scores at the moment")


    def init(self, sentlen):
        self.len = sentlen
        self.rootid: int | None = None
        self.queue: list[qelement] = []
        self.actives: dict[str, list[tuple[ActiveItem, backtrace, float]]] = {}
        self.filtering = False
        self.stop_early: float | bool = True

        self.expanded: dict[ActiveItem|PassiveItem, int] = dict()
        self.from_lhs: dict[str, list[tuple[disco_span, int, float]]] = defaultdict(list)
        self.backtraces: list[list[backtrace]] = []  
        self.items: list[PassiveItem] = []
        
        self.ruleweights = dict()
        self.itemweights = list()


    def add_rules(self, *rules_per_position: Iterable[tuple[int, float]]):
        assert not self.queue
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
                self.ruleweights[(rid, i)] = weight
        heapify(self.queue)

    
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


    def fill_chart(self, stop_early: bool = False) -> None:
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
            if qi.item in self.expanded:
                if isinstance(qi.item, PassiveItem):
                    self.backtraces[self.expanded[qi.item]].append(qi.bt)
                continue
            self.expanded[qi.item] = len(self.backtraces)

            if isinstance(qi.item, PassiveItem):
                backtrace_id = len(self.backtraces)
                self.backtraces.append([qi.bt])
                self.items.append(qi.item)
                self.itemweights.append(qi.weight)

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
                    if stop_early:
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

    def get_best_iter(self):
        if self.rootid is None:
            yield [Tree("NOPARSE", list(range(self.len)))], 0.0
            return
      
        self.kbestchart = KbestChart(
            self.backtraces,
            self.ruleweights,
            self.grammar.rules,
            self.itemweights,
            self.rootid
        )

        for t in iter(self.kbestchart):
            yield t
    
    def get_best(self):
        return next(self.get_best_iter())[0]

    # todo: merge with function below
    def get_best_deriv(self, item = None):
        if item is None:
            item = self.rootid
        bt: backtrace = self.backtraces[item][0]
        return Tree((bt.leaf, self.grammar.rules[bt.rid]), (self.get_best_deriv(c) for c in bt.children))

    def get_best_derivation(self, item=None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return None, None
        bt: backtrace = self.backtraces[item][0]
        return Tree((bt.rid, bt.leaf), [self.get_best_derivation(i) for i in bt.children])