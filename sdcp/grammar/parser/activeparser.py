from collections import defaultdict
from heapq import heapify, heappush, heappop
from typing import Iterable


from ..extract_head import Tree
from ..sdcp import grammar, rule
from ..lcfrs import disco_span
from .item import qelement, PassiveItem, ActiveItem, backtrace, item
from .kbestchart import KbestChart


class ActiveParser:
    def __init__(self, grammar: grammar):
        self.grammar = grammar

    def init(self, sentlen):
        self.len = sentlen
        self.rootid: int | None = None
        self.queue: list[qelement] = []
        self.actives: dict[str, list[tuple[ActiveItem, backtrace, float]]] = {}

        self.expanded: dict[ActiveItem|PassiveItem, int] = dict()
        self.from_lhs: dict[str, list[tuple[disco_span, int, float]]] = defaultdict(list)
        self.backtraces: list[list[backtrace]] = []  
        self.items: list[PassiveItem] = []
        
        self.ruleweights = dict()
        self.itemweights = list()


    def add_rules(self, *rules_per_position: Iterable[tuple[int, float]]):
        assert not self.queue
        for i, rules in enumerate(rules_per_position):
            minweight = min(w for _, w in rules) if rules else 0.0
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


    def add_rules_i(self, i: int, rules: tuple[int], weights: tuple[float]):
        for rid, weight in zip(rules, weights):
            r: rule = self.grammar.rules[rid]
            it = item(r.lhs, disco_span(), r.scomp, r.rhs, i)
            self.queue.append(qelement(
                it,
                backtrace(rid, i, ()),
                weight
            ))
            self.ruleweights[(rid, i)] = weight



    def fill_chart(self, stop_early: bool = False) -> None:   
        heapify(self.queue)     
        while self.queue:
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
                    newitem = item(active.lhs, newpos, newcomp, active.remaining[:-1], active.leaf)
                    if newitem.leaves is None: continue
                    heappush(self.queue, qelement(
                        newitem,
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
                newitem = item(qi.item.lhs, newpos, newfunc, qi.item.remaining[:-1], qi.item.leaf)
                if newitem.leaves is None: continue
                heappush(self.queue, qelement(
                    newitem,
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

    def get_best_derivation(self, item=None):
        if item is None:
            item = self.rootid
            if self.rootid is None:
                return None, None
        bt: backtrace = self.backtraces[item][0]
        return Tree((bt.rid, bt.leaf), [self.get_best_derivation(i) for i in bt.children])