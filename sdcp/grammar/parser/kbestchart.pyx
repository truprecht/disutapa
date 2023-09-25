from dataclasses import dataclass
from discodop.tree import Tree
from heapq import heapify, heappush, heappop
from typing import Iterable

from .item import backtrace
from ..sdcp import rule


@dataclass(frozen=True)
class HeapElement:
    bt: backtrace
    successor_ks: tuple[int, ...]
    weight: float

    def __lt__(self, other):
        return self.weight < other.weight or self.bt.leaf < other.bt.leaf or self.bt.children < other.bt.children or max(self.successor_ks) < max(other.successor_ks)
    
    def __eq__(self, other):
        return self.bt == other.bt and self.successor_ks == other.successor_ks


def successor_indices(idxs: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    for i in range(len(idxs)):
        yield idxs[:i] + (idxs[i]+1,) + idxs[i+1:]


class KbestChart:
    def __init__(self, chart: list[list[backtrace]], rweights: dict[tuple[int, int], float], rules: list[rule], iweights: list[float], rootid: int):
        self.rootid = rootid
        self.chart = chart
        self.rweights = rweights
        self.iweights = iweights
        self.rules = rules
        self.heaps: dict[int, list[HeapElement]] = {}
        self.inheap: dict[int, set[tuple[backtrace, tuple[int, ...]]]] = {}
        self.kchart: dict[int, list[HeapElement]] = {}


    def _exists(self, item: int, k: int) -> bool:
        if not item in self.heaps:
            heap = [
                HeapElement(b, (0,)*len(b.children), self.rweights[(b.rid, b.leaf)] + sum(self.iweights[s] for s in b.children))
                for b in self.chart[item]
            ]
            inheap = set((he.bt, he.successor_ks) for he in heap)
            self.heaps[item] = heap
            self.inheap[item] = inheap
            heapify(heap)
        else:
            heap = self.heaps[item]
            inheap = self.inheap[item]
        kchart = self.kchart.setdefault(item, [])

        while len(kchart) <= k:
            if not heap:
                return False
            element = heappop(heap)
            kchart.append(element)
            for si in successor_indices(element.successor_ks):
                if (element.bt, si) in inheap or \
                        (weight := self._compositional_weight(element.bt, si)) is None:
                    continue
                heappush(heap, HeapElement(element.bt, si, weight))
                inheap.add((element.bt, si))
        return True


    def kthweight(self, item, k) -> float | None:
        if not self._exists(item, k):
            return None
        return self.kchart[item][k].weight
        

    def _compositional_weight(self, bt: backtrace, successor_ks: tuple[int, ...]) -> float:
        weight = self.rweights[(bt.rid, bt.leaf)]
        for s, sk in zip(bt.children, successor_ks):
            if (sweight := self.kthweight(s, sk)) is None:
                return None
            weight += sweight
        return weight


    def kthbest(self, item: int, k: int, pushed: int = None) -> tuple[list[Tree], float]:
        assert item in self.kchart and len(self.kchart[item]) > k
        he = self.kchart[item][k]
        bt: backtrace = he.bt
        fn, push = self.rules[bt.rid].dcp(bt.leaf, pushed)
        successors = (
            self.kthbest(c, cidx, pushed=p)[0]
            for c, cidx, p in zip(bt.children, he.successor_ks, push)
        )
        return fn(*successors), he.weight


    def __iter__(self):
        i = 0
        while self._exists(self.rootid, i):
            yield self.kthbest(self.rootid, i)
            i += 1