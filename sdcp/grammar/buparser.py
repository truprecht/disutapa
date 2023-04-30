from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .sdcp import grammar
from queue import PriorityQueue
from bitarray import frozenbitarray, bitarray
from bitarray.util import count_and
from sortedcontainers import SortedList
from collections import defaultdict


a01 = frozenbitarray('01')
a1 = frozenbitarray('1')
a0 = frozenbitarray('0')
@dataclass
class BitSpan:
    leaves: bitarray
    leftmost: int = field(init=False, repr=False, hash=False, compare=False)
    gaps: int = field(init=False, repr=False, hash=False, compare=False)
    gapslen: int = field(init=False, repr=False, hash=False, compare=False)
    firstgap: int = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        self.__dict__["leftmost"] = self._leftmost()
        self.__dict__["gaps"] = self.numgaps()
        self.__dict__["firstgap"] = self._firstgap()

    def isdisjoint(self, other: "BitSpan") -> bool:
        return count_and(self.leaves, other.leaves) == 0

    def union(self, other: "BitSpan", and_leaf: int = None) -> "BitSpan":
        v = self.leaves | other.leaves
        if not and_leaf is None:
            v[and_leaf] = 1
        return BitSpan(v)

    def _leftmost(self) -> int:
        return self.leaves.find(a1)

    def _firstgap(self) -> int:
        if (m := self.leaves[self.leftmost:].find(a0)) != -1:
            return self.leftmost+m
        return len(self.leaves)

    @classmethod
    def fromit(cls, ps: Iterable[int], len: int):
        bv = [False] * len
        for p in ps:
            bv[p] = True
        return cls(bitarray(bv))

    def numgaps(self):
        return sum(1 for _ in self.leaves[self.leftmost:].itersearch(a01))

    def __bool__(self):
        return self.leaves.any()

    def __iter__(self):
        return (i for i,b in enumerate(self.leaves) if b)

    def __str__(self):
        return str(self.leaves)

    def __repr__(self) -> str:
        return str(self.leaves)

    def numleaves(self):
        return self.leaves.count(1)

    def __lt__(self, other: "BitSpan"):
        #return (self.leftmost, self.leaves[self.leftmost:]) > (other.leftmost, other.leaves[other.leftmost:])
        return self.leaves > other.leaves

    def __contains__(self, i: int) -> bool:
        return self.leaves[i] == 1

    def with_leaf(self, i: int) -> "BitSpan":
        v = self.leaves.copy()
        v[i] = 1
        return BitSpan(v)

    def freeze(self) -> bytes:
        return self.leaves.tobytes()

    def __len__(self) -> int:
        return self.leaves.count(1)
    
    def fences(self) -> Iterable[tuple[int, int]]:
        left, right = None, None
        for i in self:
            if left is None:
                left, right = i, i+1
            elif right == i:
                right += 1
            else:
                yield (left, right)
                left, right = i, i+1
        if not left is None:
            yield (left, right)

@dataclass
class PassiveItem:
    lhs: str
    leaves: BitSpan
    maxfo: int

    def freeze(self):
        return (self.lhs, self.leaves.freeze(), self.maxfo)

    def __gt__(self, other: "PassiveItem") -> bool:
        return (self.lhs, self.leaves, self.maxfo) > (other.lhs, other.leaves, other.maxfo)

@dataclass
class backtrace:
    rid: int
    leaf: int
    children: tuple[int]

    def as_tuple(self):
        return self.rid, self.children


@dataclass(eq=False, order=False)
class qelement:
    item: PassiveItem
    bt: backtrace
    weight: float

    def __gt__(self, other):
        return (self.weight, self.item) > (other.weight,  other.item)

    def tup(self):
        return self.item, self.bt, self.weight