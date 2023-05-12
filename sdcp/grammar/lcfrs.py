from array import array
from dataclasses import dataclass
from typing import Iterable
from itertools import chain

def fanout(leaves: set[int]) -> int:
    # ol = sorted(leaves)
    return 1+sum(1 for x,y in zip(leaves[:-1], leaves[1:]) if x+1 != y)

@dataclass(init=False, frozen=True)
class disco_span:
    spans: array
    len: int

    def __hash__(self) -> int:
        return hash((*self.spans, self.len))

    def __init__(self, *spans: tuple[int, int]):
        self.__dict__["spans"] = array('H', (i for lr in spans for i in lr))
        self.__dict__["len"] = len(spans)

    def __getitem__(self, idx) -> tuple[int, int]:
        if idx >= self.len:
            raise IndexError()
        l, r = self.spans[2*idx:2*(idx+1)]
        return (l, r)
    
    def __bool__(self):
        return bool(self.len)
    
    def __len__(self):
        return self.len
    
    @classmethod
    def singleton(cls, idx: int) -> "disco_span":
        return cls((idx, idx+1))
    
    def exclusive_union(self, other: "disco_span"):
        xi, yi = 0, 0
        spans, current_l, current_r = [], None, None
        while xi < self.len or yi < other.len:
            if not yi < other.len or xi<self.len and self[xi] < other[yi]:
                l,r = self[xi]
                xi += 1
            else:
                l,r =  other[yi]
                yi += 1

            if current_l is None:
                current_l, current_r = l, r
            elif current_r == l:
                current_r = r
            elif current_r > l:
                return None
            else:
                spans.append((current_l, current_r))
                current_r, current_l = r, l
        spans.append((current_l, current_r))
        return self.__class__(*spans)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(str(t) for t in self)})"
    
    def __lt__(self, other) -> bool:
        for (s1, s2) in zip(self, other):
            if s1 < s2: return True
            if s1 > s2: return False
        return self.len <= other.len
    
    def spanlen(self) -> int:
        return sum(r-l for l,r in self)
    
    def __contains__(self, position: int) -> bool:
        return any(l <= position < r for l,r in self)


@dataclass(init=False, frozen=True)
class ordered_union_composition:
    # implement ordered union of leaves without explicit composition function
    order_and_fanout: bytes

    def __init__(self, order: Iterable[int] | str, fanout: int = 1):
        if isinstance(order, str):
            order = map(int, order)
        self.__dict__["order_and_fanout"] = bytes(chain(order, (fanout,)))

    @property
    def fanout(self):
        return self.order_and_fanout[-1]

    def reorder_rhs(self, rhs: tuple[str], leaf: int):
        rhs = (leaf,) + rhs
        canon_composition = self.__class__([], self.order_and_fanout[-1])
        reordered_rhs = tuple(rhs[i] for i in self.order_and_fanout[:-1])
        return canon_composition, reordered_rhs
    
    def undo_reorder(self, successors: tuple) -> tuple:
        varpos = (v for v in self.order_and_fanout[:-1] if not v == 0)
        indices = (i for i, _ in sorted(enumerate(varpos), key=lambda x: x[1]))
        return (successors[i] for i in indices)

    @classmethod
    def from_positions(cls, positions, successor_positions: list[set]):
        lex = next(p for p in positions if all((not p in spos) for spos in successor_positions))
        succs = {p: i+1 for i, p in enumerate(spos[0] for spos in successor_positions)}
        succs[lex] = 0
        order = (succs[p] for p in sorted(succs.keys()))
        return cls(order, fanout(positions))
        
    def partial(self, x: disco_span, y: disco_span) -> tuple[disco_span, "ordered_union_composition"]:
        if not x:
            return y, self
        if len(x) >= self.fanout and x[self.fanout-1][1] < y[0][0]:
            return None, None
        if x < y and not (spans := x.exclusive_union(y)) is None:
            return spans, self
        return None, None
    
    def __str__(self) -> str:
        suffix = "" if self.order_and_fanout[-1] == 1 else f", fanout={self.order_and_fanout[-1]}"
        if len(self.order_and_fanout) <= 10:
            orderstr = "'" + "".join(map(str, self.order_and_fanout[:-1])) + "'"
        else:
            orderstr = str(tuple(self.order_and_fanout[:-1]))
        return orderstr + suffix
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"


@dataclass(init=False, frozen=True)
class lcfrs_composition:
    inner: bytes

    def __init__(self, vars: Iterable[int | str] | bytes):
        if not isinstance(vars, bytes):
            vars = (
                int(v) if isinstance(v, int) or "0" <= v <= "9" else 255
                for v in vars)
            vars = bytes(vars)
        self.__dict__["inner"] = vars

    @property
    def fanout(self):
        return sum(1 for c in self.inner if c == 255)+1
    
    def reorder_rhs(self, rhs, leaf):
        return self, (leaf,)+ rhs
    
    @classmethod
    def from_positions(cls, positions: Iterable[int], successor_positions: list[set]):
        last_position = None
        vars = []
        for p in positions:
            if not last_position is None and last_position+1 < p:
                vars.append(255)
            var = 0
            for i, sp in enumerate(successor_positions):
                if p in sp:
                    var = i+1
            if not vars or vars[-1] != var:
                vars.append(var)
            last_position = p
        return cls(vars)
        
    
    def partial(self, part: disco_span, x: disco_span) -> tuple[disco_span, "lcfrs_composition"]:
        if not part:
            return x, self
        xs = [iter(part), iter(x)]
        fs = [len(part), len(x)]
        total, current_l, current_r = [], None, None
        remainder = []
        for var in chain(self.inner, (255,)):
            if var > 1:
                if not current_l is None:
                    total.append((current_l, current_r))
                    current_r = current_l = None
                remainder.append(var-1 if var < 255 else 255)
                continue
            
            if not remainder or remainder[-1] != 0:
                remainder.append(0)

            if fs[var] == 0: return None, None
            fs[var] -= 1
            if current_l is None:
                current_l, current_r = next(xs[var])
                if total and total[-1][1] >= current_l:
                    return None, None
            else:
                l,r = next(xs[var])
                if current_r == l:
                    current_r = r
                else:
                    return None, None
        if any(f!=0 for f in fs): return None, None
        return disco_span(*total), lcfrs_composition(remainder[:-1])


    def __str__(self) -> str:
        maxvar = max(v for v in self.inner if v < 255)
        if maxvar < 10:
            compstr = "".join((str(v) if v < 255 else ",") for v in self.inner)
            compstr = "'" + compstr + "'"
        else:
            compstr = ",".join(str(v) for v in self.inner)
            compstr = "[" + compstr + "]"
        return compstr


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"
    
    def undo_reorder(self, successors):
        return successors