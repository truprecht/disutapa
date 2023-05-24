from dataclasses import dataclass
from typing import Iterable, Union, cast
from itertools import chain
from sortedcontainers import SortedSet  # type: ignore

def fanout(leaves: SortedSet[int]) -> int:
    return 1+sum(1 for x,y in zip(leaves[:-1], leaves[1:]) if x+1 != y)


class spanzip:
    def __init__(self, xs: tuple[tuple[int, int], ...], ys: tuple[tuple[int, int], ...]):
        self.xi, self.yi = 0, 0
        self.xs, self.ys = xs, ys

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, int]:
        if self.yi >= len(self.ys) and self.xi >= len(self.xs):
            raise StopIteration()
        if not self.yi < len(self.ys) or self.xi < len(self.xs) and (self.xs[self.xi]) < self.ys[self.yi]:
            x = self.xs[self.xi]
            self.xi += 1
            return x
        else:
            y = self.ys[self.yi]
            self.yi += 1
            return y
    
    def finished(self):
        return self.xi == len(self.xs) and self.yi == len(self.ys)


@dataclass(init=False, frozen=True)
class disco_span:
    spans: tuple[tuple[int, int], ...]

    def __init__(self, *spans: tuple[int, int]):
        self.__dict__["spans"] = spans

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.spans[idx]

    def __iter__(self):
        return iter(self.spans)
    
    def __bool__(self):
        return bool(self.spans)
    
    def __len__(self):
        return len(self.spans)
    
    @classmethod
    def singleton(cls, idx: int) -> "disco_span":
        return cls((idx, idx+1))
    
    def exclusive_union(self, other: "disco_span") -> Union["disco_span", None]:
        spans: list[tuple[int, int]] = []
        spanit = spanzip(self.spans, other.spans)
        current_l, current_r = next(spanit)
        for l, r in spanit:
            if current_r == l:
                current_r = r
            elif current_r > l:
                return None
            else:
                spans.append((current_l, current_r))
                current_r, current_l = r, l
        spans.append((current_l, current_r))
        return self.__class__(*spans)
    
    def spanlen(self) -> int:
        return sum(r-l for l,r in self)
    
    def __contains__(self, position: int) -> bool:
        return any(l <= position < r for l,r in self)
    
    def __lt__(self, other: Union[int, tuple[int, int], "disco_span"]) -> bool:
        if isinstance(other, int):
            return self.spans[0][0] < other and not other in self
        if isinstance(other, tuple):
            return self.spans[0][0] < other[0]
        return self.spans < other.spans


@dataclass(frozen=True)
class NtOrLeaf:
    payload: str | int
    is_leaf: bool

    def get_leaf(self) -> int:
        if not self.is_leaf:
            raise ValueError()
        return cast(int, self.payload)
    
    def get_nt(self) -> str:
        if self.is_leaf:
            raise ValueError()
        return cast(str, self.payload)
    
    @classmethod
    def nt(cls, n: str) -> "NtOrLeaf":
        return cls(n, False)
    
    @classmethod
    def leaf(cls, i: int) -> "NtOrLeaf":
        return cls(i, True)


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

    def reorder_rhs(self, rhs: tuple[str, ...], leaf: int) -> tuple["ordered_union_composition", tuple[NtOrLeaf, ...]]:
        original_order: tuple[NtOrLeaf, ...] = (NtOrLeaf(leaf, is_leaf=True), *(NtOrLeaf(nt, is_leaf=False) for nt in rhs))
        canon_composition = self.__class__([], self.order_and_fanout[-1])
        reordered_rhs = tuple(original_order[i] for i in self.order_and_fanout[:-1])
        return canon_composition, reordered_rhs
    
    def undo_reorder(self, successors: tuple) -> Iterable[int]:
        varpos = (v for v in self.order_and_fanout[:-1] if not v == 0)
        indices = (i for i, _ in sorted(enumerate(varpos), key=lambda x: x[1]))
        return (successors[i] for i in indices)

    @classmethod
    def from_positions(cls, positions, successor_positions: list[SortedSet]):
        lex = next(p for p in positions if all((not p in spos) for spos in successor_positions))
        succs = {p: i+1 for i, p in enumerate(spos[0] for spos in successor_positions)}
        succs[lex] = 0
        order = (succs[p] for p in sorted(succs.keys()))
        return cls(order, fanout(positions))
        
    def partial(self, x: disco_span, y: disco_span) -> tuple[disco_span | None, Union["ordered_union_composition", None]]:
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
    
    @classmethod
    def default(cls, arity: int):
        return cls(i for i in (1,0,*range(2,arity+1)) if i <= arity)

    @property
    def fanout(self):
        return sum(1 for c in self.inner if c == 255)+1
    
    def reorder_rhs(self, rhs: tuple[str|int, ...], leaf: int) -> tuple["lcfrs_composition", tuple[NtOrLeaf, ...]]:
        original_order: tuple[NtOrLeaf, ...] = (NtOrLeaf(leaf, is_leaf=True), *(NtOrLeaf(nt, is_leaf=False) for nt in rhs))
        occs = sorted(range(len(original_order)), key=lambda x: next(i for i,v in enumerate(self.inner) if v ==x))
        revoccs = {oldpos: newpos for newpos, oldpos in enumerate(occs)}
        revoccs[255] = 255
        comp = self.__class__(revoccs[v] for v in self.inner)
        return comp, tuple(original_order[i] for i in occs)
    
    @classmethod
    def from_positions(cls, positions: Iterable[int], successor_positions: list[SortedSet]):
        last_position: int | None = None
        vars: list[int] = []
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
        
    
    def partial(self, part: disco_span, x: disco_span) -> tuple[disco_span, "lcfrs_composition"] | tuple[None, None]:
        if not part:
            return x, self
        xs = [iter(part), iter(x)]
        fs = [len(part), len(x)]
        total: list[tuple[int, int]] = []
        current_l, current_r = None, None
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
        occs = sorted(range(len(successors)), key=lambda x: next(i for i,v in enumerate(self.inner) if v==x+1))
        revoccs = {oldpos: newpos for newpos, oldpos in enumerate(occs)}
        return tuple(successors[revoccs[i]] for i in range(len(successors)))