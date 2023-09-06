from dataclasses import dataclass
from typing import Iterable, Union, cast
from itertools import chain
from sortedcontainers import SortedSet  # type: ignore

from .parser.span import Discospan

def fanout(leaves: SortedSet[int]) -> int:
    return 1+sum(1 for x,y in zip(leaves[:-1], leaves[1:]) if x+1 != y)


@dataclass(frozen=True)
class ordered_union_composition:
    # implement ordered union of leaves without explicit composition function
    fanout: int = 1

    @classmethod
    def from_positions(
            cls,
            positions,
            successor_positions: list[SortedSet]
            ) -> tuple["ordered_union_composition", Iterable[int]]:
        lex = next(p for p in positions if all((not p in spos) for spos in successor_positions))
        succs = {p: i+1 for i, p in enumerate(spos[0] for spos in successor_positions)}
        succs[lex] = 0
        order = [succs[p] for p in sorted(succs.keys())]
        return cls(fanout(positions)), order
        
    def partial(self, x: Discospan, y: Discospan) -> tuple[Discospan | None, Union["ordered_union_composition", None]]:
        if not y:
            return x, self
        if not (spans := x.exclusive_union(y)) is None:
            return spans, self
        return None, None

    def finalize(self, span: Discospan):
        return span if len(span) == self.fanout else None


@dataclass(init=False, frozen=True)
class lcfrs_composition:
    inner: bytes
    arity: int

    def __init__(self, vars: Iterable[int | str] | bytes):
        if not isinstance(vars, bytes):
            vars = (
                int(v) if isinstance(v, int) or "0" <= v <= "9" else 255
                for v in vars)
            vars = bytes(vars)
        # check ordered variables
        seen = SortedSet()
        for v in vars:
            if v != 255 and not v in seen:
                assert not seen or v > seen[-1], f"lcfrs composition {vars!r} is not ordered"
                seen.add(v)
        self.__dict__["inner"] = vars
        self.__dict__["arity"] = seen[-1]
    
    @classmethod
    def default(cls, arity: int):
        return cls(range(arity))

    @property
    def fanout(self):
        return sum(1 for c in self.inner if c == 255)+1
    
    @classmethod
    def from_positions(cls,
            positions: Iterable[int],
            successor_positions: list[SortedSet],
            ) -> tuple["lcfrs_composition", Iterable[int]]:
        last_position: int | None = None
        vars: list[int] = []
        revorder: dict[int, int] = {}
        for p in positions:
            if not last_position is None and last_position+1 < p:
                vars.append(255)
            var = 0
            for i, sp in enumerate(successor_positions):
                if p in sp:
                    var = i+1
            var = revorder.setdefault(var, len(revorder))
            if not vars or vars[-1] != var:
                vars.append(var)
            last_position = p
        return cls(vars), revorder.keys()
    

    def partial(self, x: Discospan, accumulator: Discospan) -> tuple[Discospan, "lcfrs_composition"] | tuple[None, None]:
        if not accumulator:
            return x, self
        xs = [iter(x), iter(accumulator)]
        fs = [len(x), len(accumulator)]
        total: list[tuple[int, int]] = []
        current_l, current_r = None, None
        remainder = []
        for var in chain(self.inner, (255,)):
            if not var in (self.arity, self.arity-1):
                if not current_l is None:
                    total.append((current_l, current_r))
                    current_r = current_l = None
                remainder.append(var)
                continue
            
            if not remainder or remainder[-1] != self.arity-1:
                remainder.append(self.arity-1)

            var -= self.arity-1
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
        return Discospan(*total), lcfrs_composition(remainder[:-1])


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
        return "lcfrs_composition(" + str(self) + ")"
    
    def finalize(self, span: Discospan) -> Discospan:
        return span