# cython: profile=True
# cython: linetrace=True
from dataclasses import dataclass
from typing import Iterable
from sortedcontainers import SortedSet  # type: ignore
import cython

def fanout(leaves: SortedSet[int]) -> int:
    return 1+sum(1 for x,y in zip(leaves[:-1], leaves[1:]) if x+1 != y)

@dataclass(frozen=True)
cdef class Composition:
    @classmethod
    def union(cls, fanout: cython.int = 1):
        return cls(fanout, 0, bytes())

    @classmethod
    def lcfrs(cls, variables: Iterable[int] | str):
        if isinstance(variables, str):
            variables = [
                int(v) if "0" <= v <= "9" else 255
                for v in variables
            ]
        arity: cython.int = max(v for v in variables if not v == 255)+1
        fanout: cython.int = sum(1 for c in variables if c == 255)+1
        return cls(fanout, arity, bytes(variables))

    def __repr__(self) -> str:
        if not self.variables:
            return "Composition.union(" + str(self.fanout) + ")"
        if self.arity < 10:
            compstr = "".join((str(v) if v < 255 else ",") for v in self.variables)
            compstr = "'" + compstr + "'"
        else:
            compstr = ",".join(str(v) for v in self.variables)
            compstr = "[" + compstr + "]"
        return "Composition.lcfrs(" + compstr + ")"

    cpdef CompositionView view(self, cython.int arg = -1) noexcept:
        return CompositionView(self, arg)


@dataclass(init=False, frozen=True)
cdef class CompositionView(Composition):
    def __init__(self, c: Composition, na: cython.int = -1):
        self.fanout = c.fanout
        self.arity = c.arity
        self.variables = c.variables
        self.next_arg = na if na != -1 else c.arity-1

    cdef CompositionView next(self) noexcept:
        if not self.variables:
            return self
        return CompositionView(self, self.next_arg-1)

    cpdef Discospan partial(self, arg: Discospan, acc: Discospan) noexcept:
        total: list[cython.int] = []
        vars_len: cython.int = len(self.variables)
        arg_idx: cython.int = 0
        acc_idx: cython.int = 0
        arg_len: cython.int = len(arg)
        acc_len: cython.int = len(acc)
        current: tuple[cython.int, cython.int] = (-1, -1)
        lr: tuple[cython.int, cython.int] = (-1, -1)
        last_var: cython.int = -1
        vidx: cython.int
        cvar: cython.int

        if vars_len == 0:
            return arg.exclusive_union(acc)

        if self.next_arg < 0:
            return None
        
        for vidx in range(vars_len):
            cvar = self.variables[vidx]
            if cvar != 255 and last_var != 255 and last_var > self.next_arg and cvar > self.next_arg:
                continue
            last_var = cvar

            if not cvar >= self.next_arg or cvar == 255:
                if not current[0] == -1:
                    total.append(current[0])
                    total.append(current[1])
                    current = (-1, -1)
                continue            

            if cvar > self.next_arg and acc_len == acc_idx or \
                    cvar == self.next_arg and arg_len == arg_idx:
                return None
            if cvar > self.next_arg:
                lr = acc[acc_idx]
                acc_idx += 1
            else:
                lr = arg[arg_idx]
                arg_idx += 1
            
            if current[0] == -1:
                current = lr
                continue

            if current[1] != lr[0]:
                return None
            current[1] = lr[1]

        if not current[0] == -1:
            total.append(current[0])
            total.append(current[1])

        if arg_idx < arg_len or acc_idx < acc_len:
            return None
        return Discospan(tuple(total))

    cdef Discospan finalize(self, Discospan acc) noexcept:
        return acc if len(acc) == self.fanout else None


def union_from_positions(
        positions: Iterable[int],
        successor_positions: Iterable[SortedSet]
        ) -> tuple[Composition, Iterable[int]]:
    lex = next(p for p in positions if all((not p in spos) for spos in successor_positions))
    succs = {p: i+1 for i, p in enumerate(spos[0] for spos in successor_positions)}
    succs[lex] = 0
    order = [succs[p] for p in sorted(succs.keys())]
    return Composition.union(fanout(positions)), order


def default_lcfrs(arity: cython.int) -> Composition:
    return Composition(1, arity, bytes(range(arity)))


def lcfrs_from_positions(
        positions: Iterable[int],
        successor_positions: Iterable[SortedSet],
        ) -> tuple[Composition, Iterable[int]]:
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
    return Composition.lcfrs(vars), revorder.keys()

def lcfrs_composition(vars: str|Iterable[int]):
    return Composition.lcfrs(vars)

def ordered_union_composition(fanout: cython.int = 1):
    return Composition.union(fanout)