from dataclasses import dataclass
from ..lcfrs import disco_span, lcfrs_composition

@dataclass
class backtrace:
    rid: int
    leaf: int
    children: tuple[int]

    def as_tuple(self):
        return self.rid, self.children


@dataclass
class PassiveItem:
    lhs: str
    leaves: disco_span

    def freeze(self):
        return (self.lhs, self.leaves)

    def __gt__(self, other: "PassiveItem") -> bool:
        if isinstance(other, ActiveItem): return False
        return other.leaves > self.leaves


@dataclass(eq=False, order=False)
class qelement:
    item: int
    bt: backtrace
    weight: float

    # priority queue retrieves lowest weighted elements first
    # with nll values, lower weights are better
    def __gt__(self, other):
        return self.weight > other.weight
        # return (self.weight, self.item) > (other.weight,  other.item)

    def tup(self):
        return self.item, self.bt, self.weight


@dataclass
class ActiveItem:
    lhs: str
    leaves: disco_span
    remaining_function: lcfrs_composition
    remaining: tuple[str]

    def freeze(self):
        return (self.lhs, self.leaves, self.remaining, self.remaining_function)

    def __gt__(self, other: "ActiveItem") -> bool:
        if isinstance(other, PassiveItem): return True
        return (other.leaves, len(self.remaining)) > (self.leaves, len(other.remaining))


def item(lhs: str, leaves: disco_span, remaining_function: lcfrs_composition, remaining_rhs: tuple[str | int]):
    if remaining_rhs and isinstance(remaining_rhs[0], int):
        leaves, remaining_function = remaining_function.partial(leaves, disco_span.singleton(remaining_rhs[0]))
        remaining_rhs = remaining_rhs[1:]
    if not remaining_rhs:
        return PassiveItem(lhs, leaves)
    return ActiveItem(lhs, leaves, remaining_function, remaining_rhs)