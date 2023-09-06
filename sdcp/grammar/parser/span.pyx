# cython: profile=True
# cython: linetrace=True
import cython

@cython.cclass
class Discospan:
    def __init__(self, *spans: tuple[int, int]):
        self.borders = tuple(b for bs in spans for b in bs)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        idx = 2*idx
        return self.borders[idx:idx+2]

    def __iter__(self):
        idx: cython.int = 0
        while idx < len(self.borders):
            yield self.borders[idx:idx+2]
            idx += 2
    
    def __bool__(self):
        return bool(self.borders)
    
    def __len__(self):
        return len(self.borders)//2
    
    def exclusive_union(self, other: "Discospan") -> "Discospan":
        spans: list[tuple[int, int]] = []
        spanit = spanzip(self.borders, other.borders)
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
    
    def __contains__(self, position: int) -> bool:
        return any(l <= position < r for l,r in self)

    def __str__(self) -> str:
        return f"<{', '.join(str((i,j)) for (i,j) in self)}>"

    def __repr__(self) -> str:
        return f"Discospan({', '.join(str((i,j)) for (i,j) in self)})"

    def __gt__(self, other: "Discospan") -> bool:
        return self.borders > other.borders

    def gt_leaf(self, other: cython.int) -> bool:
        return self.borders and self.borders[0] > other

    def __eq__(self, other: "Discospan") -> bool:
        return self.borders == other.borders

    def __hash__(self) -> int:
        return hash(self.borders)


def singleton(idx: int) -> Discospan:
    return Discospan((idx, idx+1))

class spanzip:
    def __init__(self, xs: tuple[int, ...], ys: tuple[int, ...]):
        self.xi, self.yi = 0, 0
        self.xs, self.ys = xs, ys

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, int]:
        if self.yi >= len(self.ys) and self.xi >= len(self.xs):
            raise StopIteration()
        if not self.yi < len(self.ys) or self.xi < len(self.xs) and (self.xs[self.xi]) < self.ys[self.yi]:
            x = self.xs[self.xi:self.xi+2]
            self.xi += 2
            return x
        else:
            y = self.ys[self.yi:self.yi+2]
            self.yi += 2
            return y
    
    def finished(self):
        return self.xi == len(self.xs) and self.yi == len(self.ys)