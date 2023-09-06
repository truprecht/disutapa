# cython: profile=True
# cython: linetrace=True
import cython

@cython.cclass
class Discospan:
    def __init__(self, borders: tuple[cython.int, ...]):
        self.borders = borders
        self.length = len(borders)

    def __getitem__(self, idx: cython.int) -> tuple[int, int]:
        start: cython.int = 2*idx
        end: cython.int = start + 2
        return self.borders[start:end]

    def __iter__(self):
        idx: cython.int
        for idx in range(0, self.length, 2):
            yield self.borders[idx:idx+2]
    
    def __bool__(self):
        return self.length > 0
    
    def __len__(self):
        return self.length // 2
    
    def exclusive_union(self, other: Discospan) -> Discospan:
        current_l: cython.int
        current_r: cython.int
        l: cython.int
        r: cython.int
        borders: list[cython.int] = []
        spanit: spanzip = spanzip(self.borders, other.borders)
        current_l, current_r = next(spanit)
        for l, r in spanit:
            if current_r == l:
                current_r = r
            elif current_r > l:
                return None
            else:
                borders.append(current_l)
                borders.append(current_r)
                current_r, current_l = r, l
        borders.append(current_l)
        borders.append(current_r)
        return Discospan(tuple(borders))
    
    def __contains__(self, position: cython.int) -> cython.bint:

        return any(l <= position < r for l,r in self)

    def __str__(self) -> str:
        return f"<{', '.join(str((i,j)) for (i,j) in self)}>"

    def __repr__(self) -> str:
        return f"Discospan({', '.join(str((i,j)) for (i,j) in self)})"

    def __gt__(self, other: Discospan) -> bool:
        return self.borders > other.borders

    def gt_leaf(self, other: cython.int) -> bool:
        return self.length > 0 and self.borders[0] > other

    def __eq__(self, other: Discospan) -> bool:
        return self.borders == other.borders

    def __hash__(self) -> int:
        return hash(self.borders)

    @classmethod
    def singleton(cls, idx: int) -> Discospan:
        return cls((idx, idx+1))

    @classmethod
    def from_tuples(cls, *tups: tuple[int, int]) -> Discospan:
        return cls(tuple(b for bs in tups for b in bs))

    @classmethod
    def empty(cls):
        return cls(())


@cython.cclass
class spanzip:
    xi: cython.int
    yi: cython.int
    xs: tuple
    ys: tuple
    lenx: cython.int
    leny: cython.int

    def __init__(self, xs: tuple[cython.int, ...], ys: tuple[cython.int, ...]):
        self.xi, self.yi = 0, 0
        self.xs, self.ys = xs, ys
        self.lenx, self.leny = len(xs), len(ys)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[cython.int, cython.int]:
        x: tuple[cython.int, cython.int]
        y: tuple[cython.int, cython.int]
        if self.yi >= self.leny and self.xi >= self.lenx:
            raise StopIteration()
        if not self.yi < self.leny or self.xi < self.lenx and self.xs[self.xi] < self.ys[self.yi]:
            x = self.xs[self.xi:self.xi+2]
            self.xi += 2
            return x
        else:
            y = self.ys[self.yi:self.yi+2]
            self.yi += 2
            return y
    
    @cython.cfunc
    @cython.returns(cython.bint)
    @cython.exceptval(check=False)
    def finished(self):
        return self.xi == self.lenx and self.yi == self.leny