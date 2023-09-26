from ..composition import fanout
from re import escape, compile, Match

class MultiKeyReplacement:
    def __init__(self, map: dict[str, str]):
        self.regex = compile(
            "|".join(escape(k) for k in map)
        )
        self.map = map

    def _replace_single_match(self, m: Match) -> str:
        return self.map[m.group()]

    def __call__(self, string: str) -> str:
        return self.regex.sub(self._replace_single_match, string)
    

def firstCharReplacement(string: str):
    if not "|<" in string:
        return string[0]
    head, tail = string[:-1].split("|<")
    markovsuffix = ",".join(nt[0] for nt in tail.replace("$,", "$").split(","))
    return f"{head[0]}|<{markovsuffix}>"


class NtConstructor:
    def __init__(self, type: str, coarsetab: dict[str, str] | None = None):
        self.type = type
        self.coarsetab = MultiKeyReplacement(coarsetab) \
            if not coarsetab is None else firstCharReplacement
    
    def __call__(self, ctree, deriv_yield):
        match self.type:
            case "vanilla":
                oldfanout = fanout(ctree.leaves())
                if ctree.leaves() != deriv_yield:
                    newfanout = fanout(deriv_yield)
                    return f"{ctree.label}/{oldfanout}/{newfanout-oldfanout}"
                return f"{ctree.label}/{oldfanout}"
            case "classic":
                # binarization nodes do not contain "+"-merged unary constituents
                baselabel = ctree.label.split("+")[0]
                return f"{baselabel}/{fanout(deriv_yield)}"
            case "coarse":
                # binarization nodes do not contain "+"-merged unary constituents
                baselabel = ctree.label.split("+")[0]
                return f"{self.coarsetab(baselabel)}/{fanout(deriv_yield)}"