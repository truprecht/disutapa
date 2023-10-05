from ..composition import fanout
from re import escape, compile, Match

def read_clusters(filename: str):
    label_to_clusterid = {}
    with open(filename, "r") as cfile:
        for line in cfile:
            array = line.strip().split()
            clusterid = array[0]
            for label in array[1:]:
                assert not label in label_to_clusterid, f"label {label} appears multiple times in {filename}"
                label_to_clusterid[label] = clusterid
    return label_to_clusterid

class MultiKeyReplacement:
    def __init__(self, map: dict[str, str]):
        # sort keys in decreasing length to prefer substituting whole constituent symbols
        keys = sorted(map.keys(), key=lambda s: len(s), reverse=True)
        self.regex = compile("|".join(escape(k) for k in keys))
        self.map = map

    def _replace_single_match(self, m: Match) -> str:
        return self.map[m.group()]

    def __call__(self, string: str) -> str:
        return self.regex.sub(self._replace_single_match, string)
    

def firstCharReplacement(string: str):
    if not "|<" in string:
        return string[0]
    head, tail = string[:-1].split("|<")
    markovsuffix = ",".join((nt[0] if nt else "") for nt in tail.replace("$,", "$").split(",")) \
        if tail else ""
    return f"{head[0]}|<{markovsuffix}>"


class NtConstructor:
    def __init__(self, type: str, coarsetab: dict[str, str] | None = None):
        if not "-" in type:
            type += "-"
        self.type, self.decoration = type.split("-")
        self.coarsetab = MultiKeyReplacement(coarsetab) \
            if not coarsetab is None else firstCharReplacement
        
    def leaf(self, parent: str):
        match self.type:
            case "vanilla":
                return f"arg({parent})"
            case _:
                return f"arg"
    
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
            case "coarse":
                # binarization nodes do not contain "+"-merged unary constituents
                baselabel = self.coarsetab(ctree.label.split("+")[0])
        
        match self.decoration:
            case "nof":
                return baselabel
            case "disc":
                decor = "D" if fanout(deriv_yield) > 1 else ""
            case "":
                decor = str(fanout(deriv_yield))
        
        return f"{baselabel}/{decor}" if decor else baselabel
