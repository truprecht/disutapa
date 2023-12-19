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
    

R_CSYMBOL = compile(r"$[,.()]|\w+")
def firstCharReplacement(string: str):
    return R_CSYMBOL.sub(lambda m: m.group()[0], string)


class NtConstructor:
    def __init__(self, type: str, coarsetab: dict[str, str] | None = None):
        if not "-" in type:
            type += "-"
        self.type, self.decoration = type.split("-")
        self.coarsetab = MultiKeyReplacement(coarsetab) \
            if not coarsetab is None else firstCharReplacement
        
    def leaf(self, parent: str):
        if self.type == "vanilla":
            return f"arg({parent})"
        parent = self.__class__.rmchain(parent)
        if self.type == "coarse":
            parent = self.coarsetab(parent)
        return f"arg({parent})"

    @classmethod
    def rmchain(cls, cnst: str):
        # remove bottom merged unary nodes
        # markovization annotations do not contain "+"-merged unary constituents
        if "+" in cnst:
            cnst, tail = cnst.split("+", 1)
            if "^<" in tail:
                cnst += "^<" + tail.split("^<", 1)[1]
        return cnst
    
    def __call__(self, ctree, deriv_yield):
        if self.type == "vanilla":
            oldfanout = fanout(ctree.leaves())
            if ctree.leaves() != deriv_yield:
                newfanout = fanout(deriv_yield)
                return f"{ctree.label}/{oldfanout}/{newfanout-oldfanout}"
            return f"{ctree.label}/{oldfanout}"
        
        baselabel = self.__class__.rmchain(ctree.label)
        if self.type == "coarse":
            baselabel = self.coarsetab(baselabel)
        
        match self.decoration:
            case "nof":
                return baselabel
            case "disc":
                decor = "D" if fanout(deriv_yield) > 1 else ""
            case "":
                decor = str(fanout(deriv_yield))
        
        return f"{baselabel}/{decor}" if decor else baselabel
