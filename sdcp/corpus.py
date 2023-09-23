from typing import Iterable, Tuple
from discodop.treebank import READERS, CorpusReader  # type: ignore
# from discodop.treetransforms import binarize, collapseunary  # type: ignore
from discodop.tree import Tree  # type: ignore
from collections import defaultdict
from dataclasses import dataclass

from .grammar.extract import extract, singleton, rule
from .grammar.extract_head import Extractor
from .autotree import AutoTree
from .ranktransform import Binarizer


PTB_SPECIAL_TOKENS = {
    "-RRB-": ")",
    "-LRB-": "(",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    # "``": '"',
    # "''": '"',
}

class corpus_extractor:
    def __init__(self, filename_or_iterator: str | Iterable[Tuple[Tree, Iterable[str]]], headrules: str | None = None, guide="strict", cmode="lcfrs", ntmode="plain", **binparams):
        self.norm_token = lambda x: x
        if isinstance(filename_or_iterator, str):
            filetype = filename_or_iterator.split(".")[-1]
            if filetype == "xml":
                filetype = "tiger"
            encoding = "iso-8859-1" if filetype == "export" else "utf8"
            self.trees = READERS[filetype](filename_or_iterator, encoding=encoding, punct="move", headrules=headrules)
            if "ptb" in filename_or_iterator:
                self.norm_token = lambda x: PTB_SPECIAL_TOKENS.get(x, x)
        else:
            self.trees = filename_or_iterator
        self.rules: dict[rule, int] = defaultdict(lambda: len(self.rules))
        self.nonterminals: dict[str, int] = defaultdict(lambda: len(self.nonterminals))
        # ensure root is mapped to integer 0
        self.nonterminals["ROOT"]
        self.sentences: list[tuple[str, ...]] = []
        self.goldtrees: list[Tree] = []
        self.goldrules: list[tuple[rule, ...]] = []
        self.goldpos: list[tuple[str, ...]] = []
        self.goldderivs: list[Tree] = []
        self.idx: dict[int, int] = {}
        self.binarizer = Binarizer(
            vmarkov = binparams.get("vertmarkov", 1),
            hmarkov = binparams.get("horzmarkov", 2),
            head_outward = (guide == "dependent")
        )
        # todo: merge headed and strict extraction and remove this
        self._binparams = binparams
        self.guide = guide
        self.cmode = cmode
        self.ntmode = ntmode


    def store_rule(self, r: rule):
        irhs = tuple(
            -1 if nt is None or nt == -1 else self.nonterminals[nt]
            for nt in r.rhs
        )
        irule = rule(self.nonterminals[r.lhs], irhs, r.scomp, r.dcp)
        return self.rules[irule]


    def read(self, lrange: range | None = None):
        if isinstance(self.trees, CorpusReader):
            start, stop = None, None
            if not lrange is None:
                start, stop = lrange.start, lrange.stop
            trees_sents = ((Tree.convert(item.tree), item.sent) for _, item in self.trees.itertrees(start, stop))
            treeit: Iterable[tuple[int, tuple[Tree, list[str]]]] \
                = zip(lrange, trees_sents) if not lrange is None else enumerate(trees_sents)
        else:
            treeit = enumerate(self.trees)
        for i, (tree, sent) in treeit:
            if not isinstance(self.trees, CorpusReader) and not (lrange is None or i in lrange): continue
            self.idx[i] = len(self.goldtrees)
            if self.guide == "head":
                ht: AutoTree = AutoTree.convert(tree)
                rules, deriv = Extractor(**self._binparams, composition=self.cmode, mark=self.ntmode)(ht)
                pos = tuple(p for _, p in sorted(ht.postags.items()))
            else:
                if len(sent) == 1:
                    stree = self.binarizer(tree)
                    rules, pos = singleton(stree)
                    deriv = 0
                else:
                    bintree: AutoTree = AutoTree.convert(self.binarizer(tree))
                    rules, deriv = extract(bintree, ctype=self.cmode, ntype=self.ntmode, gtype=self.guide)
                    for node in deriv.subtrees():
                        node.children = [(c if len(c) > 0 else c.label) for c in node]
                    pos = tuple(p for _, p in sorted(bintree.postags.items()))
            rules = tuple(
                self.store_rule(gr) for gr in rules
            )
            self.goldtrees.append(tree)
            self.sentences.append(tuple(self.norm_token(tok) for tok in sent))
            self.goldpos.append(pos)
            self.goldderivs.append(deriv)
            self.goldrules.append(rules)
    
    def __getitem__(self, idx):
        idx = self.idx[idx]
        return (
            self.goldtrees[idx],
            self.sentences[idx],
            self.goldpos[idx],
            self.goldrules[idx],
            self.goldderivs[idx],
        )


@dataclass(init=False)
class Split:
    train: range
    dev: range
    test: range

    def __init__(self, train = range(1), dev = range(1), test = range(1)):
        self.train = train
        self.dev = dev
        self.test = test

    def nonoverlapping(self) -> Iterable[range]:
        bymin = iter(sorted((r for r in (self.train, self.dev, self.test)), key=lambda x: (x.start, x.stop)))
        crange = next(bymin)
        for r in bymin:
            if r.start > crange.stop:
                yield crange
                crange = r
            else:
                crange = range(crange.start, r.stop)
        yield crange

    def __contains__(self, idx: int):
        return any(idx in r for r in (self.train, self.dev, self.test))

    def items(self):
        return self.__dict__.items()