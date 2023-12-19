from typing import Iterable, Tuple
from discodop.treebank import READERS, CorpusReader  # type: ignore
# from discodop.treetransforms import binarize, collapseunary  # type: ignore
from discodop.tree import Tree  # type: ignore
from collections import defaultdict
from dataclasses import dataclass

from .extract import extract, singleton, rule
from .extract_head import Extractor
from ...autotree import AutoTree
from .ranktransform import Binarizer


def splitstr(s: str) -> dict:
    return eval(s)


@dataclass
class ExtractionParameter:
    hmarkov: int = 999
    vmarkov: int = 1
    factor: str = "right"
    coarsents: str = None
    composition: str = "lcfrs"
    nts: str = "classic"
    guide: str = "strict"
    penn_treebank_replacements: bool = None

    def __post_init__(self):
        assert self.hmarkov >= 0 and self.vmarkov > 0
        assert self.factor in ("right", "left", "headoutward")
        assert self.composition in ("lcfrs", "dcp")
        assert self.nts in ("vanilla", "classic", "coarse")
        if not self.coarsents is None:
            assert self.nts == "coarse"
        assert self.guide in ("strict", "vanilla", "dependent", "head", "least", "near")


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
    def __init__(self, config: ExtractionParameter):
        self.rules: dict[str, None] = dict()
        self.postags: dict[str, None] = dict()
        self.binarizer = Binarizer(
            vmarkov = config.vmarkov,
            hmarkov = config.hmarkov,
            factor = config.factor if not config.guide == "dependent" else "headoutward"
        )
        # todo: merge headed and strict extraction and remove this
        self.params = config


    def read_tree(self, tree):
        if self.params.guide == "head":
            ht: AutoTree = AutoTree.convert(tree)
            rules, deriv = Extractor(
                composition=self.params.composition,
                ntype=self.params.nts,
                hmarkov=self.params.hmarkov,
                vmarkov=self.params.vmarkov,
            )(ht)
            pos = tuple(p for _, p in sorted(ht.postags.items()))
        else:
            bintree = self.binarizer(tree)
            if len(bintree) == 1 and not isinstance(bintree[0], Tree):
                rules, pos = singleton(bintree)
                deriv = 0
            else:
                bintree: AutoTree = AutoTree.convert(bintree)
                rules, deriv = extract(bintree, ctype=self.params.composition, ntype=self.params.nts, gtype=self.params.guide, nt_tab=self.params.coarsents)
                for node in deriv.subtrees():
                    node.children = [(c if len(c) > 0 else c.label) for c in node]
                pos = tuple(p for _, p in sorted(bintree.postags.items()))
        rules = tuple(repr(gr) for gr in rules)
        for r in rules:
            self.rules.setdefault(r, None)
        for p in pos:
            self.postags.setdefault(p, None)
        return rules, pos
    
    @staticmethod
    def ptb_sentence(sentence: Iterable[str]):
        return tuple(PTB_SPECIAL_TOKENS.get(tok, tok) for tok in sentence)


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