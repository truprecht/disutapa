from typing import Iterable, Tuple
from discodop.treebank import READERS, CorpusReader
from discodop.treetransforms import binarize, collapseunary
from discodop.tree import Tree
from collections import defaultdict
from dataclasses import dataclass

from .grammar.extract import extract, singleton
from .grammar.extract_head import Extractor
from .autotree import AutoTree
from .headed_tree import HeadedTree


class corpus_extractor:
    def __init__(self, filename_or_iterator: str | Iterable[Tuple[Tree, Iterable[str]]], headrules: str = None, guide="head", cmode="lcfrs", **binparams):
        if isinstance(filename_or_iterator, str):
            filetype = filename_or_iterator.split(".")[-1]
            if filetype == "xml":
                filetype = "tiger"
            encoding = "iso-8859-1" if filetype == "export" else "utf8"
            self.trees = READERS[filetype](filename_or_iterator, encoding=encoding, punct="move", headrules=headrules)
        else:
            self.trees = filename_or_iterator
        self.rules = defaultdict(lambda: len(self.rules))
        self.sentences = []
        self.goldtrees = []
        self.goldrules = []
        self.goldpos = []
        self.goldderivs = []
        self.idx = {}
        self._binparams = binparams
        # todo: merge headed and strict extraction and remove this
        if not headrules:
            self._binparams = {"vertmarkov": binparams.get("vertmarkov", 1), "horzmarkov": binparams.get("horzmarkov", 2)}
        self.guide = "head" if not headrules is None else "inorder"
        self.cmode = cmode

    def read(self, lrange: range = None):
        if isinstance(self.trees, CorpusReader):
            start, stop = None, None
            if not lrange is None:
                start, stop = lrange.start, lrange.stop
            treeit = ((Tree.convert(item.tree), item.sent) for _, item in self.trees.itertrees(start, stop))
            treeit = zip(lrange, treeit) if not lrange is None else enumerate(treeit)
        else:
            treeit = enumerate(self.trees)
        for i, (tree, sent) in treeit:
            if not isinstance(self.trees, CorpusReader) and not (lrange is None or i in lrange): continue
            self.idx[i] = len(self.goldtrees)
            if self.guide == "head":
                ht = HeadedTree.convert(tree)
                rules, deriv = Extractor(**self._binparams, composition=self.cmode)(ht)
                rules = tuple(self.rules[gr] for gr in rules)
                pos = tuple(p for _, p in sorted(ht.postags.items()))
            else:
                if len(sent) == 1:
                    stree = collapseunary(Tree.convert(tree), collapsepos=True, collapseroot=True)
                    rules, pos = singleton(stree)
                    rules = tuple(self.rules[gr] for gr in rules)
                    deriv = 0
                else:
                    bintree = binarize(
                        collapseunary(Tree.convert(tree), collapsepos=True, collapseroot=True),
                        **self._binparams)
                    bintree = AutoTree.convert(bintree)
                    rules, deriv = extract(bintree, ctype=self.cmode)
                    rules = tuple(self.rules[gr] for gr in rules)
                    for node in deriv.subtrees():
                        # node.label = rules[node.label]
                        node.children = [(c if len(c) > 0 else c.label) for c in node]
                    pos = tuple(p for _, p in sorted(bintree.postags.items()))
            self.goldtrees.append(tree)
            self.sentences.append(tuple(sent))
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