from typing import Iterable, Tuple
from discodop.treebank import READERS
from discodop.treetransforms import binarize, collapseunary
from discodop.tree import Tree
from collections import defaultdict

from .grammar.extract import extract, singleton
from .autotree import AutoTree


class corpus_extractor:
    def __init__(self, filename_or_iterator: str | Iterable[Tuple[Tree, Iterable[str]]], **binparams):
        if isinstance(filename_or_iterator, str):
            filetype = filename_or_iterator.split(".")[-1]
            if filetype == "xml":
                filetype = "tiger"
            encoding = "iso-8859-1" if filetype == "export" else "utf8"
            reader = READERS[filetype](filename_or_iterator, encoding=encoding, punct="move")
            self.trees = ((Tree.convert(item.tree), item.sent) for _, item in reader.itertrees())
        else:
            self.trees = filename_or_iterator
        self.rules = defaultdict(lambda: len(self.rules))
        self.sentences = []
        self.goldtrees = []
        self.goldrules = []
        self.goldpos = []
        self._binparams = binparams

    def read(self):
        for tree, sent in self.trees:
            if len(sent) == 1:
                stree = collapseunary(Tree.convert(tree), collapsepos=True, collapseroot=True)
                rules, pos = singleton(stree)
                rules = tuple(self.rules[gr] for gr in rules)
            else:
                bintree = binarize(
                    collapseunary(Tree.convert(tree), collapsepos=True, collapseroot=True),
                    **self._binparams)
                bintree = AutoTree.convert(bintree)
                rules = tuple(self.rules[gr] for gr in extract(bintree))
                pos = tuple(p for _, p in sorted(bintree.postags.items()))
            self.goldtrees.append(tree)
            self.sentences.append(tuple(sent))
            self.goldpos.append(pos)
            self.goldrules.append(rules)