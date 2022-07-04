from typing import Iterable
from discodop.treebank import READERS
from discodop.treetransforms import binarize, collapseunary
from discodop.tree import Tree, DiscTree
from collections import defaultdict

from .grammar.extract import extract

class corpus_extractor:
    def __init__(self, filename_or_iterator: str | Iterable[DiscTree]):
        if isinstance(filename_or_iterator, str):
            filetype = filename_or_iterator.split(".")[-1]
            if filetype == "xml":
                filetype = "tiger"
            encoding = "iso-8859-1" if filetype == "export" else "utf8"
            reader = READERS[filetype](filename_or_iterator, encoding=encoding, punct="move")
            self.trees = (item for _, item in reader.itertrees())
        else:
            self.trees = filename_or_iterator
        self.rules = defaultdict(lambda: len(self.rules))
        self.sentences = []
        self.goldtrees = []
        self.goldrules = []

    def read(self):
        for disctree in self.trees:
            bintree = binarize(
                collapseunary(Tree.convert(disctree), collapsepos=True, collapseroot=True),
                horzmarkov=0, vertmarkov=1)
            self.goldrules.append(
                [self.rules[gr] for gr in extract(bintree)]
            )
            self.goldtrees.append(disctree)
            self.sentences.append(disctree.sent)