from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Any, Callable
from discodop.tree import Tree, ImmutableTree
from math import log
from tqdm import tqdm
from ..grammar.lcfrs import lcfrs_composition, SortedSet
from multiprocessing import Pool

LcfrsRule = tuple[str, tuple[str, ...], lcfrs_composition]

@dataclass
class Derivation:
    rule: int
    position: int
    children: tuple["Derivation"]


treeposition = 0
def into_derivation(
        tree: Tree,
        integerize: Callable[[Any], int] = None
        ) -> tuple[Derivation, SortedSet]:
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        return None, SortedSet((tree[0],))
    global treeposition
    nodeposition = (treeposition := treeposition + 1)
    children, positions = zip(*(into_derivation(c, integerize) for c in tree))
    allpositions = positions[0].union(*positions[1:])
    composition, _ = lcfrs_composition.from_positions(allpositions, positions)
    rule = (tree.label, tuple(c.label for c in tree), composition)
    if not integerize is None:
        rule = integerize(rule)
    return Derivation(
        rule,
        nodeposition,
        children
    ), allpositions


def largest_common_fragment(s: Derivation, t: Derivation, excludes: set[tuple[int, int]]) -> Tree:
    if s is None or t is None or s.rule != t.rule:
        return None
    excludes.add((s.position, t.position))
    return ImmutableTree(
        s.rule,
        [
            largest_common_fragment(s_, t_, excludes)
            for s_, t_ in zip(s.children, t.children)
        ]
    )


 


class Dop:
    def get_tree_fragments(self, derivation):
        dop_fragments = Counter()
        exclude_combinations = set()
        queue = [derivation]
        while queue:
            already_counted = set()
            subderivation = queue.pop(0)
            queue.extend(c for c in subderivation.children if not c is None)
            for other_occurrence in self.rule_index[subderivation.rule]:
                if other_occurrence.position >= subderivation.position:
                    break
                if (subderivation.position, other_occurrence.position) in exclude_combinations:
                    continue
                assert other_occurrence.rule == subderivation.rule
                fragment = largest_common_fragment(subderivation, other_occurrence, exclude_combinations)
                if fragment in already_counted:
                    continue
                already_counted.add(fragment)
                dop_fragments[fragment] += 1
        return dop_fragments


    def __init__(self, training_set: Iterable[Tree], prior: float = 0.1):
        self.rule2idx = {}
        self.idx2lhs = []
        derivations: list[Tree] = []
        self.rule_index: dict[int, Derivation] = defaultdict(list)
        for tree in training_set:
            derivation, _ = into_derivation(tree, integerize=self.integerize)
            derivations.append(derivation)
            queue = [derivation]
            while queue:
                subderivation = queue.pop(0)
                self.rule_index[subderivation.rule].append(subderivation)
                queue.extend(c for c in subderivation.children if not c is None)

        dop_fragments = Counter()
        pool = Pool()
        for counts in pool.imap_unordered(self.get_tree_fragments, tqdm(derivations, "extract fragments"), chunksize=128):
            dop_fragments += counts            

        self.rules = defaultdict(list)
        denominators = Counter()
        for fragment, c in dop_fragments.items():
            denominators[self.idx2lhs[fragment.label]] += c + 1 + prior
            self.rules[fragment.label].append(fragment)
        self.weights = {
            fragment: log(denominators[self.idx2lhs[fragment.label]] + prior) - log(c + 1 + prior)
            for fragment, c in dop_fragments.items()
        }
        self.fallback_weight = {
            label: log(denom + prior) - log(prior) if prior else float("-inf")
            for label, denom in denominators.items()
        }


    def integerize(self, rule) -> int:
        if not rule in self.rule2idx:
            self.idx2lhs.append(rule[0])
        return self.rule2idx.setdefault(rule, len(self.idx2lhs)-1)


    def match(self, tree: Tree) -> float:
        parser = DopTreeParser(self)
        parser.fill_chart(tree)
        return parser.chart[1]
    
    def select(self, trees: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        bestidx, besttree, bestweight = None, None, None
        for i, (tree, _) in enumerate(trees):
            weight = self.match(tree)
            if bestweight is None or weight > bestweight:
                bestidx, besttree, bestweight = i, tree, weight
        return bestidx, besttree


def children_with_positions(tree, position, skip_none: bool = True):
    return (
        (c, position+(i,))
        for i,c in enumerate(tree)
        if not (skip_none and c is None)
    )

@dataclass
class DopTreeParser:
    grammar: Dop
    chart: dict[tuple[int, ...], float] = field(default_factory=dict)
    
    @classmethod
    def _match(cls, fragment, derivation) -> Iterable[Derivation] | None:
        if fragment.label != derivation.rule:
            return None
        child_derivations = []
        for subfragment, subderivation in zip(fragment, derivation.children):
            if subfragment is None:
                if not subderivation is None:
                    child_derivations.append(subderivation)
                continue
            if not (children := cls._match(subfragment, subderivation)) is None:
                child_derivations.extend(children)
                continue
            return None
        return child_derivations

    def fill_chart(self, tree: Tree):
        global treeposition
        treeposition = 0
        derivation, _ = into_derivation(tree, self.grammar.integerize)
        queue = [derivation]
        while queue:
            subderivation = queue.pop()
            if not all(c is None or c.position in self.chart for c in subderivation.children):
                queue.append(subderivation)
                queue.extend(c for c in subderivation.children if not c is None)
                continue
            if not subderivation.rule in self.grammar.rules:
                weight = sum(self.chart[c.position] for c in subderivation.children if not c is None)
                weight += self.grammar.fallback_weight.get(self.grammar.idx2lhs[subderivation.rule], 0)
                if not subderivation.position in self.chart or self.chart[subderivation.position] > weight:
                    self.chart[subderivation.position] = weight
                continue
            for doprule in self.grammar.rules[subderivation.rule]:
                children = self._match(doprule, subderivation)
                if children is None:
                    continue
                weight = sum(self.chart[c.position] for c in children)
                weight += self.grammar.weights[doprule]
                if not subderivation.position in self.chart or self.chart[subderivation.position] > weight:
                    self.chart[subderivation.position] = weight