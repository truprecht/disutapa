from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Any, Callable
from discodop.tree import Tree, ImmutableTree
from math import log
from itertools import chain, product
from tqdm import tqdm

from ..grammar.lcfrs import lcfrs_composition, SortedSet

LcfrsRule = tuple[str, tuple[str, ...], lcfrs_composition]


def into_derivation(tree: Tree, integerize: Callable[[Any], int] = None) -> tuple[Tree, SortedSet]:
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        return None, SortedSet((tree[0],))
    children, positions = zip(*(into_derivation(c, integerize) for c in tree))
    allpositions = positions[0].union(*positions[1:])
    composition, _ = lcfrs_composition.from_positions(allpositions, positions)
    rule = (tree.label, tuple(c.label for c in tree), composition)
    if not integerize is None:
        rule = integerize(rule)
    return ImmutableTree(
        rule,
        children
    ), allpositions

@dataclass(frozen=True)
class DopRule:
    decomposition: Tree

    @classmethod
    def at_root(cls, tree: Tree) -> Iterable["DopRule"]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return
        root = tree.label
        children = tuple(n.label for n in tree)
        yield cls(root, children)
        for i, child in enumerate(tree):
            if len(child) == 1 and not isinstance(child[0], Tree):
                continue
            grandchildren = tuple(n.label for n in child)
            yield cls(root, children, (i, grandchildren))


def largest_common_fragment(s: Tree, spos: tuple[int, ...], t: Tree, tpos: tuple[int, ...], tidx: int, excludes: set[tuple[tuple[int, ...], int, tuple[int, ...]]]) -> DopRule:
    if s is None or t is None or not s.label == t.label:
        return None
    excludes.add((spos, tidx, tpos))
    return ImmutableTree(
        s.label,
        [
            largest_common_fragment(*s_, *t_, tidx, excludes)
            for s_, t_ in zip(children_with_positions(s, spos, False), children_with_positions(t, tpos, False))
        ]
    )


class Dop:
    def __init__(self, training_set: Iterable[Tree], prior: float = 0.1):
        self.rule2idx = defaultdict(lambda: len(self.rule2idx))
        self.idx2lhs = []
        derivations: list[Tree] = []
        rule_index: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for tree_index, tree in enumerate(training_set):
            derivation, _ = into_derivation(tree, integerize=self.integerize)
            derivations.append(derivation)
            queue = [(derivation, ())]
            while queue:
                subderivation, position = queue.pop(0)
                rule_index[subderivation.label].append((tree_index, position))
                queue.extend((c, position+(i,)) for i,c in enumerate(subderivation) if not c is None)

        dop_fragments = Counter()
        for derivation_index, derivation in enumerate(tqdm(derivations, "extracting dop fragments")):
            exclude_combinations = set()
            queue = [(derivation, ())]
            while queue:
                already_counted = set()
                subderivation, position = queue.pop(0)
                queue.extend(children_with_positions(subderivation, position))
                for other_occurrence in rule_index[subderivation.label]:
                    other_tidx, other_position = other_occurrence
                    if other_tidx >= derivation_index:
                        break
                    if (position, other_tidx, other_position) in exclude_combinations:
                        continue
                    other_occurrence = derivations[other_tidx][other_position]
                    assert other_occurrence.label == subderivation.label
                    fragment = largest_common_fragment(subderivation, position, other_occurrence, other_position, other_tidx, exclude_combinations)
                    if fragment in already_counted:
                        continue
                    already_counted.add(fragment)
                    dop_fragments[fragment] += 1 if fragment in dop_fragments else 2

        self.rules = defaultdict(list)
        denominators = Counter()
        for fragment, c in dop_fragments.items():
            denominators[self.idx2lhs[fragment.label]] += c + prior
            self.rules[fragment.label].append(fragment)
        self.weights = {
            fragment: log(denominators[self.idx2lhs[fragment.label]] + prior) - log(c + prior)
            for fragment, c in dop_fragments.items()
        }
        self.fallback_weight = {
            label: log(denom + prior) - log(prior) if prior else float("-inf")
            for label, denom in denominators.items()
        }


    def integerize(self, rule) -> int:
        if not rule in self.rule2idx:
            self.idx2lhs.append(rule[0])
        return self.rule2idx[rule]


    def match(self, tree: Tree) -> float:
        parser = DopTreeParser(self)
        parser.fill_chart(tree)
        return parser.chart[()]
    
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
    def _match(cls, fragment, derivation, position) -> Iterable[tuple[Tree, tuple[int,...]]] | None:
        if fragment.label != derivation.label:
            return None
        child_derivations = []
        subpositions = (position+(i,) for i in range(len(fragment)))
        for subfragment, subderivation, subposition in zip(fragment, derivation, subpositions):
            if subfragment is None:
                if not subderivation is None:
                    child_derivations.append((subderivation, subposition))
                continue
            if not (children := cls._match(subfragment, subderivation, subposition)) is None:
                child_derivations.extend(children)
                continue
            return None
        return child_derivations

    def fill_chart(self, tree: Tree):
        derivation, _ = into_derivation(tree, self.grammar.integerize)
        queue = [(derivation, ())]
        while queue:
            subderivation, position = queue.pop()
            if not all(c is None or p in self.chart for c,p in children_with_positions(subderivation, position)):
                queue.append((subderivation, position))
                queue.extend(children_with_positions(subderivation, position))
                continue
            if not subderivation.label in self.grammar.rules:
                weight = sum(self.chart[p] for _, p in children_with_positions(subderivation, position))
                weight += self.grammar.fallback_weight.get(self.grammar.idx2lhs[subderivation.label], 0)
                if not position in self.chart or self.chart[position] > weight:
                    self.chart[position] = weight
                continue
            for doprule in self.grammar.rules[subderivation.label]:
                children = self._match(doprule, subderivation, position)
                if children is None:
                    continue
                weight = sum(self.chart[p] for _, p in children)
                weight += self.grammar.weights[doprule]
                if not position in self.chart or self.chart[position] > weight:
                    self.chart[position] = weight