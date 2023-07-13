from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable
from discodop.tree import Tree, ImmutableTree
from math import log
from itertools import chain, product
from tqdm import tqdm

from ..grammar.lcfrs import lcfrs_composition, SortedSet

LcfrsRule = tuple[str, tuple[str, ...], lcfrs_composition]

def into_derivation(tree: Tree) -> tuple[Tree, SortedSet]:
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        # return ImmutableTree((tree.label, (), lcfrs_composition("0")), []), SortedSet((tree[0],))
        return None, SortedSet((tree[0],))
    children, positions = zip(*(into_derivation(c) for c in tree))
    allpositions = positions[0].union(*positions[1:])
    composition, _ = lcfrs_composition.from_positions(allpositions, positions)
    return ImmutableTree(
        (tree.label, tuple(c.label for c in tree), composition),
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


def common_fragments(s: Tree, t: Tree) -> Iterable[DopRule]:
    if s is None or t is None or not s.label == t.label:
        return
    root = s.label
    child_combinations = (
        chain((None,), common_fragments(s_, t_))
        for s_,t_ in zip(s, t)
    )
    for children in product(*child_combinations):
        yield ImmutableTree(root, children)


class Dop:
    def __init__(self, training_set: Iterable[Tree], prior: float = 0.1):
        derivations: list[Tree] = []
        rule_index: dict[LcfrsRule, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
        for tree_index, tree in enumerate(training_set):
            derivation, _ = into_derivation(tree)
            derivations.append(derivation)
            queue = [(derivation, ())]
            while queue:
                subderivation, position = queue.pop(0)
                rule_index[subderivation.label].append((tree_index, position))
                queue.extend((c, position+(i,)) for i,c in enumerate(subderivation) if not c is None)

        dop_fragments = Counter()
        for derivation_index, derivation in enumerate(tqdm(derivations, "extracting dop fragments")):
            queue = [(derivation, ())]
            while queue:
                subderivation, position = queue.pop(0)
                queue.extend((c, position+(i,)) for i,c in enumerate(subderivation) if not c is None)
                dop_fragments_in_occurrence = set()
                for other_occurrence in rule_index[subderivation.label]:
                    if (derivation_index, position) == other_occurrence:
                        continue
                    other_occurrence = derivations[other_occurrence[0]][other_occurrence[1]]
                    assert other_occurrence.label == subderivation.label
                    for common_fragment in common_fragments(subderivation, other_occurrence):
                        dop_fragments_in_occurrence.add(common_fragment)
                dop_fragments.update(dop_fragments_in_occurrence)

        self.rules = defaultdict(list)
        denominators = Counter()
        for fragment, c in dop_fragments.items():
            denominators[fragment.label[0]] += c + prior
            self.rules[fragment.label].append(fragment)
        self.weights = {
            fragment: log(denominators[fragment.label[0]] + prior) - log(c + prior)
            for fragment, c in dop_fragments.items()
        }
        self.fallback_weight = {
            label: log(denom + prior) - log(prior) if prior else float("-inf")
            for label, denom in denominators.items()
        }


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


def children_with_positions(tree, position):
    return ((c, position+(i,)) for i,c in enumerate(tree) if not c is None)

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
        derivation, _ = into_derivation(tree)
        queue = [(derivation, ())]
        while queue:
            subderivation, position = queue.pop()
            if not all(c is None or p in self.chart for c,p in children_with_positions(subderivation, position)):
                queue.append((subderivation, position))
                queue.extend(children_with_positions(subderivation, position))
                continue
            if not subderivation.label in self.grammar.rules:
                weight = sum(self.chart[p] for _, p in children_with_positions(subderivation, position))
                weight += self.grammar.fallback_weight.get(subderivation.label[0], 0)
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