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

    def bottom_up(self) -> Iterable["Derivation"]:
        queue = [self]
        visited = set()
        while queue:
            node = queue.pop()
            if not node.position in visited and not all(c is None for c in node.children):
                queue.append(node)
                queue.extend(c for c in node.children if not c is None)
                visited.add(node.position)
                continue
            yield node


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


def fragment_to_transitions(
        fragment: Tree,
        id: int,
        id2rule: dict[int, LcfrsRule],
        weight: float
        ) -> Iterable[tuple[tuple[str, ...], int, str, float]]:
    queue = [(fragment, ())]
    while queue:
        node, position = queue.pop()

        w = weight if position == () else 0.0
        symbol: int = node.label
        grammar_rule = id2rule[symbol]
        top = grammar_rule[0] if position == () else f"{id}-{position}"
        bottoms = tuple(
            rhsnt if c is None else f"{id}-{position+(i,)}"
            for i, (c, rhsnt) in enumerate(zip(node.children, grammar_rule[1]))
        )

        yield (bottoms, symbol, top, w)

        queue.extend(
            (c, position+(i,))
            for i, c in enumerate(node.children)
            if not c is None
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
        self.tops = set()
        derivations: list[Tree] = []
        self.rule_index: dict[int, Derivation] = defaultdict(list)
        for tree in training_set:
            self.tops.add(tree.label)
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

        self.transitions = defaultdict(list)
        idx2rule = list(self.rule2idx.keys())
        denominators = Counter()
        for fragment, c in dop_fragments.items():
            denominators[self.idx2lhs[fragment.label]] += c + 1 + prior
        for idx, (fragment, c) in enumerate(dop_fragments.items()):
            weight = log(denominators[self.idx2lhs[fragment.label]] + prior) - log(c + 1 + prior)
            for (ps, symb, top, w) in fragment_to_transitions(fragment, idx, idx2rule, weight):
                self.transitions[symb].append((ps, top, w))
        self.fallback_weight = {
            label: log(denom + prior) - log(prior) if prior else float("inf")
            for label, denom in denominators.items()
        }

    def integerize(self, rule) -> int:
        if not rule in self.rule2idx:
            self.idx2lhs.append(rule[0])
        return self.rule2idx.setdefault(rule, len(self.idx2lhs)-1)

    def match(self, tree: Tree) -> float:
        parser = DopTreeParser(self)
        parser.fill_chart(tree)
        return parser.chart.get((1, tree.label), float("inf"))
    
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

    def _update(self, key, value):
        if self.chart.setdefault(key, value) > value:
            self.chart[key] = value

    def fill_chart(self, tree: Tree):
        global treeposition
        treeposition = 0
        derivation, _ = into_derivation(tree, self.grammar.integerize)
        for subderivation in derivation.bottom_up():
            add_fallback_rule = True
            for (bottoms, top, w) in self.grammar.transitions[subderivation.rule]:
                weight = 0
                for (child, childstate) in zip(subderivation.children, bottoms):
                    if child is None: continue
                    if not (childweight := self.chart.get((child.position, childstate))) is None:
                        weight += childweight
                    else:
                        weight = None
                        break
                if weight is None: continue
                self._update((subderivation.position, top), weight+w)
                if top == self.grammar.idx2lhs[subderivation.rule]:
                    add_fallback_rule = False
            if add_fallback_rule:
                lhs = self.grammar.idx2lhs[subderivation.rule]
                children = (
                    (c.position, self.grammar.idx2lhs[c.rule])
                    for c in subderivation.children
                    if not c is None
                )
                weight = sum(self.chart[child] for child in children)
                weight += self.grammar.fallback_weight.get(lhs, float("inf"))
                self._update((subderivation.position, lhs), weight)
                continue
                