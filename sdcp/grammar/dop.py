from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Any
from discodop.tree import Tree
from discodop.treetransforms import binarize
from math import log, exp
from tqdm import tqdm
from multiprocessing import Pool

from .lcfrs import lcfrs_composition, SortedSet

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

@dataclass(frozen=True)
class Fragment:
    rule: int
    children: tuple["Fragment"]
    _precomputed_str: str = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        if not self.children:
            self.__dict__ ["_precomputed_str"] = str(self.rule)
            return
        children = " ".join(
            "*" if child is None else child._precomputed_str
            for child in self.children
        )
        self.__dict__ ["_precomputed_str"] = f"({self.rule} {children})"

    def __str__(self):
        return self._precomputed_str



class DerivationFactory:
    def __init__(self, markovization: tuple[str, int, int] = ("right", 0, 1)):
        self.rule2idx: dict[LcfrsRule, int] = dict()
        self.idx2lhs: list[str] = []
        self.markovization = markovization
        self.seen_positions: int = 0

    def _integerize(self, rule: LcfrsRule) -> int:
        if not rule in self.rule2idx:
            self.idx2lhs.append(rule[0])
        return self.rule2idx.setdefault(rule, len(self.idx2lhs)-1)
    
    def _produce_with_positions(self, tree) -> tuple[Derivation, SortedSet]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return None, SortedSet((tree[0],))
        nodeposition = self.seen_positions
        self.seen_positions += 1
        children, positions = zip(*(self._produce_with_positions(c) for c in tree))
        allpositions = positions[0].union(*positions[1:])
        composition, _ = lcfrs_composition.from_positions(allpositions, positions)
        rule = self._integerize((tree.label, tuple(c.label for c in tree), composition))
        return Derivation(
            rule,
            nodeposition,
            children
        ), allpositions

    def __call__(self, tree) -> Derivation:
        if not self.markovization is None:
            tree = binarize(tree, *self.markovization)
        return self._produce_with_positions(tree)[0]
    
    def get_state(self) -> Any:
        return (self.rule2idx, self.idx2lhs, self.markovization)
    
    @classmethod
    def from_state(cls, args: Any) -> "DerivationFactory":
        obj = cls()
        obj.rule2idx, obj.idx2lhs, obj.markovization = args
        return obj


def largest_common_fragment(s: Derivation, t: Derivation, excludes: set[tuple[int, int]]) -> Fragment:
    if s is None or t is None or s.rule != t.rule:
        return None
    excludes.add((s.position, t.position))
    return Fragment(
        s.rule,
        tuple(
            largest_common_fragment(s_, t_, excludes)
            for s_, t_ in zip(s.children, t.children)
        )
    )


def fragment_to_transitions(
        fragment: Fragment,
        id2rule: dict[int, LcfrsRule],
        weight: float
        ) -> Iterable[tuple[tuple[str, ...], int, str, float]]:
    queue = [(fragment, True)]
    while queue:
        node, is_root = queue.pop()

        w = weight if is_root else 0.0
        symbol: int = node.rule
        grammar_rule = id2rule[symbol]
        top = grammar_rule[0] if is_root else f"F {node}"
        bottoms = tuple(
            rhsnt if c is None else f"F {c}"
            for c, rhsnt in zip(node.children, grammar_rule[1])
        )

        yield (bottoms, symbol, top, w)

        queue.extend(
            (c, False)
            for c in node.children
            if not c is None
        )


class Dop:
    def get_tree_fragments(self, derivation: Derivation) -> Counter[Fragment]:
        dop_fragments = Counter()
        exclude_combinations = set()
        queue: list[Derivation] = [derivation]
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
                if not fragment in already_counted:
                    already_counted.add(fragment)
                    dop_fragments[fragment] += 1
            # ensure singleton fragment is counted
            rule_fragment = Fragment(subderivation.rule, (None,)*len(subderivation.children))
            if not rule_fragment in already_counted:
                dop_fragments[rule_fragment] += 1
        return dop_fragments


    def __init__(self, training_set: Iterable[Tree], prior: float = 0.1, markovization: None | tuple[str,int,int] = None):
        self.tops = set()
        derivations: list[Tree] = []
        self.rule_index: dict[int, Derivation] = defaultdict(list)
        into_derivation = DerivationFactory(markovization)
        for tree in training_set:
            self.tops.add(tree.label)
            derivation = into_derivation(tree)
            derivations.append(derivation)
            queue = [derivation]
            while queue:
                subderivation = queue.pop(0)
                self.rule_index[subderivation.rule].append(subderivation)
                queue.extend(c for c in subderivation.children if not c is None)

        dop_fragments: Counter[Fragment] = Counter()
        pool = Pool()
        for counts in pool.imap_unordered(self.get_tree_fragments, tqdm(derivations, "extract fragments"), chunksize=128):
            dop_fragments += counts            

        self.transitions = defaultdict(set)
        idx2rule = list(into_derivation.rule2idx.keys())
        denominators = Counter()
        for fragment in dop_fragments:
            if not all(child is None for child in fragment.children):
                dop_fragments[fragment] += 1
            count = dop_fragments[fragment]
            denominators[into_derivation.idx2lhs[fragment.rule]] += count + prior
        for fragment, c in dop_fragments.items():
            weight = log(denominators[into_derivation.idx2lhs[fragment.rule]] + prior) - log(c + prior)
            for (ps, symb, top, w) in fragment_to_transitions(fragment, idx2rule, weight):
                self.transitions[symb].add((ps, top, w))
        self.fallback_weight = {
            label: log(denom + prior) - log(prior) if prior else float("inf")
            for label, denom in denominators.items()
        }
        self.derivation_factory_state = into_derivation.get_state()

    def match(self, tree: Tree) -> float:
        parser = DopTreeParser(self)
        parser.fill_chart(Tree.convert(tree), self.derivation_factory_state)
        return parser.chart.get((0, tree.label), float("inf"))
    
    def select(self, trees: Iterable[tuple[Tree, float]]) -> tuple[int, Tree]:
        bestidx, besttree, bestweight = None, None, None
        for i, (tree, _) in enumerate(trees):
            weight = self.match(tree)
            if bestweight is None or weight < bestweight:
                bestidx, besttree, bestweight = i, tree, weight
        return bestidx, besttree


@dataclass
class DopTreeParser:
    grammar: Dop
    reduction: str = "sum"
    chart: dict[tuple[int, ...], float] = field(default_factory=Counter)

    def _update(self, local_chart, key, value):
        if self.reduction == "min" and local_chart.setdefault(key, value) > value:
            local_chart[key] = value
        if self.reduction == "sum":
            local_chart[key] += exp(-value)

    def assimilate_local_chart(self, local_chart, node):
        for k, v in local_chart.items():
            if self.reduction == "sum":
                v = -log(v) if v > 0 else float("inf")
            self.chart[(node.position, k)] = v

    def fill_chart(self, tree: Tree, factory_state: Any):
        into_derivation = DerivationFactory.from_state(factory_state)
        derivation = into_derivation(tree)
        for subderivation in derivation.bottom_up():
            newweights = Counter()
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
                self._update(newweights, top, weight+w)
            if not self.grammar.transitions[subderivation.rule]:
                lhs = into_derivation.idx2lhs[subderivation.rule]
                children = (
                    (c.position, into_derivation.idx2lhs[c.rule])
                    for c in subderivation.children
                    if not c is None
                )
                weight = sum(self.chart[child] for child in children)
                weight += self.grammar.fallback_weight.get(lhs, float("inf"))
                self._update(newweights, lhs, weight)
            self.assimilate_local_chart(newweights, subderivation)
                