from dataclasses import dataclass
from typing import Callable, Tuple
from itertools import chain
from discodop.tree import Tree

class AutoTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        q = [self]
        while q:
            node = q.pop()
            node.children.sort(key=lambda node: min(node.leaves()) if isinstance(node, Tree) else node)
            q.extend(n for n in node.children if isinstance(n, Tree))


class node_constructor:
    def __init__(self, label, *fixed_children):
        self.label = label
        self.fixed_children = fixed_children

    def __call__(self, *children) -> str:
        childstrs = chain((str(i) for i in self.fixed_children if not i is None), children)
        childstrs = ' '.join(childstrs)
        if self.label:
            childstrs = f"({self.label} {childstrs})"
        return childstrs

    def __str__(self):
        return f"~{self()}"

    def __repr__(self):
        return str(self)

    def __eq__(self, o) -> bool:
        return self.label == o.label and [l for l in self.fixed_children if not l is None] == [l for l in o.fixed_children if not l is None]


@dataclass
class sdcp_clause:
    label: str
    arity: int
    push_idx: int = 1

    def __call__(self, lex: int, pushed: int) -> Tuple[node_constructor, Tuple[int, ...]]:
        if self.push_idx == 0:
            lex, pushed = pushed, lex
        match self.arity:
            case 2:
                return node_constructor(self.label), (pushed, lex)
            case 1:
                return node_constructor(self.label, lex), (pushed,)
            case 0:
                return node_constructor(self.label, pushed, lex), ()
        return NotImplementedError()


@dataclass
class rule:
    lhs: str
    fn: sdcp_clause
    rhs: tuple[str]

    def as_tuple(self):
        return self.lhs, self.rhs


@dataclass
class grammar:
    root: str
    rules: list[rule]


@dataclass
class backtrace:
    rid: int
    leaf: int
    child_leafs: tuple[frozenset]

    def as_tuple(self):
        return self.rid, self.child_leafs


def gaps(positions):
    ps = sorted(positions)
    for (last, next) in zip(ps[:-1], ps[1:]):
        if last+1 != next:
            yield next-last-1


class parser:
    def __init__(self, grammar: grammar, gap_panelty: Callable = lambda gaps: sum(g+1 for g in gaps)):
        self.grammar = grammar
        self.gap_panelty = gap_panelty


    def init(self, *rules_per_position):
        self.len = len(rules_per_position)
        self.chart = {}
        self.weight = {}
        self.unaries = {}
        self.from_left = {}
        self.from_right = {}
        for i, rules in enumerate(rules_per_position):
            nullary_entries = []
            for rid in rules:
                match self.grammar.rules[rid].as_tuple():
                    case (lhs, ()):
                        nullary_entries.append((lhs, backtrace(rid, i, ())))
                    case (lhs, (r1,)):
                        self.unaries.setdefault(r1, []).append((rid, i))
                    case (lhs, (r1, r2)):
                        self.from_left.setdefault(r1, []).append((rid, i))
                        self.from_right.setdefault(r2, []).append((rid, i))
            for (lhs, bt) in nullary_entries:
                self.chart[(lhs, frozenset([i]))] = bt
                self.weight[(lhs, frozenset([i]))] = self.gap_panelty([])


    def save_backtrace(self, item, backtrace):
        weight = self.gap_panelty(gaps(item[1]))
        if not item in self.weight or self.weight[item] > weight:
            self.weight[item] = weight
            self.chart[item] = backtrace
            return True
        return False


    def fill_chart(self):
        queue = list(self.chart)
        while queue:
            lhs, positions = queue.pop()
            new_elements = []
            for rid, i in self.unaries.get(lhs, []):
                if i in positions:
                    continue
                newpos = positions.union({i})
                newlhs = self.grammar.rules[rid].lhs
                new_elements.append(((newlhs, newpos), backtrace(rid, i, (positions,))))
            for rid, i in self.from_left.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]
                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[1] and not i in positions2 and not positions2.intersection(positions):
                        newpos = positions.union({i}).union(positions2)
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions, positions2))))
            for rid, i in self.from_right.get(lhs, []):
                if i in positions:
                    continue
                r = self.grammar.rules[rid]

                for (rhs2, positions2) in self.chart:
                    if rhs2 == r.rhs[0] and not i in positions2 and not positions2.intersection(positions):
                        newpos = positions.union({i}).union(positions2)
                        new_elements.append(((r.lhs, newpos), backtrace(rid, i, (positions2, positions))))
            for item, bt in new_elements:
                if self.save_backtrace(item, bt):
                    queue.append(item)


    def get_best(self, item = None, pushed: int = None):
        if item is None:
            item = self.grammar.root, frozenset(range(0,self.len))
        bt = self.chart[item]
        fn, push = self.grammar.rules[bt.rid].fn(bt.leaf, pushed)
        match bt.as_tuple():
            case (rid, ()):
                return fn()
            case (rid, (pos,)):
                childitem = self.grammar.rules[rid].rhs[0], pos
                return fn(self.get_best(childitem, push[0]))
            case (rid, (pos1,pos2)):
                childitem1 = self.grammar.rules[rid].rhs[0], pos1
                childitem2 = self.grammar.rules[rid].rhs[1], pos2
                return fn(self.get_best(childitem1, push[0]), self.get_best(childitem2, push[1]))


def test_sdcp_fn():
    functions = [
        sdcp_clause(None, 0),
        sdcp_clause("SBAR+S", 2, push_idx=1),
        sdcp_clause("NP", 0),
        sdcp_clause("VP", 1, push_idx=1),
        sdcp_clause("VP", 2, push_idx=1),
        sdcp_clause(None, 0)
    ]

    consts = [
        (node_constructor(None, 0), ()),
        (node_constructor("SBAR+S"), (None, 1)),
        (node_constructor("NP", 1, 2), ()),
        (node_constructor("VP", 3), (None,)),
        (node_constructor("VP"), (None, 4)),
        (node_constructor(None, 4, 5), ()),
    ]

    assert functions[0](0, None) == consts[0]
    assert functions[1](1, None) == consts[1]
    assert functions[2](2, 1) == consts[2]
    assert functions[3](3, None) == consts[3]
    assert functions[4](4, None) == consts[4]
    assert functions[5](5, 4) == consts[5]


def test():
    rules = [
        rule("L-VP", sdcp_clause(None, 0), ()),
        rule("SBAR+S", sdcp_clause("SBAR+S", 2, push_idx=1), ("VP", "NP")),
        rule("NP", sdcp_clause("NP", 0), ()),
        rule("VP", sdcp_clause("VP", 1, push_idx=1), ("VP",)),
        rule("VP", sdcp_clause("VP", 2, push_idx=1), ("L-VP", "VP|<>")),
        rule("VP|<>", sdcp_clause(None, 0), ()),
    ]
    parse = parser(grammar("SBAR+S", rules))
    parse.init(*([rid] for rid in range(6)))
    parse.fill_chart()
    assert AutoTree(parse.get_best()) == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")

def test_tree():
    assert AutoTree("(S 0 1 2)") == AutoTree("(S 2 0 1)")
    assert AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))") == AutoTree("(SBAR+S (NP 1 2) (VP 3 (VP 0 4 5)))")
