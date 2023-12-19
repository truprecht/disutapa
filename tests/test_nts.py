from disutapa.grammar.extraction.nonterminal import NtConstructor
from sortedcontainers import SortedSet
from disutapa.autotree import AutoTree

tree = AutoTree("(SBAR+S (VP (VP (WRB 0) (VP|<VBD,PT> (VBD 4) (PT 5))) (VBN 3)) (NP (DT 1) (NN 2)))")

def test_vanilla():
    nt = NtConstructor("vanilla")
    assert nt(tree, SortedSet(range(6))) == "SBAR+S/1"
    assert nt(tree[0], SortedSet([0,3,4,5])) == "VP/2"
    assert nt(tree[(0,0,1)], SortedSet([5])) == "VP|<VBD,PT>/1/0"
    assert nt(tree[1], SortedSet([2])) == "NP/1/0"

    assert nt(tree[(0,0)], SortedSet([4,5])) == "VP/2/-1"

def test_classic():
    nt = NtConstructor("classic")
    assert nt(tree, SortedSet(range(6))) == "SBAR/1"
    assert nt(tree[0], SortedSet([0,3,4,5])) == "VP/2"
    assert nt(tree[(0,0,1)], SortedSet([5])) == "VP|<VBD,PT>/1"
    assert nt(tree[1], SortedSet([2])) == "NP/1"

    assert nt(tree[(0,0)], SortedSet([4,5])) == "VP/1"

def test_coarse():
    nt = NtConstructor("coarse", {"SBAR": "S", "NP": "N", "VP": "S", "VBD": "P", "PT": "P"})
    assert nt(tree, SortedSet(range(6))) == "S/1"
    assert nt(tree[0], SortedSet([0,3,4,5])) == "S/2"
    assert nt(tree[(0,0,1)], SortedSet([5])) == "S|<P,P>/1"
    assert nt(tree[1], SortedSet([2])) == "N/1"

    assert nt(tree[(0,0)], SortedSet([4,5])) == "S/1"

    nt = NtConstructor("coarse")
    assert nt(tree, SortedSet(range(6))) == "S/1"
    assert nt(tree[0], SortedSet([0,3,4,5])) == "V/2"
    assert nt(tree[(0,0,1)], SortedSet([5])) == "V|<V,P>/1"
    assert nt(tree[1], SortedSet([2])) == "N/1"

    assert nt(tree[(0,0)], SortedSet([4,5])) == "V/1"
