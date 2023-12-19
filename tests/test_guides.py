from disutapa.grammar.extraction.guide import *
from discodop.tree import Tree, HEAD

def test_guides():
    t = Tree("(SBAR+S (VP (VP (WRB 0) (VP|<> (VBD 4) (PT 5))) (VBN 3)) (NP (DT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0,1)].type = HEAD
    t[(0,0,1)].type = HEAD
    t[(0,0,1,0)].type = HEAD
    t[(1,1)].type = HEAD
    tree = AutoTree.convert(t)

    strict_guide = Guide.construct("strict", tree)
    assert strict_guide(tree) == 1
    assert strict_guide(tree[0]) == 3
    assert strict_guide(tree[(0,0)]) == 4
    assert strict_guide(tree[(0,0,1)]) == 5
    assert strict_guide(tree[1]) == 2

    dep_guide = Guide.construct("dependent", tree)
    assert dep_guide(tree) == 2
    assert dep_guide(tree[0]) == 4
    assert dep_guide(tree[(0,0)]) == 0
    assert dep_guide(tree[(0,0,1)]) == 5
    assert dep_guide(tree[1]) == 1

    van_guide = Guide.construct("vanilla", tree)
    assert van_guide(tree) == 1
    assert van_guide(tree[0]) == 3
    assert van_guide(tree[(0,0)]) == 0
    assert van_guide(tree[(0,0,1)]) == 5
    assert van_guide(tree[1]) == 2

    least_guide = Guide.construct("least", tree)
    assert least_guide(tree) == 2
    assert least_guide(tree[0]) == 3
    assert least_guide(tree[(0,0)]) == 0
    assert least_guide(tree[(0,0,1)]) == 4
    assert least_guide(tree[1]) == 1

    near_guide = Guide.construct("near", tree)
    assert near_guide(tree) == 3
    assert near_guide(tree[0]) == 0
    assert near_guide(tree[(0,0)]) == 4
    assert near_guide(tree[(0,0,1)]) == 5
    assert near_guide(tree[1]) == 1

