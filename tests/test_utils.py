from sdcp.autotree import AutoTree
from sdcp.headed_tree import HeadedTree
from discodop.tree import Tree, HEAD

def test_tree():
    t = AutoTree("(WRB 0)")
    assert t.label == "WRB"
    assert t.children == [0]
    assert t.postags == {}

    t = AutoTree("(S (WRB 0) (NN 1))")
    assert t.label == "S"
    assert t.children == [0, 1]
    assert t.postags == {0: "WRB", 1: "NN"}

    t = AutoTree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    assert t == AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))")
    assert t.postags == {0: "WRB", 1: "PT", 2: "NN", 3: "VBD", 4: "VBN", 5: "RP"}
    assert t.tree() == Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")

    assert AutoTree("(S 0 1 2)") == AutoTree("(S 2 0 1)")
    assert AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))") == AutoTree("(SBAR+S (NP 1 2) (VP 3 (VP 0 4 5)))")


def test_unary_singleton():
    from discodop.treetransforms import collapseunary
    
    t = AutoTree("(ROOT (S ($. 0)))")
    assert t == Tree("(ROOT (S 0))")
    assert t.postags == { 0: "$." }
    assert t.tree() == Tree("(ROOT (S ($. 0)))")
    assert t.tree(override_postags=["A+B"]) == Tree("(ROOT (S (A (B 0))))")

    t = collapseunary(t.tree(), collapsepos=True, collapseroot=True)
    assert t == Tree("(ROOT+S+$. 0)")


def test_headed_tree():
    t = HeadedTree("(WRB 0)")
    assert t.label == "WRB"
    assert t.children == [0]
    assert t.postags == {}
    assert t.type != HEAD
    assert t.headidx == 0
    assert t.headterm == 0

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    assert t.label == "S"
    assert t.children == [0, 1]
    assert t.postags == {0: "WRB", 1: "NN"}
    assert t.type != HEAD
    assert t.headidx == 1
    assert t.headterm == 1

    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    t = HeadedTree.convert(t)
    assert t.headidx == 0
    assert t.headterm == 3
    assert t[0].headidx == 1
    assert t[0].headterm == 3
    assert t[(0,0)].headidx == 1
    assert t[(0,0)].headterm == 4