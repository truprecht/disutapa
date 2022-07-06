from sdcp.autotree import AutoTree
from discodop.tree import Tree

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
    assert t.tree() == Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")

    assert AutoTree("(S 0 1 2)") == AutoTree("(S 2 0 1)")
    assert AutoTree("(SBAR+S (VP (VP 0 4 5) 3) (NP 1 2))") == AutoTree("(SBAR+S (NP 1 2) (VP 3 (VP 0 4 5)))")