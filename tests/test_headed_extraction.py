from sdcp.grammar.extract_head import read_spine, extract_node, headed_clause, headed_rule
from sdcp.headed_tree import HeadedTree, Tree, HEAD

def test_read_spine():
    t = HeadedTree("(WRB 0)")
    clause, succs, _ = read_spine(t)
    assert clause == Tree("WRB", [0])
    assert succs == []

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    clause, succs, _ = read_spine(t)
    assert clause == Tree("S", [1, 0])
    assert succs == [("S", [t[0]])]

    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    t = HeadedTree.convert(t)
    clause, succs, _ = read_spine(t)
    assert clause == Tree("SBAR+S", [Tree("VP", [1, 0]), 2])
    assert succs == [("VP", [t[(0,0)]]), ("SBAR+S", [t[1]])]

def test_extract():
    t = HeadedTree("(WRB 0)")
    clause = headed_clause(Tree("WRB", [0]))
    assert extract_node(t, "ROOT") == Tree((0, headed_rule("ROOT", (), clause, 0)), [])

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    r1 = headed_rule("ROOT", ["S|<>"], headed_clause(Tree("S", [1, 0])), 0)
    r2 = headed_rule("S|<>", [], headed_clause(0), 0)
    assert extract_node(t, "ROOT") == Tree((1, r1), [Tree((0, r2), [])])

    t = Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")
    t[0].type = HEAD
    t[(0,0)].type = HEAD
    t[(0,0, 1)].type = HEAD
    t[(0,0, 0, 1)].type = HEAD
    t[(0,1, 1)].type = HEAD
    t = HeadedTree.convert(t)
    deriv = extract_node(t, "ROOT")
    assert deriv.label == (3, headed_rule("ROOT", ["VP|<>", "S|<>"], headed_clause(Tree("SBAR", [Tree("S", [Tree("VP", [1, 0]), 2])])), 0))
    assert deriv[0].label == (4, headed_rule("VP|<>", ["VP|<>", "VP|<>"], headed_clause(Tree("VP", [1, 0, 2])), 0))
    assert deriv[(0,0)].label == (0, headed_rule("VP|<>", [], headed_clause(0), 0))
    assert deriv[(0,1)].label == (5, headed_rule("VP|<>", [], headed_clause(0), 0))
    assert deriv[1].label == (2, headed_rule("S|<>", ["NP|<>"], headed_clause(Tree("NP", [1, 0])), 0))
    assert deriv[(1,0)].label == (1, headed_rule("NP|<>", [], headed_clause(0), 0))

def test_assembly():
    constructors = [
        headed_clause(Tree("SBAR+S", [Tree("VP", [1, 0]), 2]))(3),
        headed_clause(Tree("VP", [1, 0, 2]))(4),
        headed_clause(0)(0),
        headed_clause(0)(5),
        headed_clause(Tree("NP", [1, 0]))(2),
        headed_clause(0)(1),
    ]

    assert constructors[2]() == 0
    assert constructors[3]() == 5
    assert constructors[1](0, 5) == Tree("VP", [0, 4, 5])
    assert constructors[5]() == 1
    assert constructors[4](1) == Tree("NP", [1, 2])
    assert constructors[0](Tree("VP", [0, 4, 5]), Tree("NP", [1, 2])) == Tree("SBAR+S", [Tree("VP", [Tree("VP", [0, 4, 5]), 3]), Tree("NP", [1, 2])])