from sdcp.grammar.extract_head import read_spine, extract_node, headed_clause, headed_rule, extract_head
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


def test_reorder():
    r = headed_rule("A", "abcd", "(A 4 3 (B 0) 2 1)")
    positions = (2,0,1,3,4)
    assert r.reorder(positions) ==headed_rule("A", "abcd", "(A 4 3 (B 0) 2 1)", lexidx=2)
    positions = (0,1,3,2,4)
    assert r.reorder(positions) == headed_rule("A", "acbd", "(A 4 2 (B 0) 3 1)", lexidx=0)
    positions = (3,5,4,2,1)
    assert r.reorder(positions) == headed_rule("A", "dcba", "(A 1 2 (B 0) 3 4)", lexidx=2)


def test_extract():
    t = HeadedTree("(WRB 0)")
    clause = headed_clause(Tree("WRB", [0]))
    assert extract_node(t, "ROOT", hmarkov=0) == Tree((0, 0, headed_rule("ROOT", (), clause, 1)), [])

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    r1 = headed_rule("ROOT", ["S|<>"], headed_clause(Tree("S", [1, 0])), 1)
    r2 = headed_rule("S|<>", [], headed_clause(0), 1)
    assert extract_node(t, "ROOT", hmarkov=0) == Tree((1, 0, r1), [Tree((0, 0, r2), [])])

    t = Tree("(S (SBAR (WRB 0)) (NN 1))")
    t[1].type = HEAD
    t[(0,0)].type = HEAD
    t = HeadedTree.convert(t)
    r1 = headed_rule("ROOT", ["S|<>"], "(S 1 0)")
    r2 = headed_rule("S|<>", [], "(SBAR 0)")
    assert extract_node(t, "ROOT", hmarkov=0) == Tree((1, 0, r1), [Tree((0, 0, r2), [])])

    t = Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")
    t[0].type = HEAD
    t[(0,0)].type = HEAD
    t[(0,0, 1)].type = HEAD
    t[(0,0, 0, 1)].type = HEAD
    t[(0,1, 1)].type = HEAD
    t = HeadedTree.convert(t)
    deriv = extract_node(t, "ROOT", hmarkov=0)
    assert deriv.label == (3, 0, headed_rule("ROOT", ["VP|<>", "S|<>"], headed_clause(Tree("SBAR", [Tree("S", [Tree("VP", [1, 0]), 2])])), 1, lexidx=2))
    assert deriv[0].label == (4, 0, headed_rule("VP|<>", ["VP|<>", "VP|<>"], headed_clause(Tree("VP", [1, 0, 2])), 2))
    assert deriv[(0,0)].label == (0, 0, headed_rule("VP|<>", [], headed_clause(0), 1))
    assert deriv[(0,1)].label == (5, 5, headed_rule("VP|<>", [], headed_clause(0), 1))
    assert deriv[1].label == (2, 1, headed_rule("S|<>", ["NP|<>"], headed_clause(Tree("NP", [1, 0])), 1))
    assert deriv[(1,0)].label == (1, 1, headed_rule("NP|<>", [], headed_clause(0), 1))


    deriv = extract_node(t, "ROOT", hmarkov=0, markendpoint=False)
    assert deriv.label == (3, 0, headed_rule("ROOT", ["VP", "NP"], headed_clause(Tree("SBAR", [Tree("S", [Tree("VP", [1, 0]), 2])])), 1,  lexidx=2))
    assert deriv[0].label == (4, 0, headed_rule("VP", ["ARG", "ARG"], headed_clause(Tree("VP", [1, 0, 2])), 2))
    assert deriv[(0,0)].label == (0, 0, headed_rule("ARG", [], headed_clause(0), 1))
    assert deriv[(0,1)].label == (5, 5, headed_rule("ARG", [], headed_clause(0), 1))
    assert deriv[1].label == (2, 1, headed_rule("NP", ["ARG"], headed_clause(Tree("NP", [1, 0])), 1))
    assert deriv[(1,0)].label == (1, 1, headed_rule("ARG", [], headed_clause(0), 1))

def test_nonbin_extraction():
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    assert list(extract_head(t, horzmarkov=0)) == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert list(extract_head(t, horzmarkov=0)) == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("T|<>", "T|<>", "S|<>",), clause="(_|<> (T 1 0 2) 3)"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert list(extract_head(t, horzmarkov=0)) == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("T|<>", "S|<>", "T|<>",), clause="(_|<> (T 1 0 3) 2)"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
        headed_rule("T|<>", ()),
    ]
    
    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert list(extract_head(t, horzmarkov=0, rightmostunary=False)) == [
        headed_rule("ARG", ()),
        headed_rule("ROOT", ("ARG", "S|<>"), clause="(S 1 0 2)"),
        headed_rule("ARG", ()),
        headed_rule("S|<>", ("ARG", "S|<>", "ARG",), clause="(_|<> (T 1 0 3) 2)"),
        headed_rule("S|<>", ("ARG",), clause="(_|<> 0 1)"),
        headed_rule("ARG", ()),
        headed_rule("ARG", ()),
    ]

def test_assembly():
    constructors = [
        headed_clause(Tree("SBAR+S", [Tree("VP", [1, 0]), 2]))(3),
        headed_clause(Tree("VP", [1, 0, 2]))(4),
        headed_clause(0)(0),
        headed_clause(0)(5),
        headed_clause(Tree("NP", [1, 0]))(2),
        headed_clause(0)(1),
    ]

    assert constructors[2]() == [0]
    assert constructors[3]() == [5]
    assert constructors[1]([0], [5]) == [Tree("VP", [0, 4, 5])]
    assert constructors[5]() == [1]
    assert constructors[4]([1]) == [Tree("NP", [1, 2])]
    assert constructors[0]([Tree("VP", [0, 4, 5])], [Tree("NP", [1, 2])]) == [Tree("SBAR+S", [Tree("VP", [Tree("VP", [0, 4, 5]), 3]), Tree("NP", [1, 2])])]

    constructors = [
        headed_clause(0)(0),
        headed_clause("(S 1 0 2)")(1),
        headed_clause(0)(2),
        headed_clause("(_|<> (T 1 0 3) 2)")(3),
        headed_clause("(_|<> 0 1)")(4),
        headed_clause(0)(5),
        headed_clause(0)(6),
    ]
    assert constructors[0]() == [0]
    assert constructors[2]() == [2]
    assert constructors[5]() == [5]
    assert constructors[6]() == [6]
    
    assert constructors[4]([5]) == [4, 5]
    assert constructors[3]([2], [4, 5], [6]) == [Tree("(T 2 3 6)"), 4, 5]
    assert constructors[1]([0], [Tree("(T 2 3 6)"), 4, 5]) == [Tree("(S 0 1 (T 2 3 6) 4 5)")]