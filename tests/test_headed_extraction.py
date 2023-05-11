from sdcp.grammar.extract_head import headed_clause, headed_rule, Extractor, Nonterminal, extraction_result, SortedSet, lcfrs_composition
from sdcp.headed_tree import HeadedTree, Tree, HEAD

def test_read_spine():
    e = Extractor()
    t = HeadedTree("(WRB 0)")
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("WRB", [0])
    assert succs == []

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("S", [1, 0])
    assert succs == [(("S",), [t[0]], -1)]

    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    t = HeadedTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("SBAR+S", [Tree("VP", [1, 0]), 2])
    assert succs == [(("SBAR+S", "VP"), [t[(0,0)]], -1), (("SBAR+S",), [t[1]], +1)]

    e = Extractor(horzmarkov=0)
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("S", [1, 0, 2])
    assert succs == [(("S",), [t[0]], -1), (("S",), t[2:], +1)]


def test_extract():
    e = Extractor(horzmarkov=0, rightmostunary=True, markrepeats=True)
    t = HeadedTree("(WRB 0)")
    clause = headed_clause(Tree("WRB", [0]))
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(0, SortedSet([0]), headed_rule("ROOT", (), clause)), [])

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    r1 = headed_rule("ROOT", ["S|<>"], headed_clause(Tree("S", [1, 0])), lcfrs_composition("10"))
    r2 = headed_rule("S|<>", [], headed_clause(0))
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(1, SortedSet([0, 1]), r1), [Tree(
            extraction_result(0, SortedSet([0]), r2), [])])

    t = Tree("(S (SBAR (WRB 0)) (NN 1))")
    t[1].type = HEAD
    t[(0,0)].type = HEAD
    t = HeadedTree.convert(t)
    r1 = headed_rule("ROOT", ["S|<>"], "(S 1 0)", lcfrs_composition("10"))
    r2 = headed_rule("S|<>", [], "(SBAR 0)")
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(1, SortedSet([0, 1]), r1), [Tree(
            extraction_result(0, SortedSet([0]), r2), [])])

    t = Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")
    t[0].type = HEAD
    t[(0,0)].type = HEAD
    t[(0,0,1)].type = HEAD
    t[(0,0,0,1)].type = HEAD
    t[(0,1,1)].type = HEAD
    t = HeadedTree.convert(t)
    deriv = e.extract_node(t, "ROOT")
    assert deriv.label == extraction_result(3, SortedSet(range(6)),
                                headed_rule("ROOT", ["VP|<>", "S|<>"], headed_clause(Tree("SBAR", [Tree("S", [Tree("VP", [1, 0]), 2])])), lcfrs_composition("1201")))
    assert deriv[0].label == extraction_result(4, SortedSet([0,4,5]),
                                headed_rule("VP|<>", ["VP+|<>", "VP+|<>"], headed_clause(Tree("VP", [1, 0, 2])), lcfrs_composition("1,02")))
    assert deriv[(0,0)].label == extraction_result(0, SortedSet([0]),
                                headed_rule("VP+|<>", [], headed_clause(0)))
    assert deriv[(0,1)].label == extraction_result(5, SortedSet([5]),
                                headed_rule("VP+|<>", [], headed_clause(0)))
    assert deriv[1].label == extraction_result(2, SortedSet([1, 2]),
                                headed_rule("S|<>", ["NP|<>"], headed_clause(Tree("NP", [1, 0])), lcfrs_composition("10")))
    assert deriv[(1,0)].label == extraction_result(1, SortedSet([1]), headed_rule("NP|<>", [], headed_clause(0)))

    e = Extractor(horzmarkov=0, rightmostunary=False, markrepeats=True)
    deriv = e.extract_node(t, "ROOT")
    assert deriv.label == extraction_result(3, SortedSet(range(6)),
                                headed_rule("ROOT", ["VP+", "NP"], headed_clause(Tree("SBAR", [Tree("S", [Tree("VP", [1, 0]), 2])])), lcfrs_composition("1201")))
    assert deriv[0].label == extraction_result(4, SortedSet([0,4,5]),
                                headed_rule("VP+", ["ARG(VP)", "ARG(VP)"], headed_clause(Tree("VP", [1, 0, 2])), lcfrs_composition("1,02")))
    assert deriv[(0,0)].label == extraction_result(0, SortedSet([0]), headed_rule("ARG(VP)", [], headed_clause(0)))
    assert deriv[(0,1)].label == extraction_result(5, SortedSet([5]), headed_rule("ARG(VP)", [], headed_clause(0)))
    assert deriv[1].label == extraction_result(2, SortedSet([1,2]), headed_rule("NP", ["ARG(NP)"],
                                headed_clause(Tree("NP", [1, 0])), lcfrs_composition("10")))
    assert deriv[(1,0)].label == extraction_result(1, SortedSet([1]), headed_rule("ARG(NP)", [], headed_clause(0)))

def test_nonbin_extraction():
    e = Extractor(horzmarkov=0, rightmostunary=True)
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)", composition="102"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)", composition="102"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("T|<>", "T|<>", "S|<>",), clause="(_|<> (T 1 0 2) 3)", composition="1023"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("S|<>", ()),
        headed_rule("ROOT", ("S|<>", "S|<>"), clause="(S 1 0 2)", composition="102"),
        headed_rule("T|<>", ()),
        headed_rule("S|<>", ("T|<>", "T|<>", "S|<>"), clause="(_|<> (T 1 0 2) 3)", composition="1032"),
        headed_rule("S|<>", ("S|<>",), clause="(_|<> 0 1)"),
        headed_rule("S|<>", ()),
        headed_rule("T|<>", ()),
    ]
    
    e = Extractor(horzmarkov=0, rightmostunary=False)
    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = HeadedTree.convert(t)
    assert e(t)[0] == [
        headed_rule("ARG(S)", ()),
        headed_rule("ROOT", ("ARG(S)", "S|<>"), clause="(S 1 0 2)", composition="102"),
        headed_rule("ARG(T)", ()),
        headed_rule("S|<>", ("ARG(T)", "ARG(T)", "S|<>"), clause="(_|<> (T 1 0 2) 3)", composition="1032"),
        headed_rule("S|<>", ("ARG(S)",), clause="(_|<> 0 1)"),
        headed_rule("ARG(S)", ()),
        headed_rule("ARG(T)", ()),
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