from sdcp.grammar.extraction.extract_head import Extractor, Nonterminal, extraction_result, SortedSet, lcfrs_composition, rule, sdcp_clause
from sdcp.autotree import AutoTree, Tree, HEAD

def test_read_spine():
    e = Extractor()
    t = AutoTree("(WRB 0)")
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("WRB", [0])
    assert succs == []

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("S", [1, 0])
    assert succs == [(("S",), [t[0]], -1)]

    t = Tree("(SBAR+S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))")
    t[0].type = HEAD
    t[(0, 1)].type = HEAD
    t[(0, 0, 1)].type = HEAD
    t[(1, 1)].type = HEAD
    t = AutoTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("SBAR+S", [Tree("VP", [1, 0]), 2])
    assert succs == [(("SBAR+S", "VP"), [t[(0,0)]], -1), (("SBAR+S",), [t[1]], +1)]

    e = Extractor(hmarkov=0)
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    clause, succs, _ = e.read_spine(t, ())
    assert clause == Tree("S", [1, 0, 2])
    assert succs == [(("S",), [t[0]], -1), (("S",), t[2:], +1)]


def test_extract():
    e = Extractor(hmarkov=0, rightmostunary=True, markrepeats=True)
    t = AutoTree("(WRB 0)")
    clause = sdcp_clause.spine(Tree("WRB", [0]))
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(0, SortedSet([0]), rule("ROOT", dcp=clause)), [])

    t = Tree("(S (WRB 0) (NN 1))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    r1 = rule("ROOT", ["S|<>", None], dcp=sdcp_clause.spine(Tree("S", [1, 0])))
    r2 = rule("S|<>", dcp=sdcp_clause.spine(0))
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(1, SortedSet([0, 1]), r1), [Tree(
            extraction_result(0, SortedSet([0]), r2), [])])

    t = Tree("(S (SBAR (WRB 0)) (NN 1))")
    t[1].type = HEAD
    t[(0,0)].type = HEAD
    t = AutoTree.convert(t)
    r1 = rule("ROOT", ["S|<>", None], dcp=sdcp_clause.spine("(S 1 0)"))
    r2 = rule("S|<>", dcp=sdcp_clause.spine("(SBAR 0)"))
    assert e.extract_node(t, "ROOT") == Tree(
        extraction_result(1, SortedSet([0, 1]), r1), [Tree(
            extraction_result(0, SortedSet([0]), r2), [])])

    t = Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))")
    t[0].type = HEAD
    t[(0,0)].type = HEAD
    t[(0,0,1)].type = HEAD
    t[(0,0,0,1)].type = HEAD
    t[(0,1,1)].type = HEAD
    t = AutoTree.convert(t)
    deriv = e.extract_node(t, "ROOT")
    assert deriv.label == extraction_result(3, SortedSet(range(6)),
                                rule("ROOT", ["VP|<>", "S|<>", None], dcp=sdcp_clause.spine("(SBAR (S (VP 1 0) 2))"), scomp=lcfrs_composition("0120")))
    assert deriv[0].label == extraction_result(4, SortedSet([0,4,5]),
                                rule("VP|<>", ["VP+|<>", None, "VP+|<>"], dcp=sdcp_clause.spine("(VP 1 0 2)"), scomp=lcfrs_composition("0,12")))
    assert deriv[(0,0)].label == extraction_result(0, SortedSet([0]),
                                rule("VP+|<>", dcp=sdcp_clause.spine(0)))
    assert deriv[(0,1)].label == extraction_result(5, SortedSet([5]),
                                rule("VP+|<>", dcp=sdcp_clause.spine(0)))
    assert deriv[1].label == extraction_result(2, SortedSet([1, 2]),
                                rule("S|<>", ["NP|<>", None], dcp=sdcp_clause.spine("(NP 1 0)")))
    assert deriv[(1,0)].label == extraction_result(1, SortedSet([1]), rule("NP|<>", dcp=sdcp_clause.spine(0)))

    e = Extractor(hmarkov=0, rightmostunary=False, markrepeats=True)
    deriv = e.extract_node(t, "ROOT")
    assert deriv.label == extraction_result(3, SortedSet(range(6)),
                                rule("ROOT", ["VP+", "NP", None], dcp=sdcp_clause.spine("(SBAR (S (VP 1 0) 2))"), scomp=lcfrs_composition("0120")))
    assert deriv[0].label == extraction_result(4, SortedSet([0,4,5]),
                                rule("VP+", ["ARG(VP)", None, "ARG(VP)"], dcp=sdcp_clause.spine("(VP 1 0 2)"), scomp=lcfrs_composition("0,12")))
    assert deriv[(0,0)].label == extraction_result(0, SortedSet([0]), rule("ARG(VP)", dcp=sdcp_clause.spine(0)))
    assert deriv[(0,1)].label == extraction_result(5, SortedSet([5]), rule("ARG(VP)", dcp=sdcp_clause.spine(0)))
    assert deriv[1].label == extraction_result(2, SortedSet([1,2]), rule("NP", ["ARG(NP)", None],
                                dcp=sdcp_clause.spine("(NP 1 0)")))
    assert deriv[(1,0)].label == extraction_result(1, SortedSet([1]), rule("ARG(NP)", dcp=sdcp_clause.spine(0)))

def test_nonbin_extraction():
    e = Extractor(hmarkov=0, rightmostunary=True)
    t = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4))")
    t[1].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>"),
        rule("ROOT", ("S|<>", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)")),
        rule("S|<>", (None, "S|<>"), dcp=sdcp_clause.default(1)),
        rule("S|<>", (None, "S|<>"), dcp=sdcp_clause.default(1)),
        rule("S|<>"),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 4)) (D 5) (E 6))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>"),
        rule("ROOT", ("S|<>", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)")),
        rule("T|<>"),
        rule("S|<>", ("T|<>", None, "T|<>", "S|<>",), dcp=sdcp_clause.spine("(T 1 0 2)", 3)),
        rule("T|<>"),
        rule("S|<>", (None, "S|<>",), dcp=sdcp_clause.default(1)),
        rule("S|<>"),
    ]

    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("S|<>"),
        rule("ROOT", ("S|<>", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)")),
        rule("T|<>"),
        rule("S|<>", ("T|<>", None, "S|<>", "T|<>"), dcp=sdcp_clause.spine("(T 1 0 3)", 2)),
        rule("S|<>", (None, "S|<>",), dcp=sdcp_clause.default(1)),
        rule("S|<>"),
        rule("T|<>"),
    ]
    
    e = Extractor(hmarkov=0, rightmostunary=False)
    t = Tree("(S (A 0) (B 1) (T (C 2) (D 3) (E 6)) (D 4) (E 5))")
    t[1].type = HEAD
    t[(2,1)].type = HEAD
    t = AutoTree.convert(t)
    assert e(t)[0] == [
        rule("ARG(S)"),
        rule("ROOT", ("ARG(S)", None, "S|<>"), dcp=sdcp_clause.spine("(S 1 0 2)")),
        rule("ARG(T)"),
        rule("S|<>", ("ARG(T)", None, "S|<>", "ARG(T)"), dcp=sdcp_clause.spine("(T 1 0 3)", 2)),
        rule("S|<>", (None, "ARG(S)",), dcp=sdcp_clause.default(1)),
        rule("ARG(S)"),
        rule("ARG(T)"),
    ]

def test_assembly():
    constructors = [
        sdcp_clause.spine(Tree("SBAR+S", [Tree("VP", [1, 0]), 2]))(3),
        sdcp_clause.spine(Tree("VP", [1, 0, 2]))(4),
        sdcp_clause.spine(0)(0),
        sdcp_clause.spine(0)(5),
        sdcp_clause.spine(Tree("NP", [1, 0]))(2),
        sdcp_clause.spine(0)(1),
    ]

    assert constructors[2][0]() == [0]
    assert constructors[3][0]() == [5]
    assert constructors[1][0]([0], [5]) == [Tree("VP", [0, 4, 5])]
    assert constructors[5][0]() == [1]
    assert constructors[4][0]([1]) == [Tree("NP", [1, 2])]
    assert constructors[0][0]([Tree("VP", [0, 4, 5])], [Tree("NP", [1, 2])]) == [Tree("(SBAR (S (VP (VP 0 4 5) 3) (NP 1 2)))")]

    constructors = [
        sdcp_clause.spine(0)(0),
        sdcp_clause.spine("(S 1 0 2)")(1),
        sdcp_clause.spine(0)(2),
        sdcp_clause.spine("(T 1 0 3)", 2)(3),
        sdcp_clause.spine(0, 1)(4),
        sdcp_clause.spine(0)(5),
        sdcp_clause.spine(0)(6),
    ]
    assert constructors[0][0]() == [0]
    assert constructors[2][0]() == [2]
    assert constructors[5][0]() == [5]
    assert constructors[6][0]() == [6]
    
    assert constructors[4][0]([5]) == [4, 5]
    assert constructors[3][0]([2], [4, 5], [6]) == [Tree("(T 2 3 6)"), 4, 5]
    assert constructors[1][0]([0], [Tree("(T 2 3 6)"), 4, 5]) == [Tree("(S 0 1 (T 2 3 6) 4 5)")]