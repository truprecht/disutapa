from sdcp.grammar.lcfrs import *

def test_fanout():
    assert fanout([1]) == 1
    assert fanout(range(7)) == 1
    assert fanout([0,2]) == 2
    assert fanout([0,1,2,3,6,7,8]) == 2
    assert fanout([0,4,5,6,9]) == 3

def test_spans():
    sp1 = disco_span((1,2), (3,4))
    sp2 = disco_span((2,3))
    sp3 = disco_span.singleton(2)
    sp4 = disco_span((2,3), (4,8))
    sp5 = disco_span((1,2), (3,6))
    
    assert sp1.len == 2 and sp2.len == 1
    assert list(sp1.spans) == [1,2,3,4]
    assert list(sp2.spans) == [2,3]
    
    assert sp1.exclusive_union(sp2) == disco_span((1,4))
    assert sp2.exclusive_union(sp3) is None
    assert sp1.exclusive_union(sp4) == disco_span((1,8))
    assert sp5.exclusive_union(sp4) is None

    assert repr(sp1) == "disco_span((1, 2), (3, 4))"
    assert repr(sp3) == "disco_span((2, 3))"

    assert sp1 < sp2
    assert sp1 < sp4
    assert sp1 < sp5


def test_composition():
    c1 = lcfrs_composition("010")
    c2 = lcfrs_composition("0,1,2,3,4")
    c3 = lcfrs_composition("01,0")
    c4 = lcfrs_composition('01023,4')

    assert c1.inner == bytes((0,1,0))
    assert c2.inner == bytes((0,255,1,255,2,255,3,255,4))
    assert c3.inner == bytes((0,1,255,0))

    assert repr(c1) == "lcfrs_composition('010')" and eval(repr(c1)) == c1
    assert repr(c3) == "lcfrs_composition('01,0')" and eval(repr(c3)) == c3

    sp1 = disco_span((1,2), (3,4))
    sp2 = disco_span((2,3))
    sp4 = disco_span((2,3), (4,8))
    sp5 = disco_span((1,2), (3,6))

    assert c1(sp1, sp2) == disco_span((1,4))
    assert c3(sp1, sp2) is None
    assert lcfrs_composition("0101")(sp1, sp4) == disco_span((1,8))
    assert lcfrs_composition("0101")(sp1, sp5) is None

    assert c1.partial(sp1, sp2) == (disco_span((1,4)), lcfrs_composition('0'))
    assert c4.partial(sp1, sp2) == (disco_span((1,4)), lcfrs_composition('012,3'))

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert lcfrs_composition.from_positions(ps, [ps0, ps1]) == lcfrs_composition("1201")
    assert lcfrs_composition.from_positions(ps0, [ps00, ps01]) == lcfrs_composition("1,02")
    assert lcfrs_composition.from_positions(ps00, []) == lcfrs_composition("0")
    assert lcfrs_composition.from_positions(ps01, []) == lcfrs_composition("0")
    assert lcfrs_composition.from_positions(ps1, [[1]]) == lcfrs_composition("10")


def test_union_composition():
    c1 = ordered_union_composition("01")
    c2 = ordered_union_composition("01234", fanout=5)
    c3 = ordered_union_composition("01", fanout=2)
    c4 = ordered_union_composition('01234', fanout=2)

    assert c1.order_and_fanout == bytes((0,1,1))
    assert c2.order_and_fanout == bytes((0,1,2,3,4,5))
    assert c3.order_and_fanout == bytes((0,1,2))
    assert c4.order_and_fanout == bytes((0,1,2,3,4,2))

    assert repr(c1) == "ordered_union_composition('01')" and eval(repr(c1)) == c1
    assert repr(c3) == "ordered_union_composition('01', fanout=2)" and eval(repr(c3)) == c3

    sp1 = disco_span((1,2), (3,4))
    sp2 = disco_span((2,3))
    sp4 = disco_span((2,3), (4,8))
    sp5 = disco_span((1,2), (3,6))

    assert c1(sp1, sp2) == disco_span((1,4))
    assert c3(sp1, sp2) is None
    assert c1(sp1, sp4) == disco_span((1,8))
    assert c1(sp1, sp5) is None

    assert c1.partial(sp1, sp2) == (disco_span((1,4)), ordered_union_composition('01'))
    assert c4.partial(sp1, sp2) == (disco_span((1,4)), ordered_union_composition('01234', fanout=2))

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert ordered_union_composition.from_positions(ps, [ps0, ps1]) == ordered_union_composition("120")
    assert ordered_union_composition.from_positions(ps0, [ps00, ps01]) == ordered_union_composition("102", fanout=2)
    assert ordered_union_composition.from_positions(ps00, []) == ordered_union_composition("0")
    assert ordered_union_composition.from_positions(ps01, []) == ordered_union_composition("0")
    assert ordered_union_composition.from_positions(ps1, [[1]]) == ordered_union_composition("10")