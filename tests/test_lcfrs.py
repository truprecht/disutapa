from sdcp.grammar.lcfrs import fanout, lcfrs_composition, ordered_union_composition
from sdcp.grammar.parser.span import Discospan, singleton

def test_fanout():
    assert fanout([1]) == 1
    assert fanout(range(7)) == 1
    assert fanout([0,2]) == 2
    assert fanout([0,1,2,3,6,7,8]) == 2
    assert fanout([0,4,5,6,9]) == 3

def test_spans():
    sp1 = Discospan((1,2), (3,4))
    sp2 = Discospan((2,3))
    sp3 = singleton(2)
    sp4 = Discospan((2,3), (4,8))
    sp5 = Discospan((1,2), (3,6))
    
    assert len(sp1) == 2 and len(sp2) == 1
    
    assert sp1.exclusive_union(sp2) == Discospan((1,4))
    assert sp2.exclusive_union(sp3) is None
    assert sp1.exclusive_union(sp4) == Discospan((1,8))
    assert sp5.exclusive_union(sp4) is None

    print(sp1, sp2, sp3, sp4, sp5)
    assert sp1 < sp2
    assert sp1 < sp4
    assert sp1 < sp5


def test_composition():
    assert lcfrs_composition.default(1) == lcfrs_composition("0")
    assert list(lcfrs_composition.default(1).inner) == [0]
    assert str(lcfrs_composition.default(1)) == "'0'"
    assert lcfrs_composition.default(2) == lcfrs_composition("01")
    assert list(lcfrs_composition.default(2).inner) == [0,1]
    assert lcfrs_composition.default(3) == lcfrs_composition("012")
    assert list(lcfrs_composition.default(3).inner) == [0,1,2]

    c1 = lcfrs_composition("010")
    c2 = lcfrs_composition("0,1,2,3,4")
    c3 = lcfrs_composition("01,0")
    c4 = lcfrs_composition('01023,4')
    c5 = lcfrs_composition('01,232')

    assert c1.inner == bytes((0,1,0))
    assert c2.inner == bytes((0,255,1,255,2,255,3,255,4))
    assert c3.inner == bytes((0,1,255,0))

    assert repr(c1) == "lcfrs_composition('010')" and eval(repr(c1)) == c1
    assert repr(c3) == "lcfrs_composition('01,0')" and eval(repr(c3)) == c3

    sp1 = Discospan((1,2), (3,4))
    sp2 = Discospan((2,3))

    assert c1.partial(sp1, sp2) == (Discospan((1,4)), lcfrs_composition('0'))
    assert c4.partial(sp1, sp2) == (None, None)
    assert c5.partial(sp1, sp2) == (Discospan((1,4)), lcfrs_composition('01,2'))

    assert lcfrs_composition("010").partial(Discospan((0, 13), (15,16)), Discospan((14, 20))) == (None, None)
    assert lcfrs_composition("01").partial(Discospan((0, 13), (15,16)), Discospan((14, 20))) == (None, None)
    assert lcfrs_composition("012").partial(Discospan((0, 13), (15,16)),Discospan((13, 14))) == (None, None)

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert lcfrs_composition.from_positions(ps, [ps0, ps1])[0] == lcfrs_composition("0120")
    assert lcfrs_composition.from_positions(ps0, [ps00, ps01])[0] == lcfrs_composition("0,12")
    assert lcfrs_composition.from_positions(ps00, [])[0] == lcfrs_composition("0")
    assert lcfrs_composition.from_positions(ps01, [])[0] == lcfrs_composition("0")
    assert lcfrs_composition.from_positions(ps1, [[1]])[0] == lcfrs_composition("01")


def test_union_composition():
    c1 = ordered_union_composition()
    c3 = ordered_union_composition(fanout=2)

    assert repr(c1) == "ordered_union_composition(fanout=1)" and eval(repr(c1)) == c1
    assert repr(c3) == "ordered_union_composition(fanout=2)" and eval(repr(c3)) == c3

    sp1 = Discospan((1,2), (3,4))
    sp2 = Discospan((2,3))

    assert c1.partial(sp1, sp2) == (Discospan((1,4)), ordered_union_composition())
    assert c3.partial(sp1, sp2) == (Discospan((1,4)), ordered_union_composition(fanout=2))

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert ordered_union_composition.from_positions(ps, [ps0, ps1]) == (ordered_union_composition(), [1,2,0])
    assert ordered_union_composition.from_positions(ps0, [ps00, ps01]) == (ordered_union_composition(fanout=2), [1,0,2])
    assert ordered_union_composition.from_positions(ps00, []) == (ordered_union_composition(), [0])
    assert ordered_union_composition.from_positions(ps01, []) == (ordered_union_composition(), [0])
    assert ordered_union_composition.from_positions(ps1, [[1]]) == (ordered_union_composition(), [1,0])