from sdcp.grammar.composition import fanout, Composition, default_lcfrs, lcfrs_from_positions, union_from_positions
from sdcp.grammar.parser.span import Discospan

def test_fanout():
    assert fanout([1]) == 1
    assert fanout(range(7)) == 1
    assert fanout([0,2]) == 2
    assert fanout([0,1,2,3,6,7,8]) == 2
    assert fanout([0,4,5,6,9]) == 3

def test_spans():
    sp1 = Discospan.from_tuples((1,2), (3,4))
    sp2 = Discospan.from_tuples((2,3))
    sp3 = Discospan.singleton(2)
    sp4 = Discospan.from_tuples((2,3), (4,8))
    sp5 = Discospan.from_tuples((1,2), (3,6))
    
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
    assert default_lcfrs(1) == Composition.lcfrs([0])
    assert default_lcfrs(2) == Composition.lcfrs([0,1])
    assert default_lcfrs(3) == Composition.lcfrs([0,1,2])

    c1 = Composition.lcfrs("010")
    c2 = Composition.lcfrs("0,1,2,3,4")
    c3 = Composition.lcfrs("01,0")
    c4 = Composition.lcfrs('01023,4')
    c5 = Composition.lcfrs('01,232')

    assert c1 == Composition(1, 2, bytes([0,1,0]))
    assert c2 == Composition(5, 5, bytes([0,255,1,255,2,255,3,255,4]))
    assert c3 == Composition(2, 2, bytes([0,1,255,0]))

    assert repr(c1) == "Composition.lcfrs('010')" and eval(repr(c1)) == c1
    assert repr(c3) == "Composition.lcfrs('01,0')" and eval(repr(c3)) == c3

    sp1 = Discospan.from_tuples((1,2), (3,4))
    sp2 = Discospan.from_tuples((2,3))

    assert c1.view().partial(sp2, Discospan.empty()) == sp2
    assert c1.view(0).partial(sp1, sp2) == Discospan((1,4))

    assert c4.view().partial(sp2, Discospan.empty()) == sp2
    assert c4.view(3).partial(sp1, sp2) is None

    assert c5.view().partial(sp2, Discospan.empty()) == sp2
    assert c5.view(2).partial(sp1, sp2) == Discospan((1,4))

    assert Composition.lcfrs("010").view(0).partial(Discospan((14, 20)), Discospan.empty()) is None

    assert Composition.lcfrs("010").view(0).partial(Discospan.from_tuples((0, 13), (15,16)), Discospan((14, 20))) is None
    assert Composition.lcfrs("01").view(0).partial(Discospan.from_tuples((0, 13), (15,16)), Discospan((14, 20))) is None
    assert Composition.lcfrs("012").view(0).partial(Discospan.from_tuples((0, 13), (15,16)),Discospan((13, 14))) is None

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert lcfrs_from_positions(ps, [ps0, ps1])[0] == Composition.lcfrs("0120")
    assert lcfrs_from_positions(ps0, [ps00, ps01])[0] == Composition.lcfrs("0,12")
    assert lcfrs_from_positions(ps00, [])[0] == Composition.lcfrs("0")
    assert lcfrs_from_positions(ps01, [])[0] == Composition.lcfrs("0")
    assert lcfrs_from_positions(ps1, [[1]])[0] == Composition.lcfrs("01")


def test_union_composition():
    c1 = Composition.union()
    c3 = Composition.union(2)

    assert repr(c1) == "Composition.union(1)" and eval(repr(c1)) == c1
    assert repr(c3) == "Composition.union(2)" and eval(repr(c3)) == c3

    sp1 = Discospan.from_tuples((1,2), (3,4))
    sp2 = Discospan((2,3))

    assert c1.view().partial(sp1, sp2) == Discospan((1,4))
    assert c3.view().partial(sp1, sp2) == Discospan((1,4))

    ps = list(range(6))
    ps0 = [0,4,5]
    ps00 = [0]
    ps01 = [5]
    ps1 = [1,2]

    assert union_from_positions(ps, [ps0, ps1]) == (c1, [1,2,0])
    assert union_from_positions(ps0, [ps00, ps01]) == (c3, [1,0,2])
    assert union_from_positions(ps00, []) == (c1, [0])
    assert union_from_positions(ps01, []) == (c1, [0])
    assert union_from_positions(ps1, [[1]]) == (c1, [1,0])