from discodop.tree import ImmutableTree, ParentedTree, Tree
from sdcp.reranking import classifier, pairwise, boost


def test_extraction():
    tree = Tree("(S (S (A 0) (B 1)) (S (A 2) (C 3) (S (D 4))))")
    extractor = classifier.FeatureExtractor()

    vec = extractor.extract(tree)

    assert set(extractor.objects.keys()) == {"rules", "parent_rules", "bigrams", "parent_bigrams", "rightbranch", "ranks", "branching_direction"}
    assert set(extractor.objects["rules"]) == {("S", "S", "S"), ("S", "A", "B"), ("S", "A", "C", "S"), ("S", "D")}
    assert set(extractor.objects["parent_rules"]) == {("TOP", "S", "S", "S"), ("S", "S", "A", "B"), ("S", "S", "A", "C", "S"), ("S", "S", "D")}
    assert set(extractor.objects["bigrams"]) == {("S", "S", "END"), ("S", "S", "S"), ("S", "A", "B"), ("S", "B", "END"), ("S", "A", "C"), ("S", "C", "S"), ("S", "S", "END"), ("S", "D", "END")}
    assert set(extractor.objects["parent_bigrams"]) == {("TOP", "S", "S", "END"), ("TOP", "S", "S", "S"), ("S", "S", "A", "B"), ("S", "S", "B", "END"), ("S", "S", "A", "C"), ("S", "S", "C", "S"), ("S", "S", "S", "END"), ("S", "S", "D", "END")}
    assert set(extractor.objects["ranks"]) == { ("S", 2), ("S", 3), ("S", 1) }
    assert set(extractor.objects["branching_direction"]) == { ("S", "Multi", 2), ("S", "Right"), ("S", "None") }

    features = { (k, i): 1 for k in extractor.objects for i in range(len(extractor.objects[k])) }
    features[("bigrams", extractor.objects["bigrams"][("S", "S", "END")])] = 2
    features[("ranks", extractor.objects["ranks"][("S", 2)])] = 2
    features[("branching_direction", extractor.objects["branching_direction"][("S", "None")])] = 2
    features[("rightbranch", 0)] = 4
    features[("rightbranch", 1)] = 5

    assert  vec.features == features

    extractor.truncate(mincount=2)
    assert vec.tup(extractor) == (2,4,5,2,2)


def test_ranker():
    Tree = ImmutableTree
    goldtrees = [
        Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"),
        Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))")
    ]
    silvertrees = [
        [
            (Tree("(SBAR (S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5))))"), 1.0),
            (Tree("(SBAR (S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBD 3) (VBN 4) (RP 5))) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2)))"), 1.0),
        ],
        [
            (Tree("(S (VP (NP (DT 0) (NN 1) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (DT 0) (NN 1)) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NN 7))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7)))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 7)))) (VBN 3) (NP (NN 6))) (VBZ 2))"), 3.7),
        ]
    ]
    goldindices = [2, 6]

    ranker = classifier.TreeRanker(min_feature_count=1)
    for gold, silvers in zip(goldtrees, silvertrees):
        ranker.add_tree(gold, silvers)
    
    ranker.fit(20, devset=list(zip(silvertrees, goldtrees)))
    for gold, goldidx, silvers in zip(goldtrees, goldindices, silvertrees):
        i, tree = ranker.select(silvers)
        assert i == goldidx, f"chose tree {tree} instead of {gold}"


def test_pairwise():
    Tree = ImmutableTree
    goldtrees = [
        Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"),
        Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))")
    ]
    silvertrees = [
        [
            (Tree("(SBAR (S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5))))"), 1.0),
            (Tree("(SBAR (S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBD 3) (VBN 4) (RP 5))) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2)))"), 1.0),
        ],
        [
            (Tree("(S (VP (NP (DT 0) (NN 1) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (DT 0) (NN 1)) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NN 7))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7)))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 7)))) (VBN 3) (NP (NN 6))) (VBZ 2))"), 3.7),
        ]
    ]
    goldindices = [2, 6]

    ranker = pairwise.PairwiseTreeRanker(min_feature_count=1)
    for gold, silvers in zip(goldtrees, silvertrees):
        ranker.add_tree(gold, silvers)
    
    ranker.fit(20, 0.0, devset=list(zip(silvertrees, goldtrees)))
    for gold, goldidx, silvers in zip(goldtrees, goldindices, silvertrees):
        i, tree = ranker.select(silvers)
        assert i == goldidx, f"chose tree {tree} instead of {gold}"

def test_boost():
    Tree = ImmutableTree
    goldtrees = [
        Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"),
        Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))")
    ]
    silvertrees = [
        [
            (Tree("(SBAR (S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5))))"), 1.0),
            (Tree("(SBAR (S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(SBAR (S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2))))"), 1.0),
            (Tree("(S (VP (WRB 0) (NP (PT 1) (NN 2)) (VBD 3) (VBN 4) (RP 5)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBN 4) (RP 5)) (VBD 3)) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (VP (WRB 0) (VBD 3) (VBN 4) (RP 5))) (NP (PT 1) (NN 2)))"), 1.0),
            (Tree("(S (VP (WRB 0) (VBD 3) (VBN 4) (RP 5)) (NP (PT 1) (NN 2)))"), 1.0),
        ],
        [
            (Tree("(S (VP (NP (DT 0) (NN 1) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (DT 0) (NN 1)) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NN 7))"), 1.0),
            (Tree("(S (NP (DT 0) (NN 1)) (VBZ 2) (VBN 3) (PP (IN 4) (NP (DT 5) (NN 6))) (NP (NN 7)))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6) (NN 7)))) (VBN 3)) (VBZ 2))"), 0.9),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 6)))) (VBN 3) (NP (NN 7))) (VBZ 2))"), 1.0),
            (Tree("(S (VP (NP (NP (DT 0) (NN 1)) (PP (IN 4) (NP (DT 5) (NN 7)))) (VBN 3) (NP (NN 6))) (VBZ 2))"), 3.7),
        ]
    ]
    goldindices = [2, 6]

    ranker = boost.BoostTreeRanker(min_feature_count=2)
    for gold, silvers in zip(goldtrees, silvertrees):
        ranker.add_tree(gold, silvers)
    
    ranker.fit(10, devset=list(zip(silvertrees, goldtrees)))
    for gold, goldidx, silvers in zip(goldtrees, goldindices, silvertrees):
        i, tree = ranker.select(silvers)
        assert i == goldidx, f"chose tree {tree} instead of {gold}"