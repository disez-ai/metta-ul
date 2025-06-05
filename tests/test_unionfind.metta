! (import! &self metta_ul:unionfind)

(=
    (UnionFind.init.test)
    (let ($parent $count) (UnionFind.init 2)
        (assertEqual
            (
                (np.assertAllClose $parent (np.arange 2))
                (assertEqual $count 2)
            )
            (() ())
        )
    )
)
(Test UnionFind.init.test)

(=
    (UnionFind.find.test)
    (assertEqual
        (UnionFind.find (UnionFind.init 2) 0)
        0
    )
)
(Test UnionFind.find.test)

(=
    (UnionFind.union.test)
    (let ($parent $count) (UnionFind.union (UnionFind.init 2) 0 1)
        (assertEqual 
            (
                (np.assertAllClose $parent (np.vector 1 1))
                (assertEqual $count 1)
            )
            (() ())
        )
    )
)
(Test UnionFind.union.test)