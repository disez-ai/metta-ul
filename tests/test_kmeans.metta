! (import! &self metta_ul:cluster:kmeans)
! (np.random.seed 1)

(=
    (X)
    ((1 0) (0 1))
)

(=
    (assignments)
    ((1 0) (0 1))
)

(=
    (centroids)
    ((1 0) (0 1))
)

;; Test kmeans.update
(=
    (kmeans.update.test)
    (np.assertAllClose
        (kmeans.update (np.array (X)) (np.array (assignments)))
        (np.array (X))
    )
)
(Test kmeans.update.test)

;; Test kmeans.assign
(=
    (kmeans.assign.test)
    (np.assertAllClose
        (kmeans.assign (np.array (X)) (np.array (centroids)))
        (np.array (assignments))
    )
)
(Test kmeans.assign.test)

;; Test kmeans.recursion
(=
    (kmeans.recursion.testNoRecursion)
    (np.assertAllClose
        (kmeans.recursion (np.array (X)) (np.array (centroids)) 0)
        (np.array (centroids))
    )
)
(Test kmeans.recursion.testNoRecursion)

(=
    (kmeans.recursion.testOneRecursion)
    (np.assertAllClose
        (kmeans.recursion (np.array (X)) (np.array (centroids)) 1)
        (np.array (centroids))
    )
)
(Test kmeans.recursion.testOneRecursion)

;; Test kmeans
(=
    (kmeans.test)
    (np.assertAllClose
        (kmeans (np.array (X)) 2)
        (np.array (centroids))
    )
)
(Test kmeans.test)