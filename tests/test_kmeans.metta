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
    (assertEqual
        (np.allclose (kmeans.update (np.array (X)) (np.array (assignments))) (np.array (X)))
        True
    )
)
(Test kmeans.update.test)

;; Test kmeans.assign
(=
    (kmeans.assign.test)
    (assertEqual
        (np.allclose (kmeans.assign (np.array (X)) (np.array (centroids))) (np.array (assignments)))
        True
    )
)
(Test kmeans.assign.test)

;; Test kmeans.recursion
(=
    (kmeans.recursion.testNoRecursion)
    (assertEqual
        (np.allclose (kmeans.recursion (np.array (X)) (np.array (centroids)) 0) (np.array (centroids)))
        True
    )
)
(Test kmeans.recursion.testNoRecursion)

(=
    (kmeans.recursion.testOneRecursion)
    (assertEqual
        (np.allclose (kmeans.recursion (np.array (X)) (np.array (centroids)) 1) (np.array (centroids)))
        True
    )
)
(Test kmeans.recursion.testOneRecursion)

;; Test kmeans
(=
    (kmeans.test)
    (assertEqual
        (np.allclose (kmeans (np.array (X)) 2) (np.array (centroids)))
        True
    )
)
(Test kmeans.test)