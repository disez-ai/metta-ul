! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

(=
    (X)
    ((1.0 0.0) (0.0 1.0))
)

(=
    (assignments)
    ((1.0 0.0) (0.0 1.0))
)

(=
    (centroids)
    ((1.0 0.0) (0.0 1.0))
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
        (kmeans.recursion (np.array (X)) (np.array (centroids)) 0 0.0001)
        (np.array (centroids))
    )
)
(Test kmeans.recursion.testNoRecursion)

(=
    (kmeans.recursion.testOneRecursion)
    (np.assertAllClose
        (kmeans.recursion (np.array (X)) (np.array (centroids)) 1 0.0001)
        (np.array (centroids))
    )
)
(Test kmeans.recursion.testOneRecursion)

;; Test kmeans.fit
(=
    (kmeans.fit.test)
    (let $_ (np.random.seed 1)
    (np.assertAllClose
        (kmeans.fit (np.array (X)) 2)
        (np.array (centroids))
    ))
)
(Test kmeans.fit.test)

;; Test kmeans.predict
(=
    (kmeans.predict.test)
    (np.assertAllClose
        (kmeans.predict (np.array (X)) (np.array (centroids)))
        (np.array (0 1))
    )
)
(Test kmeans.predict.test)