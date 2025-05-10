! (import! &self metta_ul:cluster:agglomerative)

(=
    (X)
    ((1 0) (0 1))
)

(=
    (distance-matrix1)
    ((0.0 1.0) (1.0 0.0))
)

(=
    (cluster1)
    (py-list (0))
)

(=
    (cluster2)
    (py-list (1))
)

(=
    (clusters1)
    (List.fromExpression (
        (py-list (0))
        (py-list (1))
        (py-list (2))
    ))
)

(=
    (distance-matrix2)
    (
        (0.0 1.0 2.0)
        (1.0 0.0 3.0)
        (2.0 3.0 0.0)
    )
)

(=
    (clusters2)
    (List.fromExpression ((py-list (2)) (py-list (0 1))))
)

(=
    (flat-clusters)
    (List.fromExpression ((py-list (0 1 2))))
)

;; Test agglomerative.distance-matrix
(=
    (agglomerative.distance-matrix.test)
    (np.assertAllClose
        (agglomerative.distance-matrix (np.array (X)))
        (np.array (
            (0 1.414213562373095)
            (1.414213562373095 0)
        ))
    )
)
(Test agglomerative.distance-matrix.test)

;; Test agglomerative.linkage-distance
(=
    (agglomerative.linkage-distance.testSingle)
    (np.assertAllClose
        (agglomerative.linkage-distance (np.array (distance-matrix1)) (cluster1) (cluster2) "single")
        (np.array (1))
    )
)
(Test agglomerative.linkage-distance.testSingle)

(=
    (agglomerative.linkage-distance.testComplete)
    (np.assertAllClose
        (agglomerative.linkage-distance (np.array (distance-matrix1)) (cluster1) (cluster2) "complete")
        (np.array (1))
    )
)
(Test agglomerative.linkage-distance.testComplete)

(=
    (agglomerative.linkage-distance.testAverage)
    (np.assertAllClose
        (agglomerative.linkage-distance (np.array (distance-matrix1)) (cluster1) (cluster2) "average")
        (np.array (1))
    )
)
(Test agglomerative.linkage-distance.testAverage)
    
;; Test agglomerative.closest-clusters
(=
    (agglomerative.closest-clusters.test)
    (assertEqual
        (agglomerative.closest-clusters (clusters1) (np.array (distance-matrix2)) "single" py.pinf ())
        ((cluster1) (cluster2))
    )
)
(Test agglomerative.closest-clusters.test)



;; Test agglomerative.merge-clusters
(=
    (agglomerative.merge-clusters.test)
    (assertEqual
        (agglomerative.merge-clusters (clusters1) (np.array (distance-matrix2)) "single")
        (clusters2)
    )
)
(Test agglomerative.merge-clusters.test)

;; Test agglomerative.init-clusters
(=
    (agglomerative.init-clusters.test)
    (assertEqual
        (agglomerative.init-clusters 2)
        (List.fromExpression ((cluster2) (cluster1)))
    )
)
(Test agglomerative.init-clusters.test)


;; Test agglomerative.recursion
(=
    (agglomerative.recursion.testNoRecursion)
    (assertEqual
        (agglomerative.recursion 1 "single" (flat-clusters) (np.array (distance-matrix2)) 1)
        (flat-clusters)
    )
)
(Test agglomerative.recursion.testNoRecursion)

(=
    (agglomerative.recursion.testOneRecursion)
    (assertEqual
        (agglomerative.recursion 1 "single" (clusters2) (np.array (distance-matrix2)) 2)
        (List.fromExpression ((py-list (2 0 1))))
    )
)
(Test agglomerative.recursion.testOneRecursion)

;; Test agglomerative
(=
    (agglomerative.test)
    (assertEqual
        (agglomerative (np.array (X)) 1 "single")
        (List.fromExpression ((py-list (1 0))))
    )
)
(Test agglomerative.test)

;; Test agglomerative.fit-predict
(=
    (agglomerative.fit-predict.test)
    (np.assertAllClose
        (agglomerative.fit-predict (np.array (X)) 1 "single")
        (np.array (0 0))
    )
)
(Test agglomerative.fit-predict.test)