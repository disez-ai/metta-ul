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
    (np.vector 0)
)

(=
    (cluster2)
    (np.vector 1)
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

;; Test agglomerative.heapify
(=
    (agglomerative.heapify.test)
    (assertEqual
        (agglomerative.heapify (np.array (distance-matrix2)))
        (Heap.Node 1 (1.0 0 1) (Heap.Node 1 (1.0 1 0) Nil (Heap.Node 0 (2.0 2 0) Nil Nil)) Nil)
    )
)
(Test agglomerative.heapify.test)

;; Test agglomerative.heappush
(=
    (agglomerative.heappush.test)
    (assertEqual
        (agglomerative.heappush
            Nil
            (np.array (distance-matrix1))
            "single"
            ((np.vector 0 1) 2)
            0
        )
        (Heap.Node 0 (1.0 0 1) Nil Nil)
    )
)
(Test agglomerative.heappush.test)

;; Test agglomerative.cluster
(=
    (agglomerative.cluster.testTwoClusters)
    (assertEqual
        (agglomerative.cluster (np.array (X)) 2 "single")
        ((np.vector 0 1) 2)
    )
)
(Test agglomerative.cluster.testTwoClusters)

(=
    (agglomerative.cluster.testOneCluster)
    (assertEqual
        (agglomerative.cluster (np.array (X)) 1 "single")
        ((np.vector 0 0) 1)
    )
)
(Test agglomerative.cluster.testOneCluster)

;; Test agglomerative.fit-predict
(=
    (agglomerative.fit-predict.test)
    (np.assertAllClose
        (agglomerative.fit-predict (np.array (X)) 1 "single")
        (np.array (0 0))
    )
)
(Test agglomerative.fit-predict.test)