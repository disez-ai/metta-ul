! (import! &self metta_ul:cluster:bisecting_kmeans)
! (ul-import sklearn.metrics as metrics)

;; ========================================================================
;; ======================= Defining test variables ========================
;; ========================================================================
(=
    (X1)
    (np.array
        ((0 0) (1 1))
    )
)

(=
    (indices1)
    (np.array (0 1))
)

(=
    (center1)
    (np.array
        (0.5 0.5)
    )
)

(=
    (X2)
    (np.array ((2 3)))
)

(=
    (indices2)
    (np.array (0))
)

(=
    (center2)
    (np.array (2 3))
)

(=
    (X3)
    (np.array ((0 0) (1 1)))
)

(=
    (indices3)
    (np.array ())
)

(=
    (center3)
    (np.array (0.5 0.5))
)

(=
    (X4)
    (np.array
        ((1 2) (3 4) (5 6))
    )
)

(=
    (clusters1)
    (::
        (py.none py.none 10.0 py.none)
        ()
    )
)

(=
    (clusters2)
    (::
        (py.none py.none 10.0 py.none)
        (::
            (py.none py.none 20.0 py.none)
            (::
                (py.none py.none 5.0 py.none)
                ()
            )
        )
    )
)

(=
    (clusters3)
    (::
        ((np.array (0)) py.none 1.0 py.none)
        (::
            ((np.array (1)) py.none 2.0 py.none)
            (::
                ((np.array (2)) py.none 3.0 py.none)
                ()
            )
        )
    )
)

(=
    (clusters4)
    (::
        ((np.array (1)) py.none 2.0 py.none)
        (::
            ((np.array (2)) py.none 3.0 py.none)
            ()
        )
    )
)

(=
    (clusters5)
    ()
)

(=
    (X5)
    (np.array ((0.0 0.0) (0.0 1.0) (10.0 10.0) (10.0 11.0)))
)

(=
    (clusters6)
    (::
        (
            (np.array (0 1 2 3))
            (np.array (5.0 5.5))
            201.0
            py.none
        )
        ()
    )
)

(=
    (X6)
    (np.array ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0) (5.0 5.0) (5.0 6.0)))
)
;; ========================================================================
;; =================== Test bisecting-kmeans.compute-sse ==================
;; ========================================================================
(=
    (bisecting-kmeans.compute-sse.test1)
    (let
        $sse
        (bisecting-kmeans.compute-sse (X1) (indices1) (center1))
        (np.assertAllClose $sse 1.0)
    )
)
(Test bisecting-kmeans.compute-sse.test1)

(=
    (bisecting-kmeans.compute-sse.test2)
    (let
        $sse
        (bisecting-kmeans.compute-sse (X2) (indices2) (center2))
        (np.assertAllClose $sse 0.0)
    )
)
(Test bisecting-kmeans.compute-sse.test2)

(=
    (bisecting-kmeans.compute-sse.test3)
    (let
        $sse
        (bisecting-kmeans.compute-sse (X3) (indices3) (center3))
        (np.assertAllClose $sse 0.0)
    )
)
(Test bisecting-kmeans.compute-sse.test3)

;; ========================================================================
;; ============= Test bisecting-kmeans.compute-initial-cluster ============
;; ========================================================================

(=
    (bisecting-kmeans.compute-initial-cluster.test1)
    (let
        $init-cluster-list
        (bisecting-kmeans.compute-initial-cluster (X4))
        (assertEqual
            (
                (np.assertAllClose (length $init-cluster-list) 1)
                (np.assertAllClose
                    (bisecting-kmeans.get-cluster-indices (get-first $init-cluster-list))
                    (np.arange (np.shape (X4) 0))
                )
                (np.assertAllClose
                    (bisecting-kmeans.get-cluster-center (get-first $init-cluster-list))
                    (np.array (3.0 4.0))
                )
                (np.assertAllClose
                    (bisecting-kmeans.get-cluster-sse (get-first $init-cluster-list))
                    16.0
                )
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy (get-first $init-cluster-list)))
            )
            (() () () () ())
        )
    )
)
(Test bisecting-kmeans.compute-initial-cluster.test1)

;; ========================================================================
;; ================= Test bisecting-kmeans.find-max-cluster ===============
;; ========================================================================

(=
    (bisecting-kmeans.find-max-cluster.test1)
    (let
        $max-cluster
        (bisecting-kmeans.find-max-cluster (clusters1))
        (assertEqual
            (
                (py.assertIsNone (bisecting-kmeans.get-cluster-indices $max-cluster))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center $max-cluster))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse $max-cluster) 10.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy $max-cluster))
            )
            (() () () ())
        )
    )
)
(Test bisecting-kmeans.find-max-cluster.test1)

(=
    (bisecting-kmeans.find-max-cluster.test2)
    (let
        $max-cluster
        (bisecting-kmeans.find-max-cluster (clusters2))
        (assertEqual
            (
                (py.assertIsNone (bisecting-kmeans.get-cluster-indices $max-cluster))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center $max-cluster))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse $max-cluster) 20.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy $max-cluster))
            )
            (() () () ())
        )
    )
)
(Test bisecting-kmeans.find-max-cluster.test2)

;; ========================================================================
;; ================= Test bisecting-kmeans.remove-cluster =================
;; ========================================================================

(=
    (bisecting-kmeans.remove-cluster.test1)
    (let
        $new-clusters
        (bisecting-kmeans.remove-cluster (clusters3) ((np.array (1)) py.none 2.0 py.none))
        (assertEqual
            (
                (np.assertAllClose (bisecting-kmeans.get-cluster-indices (get-first $new-clusters)) (np.array (0)))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center (get-first $new-clusters)))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse (get-first $new-clusters)) 1.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy (get-first $new-clusters)))

                (np.assertAllClose (bisecting-kmeans.get-cluster-indices (get-second $new-clusters)) (np.array (2)))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center (get-second $new-clusters)))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse (get-second $new-clusters)) 3.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy (get-second $new-clusters)))
            )
            (() () () () () () () ())
        )
    )
)
(Test bisecting-kmeans.remove-cluster.test1)

(=
    (bisecting-kmeans.remove-cluster.test2)
    (let
        $new-clusters
        (bisecting-kmeans.remove-cluster (clusters4) ((np.array (99)) py.none 2.0 py.none))
        (assertEqual
            (
                (np.assertAllClose (bisecting-kmeans.get-cluster-indices (get-first $new-clusters)) (np.array (1)))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center (get-first $new-clusters)))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse (get-first $new-clusters)) 2.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy (get-first $new-clusters)))

                (np.assertAllClose (bisecting-kmeans.get-cluster-indices (get-second $new-clusters)) (np.array (2)))
                (py.assertIsNone (bisecting-kmeans.get-cluster-center (get-second $new-clusters)))
                (np.assertAllClose (bisecting-kmeans.get-cluster-sse (get-second $new-clusters)) 3.0)
                (py.assertIsNone (bisecting-kmeans.get-cluster-hierarchy (get-second $new-clusters)))
            )
            (() () () () () () () ())
        )
    )
)
(Test bisecting-kmeans.remove-cluster.test2)

(=
    (bisecting-kmeans.remove-cluster.test3)
    (let
        $new-clusters
        (bisecting-kmeans.remove-cluster (clusters5) ((np.array (99)) py.none 2.0 py.none))
        (assertEqual
            $new-clusters
            ()
        )
    )
)
(Test bisecting-kmeans.remove-cluster.test3)

;; ========================================================================
;; ================= Test bisecting-kmeans.bisect-cluster =================
;; ========================================================================

(=
    (bisecting-kmeans.bisect-cluster.test1)
    (let
        $new-clusters
        (bisecting-kmeans.bisect-cluster (X5) (bisecting-kmeans.find-max-cluster (clusters6)) 10)
        (assertEqual
            (
                (np.assertAllClose (np.sort (np.append (bisecting-kmeans.get-cluster-indices (get-first $new-clusters)) (bisecting-kmeans.get-cluster-indices (get-second $new-clusters)))) (np.array (0 1 2 3)))
                (np.assertAllClose (np.sort (np.append (bisecting-kmeans.get-cluster-center (get-first $new-clusters)) (bisecting-kmeans.get-cluster-center (get-second $new-clusters)))) (np.array (0.0 0.5 10.0 10.5)))
            )
            (() ())
        )
    )
)
(Test bisecting-kmeans.bisect-cluster.test1)

;; ========================================================================
;; ========== Test bisecting-kmeans.recursive-bisecting-kmeans ============
;; ========================================================================

(=
    (bisecting-kmeans.recursive-bisecting-kmeans.test1)
    (let
        $init-cluster
        (bisecting-kmeans.compute-initial-cluster (X6))
        (let
            $init-hierarchy
            (append py.none $init-cluster)
            (let
                $hierarchy
                (bisecting-kmeans.recursive-bisecting-kmeans (X6) $init-cluster 1 10 $init-hierarchy)
                (np.assertAllClose (len $hierarchy) 1)
            )
        )
    )
)
(Test bisecting-kmeans.recursive-bisecting-kmeans.test1)

(=
    (bisecting-kmeans.recursive-bisecting-kmeans.test2)
    (let
        $init-cluster
        (bisecting-kmeans.compute-initial-cluster (X6))
        (let
            $init-hierarchy
            (append py.none $init-cluster)
            (let
                $hierarchy
                (bisecting-kmeans.recursive-bisecting-kmeans (X6) $init-cluster 3 10 $init-hierarchy)
                (np.assertAllClose (len $hierarchy) 3)
            )
        )
    )
)
(Test bisecting-kmeans.recursive-bisecting-kmeans.test2)

;; ========================================================================
;; ======================= Test bisecting-kmeans.fit ======================
;; ========================================================================

(=
    (bisecting-kmeans.fit.test1)
    (let
        $output
        (bisecting-kmeans.fit (X6) 1 10)
        (np.assertAllClose (len $output) 1)
    )
)
(Test bisecting-kmeans.fit.test1)

(=
    (bisecting-kmeans.fit.test2)
    (let
        $output
        (bisecting-kmeans.fit (X6) 3 10)
        (np.assertAllClose (len $output) 3)
    )
)
(Test bisecting-kmeans.fit.test2)

;; ========================================================================
;; ==================== Test bisecting-kmeans.predict =====================
;; ========================================================================

(=
    (bisecting-kmeans.predict.test1)
    (let
        $hierarchy
        (bisecting-kmeans.fit (X6) 2 10)
        (let
            $labels
            (bisecting-kmeans.predict (X6) $hierarchy)
            (assertEqual
                (
                    (np.assertAllClose (np.shape $labels 0) 6)
                    (np.assertAllClose
                        (metrics.adjusted_rand_score $labels (np.array (0 0 0 0 1 1)))
                        1.0
                    )
                )
                (() ())
            )
        )
    )
)
(Test bisecting-kmeans.predict.test1)
