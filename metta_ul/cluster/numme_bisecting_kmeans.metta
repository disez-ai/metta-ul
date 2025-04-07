!(import! &self metta_ul:numme)

! (bind! np (py-atom numpy))
! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

; ================= Utils =================
(=
    (py-getitem $iterable $index)
    (
        (py-dot $iterable __getitem__)
        $index
    )
)
(=
    (to-np $x)
    ((py-dot np array) $x)
)
; =========================================

; TODO: Fix the import problem to include kmeans from kmeans.metta
; ============== KMEANS ==============
(=
    (kmeans.update $X $assignments)
    (np.div
        (np.matmul $assignments $X)
        (np.sum
            $assignments
            1
            pyNone
            pyNone
            pyTrue
        )
    )
)

(=
    (kmeans.assign $X $centroids $k)
    (np.transpose
        (np.one_hot
            (np.argmin
                (np.linalg.norm
                    (np.sub
                        (np.expand_dims $X 0)
                        (np.expand_dims $centroids 1)
                    )
                    pyNone
                    2
                )
                0
            )
            $k
        )
    )
)

(=
    (kmeans.recursion $X $centroids $k $max-iter)
    (if (> $max-iter 0)
        (kmeans.recursion
            $X
            (kmeans.update
                $X
                (kmeans.assign $X $centroids $k)
            )
            $k
            (- $max-iter 1)
        )
        $centroids
    )
)

(=
    (kmeans $X $k)
    (kmeans $X $k 100)
)

(=
    (kmeans $X $k $max-iter)
    (kmeans.recursion $X (np.choose $X $k) $k $max-iter)
)

; ===============================================

(=
    (compute-sse $X $indices $centers)
    (if (> (py-dot $indices size) 0)
        ((py-dot
            (np.power
                (np.sub
                    ((py-dot $X __getitem__) $indices)
                    $centers
                )
                2
            )
            sum
        ))
        0
    )
)

(=
    (compute-initial-cluster $X)
    (::
        (
            ; indices
            (np.arange
                (py-getitem (py-dot $X shape) 0)
            )

            ; centers
            ((py-dot $X mean) 0)

            ; sse
            (compute-sse
                $X
                (np.arange
                    (py-getitem (py-dot $X shape) 0)
                )
                ((py-dot $X mean) 0)
            )

            ; hierarchy
            pyNone
        )
        ()
    )
)

(=
    (get-cluster-indices ($indices $center $sse $hierarchy))
    $indices
)
(=
    (get-cluster-center ($indices $center $sse $hierarchy))
    $center
)
(=
    (get-cluster-sse ($indices $center $sse $hierarchy))
    $sse
)
(=
    (get-cluster-hierarchy ($indices $center $sse $hierarchy))
    $hierarchy
)

(=
    (length ())
    0
)
(=
    (length (:: ($indices $center $sse $hierarchy) $xs))
    (+ 1 (length $xs))
)
(=
    (head (:: ($indices $center $sse $hierarchy) $xs))
    ($indices $center $sse $hierarchy)
)
(=
    (tail (:: ($indices $center $sse $hierarchy) $xs))
    $xs
)


(=
    (find-max-cluster (:: ($indices $center $sse $hierarchy) ()) )
    ($indices $center $sse $hierarchy)
)

(=
    (find-max-cluster (:: ($indices1 $center1 $sse1 $hierarchy1) (:: ($indices2 $center2 $sse2 $hierarchy2) $xs)))
    (if (> $sse1 (get-cluster-sse (find-max-cluster (:: ($indices2 $center2 $sse2 $hierarchy2) $xs))))
         ($indices1 $center1 $sse1 $hierarchy1)
         (find-max-cluster (:: ($indices2 $center2 $sse2 $hierarchy2) $xs))
    )
)


(=
    (cluster-equal ($indices1 $center1 $sse1 $hierarchy1) ($indices2 $center2 $sse2 $hierarchy2))
    (if
        (and
            (and
                ((py-dot np array_equal) $indices1 $indices2)
                ((py-dot np array_equal) $center2 $center2)
            )
            (== $sse1 $sse2)
        )
        True
        False
    )
)
(=
    (remove-cluster () $target)
    ()
)
(=
    (remove-cluster (:: ($indices $center $sse $hierarchy) $xs) $target)
    (if (cluster-equal ($indices $center $sse $hierarchy) $target)
        (remove-cluster $xs $target)
        (:: ($indices $center $sse $hierarchy) (remove-cluster $xs $target))
    )
)

(=
    (get-first (:: $x $y))
    $x
)
(=
    (get-second (:: $x (:: $y ())))
    $y
)
(=
    (bisect-cluster $X $max-cluster $max-iter)
    (let
        ; compute the clusters
        $centers
        (kmeans (np.take $X (get-cluster-indices $max-cluster) 0) 2)

        ; output tuple
        (::
            ; cluster 0
            (
                (np.slice
                    ((py-dot np where)
                        (np.equal
                            (np.argmax
                                (kmeans.assign $X $centers 2)
                                0
                            )
                            0
                        )
                    )
                    [0]
                )
                (np.slice $centers [0])
                (compute-sse
                    $X
                    (np.slice
                        ((py-dot np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign $X $centers 2)
                                    0
                                )
                                0
                            )
                        )
                        [0]
                    )
                    (np.slice $centers [0])
                )
                pyNone
            )
            (::
                ; cluster 1
                (
                    (np.slice
                        ((py-dot np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign $X $centers 2)
                                    0
                                )
                                1
                            )
                        )
                        [0]
                    )
                    (np.slice $centers [1])
                    (compute-sse
                        $X
                        (np.slice
                            ((py-dot np where)
                                (np.equal
                                    (np.argmax
                                        (kmeans.assign $X $centers 2)
                                        0
                                    )
                                    1
                                )
                            )
                            [0]
                        )
                        (np.slice $centers [1])
                    )
                    pyNone
                )
                ()
            )
        )
    )
)

(=
    (extend-clusters () ($indices0 $center0 $sse0 $hierarchy0) ($indices1 $center1 $sse1 $hierarchy1))
    (::
        ($indices0 $center0 $sse0 $hierarchy0)
        (::
            ($indices1 $center1 $sse1 $hierarchy1)
            ()
        )
    )
)
(=
    (extend-clusters () () ($indices1 $center1 $sse1 $hierarchy1))
    (::
        ($indices1 $center1 $sse1 $hierarchy1)
        ()
    )
)
(=
    (extend-clusters () ($indices0 $center0 $sse0 $hierarchy0) ())
    (::
        ($indices0 $center0 $sse0 $hierarchy0)
        ()
    )
)
(=
    (extend-clusters () () ())
    ()
)
(=
    (extend-clusters (:: ($indices $center $sse $hierarchy) $xs) () ($indices1 $center1 $sse1 $hierarchy1))
    (::
        ($indices $center $sse $hierarchy)
        (::
            $xs
            (::
                ($indices1 $center1 $sse1 $hierarchy1)
                ()
            )
        )
    )
)
(=
    (extend-clusters (:: ($indices $center $sse $hierarchy) $xs) ($indices0 $center0 $sse0 $hierarchy0) ())
    (::
        ($indices $center $sse $hierarchy)
        (::
            $xs
            (::
                ($indices0 $center0 $sse0 $hierarchy0)
                ()
            )
        )
    )
)
(=
    (extend-clusters (:: ($indices $center $sse $hierarchy) $xs) () ())
    (:: ($indices $center $sse $hierarchy) $xs)
)
(=
    (extend-clusters () ($indices0 $center0 $sse0 $hierarchy0) ($indices1 $center1 $sse1 $hierarchy1))
    (::
        ($indices0 $center0 $sse0 $hierarchy0)
        (::
            ($indices1 $center1 $sse1 $hierarchy1)
            ()
        )
    )
)
(=
    (extend-clusters (:: ($indices $center $sse $hierarchy) $xs) ($indices0 $center0 $sse0 $hierarchy0) ($indices1 $center1 $sse1 $hierarchy1))
    (::
        ($indices $center $sse $hierarchy)
        (::
            $xs
            (::
                ($indices0 $center0 $sse0 $hierarchy0)
                (::
                    ($indices1 $center1 $sse1 $hierarchy1)
                    ()
                )
            )
        )
    )
)


(=
    (append-to-hierarchy pyNone (:: ($indices $center $sse $hierarchy) $xs))
    (::
        (:: ($indices $center $sse $hierarchy) $xs)
        ()
    )
)

(=
    (append-to-hierarchy (:: (:: ($indices $center $sse $hierarchy) $xs) $CS) (:: ($indices1 $center1 $sse1 $hierarchy1) $xs1))
    (::
        (:: ($indices $center $sse $hierarchy) $xs)
        (::
            $CS
            (::
                (:: ($indices1 $center1 $sse1 $hierarchy1) $xs1)
                ()
            )
        )
    )
)


(=
    (bisecting-kmeans $X $clusters $max-cluster $max-iter $hierarchy)
    ;max_cluster = find_max_cluster(clusters)
    ;(find-max-cluster $clusters)

    ;clusters = remove_cluster(clusters, max_cluster)
    ;(remove-cluster $clusters (find-max-cluster $clusters))

    ;child0, child1 = bisect_cluster(data, max_cluster, n_init)
    ;(bisect-cluster $X (find-max-cluster $clusters) $max-iter)
    (if (>= (length $clusters) $max-cluster)
        $hierarchy

        ;(remove-cluster $clusters (find-max-cluster $clusters))
        (let
            $bisect
            (bisect-cluster $X (find-max-cluster $clusters) $max-iter)

            (bisecting-kmeans
                $X
                (extend-clusters
                    (remove-cluster $clusters (find-max-cluster $clusters))
                    (get-first $bisect)
                    (get-second $bisect)
                )
                $max-cluster
                $max-iter
                (append-to-hierarchy
                    $hierarchy
                    (extend-clusters
                        (remove-cluster $clusters (find-max-cluster $clusters))
                        (get-first $bisect)
                        (get-second $bisect)
                    )
                )
            )

       )
    )
)
