!(import! &self metta_ul)

! (bind! np (py-atom numpy))
! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))
! (bind! pyINF (py-atom "float('inf')"))

; ================= Utils =================
(=
    (py-getitem $iterable $index)
    (
        (py-dot $iterable __getitem__)
        $index
    )
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
    (get-first (:: $x $y))
    $x
)
(=
    (get-second (:: $x (:: $y ())))
    $y
)
(=
    (append pyNone $a)
    (::
        $a
        ()
    )
)
(=
    (append () $a)
    (::
        $a
        ()
    )
)
(=
    (append (:: $x $xs) $a)
    (::
        $x
        (append $xs $a)
    )
)
(=
    (get-index () $idx)
    pyNone
)

(=
    (get-index (:: $x $xs) $idx)
    (if (== $idx 0)
        $x
        (get-index $xs (- $idx 1))
    )
)
(=
    (get-last ())
    pyNone
)
(=
    (get-last (:: $x ()))
    $x
)
(=
    (get-last (:: $x (:: $y $ys)))
    (get-last (:: $y $ys))
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
    (bisecting-kmeans.compute-sse $X $indices $centers)
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
    (bisecting-kmeans.compute-initial-cluster $X)
    (::
        (
            ; indices
            (np.arange
                (py-getitem (py-dot $X shape) 0)
            )

            ; centers
            ((py-dot $X mean) 0)

            ; sse
            (bisecting-kmeans.compute-sse
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
    (bisecting-kmeans.get-cluster-indices ($indices $center $sse $hierarchy))
    $indices
)
(=
    (bisecting-kmeans.get-cluster-center ($indices $center $sse $hierarchy))
    $center
)
(=
    (bisecting-kmeans.get-cluster-sse ($indices $center $sse $hierarchy))
    $sse
)
(=
    (bisecting-kmeans.get-cluster-hierarchy ($indices $center $sse $hierarchy))
    $hierarchy
)

(=
    (bisecting-kmeans.find-max-cluster (:: ($indices $center $sse $hierarchy) ()) )
    ($indices $center $sse $hierarchy)
)

(=
    (bisecting-kmeans.find-max-cluster
        (:: ($indices1 $center1 $sse1 $hierarchy1)
        (:: ($indices2 $center2 $sse2 $hierarchy2) $xs))
    )
    (if (>
        $sse1
        (bisecting-kmeans.get-cluster-sse
            (bisecting-kmeans.find-max-cluster (:: ($indices2 $center2 $sse2 $hierarchy2) $xs))
        )
    )
         ($indices1 $center1 $sse1 $hierarchy1)
         (bisecting-kmeans.find-max-cluster (:: ($indices2 $center2 $sse2 $hierarchy2) $xs))
    )
)

(=
    (bisecting-kmeans.cluster-equal
        ($indices1 $center1 $sse1 $hierarchy1)
        ($indices2 $center2 $sse2 $hierarchy2)
    )
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
    (bisecting-kmeans.remove-cluster () $target)
    ()
)
(=
    (bisecting-kmeans.remove-cluster (:: ($indices $center $sse $hierarchy) $xs) $target)
    (if (bisecting-kmeans.cluster-equal ($indices $center $sse $hierarchy) $target)
        (bisecting-kmeans.remove-cluster $xs $target)
        (:: ($indices $center $sse $hierarchy) (bisecting-kmeans.remove-cluster $xs $target))
    )
)

(=
    (bisecting-kmeans.bisect-cluster $X $max-cluster $max-iter)
    (let
        ; compute the clusters
        $centers
        (kmeans (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0) 2)

        ; output tuple
        (::
            ; cluster 0
            (
                (np.take
                    (bisecting-kmeans.get-cluster-indices $max-cluster)
                    (np.slice
                        ((py-dot np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign
                                        (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                        $centers
                                        2
                                    )
                                    0
                                )
                                0
                            )
                        )
                        [0]
                    )
                )

                (np.slice $centers [0])
                (bisecting-kmeans.compute-sse
                    (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                    (np.slice
                        ((py-dot np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign
                                        (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                        $centers
                                        2
                                    )
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
                    (np.take
                        (bisecting-kmeans.get-cluster-indices $max-cluster)
                        (np.slice
                            ((py-dot np where)
                                (np.equal
                                    (np.argmax
                                        (kmeans.assign
                                            (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                             $centers
                                             2
                                        )
                                        0
                                    )
                                    1
                                )
                            )
                            [0]
                        )
                    )
                    (np.slice $centers [1])
                    (bisecting-kmeans.compute-sse
                        (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                        (np.slice
                            ((py-dot np where)
                                (np.equal
                                    (np.argmax
                                        (kmeans.assign
                                            (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                            $centers
                                            2
                                        )
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
    (bisecting-kmeans.recursive-bisecting-kmeans $X $clusters $max-cluster $max-iter $hierarchy)
    (if (>= (length $clusters) $max-cluster)
        $hierarchy

        (let
            $bisect
            (bisecting-kmeans.bisect-cluster $X (bisecting-kmeans.find-max-cluster $clusters) $max-iter)

            (bisecting-kmeans.recursive-bisecting-kmeans
                $X
                (append
                    (append
                        (bisecting-kmeans.remove-cluster $clusters (bisecting-kmeans.find-max-cluster $clusters))
                        (get-first $bisect)
                    )
                    (get-second $bisect)
                )
                $max-cluster
                $max-iter
                (append
                    $hierarchy
                    (append
                        (append
                            (bisecting-kmeans.remove-cluster $clusters (bisecting-kmeans.find-max-cluster $clusters))
                            (get-first $bisect)
                        )
                        (get-second $bisect)
                    )
                )
            )
       )
    )
)

(=
    (bisecting-kmeans.fit $X $max-num-clusters $max-kmeans-iter)
    (let
        $init-cluster
        (bisecting-kmeans.compute-initial-cluster $X)

        (bisecting-kmeans.recursive-bisecting-kmeans
            $X
            $init-cluster
            $max-num-clusters
            $max-kmeans-iter
            (append pyNone $init-cluster)
        )
    )
)

(=
    (bisecting-kmeans.assign-point-to-cluster
        $point
        $clusters
        $best-cluster-idx
        $best-distance
        $cluster-idx
    )
    (if (>= $cluster-idx (length $clusters))
        $best-cluster-idx

        (let
            $dist
            (np.linalg.norm
                (np.sub
                    $point
                    (bisecting-kmeans.get-cluster-center (get-index $clusters $cluster-idx))
                )
            )
            (if
                (or
                    (== $best-distance pyNone)
                    (< $dist $best-distance)
                )
                (bisecting-kmeans.assign-point-to-cluster
                    $point
                    $clusters
                    $cluster-idx
                    $dist
                    (+ $cluster-idx 1)
                )
                (bisecting-kmeans.assign-point-to-cluster
                    $point
                    $clusters
                    $best-cluster-idx
                    $best-distance
                    (+ $cluster-idx 1)
                )
            )
        )
    )
)

(=
    (bisecting-kmeans.assign-all-points $X $clusters $point-idx $labels)
    (if (>= $point-idx (py-getitem (py-dot $X shape) 0))
        $labels

        (bisecting-kmeans.assign-all-points
            $X
            $clusters
            (+ $point-idx 1)
            (append
                $labels
                ; cluster_idx
                (bisecting-kmeans.assign-point-to-cluster
                    (np.take $X $point-idx 0)
                    $clusters
                    0
                    pyINF
                    0
                )
            )
        )
    )
)

(=
    (bisecting-kmeans.predict $X $hierarchy)
    (bisecting-kmeans.assign-all-points
        $X
        (get-last $hierarchy)
        0
        ()
    )
)
