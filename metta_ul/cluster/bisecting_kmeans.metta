!(import! &self metta_ul)
!(import! &self kmeans)

! (bind! bc-np (py-atom numpy))

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
    (len ())
    0
)
(=
    (len (:: $x $xs))
    (+ 1 (len $xs))
)

(: get-first (-> ClusterList Cluster))
(=
    (get-first (:: $x $y))
    $x
)

(: get-second (-> ClusterList Cluster))
(=
    (get-second (:: $x (:: $y ())))
    $y
)

(: append (-> ClusterList Cluster ClusterList))
(: append (-> Hierarchy ClusterList Hierarchy))
(: append (-> LabelList Number LabelList))
(=
    (append py.none $a)
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
    py.none
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
    py.none
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
; Python code:
; pts = data[indices]
; np.sum((pts - center) ** 2)
(: bisecting-kmeans.compute-sse (-> (NPArray ($N $D)) (NPArray ($M)) (NPArray ($C)) Number))
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

; Python code:
; n_samples = data.shape[0]
; indices = np.arange(n_samples)
; center = np.mean(data, axis=0)
; sse = compute_sse(data, indices, center)
(: bisecting-kmeans.compute-initial-cluster (-> (NPArray ($N $D)) ClusterList))
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
            py.none
        )
        ()
    )
)

(: bisecting-kmeans.get-cluster-indices (-> Cluster (NPArray ($M))))
(=
    (bisecting-kmeans.get-cluster-indices ($indices $center $sse $hierarchy))
    $indices
)

(: bisecting-kmeans.get-cluster-center (-> Cluster (NPArray ($K $D))))
(=
    (bisecting-kmeans.get-cluster-center ($indices $center $sse $hierarchy))
    $center
)

(: bisecting-kmeans.get-cluster-sse (-> Cluster Number))
(=
    (bisecting-kmeans.get-cluster-sse ($indices $center $sse $hierarchy))
    $sse
)

(: bisecting-kmeans.get-cluster-hierarchy (-> Cluster Hierarchy))
(=
    (bisecting-kmeans.get-cluster-hierarchy ($indices $center $sse $hierarchy))
    $hierarchy
)

(: bisecting-kmeans.find-max-cluster (-> ClusterList Cluster))
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


(: bisecting-kmeans.cluster-equal (-> Cluster Cluster Bool))
(=
    (bisecting-kmeans.cluster-equal
        ($indices1 $center1 $sse1 $hierarchy1)
        ($indices2 $center2 $sse2 $hierarchy2)
    )
    (if
        (and
            (and
                ((py-dot bc-np array_equal) $indices1 $indices2)
                ((py-dot bc-np array_equal) $center2 $center2)
            )
            (== $sse1 $sse2)
        )
        True
        False
    )
)

(: bisecting-kmeans.remove-cluster (-> ClusterList Cluster ClusterList))
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

(: bisecting-kmeans.bisect-cluster (-> (NPArray ($N $D)) Cluster Number ClusterList))
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
                        ((py-dot bc-np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign
                                        (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                        $centers
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
                        ((py-dot bc-np where)
                            (np.equal
                                (np.argmax
                                    (kmeans.assign
                                        (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                        $centers
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
                py.none
            )
            (::
                ; cluster 1
                (
                    (np.take
                        (bisecting-kmeans.get-cluster-indices $max-cluster)
                        (np.slice
                            ((py-dot bc-np where)
                                (np.equal
                                    (np.argmax
                                        (kmeans.assign
                                            (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                             $centers
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
                            ((py-dot bc-np where)
                                (np.equal
                                    (np.argmax
                                        (kmeans.assign
                                            (np.take $X (bisecting-kmeans.get-cluster-indices $max-cluster) 0)
                                            $centers
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
                    py.none
                )
                ()
            )
        )
    )
)

(: bisecting-kmeans.recursive-bisecting-kmeans (-> (NPArray ($N $D)) ClusterList Number Number Hierarchy Hierarchy))
(=
    (bisecting-kmeans.recursive-bisecting-kmeans $X $clusters $max-num-clusters $max-iter $hierarchy)
    (if (>= (length $clusters) $max-num-clusters)
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
                $max-num-clusters
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

(: bisecting-kmeans.fit (-> (NPArray ($N $D)) Number Number Hierarchy))
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
            (append py.none $init-cluster)
        )
    )
)

(=
    (bisecting-kmeans.extract-centers ())
    ()
)
(=
    (bisecting-kmeans.extract-centers (:: ($indices $center $sse $hierarchy) $rest))
    (:: $center (bisecting-kmeans.extract-centers $rest))
)

(=
    (bisecting-kmeans.concat-arrays ())
    (np.array ())
)
(=
    (bisecting-kmeans.concat-arrays (:: $first ()))
    (np.expand_dims $first 0)
)
(=
    (bisecting-kmeans.concat-arrays (:: $first $rest))
    (np.append
        (np.expand_dims $first 0)
        (bisecting-kmeans.concat-arrays $rest)
        0
    )
)

(=
    (bisecting-kmeans.assign $X $clusters)
    (np.argmax
        (kmeans.assign
            $X
            (bisecting-kmeans.concat-arrays
                (bisecting-kmeans.extract-centers $clusters)
            )
        )
        0
    )
)


(: bisecting-kmeans.predict (-> (NPArray ($N $D)) Hierarchy (NPArray ($N))))
(=
    (bisecting-kmeans.predict $X $hierarchy)
    (bisecting-kmeans.assign
        $X
        (get-last $hierarchy)
    )
)
