!(import! &self metta_ul)
!(import! &self metta_ul:linkedlist)

(: PyList Type)
(: py-list (-> Expression PyList))

(: agglomerative.init-clusters (-> Number (List PyList)))
(=
    (agglomerative.init-clusters $n)
    (let $n-1 (- $n 1)
        (if (> $n 0)
            (Cons (py-list ($n-1)) (agglomerative.init-clusters $n-1))
            Nil
        )
    )
)

(: agglomerative.distance-matrix (-> (NPArray ($n $d)) (NPArray ($n $n))))
(=
    (agglomerative.distance-matrix $X)
    (np.linalg.norm
        (np.sub
            (np.expand_dims $X 1)
            (np.expand_dims $X 0)
        )
        py.none
        -1
    )
)

(: agglomerative.linkage-distance (-> (NPArray ($n $n)) PyList PyList String Number))
(=
    (agglomerative.linkage-distance $distance-matrix $cluster1 $cluster2 $linkage)
    (let $distances ((py-dot $distance-matrix __getitem__) (np.ix_ $cluster1 $cluster2))
        (case $linkage
            (
                ("single" (np.min $distances))
                ("complete" (np.max $distances))
                ("average" (np.mean $distances))
            )
        )
    )
)

(: agglomerative.closest-clusters (-> (List PyList) (NPArray ($n $n)) String Number (PyList PyList) (PyList PyList)))
(= (agglomerative.closest-clusters Nil $distance-matrix $linkage $min-distance $closest-pair) $closest-pair)
(= 
    (agglomerative.closest-clusters (Cons $c $cs) $distance-matrix $linkage $min-distance $closest-pair)
    (agglomerative.closest-clusters (Cons $c $cs) $cs $distance-matrix $linkage $min-distance $closest-pair)
)
(: agglomerative.closest-clusters (-> (List PyList) (List PyList) (NPArray ($n $n)) String Number (PyList PyList) (PyList PyList)))
(=
    (agglomerative.closest-clusters $clusters1 $clusters2 $distance-matrix $linkage $min-distance $closest-pair)
    (case ($clusters1 $clusters2)
        (
            ((Nil Nil) $closest-pair)

            (
                ((Cons $c1 $cs1) Nil)
                (agglomerative.closest-clusters $cs1 $distance-matrix $linkage $min-distance $closest-pair)
            )

            (
                ((Cons $c1 $cs1) (Cons $c2 $cs2))
                (let 
                    $distance 
                    (agglomerative.linkage-distance
                        $distance-matrix
                        $c1
                        $c2
                        $linkage
                    )
                    (if (> $min-distance $distance)
                        (agglomerative.closest-clusters
                            $clusters1
                            $cs2
                            $distance-matrix
                            $linkage
                            $distance
                            ($c1 $c2)
                        )
                        (agglomerative.closest-clusters
                            $clusters1
                            $cs2
                            $distance-matrix
                            $linkage
                            $min-distance
                            $closest-pair
                        )
                    )
                )
            )
        )
    )
)

(: agglomerative.merge-clusters (-> (List PyList) (NPArray ($n $n)) String (List PyList)))
(=
    (agglomerative.merge-clusters $clusters $distance-matrix $linkage)
    (let 
        ($c1 $c2)
        (agglomerative.closest-clusters
            $clusters
            $distance-matrix
            $linkage
            py.pinf
            ()
        )
        (List.append
            ((py-dot $c1 __add__) $c2)
            (List.remove
                (List.fromExpression ($c1 $c2))
                $clusters
            )
        )
    )
)

(: agglomerative.recursion (-> Number String (List PyList) (NPArray ($n $n)) Number (List PyList)))
(=
    (agglomerative.recursion $k $linkage $clusters $distance-matrix $length)
    (if (== $k $length)
        $clusters
        (agglomerative.recursion
            $k
            $linkage
            (agglomerative.merge-clusters $clusters $distance-matrix $linkage)
            $distance-matrix
            (- $length 1)
        )
    )
)

(: agglomerative (-> (NPArray ($n $d)) Number String (List PyList)))
(=
    (agglomerative $X $k $linkage)
    (let $n (np.shape $X 0)
        (agglomerative.recursion
            $k
            $linkage
            (agglomerative.init-clusters $n)
            (agglomerative.distance-matrix $X)
            $n
        )
    )
)


(: agglomerative.assign (-> (List PyList) (NPArray ($n)) Number (NPArray ($n))))
(=
    (agglomerative.assign $clusters $assignment $index)
    (case $clusters
        (
            (Nil $assignment)
            ((Cons $c $cs)
                (let $_ (np.put $assignment $c $index)
                    (agglomerative.assign $cs $assignment (+ 1 $index))
                )
            )
        )
    )
)


(: agglomerative.fit-predict (-> (NPArray ($n $d)) Number String (NPArray ($n))))
(=
    (agglomerative.fit-predict $X $k $linkage)
    (agglomerative.assign
        (agglomerative $X $k $linkage)
        (np.ones (np.shape $X 0))
        0
    )
)