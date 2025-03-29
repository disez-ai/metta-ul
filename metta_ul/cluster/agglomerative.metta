!(import! &self metta_ul:numme)

! (bind! np (py-atom numpy))

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))
! (bind! pyPINF (py-atom "float('inf')"))

(= 
    (two-combinations ())
    ()
)
(= 
    (two-combinations (:: $x $xs))
    (two-combinations (:: $x $xs) $xs)
)
(=
    (two-combinations $list1 $list2)
    (case ($list1 $list2)
        (
            ((() ()) ())
            (((:: $x $xs) (:: $y $ys)) (:: ($x $y) (two-combinations $list1 $ys)))
            (((:: $x $xs) ()) (two-combinations $xs))
        )
    )
)
; ! (two-combinations (:: A (:: B (:: C ()))))

(= (in $x ()) False)
(=
    (in $x (:: $y $ys))
    (if (== $x $y) True (in $x $ys))
)
; ! (if (in 2 (:: 0 (:: 1 ()))) True False)

(= (remove $rm ()) ())
(= (remove $rm (:: $x $xs))
    (if (in $x $rm)
        (remove $rm $xs)
        (:: $x (remove $rm $xs))
    )
)
; ! (remove 2 (:: 1 (:: 2 (:: 5 (:: 3 ())))))

(= (append $y ()) (:: $y ()))
(= (append $y (:: $x $xs))
    (:: $x (append $y $xs))
)
; ! (append 3 (:: 1 (:: 2 ())))

(= (length ()) 0)
(= (length (:: $x $xs)) (+ 1 (length $xs)))
; ! (length (:: 1 (:: 2 ())))

(=
    (agglomerative.init-clusters $n)
    (let $n-1 (- $n 1)
        (if (> $n 0)
            (:: (py-list ($n-1)) (agglomerative.init-clusters $n-1))
            ()
        )
    )
)

(=
    (agglomerative.distance-matrix $X)
    (np.linalg.norm
        (np.sub
            (np.expand_dims $X 1)
            (np.expand_dims $X 0)
        )
        pyNone
        -1
    )
)

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

(= (agglomerative.closest-clusters () $distance-matrix $linkage $min-distance $closest-pair) $closest-pair)
(= 
    (agglomerative.closest-clusters (:: $c $cs) $distance-matrix $linkage $min-distance $closest-pair)
    (agglomerative.closest-clusters (:: $c $cs) $cs $distance-matrix $linkage $min-distance $closest-pair)
)
(=
    (agglomerative.closest-clusters $clusters1 $clusters2 $distance-matrix $linkage $min-distance $closest-pair)
    (case ($clusters1 $clusters2)
        (
            ((() ()) $closest-pair)

            (
                ((:: $c1 $cs1) ())
                (agglomerative.closest-clusters $cs1 $distance-matrix $linkage $min-distance $closest-pair)
            )

            (
                ((:: $c1 $cs1) (:: $c2 $cs2))
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

(=
    (agglomerative.merge-clusters $clusters $distance-matrix $linkage)
    (let 
        ($c1 $c2)
        (agglomerative.closest-clusters
            $clusters
            $distance-matrix
            $linkage
            pyPINF
            ()
        )
        (let $merged ((py-dot $c1 __add__) $c2)
            (append
                $merged
                (remove
                    (:: $c1 (:: $c2 ()))
                    $clusters
                )
            )
        )
    )
)

(=
    (agglomerative.recursion $linkage $clusters $distance-matrix $length)
    (if (== 1 $length)
        (:: $clusters ())
        (:: 
            $clusters
            (agglomerative.recursion
                $linkage
                (agglomerative.merge-clusters $clusters $distance-matrix $linkage)
                $distance-matrix
                (- $length 1)
            )
        )
    )
)

(=
    (agglomerative $X $linkage)
    (let $n (np.shape $X 0)
        (agglomerative.recursion
            $linkage
            (agglomerative.init-clusters $n)
            (agglomerative.distance-matrix $X)
            $n
        )
    )
)