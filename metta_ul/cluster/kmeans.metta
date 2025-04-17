!(import! &self metta_ul:numme)

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

(: kmeans.update (-> (NPArray ($n $d)) (NPArray ($k $n)) (NPArray ($k $d))))
(: kmeans.assign (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($k $n))))
(: kmeans.recursion (-> (NPArray ($n $d)) (NPArray ($k $d)) Number (NPArray ($k $d))))

(: kmeans (-> (NPArray ($n $d)) Number (NPArray ($k $d))))
(: kmeans (-> (NPArray ($n $d)) Number Number (NPArray ($k $d))))

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
    (kmeans.assign $X $centroids)
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
            (np.shape $centroids 0)
        )
    )
)

(= 
    (kmeans.recursion $X $centroids $max-iter)
    (if (> $max-iter 0)
        (kmeans.recursion
            $X
            (kmeans.update
                $X
                (kmeans.assign $X $centroids)
            )
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
    (kmeans.recursion $X (np.choose $X $k) $max-iter)
)

(=
  (kmeans.fit $X $k)
  (let* (($centers (kmeans $X $k)) ($lables (np.argmax (kmeans.assign $X $centers $k) 0))) (KmeansModel $X $centers $lables))
)

(=
  (labels (KmeansModel $A $B $C))
  $C
)


