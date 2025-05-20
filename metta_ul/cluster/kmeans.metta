!(import! &self metta_ul)

(: kmeans.update (-> (NPArray ($n $d)) (NPArray ($k $n)) (NPArray ($k $d))))
(=
    (kmeans.update $X $assignments)
    (np.div
        (np.matmul $assignments $X)
        (np.sum 
            $assignments
            1
            py.none
            py.none
            True
        )
    )
)

(: kmeans.assign (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($k $n))))
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
                    py.none
                    2
                )
                0
            )
            (np.shape $centroids 0)
        )
    )
)


(: kmeans.recursion (-> (NPArray ($n $d)) (NPArray ($k $d)) Number Number (NPArray ($k $d))))
(= 
    (kmeans.recursion $X $centroids $max-iter $tol)
    (let $new-centroids (kmeans.update $X (kmeans.assign $X $centroids))
        (if (and (> $max-iter 0) (< $tol (np.max (np.abs (np.sub $new-centroids $centroids)))))
            (kmeans.recursion
                $X
                $new-centroids
                (- $max-iter 1)
                $tol
            )
            $centroids
        )
    )
)

(: kmeans.fit (-> (NPArray ($n $d)) Number (NPArray ($k $d))))
(=
    (kmeans.fit $X $k)
    (kmeans.fit $X $k 100 0.0001)
)

(: kmeans.fit (-> (NPArray ($n $d)) Number Number Number (NPArray ($k $d))))
(=
    (kmeans.fit $X $k $max-iter $tol) 
    (kmeans.recursion $X (np.choose $X $k) $max-iter $tol)
)

(: kmeans.predict (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($n))))
(=
    (kmeans.predict $X $centers)
    (np.argmax (kmeans.assign $X $centers) 0)
)



