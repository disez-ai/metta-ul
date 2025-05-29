!(import! &self metta_ul)

(: gmm.center (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($n $k $d))))
(=
    (gmm.center $X $means)
    (np.sub
        (np.expand_dims $X 1)
        $means
    )
)


(: gmm.mahalanobis (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($k $d $d)) (NPArray ($n $k))))
(=
    (gmm.mahalanobis-term $X $means $covariances)
    (let $X-centered (gmm.center $X $means)
        (np.einsum
            "nkd,kde,nke->nk"
            $X-centered
            (np.linalg.inv $covariances)
            $X-centered
        )
    )
)


(: gmm.gaussian-pdf (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($k $d $d)) (NPArray ($n $k))))
(=
    (gmm.gaussian-pdf $X $means $covariances)
    (np.exp
        (np.mul
            -0.5
            (np.add
                (np.add
                    (gmm.mahalanobis-term $X $means $covariances)
                    (np.mul
                        (np.shape $X 1)
                        (np.log (np.mul (np.array (2)) math.pi))
                    )
                )
                (np.linalg.slogabsdet $covariances)
            )
        )
    )
)


(: gmm.log-likelihood (-> (NPArray ($n $d)) ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d))) (NPArray (1))))
(=
    (gmm.log-likelihood $X ($weights $means $covariances))
    (np.sum
        (np.log
            (np.sum
                (np.mul
                    (gmm.gaussian-pdf $X $means $covariances)
                    $weights
                )
                1
            )
        )
        0
        py.none
        py.none
        True
    )
)


(: gmm.init (-> (NPArray ($n $d)) Number ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))))
(=
    (gmm.init $X $k)
    (
        (np.div
            (np.ones $k)
            $k
        )

        (np.choose $X $k)

        (np.add
            (np.repeat
                (np.expand_dims (np.cov $X py.none False) 0)
                $k
                0
            )
            (np.mul
                0.000001
                (np.eye (np.shape $X 1))
            )
        )
    )
)


(: gmm.e-step (-> (NPArray ($n $d)) (NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)) (NPArray ($n $k))))
(=
    (gmm.e-step $X $weights $means $covariances)
    (let $weighted-pdfs
        (np.mul
            (gmm.gaussian-pdf $X $means $covariances)
            $weights
        )

        (np.div
            $weighted-pdfs
            (np.sum
                $weighted-pdfs
                1
                py.none
                py.none
                True
            )
        )
    )
)


(: gmm.m-step (-> (NPArray ($n $d)) (NPArray ($n $k)) ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))))
(=
    (gmm.m-step $X $responsibilities)
    (let $Nk (np.sum $responsibilities 0)
        (let $means
            (np.div
                (np.matmul (np.transpose $responsibilities) $X)
                (np.expand_dims $Nk 1)
            )
            (let $X-centered (gmm.center $X $means)
                (
                    (np.div $Nk (np.shape $X 0))
                    $means
                    (np.div
                        (np.einsum
                            "nk,nkd,nke->kde"
                            $responsibilities
                            $X-centered
                            $X-centered
                        )
                        (np.expand_dims (np.expand_dims $Nk -1) -1)
                    )
                )
            )
        )
    )
)


(: gmm.recursion (-> (NPArray ($n $d)) ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d))) Number Number (NPArray (1)) ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))))
(=
    (gmm.recursion $X ($weights $means $covariances) $max-iter $tol $log-likelihood)
    (if (> $max-iter 0)
        (let* 
            (
                ($new-params (gmm.m-step $X (gmm.e-step $X $weights $means $covariances)))
                ($new-log-likelihood (gmm.log-likelihood $X $new-params))
            )
            (if (< $tol (np.max (np.abs (np.sub $new-log-likelihood $log-likelihood))))
                (gmm.recursion
                    $X
                    $new-params
                    (- $max-iter 1)
                    $tol
                    $new-log-likelihood
                )
                $new-params
            )
        )
        ($weights $means $covariances)
    )
)

(: gmm.fit (-> (NPArray ($n $d)) Number Number Number ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))))
(=
    (gmm.fit $X $k $max-iter $tol)
    (let $init-params (gmm.init $X $k)
        (gmm.recursion
            $X
            $init-params
            $max-iter
            $tol
            (gmm.log-likelihood $X $init-params)
        )
    )
)

(: gmm.fit (-> (NPArray ($n $d)) Number ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))))
(=
    (gmm.fit $X $k)
    (gmm.fit $X $k 100 0.001)
)

(: gmm.predict (-> (NPArray ($n $d)) ((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d))) (NPArray ($n))))
(= (gmm.predict $x ($w $m $S)) (np.argmax (gmm.e-step $x $w $m $S) 1))
