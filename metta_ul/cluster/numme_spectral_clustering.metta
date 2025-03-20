!(import! &self metta_ul:numme)

! (bind! np (py-atom numpy))
! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

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
    (to-np $x)
    ((py-dot np array) $x)
)

(=
    (py-getitem $iterable $index)
    (
        (py-dot $iterable __getitem__)
        $index
    )
)


(=
    (square-norm $X)
    (np.sum
        (np.power $X 2)
        1
        pyNone
        pyNone
        pyTrue
    )
)

(=
    (square-distance-matrix $square-norm-X $X)
    (np.sub
        (np.add
            $square-norm-X
            (np.transpose
                $square-norm-X
            )
        )
        (np.mul
            2
            (np.matmul
                $X
                (np.transpose
                    $X
                )
            )
        )
    )
)

(=
    (rbf-affinity-matrix $sqr-distance-matrix-X $sigma)
    (np.exp
        (np.div
            (np.mul
                -1
                $sqr-distance-matrix-X
            )
            (np.mul
                2
                (np.power
                    (np.array ($sigma))
                    2
                )
            )
        )
    )
)


(=
    (degree $W)
    (np.sum $W 1)
)

(=
    (inverse-degree-matrix $degree-W)
    (np.diag
        (np.div
            1.0
            (np.sqrt
                $degree-W
            )
        )
    )
)

(=
    (normalized-laplacian $W $inverse-degree-matrix-W)
    (np.sub
        (np.eye (py-getitem (py-dot $W shape) 0))
        (np.matmul
            $inverse-degree-matrix-W
            (np.matmul
                $W
                $inverse-degree-matrix-W
            )
        )
    )
)


(=
    (eigh $X)
    ((py-dot np linalg.eigh) (to-np $X))
)
(=
    (eigenvalues $eigh-X)
    ((py-dot $eigh-X __getitem__) 0)
)

(=
    (eigenvectors $eigh-X)
    ((py-dot $eigh-X __getitem__) 1)
)

(=
    (eigval-top-k-index $eigval-L $k)
    ((py-dot (np.argsort $eigval-L) __getitem__) (np.arange $k))
)

(=
    (spectral-embeddings $eigh-I $k)
    (np.take
        (eigenvectors $eigh-I)
        ((py-dot (np.argsort (eigenvalues $eigh-I)) __getitem__) (np.arange $k))
        1
    )
)

(=
    (row-normalize $X)
    (np.div
        $X
        (np.linalg.norm
            $X
            pyNone
            1
            pyTrue
        )
    )
)

(=
    (spectral-clustering $X $k $sigma $max-iter)
    (kmeans
        (row-normalize
            (spectral-embeddings
                (eigh
                    (normalized-laplacian
                        (rbf-affinity-matrix
                            (square-distance-matrix
                                (square-norm $X)
                                $X
                            )
                            $sigma
                        )
                        (inverse-degree-matrix
                            (degree
                                (rbf-affinity-matrix
                                    (square-distance-matrix
                                        (square-norm $X)
                                        $X
                                    )
                                    $sigma
                                )
                            )
                        )
                    )
                )
                $k
            )
        )
        $k
        $max-iter
    )
)