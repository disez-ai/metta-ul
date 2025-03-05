!(import! &self metta_ul:numme)

! (bind! np (py-atom numpy))

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

(=
    (py-getitem $iterable $index)
    (
        (py-dot $iterable __getitem__)
        $index
    )
)

; X_centered = X[:, np.newaxis, :] - means
(=
    (gmm.center $X $means)
    (np.sub
        (np.expand_dims $X 1)
        $means
    )
)

; mahalanobis_term = np.einsum('nkd,kde,nke->nk', X_centered, cov_inv, X_centered)  # (N, K)
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

; exponent = -0.5 * (mahalanobis_term + D * np.log(2 * np.pi) + log_det_cov)
(=
    (gmm.exponent $X $means $covariances)
    (np.mul
        -0.5
        (np.add
            (np.add
                (gmm.mahalanobis-term $X $means $covariances)
                (np.mul
                    (py-getitem (py-dot $X shape) 1)
                    (np.log (np.mul (np.array (2)) (py-dot np pi)))
                )
            )
            (np.linalg.slogabsdet $covariances)
        )
    )
)

; weighted_pdfs = np.exp(exponent) * weights
(=
    (gmm.weighted-pdfs $exponent $weights)
    (np.mul (np.exp $exponent) $weights)
)

; log_likelihood = np.sum(np.log(np.sum(weighted_pdfs, axis=1)))
(=
    (gmm.log-likelihood $X $weights $means $covariances)
    (np.sum
        (np.log
            (np.sum
                (gmm.weighted-pdfs
                    (gmm.exponent $X $means $covariances)
                    $weights
                )
                1
            )
        )
        0
        pyNone
        pyNone
        pyTrue
    )
)