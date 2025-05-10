! (import! &self metta_ul:cluster:gmm)

(=
    (X)
    ((1 0) (0 1))
)

(=
    (means)
    ((1 0) (0 1))
)

(=
    (covariances)
    (
        ((1 0) (0 1))
        ((1 0) (0 1))
    )
)

(=
    (weights)
    (0.5 0.5)
)

(=
    (responsibilities)
    (
        (0.73105858 0.26894142)
        (0.26894142 0.73105858)
    )
)

;; Test gmm.center
(=
    (gmm.center.test)
    (np.assertAllClose
        (gmm.center (np.array (X)) (np.array (means)))
        (np.array (
            ((0 0) (1 -1))
            ((-1 1) (0 0))
        ))
    )
)
(Test gmm.center.test)

;; Test gmm.mahalanobis-term
(=
    (gmm.mahalanobis-term.test)
    (np.assertAllClose
        (gmm.mahalanobis-term (np.array (X)) (np.array (means)) (np.array (covariances)))
        (np.array (
            (0 2) (2 0)
        ))
    )
)
(Test gmm.mahalanobis-term.test)

;; Test gmm.gaussian-pdf
(=
    (gmm.gaussian-pdf.test)
    (np.assertAllClose
        (gmm.gaussian-pdf (np.array (X)) (np.array (means)) (np.array (covariances)))
        (np.array (
            (0.15915494 0.05854983)
            (0.05854983 0.15915494)
        ))
    )
)
(Test gmm.gaussian-pdf.test)

;; Test gmm.log-likelihood
(=
    (gmm.log-likelihood.test)
    (np.assertAllClose
        (gmm.log-likelihood (np.array (X)) (np.array (weights)) (np.array (means)) (np.array (covariances)))
        (np.array (-4.43552512))
    )
)
(Test gmm.log-likelihood.test)

;; Test gmm.init
(=
    (gmm.init.test)
    (let $_ (np.random.seed 1)
    (let ($weights $means $covariances) (gmm.init (np.array (X)) 2)
        (assertEqual
            (
                (np.assertAllClose $weights (np.array (weights)))
                (np.assertAllClose $means (np.array (means)))
                (np.assertAllClose $covariances (np.array (
                    ((0.500001 -0.5) (-0.5 0.500001))
                    ((0.500001 -0.5) (-0.5 0.500001))
                )))
            )
            (() () ())
        )
    ))
)
(Test gmm.init.test)

;; Test gmm.e-step
(=
    (gmm.e-step.test)
    (np.assertAllClose
        (gmm.e-step (np.array (X)) (np.array (weights)) (np.array (means)) (np.array (covariances)))
        (np.array (responsibilities))
    )
)
(Test gmm.e-step.test)

;; Test gmm.m-step
(=
    (gmm.m-step.test)
    (let ($weights $means $covariances) (gmm.m-step (np.array (X)) (np.array (responsibilities)))
        (assertEqual
            (
                (np.assertAllClose $weights (np.array (weights)))
                (np.assertAllClose $means (np.array (responsibilities)))
                (np.assertAllClose $covariances (np.array (
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                )))
            )
            (() () ())
        )
    )
)
(Test gmm.m-step.test)

;; Test gmm.recursion
(=
    (gmm.recursion.testNoRecursion)
    (let ($weights $means $covariances) (gmm.recursion (np.array (X)) ((np.array (weights)) (np.array (means)) (np.array (covariances))) 0)
        (assertEqual
            (
                (np.assertAllClose $weights (np.array (weights)))
                (np.assertAllClose $means (np.array (means)))
                (np.assertAllClose $covariances (np.array (covariances)))
            )
            (() () ())
        )
    )
)
(Test gmm.recursion.testNoRecursion)

(=
    (gmm.recursion.testOneRecursion)
    (let ($weights $means $covariances) (gmm.recursion (np.array (X)) ((np.array (weights)) (np.array (means)) (np.array (covariances))) 1)
        (assertEqual
            (
                (np.assertAllClose $weights (np.array (weights)))
                (np.assertAllClose $means (np.array (responsibilities)))
                (np.assertAllClose $covariances (np.array ((
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                ))))
            )
            (() () ())
        )
    )
)
(Test gmm.recursion.testOneRecursion)

;; Test gmm
(=
    (gmm.test)
    (let $_ (np.random.seed 1)
    (let ($weights $means $covariances) (gmm (np.array (X)) 2 1)
        (assertEqual
            (
                (np.assertAllClose $weights (np.array (weights)))
                (np.assertAllClose $means (np.array (responsibilities)))
                (np.assertAllClose $covariances (np.array ((
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                    ((0.19661193 -0.19661193) (-0.19661193 0.19661193))
                ))))
            )
            (() () ())
        )
    ))
)
(Test gmm.test)