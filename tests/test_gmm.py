from hyperon import MeTTa, Atom
import numpy as np


def test_gmm_py_getitem(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)
        '''
        )
    
    result: Atom = metta.run('! (py-getitem (py-list (1 2)) 0)')[0][0]
    element: int = result.get_object().value

    assert element == 1


def test_gmm_center(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (means)
            ((1 0) (0 1) (0 0))
        )
        '''
        )
    
    result: Atom = metta.run('! (gmm.center (np.array (X)) (np.array (means)))')[0][0]
    X_centered: np.ndarray = result.get_object().value
    X_centered_true = np.asarray([
        [
            [0, 0],
            [1, -1],
            [1, 0]
        ],
        [
            [-1, 1],
            [0, 0],
            [0, 1]
        ]
    ])

    assert np.allclose(X_centered, X_centered_true)


def test_gmm_mahalanobis_term(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (means)
            ((1 0) (0 1) (0 0))
        )

        (=
            (covariances)
            (
                ((1 0) (0 1))
                ((1 0) (0 1))
                ((1 0) (0 1))
            )
        )
        '''
        )
    
    result: Atom = metta.run('! (gmm.mahalanobis-term (np.array (X)) (np.array (means)) (np.array (covariances)))')[0][0]
    mahalanobis_term: np.ndarray = result.get_object().value
    mahalanobis_term_true = np.asarray([
        [0, 2, 1],
        [2, 0, 1]
    ])

    assert np.allclose(mahalanobis_term, mahalanobis_term_true)


def test_gmm_exponent(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (means)
            ((1 0) (0 1) (0 0))
        )

        (=
            (covariances)
            (
                ((1 0) (0 1))
                ((1 0) (0 1))
                ((1 0) (0 1))
            )
        )
        '''
        )
    
    result: Atom = metta.run('! (gmm.exponent (np.array (X)) (np.array (means)) (np.array (covariances)))')[0][0]
    exponent: np.ndarray = result.get_object().value
    exponent_true = np.asarray([
        [-1.8378770664093453, -2.8378770664093453, -2.3378770664093453],
        [-2.8378770664093453, -1.8378770664093453, -2.3378770664093453]
    ])

    assert np.allclose(exponent, exponent_true)


def test_gmm_weighted_pdfs(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (exponents)
            (
                (-1.8378770664093453 -2.8378770664093453 -2.3378770664093453)
                (-2.8378770664093453 -1.8378770664093453 -2.3378770664093453)
            )
        )

        (=
            (weights)
            ((/ 1.0 3) (/ 1.0 3) (/ 1.0 3))
        )
        '''
        )
    
    result: Atom = metta.run('! (gmm.weighted-pdfs (np.array (exponents)) (np.array (weights)))')[0][0]
    weighted_pdfs: np.ndarray = result.get_object().value
    weighted_pdfs_true = np.asarray([
        [0.05305165, 0.01951661, 0.03217745],
        [0.01951661, 0.05305165, 0.03217745]
    ])

    assert np.allclose(weighted_pdfs, weighted_pdfs_true)


def test_gmm_log_likelihood(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (means)
            ((1 0) (0 1) (0 0))
        )

        (=
            (covariances)
            (
                ((1 0) (0 1))
                ((1 0) (0 1))
                ((1 0) (0 1))
            )
        )

        (=
            (weights)
            ((/ 1.0 3) (/ 1.0 3) (/ 1.0 3))
        )
        '''
    )

    result: Atom = metta.run('! (gmm.log-likelihood (np.array (X)) (np.array (weights)) (np.array (means)) (np.array (covariances)))')[0][0]

    log_likelihood: np.ndarray = result.get_object().value
    log_likelihood_true = np.asarray([-4.5124393688714415])

    assert np.allclose(log_likelihood, log_likelihood_true)


def test_gmm(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        '''
        )
    
    result: Atom = metta.run('! (gmm (np.array (X)) 3)')[0][0]

    centroids_true = [[1,0,0], [0,1,0], [0,0,1]]
    centroids: np.ndarray = np.asarray(result.get_object().value)

    assert np.allclose(centroids.sum(axis=0), [1, 1, 1])
    assert np.allclose(centroids.sum(axis=1), [1, 1, 1])

    assert centroids[0].tolist() in centroids_true
    assert centroids[1].tolist() in centroids_true
    assert centroids[2].tolist() in centroids_true
