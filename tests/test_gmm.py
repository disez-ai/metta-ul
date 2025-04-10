from hyperon import MeTTa, Atom
import numpy as np


def test_gmm_py_getitem(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:gmm)
        """
    )

    result: Atom = metta.run("! (py-getitem (py-list (1 2)) 0)")[0][0]
    element: int = result.get_object().value

    assert element == 1


def test_gmm_center(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (means)
            ((1 0) (0 1) (0 0))
        )
        """
    )

    result: Atom = metta.run("! (gmm.center (np.array (X)) (np.array (means)))")[0][0]
    X_centered: np.ndarray = result.get_object().value
    X_centered_true = np.asarray([[[0, 0], [1, -1], [1, 0]], [[-1, 1], [0, 0], [0, 1]]])

    assert np.allclose(X_centered, X_centered_true)


def test_gmm_mahalanobis_term(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.mahalanobis-term (np.array (X)) (np.array (means)) (np.array (covariances)))"
    )[0][0]
    mahalanobis_term: np.ndarray = result.get_object().value
    mahalanobis_term_true = np.asarray([[0, 2, 1], [2, 0, 1]])

    assert np.allclose(mahalanobis_term, mahalanobis_term_true)


def test_gmm_gaussian_pdf(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.gaussian-pdf (np.array (X)) (np.array (means)) (np.array (covariances)))"
    )[0][0]
    gaussian_pdf: np.ndarray = result.get_object().value
    exponent_true = np.asarray(
        [
            [-1.8378770664093453, -2.8378770664093453, -2.3378770664093453],
            [-2.8378770664093453, -1.8378770664093453, -2.3378770664093453],
        ]
    )

    assert np.allclose(gaussian_pdf, np.exp(exponent_true))


def test_gmm_log_likelihood(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.log-likelihood (np.array (X)) (np.array (weights)) (np.array (means)) (np.array (covariances)))"
    )[0][0]

    log_likelihood: np.ndarray = result.get_object().value
    log_likelihood_true = np.asarray([-4.5124393688714415])

    assert np.allclose(log_likelihood, log_likelihood_true)


def test_gmm_init(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        """
    )

    result: Atom = metta.run("! (gmm.init (np.array (X)) 3)")[0][0]

    weights, means, covariances = (
        atom.get_object().value for atom in result.get_children()
    )
    weights_true = np.ones(3) / 3
    covariances_true = np.repeat(np.cov(np.eye(3))[np.newaxis, :, :], 3, axis=0)

    assert np.allclose(means.sum(axis=0), [1, 1, 1])
    assert np.allclose(means.sum(axis=1), [1, 1, 1])

    assert np.allclose(weights, weights_true)

    assert np.allclose(covariances, covariances_true)


def test_gmm_e_step(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.e-step (np.array (X)) (np.array (weights)) (np.array (means)) (np.array (covariances)))"
    )[0][0]

    responsibiliies = result.get_object().value
    responsibiliies_true = np.array(
        [[0.73105858, 0.26894142], [0.26894142, 0.73105858]]
    )

    assert np.allclose(responsibiliies, responsibiliies_true)


def test_gmm_m_step(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:gmm)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (responsibiliies)
            (
                (0.73105858 0.26894142)
                (0.26894142 0.73105858)
            )
        )
        """
    )

    result: Atom = metta.run(
        "! (gmm.m-step (np.array (X)) (np.array (responsibiliies)))"
    )[0][0]

    weights, means, covariances = (
        atom.get_object().value for atom in result.get_children()
    )

    weights_true = np.ones(2) / 2
    means_true = np.array([[0.73105858, 0.26894142], [0.26894142, 0.73105858]])
    covariances_true = np.repeat(
        [[[0.19661193, -0.19661193], [-0.19661193, 0.19661193]]], 2, axis=0
    )

    assert np.allclose(weights, weights_true)
    assert np.allclose(means, means_true)
    assert np.allclose(covariances, covariances_true)


def test_gmm_recursion_max_iter_0(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.recursion (np.array (X)) ((np.array (weights)) (np.array (means)) (np.array (covariances))) 0)"
    )[0][0]

    weights, means, covariances = (
        atom.get_object().value for atom in result.get_children()
    )
    weights_true = np.ones(2) / 2
    means_true = np.eye(2)
    covariances_true = np.repeat(np.eye(2)[np.newaxis, :, :], 2, axis=0)

    assert np.allclose(weights, weights_true)
    assert np.allclose(means, means_true)
    assert np.allclose(covariances, covariances_true)


def test_gmm_recursion_max_iter_1(metta: MeTTa):
    metta.run(
        """
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
        """
    )

    result: Atom = metta.run(
        "! (gmm.recursion (np.array (X)) ((np.array (weights)) (np.array (means)) (np.array (covariances))) 1)"
    )[0][0]

    weights, means, covariances = (
        atom.get_object().value for atom in result.get_children()
    )
    weights_true = np.ones(2) / 2
    means_true = np.array([[0.73105858, 0.26894142], [0.26894142, 0.73105858]])
    covariances_true = np.repeat(
        [[[0.19661193, -0.19661193], [-0.19661193, 0.19661193]]], 2, axis=0
    )

    assert np.allclose(weights, weights_true)
    assert np.allclose(means, means_true)
    assert np.allclose(covariances, covariances_true)


def test_gmm(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:gmm)
        ! (np.random.seed 0)

        (=
            (X)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run("! (gmm (np.array (X)) 2 1)")[0][0]

    weights, means, covariances = (
        atom.get_object().value for atom in result.get_children()
    )
    weights_true = np.ones(2) / 2
    means_true = np.array([[0.26894142, 0.73105858], [0.73105858, 0.26894142]])
    covariances_true = np.repeat(
        [[[0.19661193, -0.19661193], [-0.19661193, 0.19661193]]], 2, axis=0
    )

    assert np.allclose(weights, weights_true)
    assert np.allclose(means, means_true)
    assert np.allclose(covariances, covariances_true)
