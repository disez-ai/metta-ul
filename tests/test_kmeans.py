from hyperon import MeTTa, Atom
import numpy as np


def test_kmeans_update(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:kmeans)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (assignments)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run("! (kmeans.update (np.array (X)) (np.array (assignments)))")[0][0]

    centroids_true = np.asarray([[1, 0], [0, 1]])
    centroids: np.ndarray = result.get_object().value

    assert np.allclose(centroids, centroids_true)


def test_kmeans_assign(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:kmeans)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (centroids)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run("! (kmeans.assign (np.array (X)) (np.array (centroids)))")[0][0]

    assignments_true = np.asarray([[1, 0], [0, 1]])
    assignments: np.ndarray = result.get_object().value

    assert np.allclose(assignments_true, assignments)


def test_kmeans_recursion(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:kmeans)

        (=
            (X)
            ((1 0) (0 1))
        )

        (=
            (centroids)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run("! (kmeans.recursion (np.array (X)) (np.array (centroids)) 0)")[0][0]

    centroids_true = np.asarray([[1, 0], [0, 1]])
    centroids: np.ndarray = result.get_object().value

    assert np.allclose(centroids_true, centroids)

    result: Atom = metta.run("! (kmeans.recursion (np.array (X)) (np.array (centroids)) 1)")[0][0]

    centroids_true = np.asarray([[1, 0], [0, 1]])
    centroids: np.ndarray = result.get_object().value

    assert np.allclose(centroids_true, centroids)


def test_kmeans(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:kmeans)
        ! (np.random.seed 0)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        """
    )

    result: Atom = metta.run("! (kmeans (np.array (X)) 3)")[0][0]

    centroids_true = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    centroids: np.ndarray = result.get_object().value

    assert np.allclose(centroids, centroids_true)


def test_py_dot_kmeans(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:py_dot_kmeans)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        """
    )

    result: Atom = metta.run("! (kmeans (py-list (X)) 3 5)")[0][0]

    centroids_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    centroids: np.ndarray = np.asarray(result.get_object().value)

    assert np.allclose(centroids.sum(axis=0), [1, 1, 1])
    assert np.allclose(centroids.sum(axis=1), [1, 1, 1])

    assert centroids[0].tolist() in centroids_true
    assert centroids[1].tolist() in centroids_true
    assert centroids[2].tolist() in centroids_true
