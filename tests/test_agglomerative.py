from hyperon import MeTTa, Atom, ExpressionAtom, GroundedAtom
import numpy as np


def test_agglomerative_distance_matrix(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (X)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run("! (agglomerative.distance-matrix (np.array (X)))")[0][0]
    distance_matrix: np.ndarray = result.get_object().value
    distance_matrix_true = np.asarray([[0, np.sqrt(2)], [np.sqrt(2), 0]])

    assert np.allclose(distance_matrix, distance_matrix_true)


def test_agglomerative_linkage_distance(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (distance-matrix)
            ((0.0 1.0) (1.0 0.0))
        )

        (=
            (cluster1)
            (py-list (0))
        )

        (=
            (cluster2)
            (py-list (1))
        )
        """
    )

    result: Atom = metta.run('! (agglomerative.linkage-distance (np.array (distance-matrix)) (cluster1) (cluster2) "single")')[0][0]
    distance = result.get_object().value
    distance_true = 1

    assert np.allclose(distance, distance_true)

    result: Atom = metta.run('! (agglomerative.linkage-distance (np.array (distance-matrix)) (cluster1) (cluster2) "complete")')[0][0]
    distance = result.get_object().value
    distance_true = 1

    assert np.allclose(distance, distance_true)

    result: Atom = metta.run('! (agglomerative.linkage-distance (np.array (distance-matrix)) (cluster1) (cluster2) "average")')[0][0]
    distance = result.get_object().value
    distance_true = 1

    assert np.allclose(distance, distance_true)


def test_agglomerative_closest_clusters(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (clusters)
            (:: (py-list (0)) (:: (py-list (1)) (:: (py-list (2)) ())))
        )

        (=
            (distance-matrix)
            (
                (0 1 2)
                (1 0 3)
                (2 3 0)
            )
        )

        """
    )

    result: Atom = metta.run(
        '''
        ! (agglomerative.closest-clusters
            (clusters)
            (np.array (distance-matrix))
            "single"
            pyPINF
            ()
        )
        ''')[0][0]

    cluster1, cluster2 = (
        atom.get_object().value for atom in result.get_children()
    )
    cluster1_true, cluster2_true = [0], [1]

    assert cluster1 == cluster1_true
    assert cluster2 == cluster2_true


def test_agglomerative_merge_clusters(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (clusters)
            (:: (py-list (0)) (:: (py-list (1)) (:: (py-list (2)) ())))
        )

        (=
            (distance-matrix)
            (
                (0 1 2)
                (1 0 3)
                (2 3 0)
            )
        )

        """
    )

    result: Atom = metta.run(
        '''
        ! (agglomerative.merge-clusters
            (clusters)
            (np.array (distance-matrix))
            "single"
        )
        ''')[0][0]

    clusters = str(result)
    clusters_true = '(:: [2] (:: [0, 1] ()))'

    assert clusters == clusters_true


def test_agglomerative_init_clusters(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)
        """
    )

    result: Atom = metta.run('! (agglomerative.init-clusters 2)')[0][0]

    clusters = str(result)
    clusters_true = '(:: [1] (:: [0] ()))'

    assert clusters == clusters_true


def test_agglomerative_recursion(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (clusters)
            (:: (py-list (2)) (:: (py-list (0 1)) ()))
        )

        (=
            (flat-clusters)
            (:: (py-list (0 1 2)) ())
        )

        (=
            (distance-matrix)
            (
                (0 1 2)
                (1 0 3)
                (2 3 0)
            )
        )

        """
    )

    result: Atom = metta.run('! (agglomerative.recursion "single" (flat-clusters) (np.array (distance-matrix)) 1)')[0][0]
    history = str(result)
    history_true = '(:: (:: [0, 1, 2] ()) ())'
    assert history == history_true

    result: Atom = metta.run('! (agglomerative.recursion "single" (clusters) (np.array (distance-matrix)) 2)')[0][0]
    history = str(result)
    history_true = '(:: (:: [2] (:: [0, 1] ())) (:: (:: [2, 0, 1] ()) ()))'
    assert history == history_true


def test_agglomerative(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:agglomerative)

        (=
            (X)
            ((1 0) (0 1))
        )
        """
    )

    result: Atom = metta.run('! (agglomerative (np.array (X)) "single")')[0][0]

    clusters = str(result)
    clusters_true = '(:: (:: [1] (:: [0] ())) (:: (:: [1, 0] ()) ()))'

    assert clusters == clusters_true