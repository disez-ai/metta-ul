from typing import List
from hyperon import MeTTa, ExpressionAtom, E, S, ValueAtom
import numpy.random as random
from sklearn.datasets import make_blobs


def pylist_to_metta(_arr: List) -> ExpressionAtom:
    res = []
    for element in _arr:
        if isinstance(element, list):
            res.append(pylist_to_metta(element))
        else:
            res.append(ValueAtom(element))

    return E(*res)


random.seed(1)

n = 100
d = 2
k = 3

X, y, centroids_true = make_blobs(
    n_samples=n,
    n_features=2,
    centers=k,
    cluster_std=0.5,
    return_centers=True,
    random_state=0,
)  # (n,d) (n,)


def test_python_metta_direct():
    metta = MeTTa()

    metta.run("!(import! &self metta_ul:cluster:kmeans)")

    X_metta = pylist_to_metta(X.tolist())
    metta.space().add_atom(E(S("="), E(S("X")), X_metta))

    result = metta.run(
        """
    ! (kmeans (np.array (X)) 3)
    """
    )

    print(result)
    print(centroids_true)
