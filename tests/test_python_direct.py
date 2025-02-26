from typing import List
from hyperon import MeTTa, ExpressionAtom, E, S, ValueAtom
import numpy.random as random
import numpy as np
from sklearn.datasets import make_blobs


def test_python_kmeans():

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

    metta = MeTTa()

    metta.run("!(import! &self metta_ul:cluster:kmeans)")

    X_metta = pylist_to_metta(X.tolist())
    metta.space().add_atom(E(S("="), E(S("X")), X_metta))

    result = metta.run("!(kmeans (np.array (X)) 3)")

    print(result)
    print(centroids_true)


def metta_nparray_to_pylist(_metta: MeTTa, _var: str, _x: np.ndarray) -> None:
    """
    Converts a NumPy array to a MeTTa py-list and binds it to a variable in the AtomSpace.

    The function recursively converts the NumPy array into a nested Python list,
    formats it as a MeTTa py-list string, and executes a binding command in MeTTa.

    Parameters
    ----------
    _metta : MeTTa
        An instance of the MeTTa interpreter.
    _var : str
        The name of the variable to bind the converted py-list to.
    _x : np.ndarray
        The NumPy array to be converted.
    """
    def list_to_py_list(lst):
        """
        Recursively converts a nested Python list into a string in the py-list format.
        """
        if isinstance(lst, list):
            # Recursively convert each element and join with a space
            return "(" + " ".join(list_to_py_list(item) for item in lst) + ")"
        else:
            # Convert non-list element (e.g., int, float) to string
            return str(lst)

    py_list = list_to_py_list(_x.tolist())
    _metta.run(f'! (bind! {_var} (py-list {py_list}))')


def test_py_dot_kmeans():
    # setting the random seed
    random.seed(1)

    # defining the problem hyperparameters
    n = 5
    d = 2
    k = 2
    max_iter = 1

    # generating the synthetic dataset
    X, y = make_blobs(n_samples=n, n_features=2, centers=k,
                      cluster_std=0.5, random_state=0)  # (n,d) (n,)

    # initializing the MeTTa instance
    metta = MeTTa()

    # registering the input data to the Space
    metta_nparray_to_pylist(metta, 'X', X)
    metta_nparray_to_pylist(metta, 'y', y)

    metta.run("!(import! &self metta_ul:cluster:py_dot_kmeans)")
    # running the kmeans on X and k
    metta_program = f"""
! (kmeans X {k} {max_iter})
! y
    """
    print(metta_program)
    print(metta.run(metta_program))
