from typing import List
from hyperon import MeTTa, ExpressionAtom, E, S, ValueAtom
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
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

metta_preamble = """
! (import! &self numme)

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))
"""

metta_kmeans_definition = """
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

"""

metta = MeTTa()

metta.run(metta_preamble)
metta.run(metta_kmeans_definition)

X_metta = pylist_to_metta(X.tolist())
metta.space().add_atom(E(S("="), E(S("X")), X_metta))

result = metta.run(
    """
! (kmeans (np.array (X)) 3)
"""
)

print(result)
print(centroids_true)
