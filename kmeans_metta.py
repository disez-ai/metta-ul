from typing import List
from hyperon import MeTTa, ExpressionAtom, E, S, V, G, OperationAtom, ValueAtom
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def metta_nparray_to_pylist(_metta: MeTTa, _var: str, _x: np.ndarray) -> None:
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
    _metta.run(f'(= ({_var}) (np.array {py_list}))')

def pylist_to_metta(_arr: List) -> ExpressionAtom:
    res = []
    for element in _arr:
        if isinstance(element, list):
            res.append(pylist_to_metta(element))
        else:
            res.append(ValueAtom(element))

    return E(*res)

# def metta_array(_metta: MeTTa, _var: str, _arr: List) -> None:
#     _metta.run(E('=', S(_var), E('np.array')))


random.seed(0)

n = 10
d = 2
k = 3

X, y = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=0.5, random_state=0) # (n,d) (n,)
centroids = X[random.choice(X.shape[0], k, replace=False)] # (k, d)
assignments = np.eye(k)[y].T # (k, n)

metta_preamble = '''
! (import! &self numme)

(: Distance (-> Points Points Matrix))

(: euclidean Distance)
(=
    (euclidean $X $Y)

)
'''

metta_kmeans_definition = '''
(=
    (update-centroids $X $assignments)
    (np-divide
        (np-matmul ($assignments) ($X))
        (np-sum 
            ($assignments)
            (py-atom "1") 
            py-none
            py-none
            py-true
        )
    )
)

(=
    (assign $X $centroids)
    (np-transpose
        (np-one-hot-encode
            (np-argmin
                (np-linalg-norm
                    (np-subtract
                        (np-expand-dims $X (py-atom "0"))
                        (np-expand-dims $centroids (py-atom "1"))
                    )
                    py-none
                    (py-atom "-1")
                )
                (py-atom "0")
            )
            (np-eye (np-shape $centroids (py-atom "0")))
        )
    )
)

(= 
    (recursive-kmeans $X $centroids $max-iter)
    (if (> $max-iter 0)
        (recursive-kmeans
            $X
            (update-centroids
                $X
                (np-transpose
                    (np-one-hot-encode
                        (np.argmin
                            (np-linalg-norm
                                (np-subtract
                                    (np-expand-dims $X (py-atom "0"))
                                    (np-expand-dims $centroids (py-atom "1"))
                                )
                                py-none
                                (py-atom "-1")
                            )
                            (py-atom "0")
                        )
                        (np-eye (np-shape $centroids (py-atom "0")))
                    )
                )
            )
            (- $max-iter 1)
        )
        $centroids
    )
)
'''

metta = MeTTa()

metta.run(metta_preamble)
metta.run(metta_kmeans_definition)

metta_nparray_to_pylist(metta, 'X', X)
metta_nparray_to_pylist(metta, 'centroids', centroids)

# X_metta = pylist_to_metta(X.tolist())
# metta.space().add_atom(E(S('='), S('(X)'), X_metta))
# y_metta = metta_array(metta, 'y', y.tolist())
# centroids_metta = metta_array(metta, 'centroids', centroids.tolist())
# assignments_metta = metta_array(metta, 'assignments', assignments.tolist())

result = metta.run(
'''
! (update-centroids
    X
    (assign
        X
        centroids
    )               
)
'''
)

print(result)
print(centroids)