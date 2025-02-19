from typing import List
from hyperon import MeTTa, ExpressionAtom, Atom, E, S, V, G, OperationAtom
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def metta_array(_metta: MeTTa, _var: str, _arr: List) -> None:
    _metta.run(f'! (bind! {_var} (np-array (py-atom "{_arr}")))')


def unwrap_array(_atom) -> Atom:
    arr: np.ndarray = _atom.get_object().value


random.seed(0)

n = 100
d = 2
k = 3

X, y = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=0.5, random_state=0) # (n,d) (n,)
centroids = X[random.choice(X.shape[0], k, replace=False)] # (k, d)
assignments = np.eye(k)[y].T # (k, n)

kmeans = KMeans(k).fit(X)
kmeans.cluster_centers_

metta_preamble = '''
! (bind! sklearn-kmeans (py-atom sklearn.cluster.KMeans))
! (bind! np (py-atom numpy))
! (bind! np-eye (py-dot np eye))
! (bind! np-array (py-dot np asarray))
! (bind! np-sum (py-dot np sum))
! (bind! np-argmin (py-dot np argmin))
! (bind! np-linalg-norm (py-dot np linalg.norm))
! (bind! np-update-centroids (py-atom "lambda _X, _assignments, _sum: (_assignments @ _X) / _sum(_assignments, axis=1, keepdims=True)"))
! (bind! np-assign (py-atom "lambda _X, _centroids, _argmin, _norm, _eye: _eye(_centroids.shape[0])[_argmin(_norm(_X[None, :, :] - _centroids[:, None, :], axis=-1), axis=0)].T"))
'''

metta_kmeans_definition = '''
(= 
    (kmeans $X $centroids $max-iter)
    (if (> $max-iter 0)
        (kmeans
            $X
            (np-update-centroids
                $X
                (np-assign 
                    $X 
                    $centroids
                    np-argmin
                    np-linalg-norm
                    np-eye
                )
                np-sum
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

X_metta = metta_array(metta, 'X', X.tolist())
y_metta = metta_array(metta, 'y', y.tolist())
centroids_metta = metta_array(metta, 'centroids', centroids.tolist())
assignments_metta = metta_array(metta, 'assignments', assignments.tolist())

result = metta.run('''
;! (kmeans X centroids 100)
! (
    (py-dot
        (sklearn-kmeans (py-atom "3")) fit) X)
''')
print(result)
print(centroids)