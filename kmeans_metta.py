from typing import List
from hyperon import MeTTa, ExpressionAtom, E, S, V, G, OperationAtom
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# def metta_array(_metta: MeTTa, _var: str, _arr: List, add_to_space=True) -> ExpressionAtom:
#     res = _metta.parse_single(
#         f'''(= 
#                 ({_var})
#                 (np-array
#                     (py-atom "{_arr}")
#                 )
#             )''')

#     if add_to_space:
#         _metta.space().add_atom(res)

#     return res

def metta_array(_metta: MeTTa, _var: str, _arr: List) -> None:
    _metta.run(f'! (bind! {_var} (np-array (py-atom "{_arr}")))')


random.seed(0)

n = 10
d = 2
k = 3

X, y = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=0.5, random_state=0) # (n,d) (n,)
centroids = X[random.choice(X.shape[0], k, replace=False)] # (k, d)
assignments = np.eye(k)[y].T # (k, n)

metta_preamble = '''
! (bind! py-none (py-atom "None"))
! (bind! py-false (py-atom "False"))
! (bind! py-true (py-atom "True"))
! (bind! py-slice (py-atom slice))
! (bind! np (py-atom numpy))
! (bind! np-eye (py-dot np eye))
! (bind! np-array (py-dot np asarray))
! (bind! np-matmul (py-dot np matmul))
! (bind! np-sum (py-dot np sum))
! (bind! np-divide (py-dot np divide))
! (bind! np-subtract (py-dot np subtract))
! (bind! np-expand-dims (py-dot np expand_dims))
! (bind! np-argmin (py-dot np argmin))
! (bind! np-one-hot-encode (py-atom "lambda labels, eye: eye[labels]"))
! (bind! np-linalg-norm (py-dot np linalg.norm))
! (bind! np-transpose (py-dot np transpose))
! (bind! np-shape (py-atom "lambda x, i: x.shape[i]"))
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
                        (np-expand-dims ($X) (py-atom "0"))
                        (np-expand-dims ($centroids) (py-atom "1"))
                    )
                    py-none
                    (py-atom "-1")
                )
                (py-atom "0")
            )
            (np-eye (np-shape ($centroids) (py-atom "0")))
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
(=
    (list)
    ((py-dot (np-array (py-list (1.0 2.0 3.0))) tolist))
)
                   
(=
    (pop [$x $xs])
    ($x)
)
! (pop (list))
''')

print(result)
print(centroids)