import numpy as np
from sklearn.datasets import make_moons
from hyperon import MeTTa


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
    _metta.run(f"! (bind! {_var} (py-list {py_list}))")


# setting the random seed
np.random.seed(42)

# generate data
k = 2
max_iter = 10
X, y = make_moons(n_samples=10, noise=0.05, random_state=42)


# initializing the MeTTa instance
metta = MeTTa()

# binding the numpy as py-atom with alias 'np'
metta.run(
    """
    ! (bind! np (py-atom numpy))
"""
)

# registering the input data to the Space
metta_nparray_to_pylist(metta, "X", X)
metta_nparray_to_pylist(metta, "y", y)


# general utility numpy functions acting on py-list atoms
METTA_UTILS = """
(=
    (to-np $x)
    ((py-dot np array) $x)
)
(= 
    (matmul $x $y)
    (
        (py-dot                  
            ((py-dot np matmul) (to-np $x) (to-np $y))                
        tolist)
    )
)
(= 
    (shape $x $axis)
    ((py-dot (py-dot (to-np $x) shape) __getitem__) $axis)
)
(=
    (shape $x)
    (py-dot (to-np $x) shape)
)
(=
    (expand-dims $x $axis)
    ((py-dot ((py-dot np expand_dims)(Kwargs (a $x) (axis $axis))) tolist))
)
(= 
    (argmin $x $axis)
    ((py-dot 
        ((py-dot np argmin)(Kwargs (a $x) (axis $axis))) 
    tolist))

)
"""
metta.run(METTA_UTILS)


# kmeans functions
KMEANS_UTILS = """
; -> np.linalg.norm(np.expand_dims(X, axis=0) - np.expand_dims(centroids, axis=1), axis=-1)
(=
    (euclidean-distance $X $centroids)
    ((py-dot 
        (
        (py-dot np linalg.norm)(Kwargs 
            (x 
                ((py-dot 
                    ((py-dot (to-np (expand-dims $X 0)) __sub__) (to-np (expand-dims $centroids 1))) 
                tolist))
            ) 
            (axis -1)
        )
        )
    tolist))
)

; -> X[np.random.choice(X.shape[0], k, replace=False)]
(=
    (init-centroids $X $k)
    ((py-dot
        ((py-dot (to-np $X) __getitem__) ((py-dot np random.choice) (Kwargs (a (shape $X 0)) (size $k) (replace False)))) 
    tolist))
)

; -> np.matmul(assignments, X) / np.sum(assignments, axis=1, keepdims=True)
(=
    (update-centroids $assignments $X)
    (
        (py-dot            
        (
            (py-dot np divide) 
            (to-np (matmul $assignments $X)) 
            ((py-dot (to-np $assignments) sum) 1)
        )
        tolist)
    )
)

; -> 
; distances = np.linalg.norm(np.expand_dims(X, axis=0) - np.expand_dims(centroids, axis=1), axis=-1)
; labels = np.argmin(distances, axis=0)
; np.eye(centroids.shape[0])[labels].T
(=
    (assign $X $centroids)
    ((py-dot
        ((py-dot np transpose) 
            ((py-dot 
                ((py-dot np eye)(shape $centroids 0)) 
            __getitem__) (argmin (euclidean-distance $X $centroids) 0))
        )                 
    tolist))
)

; -> 
; assignments = assign(X, centroids)
; new_centroids = update_centroids(X, assignments)
; if np.allclose(centroids, new_centroids) or max_iter == 0:
;       return new_centroids
; else:
;       return recursive_kmeans(X, new_centroids, max_iter - 1)
(=
    (recursive-kmeans $X $centroids $max-iter)      
    (if
        (or 
            ((py-dot np allclose) (Kwargs (a $centroids) (b (update-centroids (assign $X $centroids) $X)))) 
            (== $max-iter 0)
        )
        (update-centroids (assign $X $centroids) $X)
    (recursive-kmeans $X $centroids (- $max-iter 1)))
)

; -> 
; centroids = init_centroids(X, k)
; centroids = recursive_kmeans(X, centroids, max_iter)
; assignments = assign(X, centroids)
(=
    (kmeans $X $k $max-iter)
    (assign 
        $X 
        (recursive-kmeans $X (init-centroids $X $k) $max-iter)
    )
)
"""
metta.run(KMEANS_UTILS)


# running the kmeans on X and k
metta_program = f"""
;! (kmeans X 3 1)
;! y

! (kmeans ((py-dot ((py-dot np eye) 3) tolist)) 3 1)
"""
# print(metta.run(metta_program))
pass
# metta_program = """
# ! X
# """
# print(metta.run(metta_program))
metta_program = """
(=
    (sqr-norm $X)
    ((py-dot 
        (
            (py-dot np sum) (
                (py-dot np pow) (to-np $X) 2) 
                (Kwargs (axis 1) (keepdims True)
            )
        )
    tolist))
)
"""
metta.run(metta_program)

metta_program = """
(=
    (sqr-distance $sqr-norm-X $X)
    ((py-dot
        ((py-dot            
            ((py-dot 
                (to-np $sqr-norm-X) 
            __add__) 
                ((py-dot np transpose) (to-np $sqr-norm-X))
            )
        __sub__) 
            ((py-dot np multiply) 2 ((py-dot np dot) (to-np $X) ((py-dot np transpose) (to-np $X))) ))
    tolist))
)
! (sqr-distance (sqr-norm X) X)
"""
metta.run(metta_program)
metta_program = """
(=
    (rbf-affinity-matrix $sqr-distance-X $sigma)
    ((py-dot 
        ((py-dot np exp)
            ((py-dot np divide) 
                ((py-dot np multiply) -1 (to-np $sqr-distance-X)) 
                ((py-dot np multiply) 2 ((py-dot np pow) $sigma 2))
            ) 
        )
    tolist))
)

;! (rbf-affinity-matrix (sqr-distance (sqr-norm X) X) 0.1)
"""
metta.run(metta_program)

metta_program = """
(=
    (degree $W)
    ((py-dot
        ((py-dot np sum) (to-np $W) 1)
    tolist))
)

(=
    (inverse-degree-matrix $degree-W)
    ((py-dot 
        ((py-dot np diag) ((py-dot np divide) 1 ((py-dot np sqrt) (to-np $degree-W))))
    tolist))
)

(=
    (normalized-laplacian $W $inverse-degree-matrix-W)
    ((py-dot ((py-dot ((py-dot np eye) (shape $W 0)) __sub__) ((py-dot (to-np $inverse-degree-matrix-W) __matmul__) ((py-dot (to-np $W) __matmul__) (to-np $inverse-degree-matrix-W)))) tolist))     
)

(= (W) (rbf-affinity-matrix (sqr-distance (sqr-norm X) X) 0.1))
(= (degree-W) (degree (W)))
(= (inverse-degree-matrix-W) (inverse-degree-matrix (degree-W)))
(= (L) (normalized-laplacian (W) (inverse-degree-matrix-W)))
"""
metta.run(metta_program)

metta_program = """
(= 
    (eigh $X)
    ((py-dot np linalg.eigh) (to-np $X))
)

(=
    (eigenvalues $eigh-X)
    ((py-dot 
        ((py-dot $eigh-X __getitem__) 0) 
    tolist))     
)

(=
    (eigenvectors $eigh-X)
    ((py-dot 
        ((py-dot $eigh-X __getitem__) 1) 
    tolist))     
)

;! ((py-dot ((py-dot np linalg.eigh) ((py-dot np eye) 3)) __getitem__) 0) 

;! ((py-dot ((py-dot np eye) 3) tolist))

(= (eigh-I-res) (eigh ((py-dot ((py-dot np eye) 3) tolist))))
! (eigh-I-res)
! (eigenvectors (eigh-I-res)) 
"""
metta.run(metta_program)

metta_program = """
(= (a) ((py-dot ((py-dot np arange) 10) tolist)))
(= (I) ((py-dot ((py-dot np eye) 3) tolist)))
;! (I)

(=
    (argsort $x)
    ((py-dot ((py-dot np argsort) (to-np $x)) tolist))
)
(= 
    (eigval-top-k-index $eigval-L $k) 
    ((py-dot 
        ((py-dot (to-np (argsort $eigval-L)) __getitem__) ((py-dot np arange) $k)) 
    tolist))
)

(= 
    (spectral-embeddings $eigval-L $eigvec-L $k) 
    ((py-dot
        ((py-dot np take)(Kwargs (a $eigvec-L) (indices (eigval-top-k-index $eigval-L $k) ) (axis 1))) 
    tolist))     
)

;! (spectral-embeddings (eigenvalues (eigh (L))) (eigenvectors (eigh (L))) 2)
"""
metta.run(metta_program)

metta_program = """
(= (U) (spectral-embeddings (eigenvalues (eigh (L))) (eigenvectors (eigh (L))) 2))
;! (U)

(=
    (row-normalize $U)
    ((py-dot 
        ((py-dot np divide) $U ((py-dot np linalg.norm)(Kwargs (x $U) (axis 1) (keepdims True))))
    tolist))   
)
;! (row-normalize (U))
"""
metta.run(metta_program)
# (normalized-laplacian (W) (inverse-degree-matrix-W))
metta_program = """
(=
    (spectral-clustering $X $k $max-iter)
    (kmeans 
        (row-normalize 
            (spectral-embeddings 
                (eigenvalues 
                    (eigh 
                        (normalized-laplacian 
                            (rbf-affinity-matrix (sqr-distance (sqr-norm $X) $X) 0.1)
                            (inverse-degree-matrix 
                                (degree 
                                    (rbf-affinity-matrix (sqr-distance (sqr-norm $X) $X) 0.1)
                                )
                            )
                        )
                    )
                ) 
                (eigenvectors 
                    (eigh 
                        (normalized-laplacian 
                            (rbf-affinity-matrix (sqr-distance (sqr-norm $X) $X) 0.1)
                            (inverse-degree-matrix 
                                (degree 
                                    (rbf-affinity-matrix (sqr-distance (sqr-norm $X) $X) 0.1)
                                )
                            )
                        )
                    )
                ) 
                $k
            )
        )
        $k 
        $max-iter
    )
)

(= (data) ((py-dot ((py-dot np eye) 3) tolist)))
! (to-np (spectral-clustering (data) 3 5))
"""
print(metta.run(metta_program))
