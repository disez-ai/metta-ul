import numpy as np
from sklearn.datasets import make_blobs
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
    _metta.run(f'! (bind! {_var} (py-list {py_list}))')


# setting the random seed
np.random.seed(0)

# defining the problem hyperparameters
n = 5
d = 2
k = 2
max_iter = 10

# generating the synthetic dataset
X, y = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=0.5, random_state=0)  # (n,d) (n,)


# initializing the MeTTa instance
metta = MeTTa()

# binding the numpy as py-atom with alias 'np'
metta.run("""
    ! (bind! np (py-atom numpy))
""")

# registering the input data to the Space
metta_nparray_to_pylist(metta, 'X', X)
metta_nparray_to_pylist(metta, 'y', y)

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
! (kmeans X {k} {max_iter})
! y
"""
print(metta.run(metta_program))

