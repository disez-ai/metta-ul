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
