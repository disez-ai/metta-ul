import numpy as np
import numpy.random as random

random.seed(0)

X: np.ndarray = random.randn(100, 2)


def kmeans(X, k, max_iter=100):
    def init_centroids(X, k):
        return X[random.choice(X.shape[0], k, replace=False)]

    def update_centroids(X, assignments) -> np.ndarray:
        """
        X: (n, d)
        assignments: (k, n)
        """
        return np.matmul(assignments, X) / np.sum(assignments, axis=1, keepdims=True)

    def assign(X, centroids):
        """
        X: (n, d)
        centroids: (k, d)
        """
        X = np.expand_dims(X, 0)  # (1, n, d)
        centroids = np.expand_dims(centroids, 1)  # (k, 1, d)
        labels = np.argmin(np.linalg.norm(X - centroids, axis=-1), axis=0)  # (n,)
        return np.eye(k)[labels].T  # (k, n)

    def recursive_kmeans(X, centroids, max_iter):
        assignments = assign(X, centroids)
        new_centroids = update_centroids(X, assignments)
        if np.allclose(centroids, new_centroids) or max_iter == 0:
            return new_centroids
        else:
            return recursive_kmeans(X, new_centroids, max_iter - 1)

    centroids = init_centroids(X, k)
    new_centroids = recursive_kmeans(X, centroids, max_iter)
    assignments = assign(X, new_centroids)
    return new_centroids, assignments
