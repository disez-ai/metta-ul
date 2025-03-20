import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from kmeans_python import kmeans


def compute_affinity(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the affinity matrix W using the Gaussian (RBF) kernel.

    The affinity matrix is defined as:
        W_{ij} = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data points.
    sigma : float
        Scaling parameter for the Gaussian kernel.

    Returns
    -------
    W : np.ndarray of shape (n_samples, n_samples)
        The computed affinity matrix.
    """
    sq_norms = np.sum(X**2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
    W = np.exp(-sq_dists / (2 * sigma**2))
    return W


def compute_normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized Laplacian matrix L_sym.

    The Laplacian is defined as:
        L_sym = I - D^{-1/2} * W * D^{-1/2}
    where D is the degree matrix with:
        D_{ii} = sum(W_{ij})

    Parameters
    ----------
    W : np.ndarray of shape (n_samples, n_samples)
        Affinity matrix.

    Returns
    -------
    L_sym : np.ndarray of shape (n_samples, n_samples)
        The symmetric normalized Laplacian.
    """
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_sym


def spectral_embedding(L_sym: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the first k eigenvectors of the Laplacian matrix.

    The eigen decomposition solves:
        L_sym * U = U * Lambda

    Parameters
    ----------
    L_sym : np.ndarray of shape (n_samples, n_samples)
        The symmetric normalized Laplacian matrix.
    k : int
        The number of eigenvectors to return.

    Returns
    -------
    U : np.ndarray of shape (n_samples, k)
        The first k eigenvectors of L_sym.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    idx = np.argsort(eigenvalues)
    U = eigenvectors[:, idx[:k]]
    return U


def row_normalize(U: np.ndarray) -> np.ndarray:
    """
    Normalize each row of the embedding matrix to have unit norm.

    Parameters
    ----------
    U : np.ndarray of shape (n_samples, k)
        Spectral embedding matrix.

    Returns
    -------
    U_norm : np.ndarray of shape (n_samples, k)
        Row-normalized spectral embedding.
    """
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return U / norms


def apply_kmeans(U_norm: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    """
    Apply k-means clustering to the spectral embedding.

    Parameters
    ----------
    U_norm : np.ndarray of shape (n_samples, k)
        Normalized spectral embedding.
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations for k-means.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments for each data point.
    """
    centroids, assignments = kmeans(X=U_norm, k=k, max_iter=max_iter)
    labels = np.argmax(assignments, axis=0)
    return labels


def spectral_clustering(
    X: np.ndarray, k: int, sigma: float = 0.1, max_iter: int = 1000
) -> np.ndarray:
    """
    Perform spectral clustering using the normalized Laplacian.

    Steps:
      1. Compute the affinity matrix W using a Gaussian kernel.
      2. Compute the symmetric normalized Laplacian L_sym.
      3. Compute the spectral embedding (first k eigenvectors).
      4. Row-normalize the embedding.
      5. Apply k-means clustering to the embedding.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data points.
    k : int
        Number of clusters.
    sigma : float, optional (default=0.1)
        Scaling parameter for the Gaussian kernel.
    max_iter : int, optional (default=1000)
        Maximum number of iterations for k-means.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments for each data point.
    """
    W = compute_affinity(X=X, sigma=sigma)
    L_sym = compute_normalized_laplacian(W=W)
    U = spectral_embedding(L_sym=L_sym, k=k)
    U_norm = row_normalize(U=U)
    labels = apply_kmeans(U_norm=U_norm, k=k, max_iter=max_iter)
    return labels


def main():
    np.random.seed(42)
    X, y = make_moons(n_samples=500, noise=0.05, random_state=30)

    labels = spectral_clustering(X, k=2, sigma=0.1, max_iter=1000)
    print("Cluster assignments:")
    print(labels)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k", s=80, alpha=0.6
    )
    plt.title("Spectral Clustering on 2D Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == "__main__":
    main()
