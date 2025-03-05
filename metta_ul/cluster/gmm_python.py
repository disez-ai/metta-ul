import numpy as np

def log_likelihood_gmm(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> float:
    """
    Compute the log-likelihood of data X under a Gaussian Mixture Model using NumPy.

    Parameters:
    X : (N, D) array - Data points
    weights : (K,) array - Mixture weights
    means : (K, D) array - Mean of each Gaussian component
    covariances : (K, D, D) array - Covariance matrices of each component

    Returns:
    log_likelihood : float - Log-likelihood of the dataset
    """
    N, D = X.shape
    K = weights.shape[0]

    # Compute determinant and inverse of covariance matrices
    cov_inv = np.linalg.inv(covariances)  # (K, D, D)
    _, log_det_cov = np.linalg.slogdet(covariances)  # (K,)

    # Compute Mahalanobis distance: (X - mu)^T @ Sigma^-1 @ (X - mu)
    X_centered = X[:, np.newaxis, :] - means  # (N, K, D)
    mahalanobis_term = np.einsum('nkd,kde,nke->nk', X_centered, cov_inv, X_centered)  # (N, K)

    # Compute exponent term of Gaussian
    exponent = -0.5 * (mahalanobis_term + D * np.log(2 * np.pi) + log_det_cov)  # (N, K)

    # Compute weighted sum over components (sum over K, then log)
    weighted_pdfs = np.exp(exponent) * weights  # (N, K)
    log_likelihood = np.sum(np.log(np.sum(weighted_pdfs, axis=1)))  # Scalar

    return log_likelihood

# Example usage:
np.random.seed(0)

N, D, K = 100, 2, 3  # 100 samples, 2D data, 3 Gaussian components
X = np.random.randn(N, D)  # Generate random data
weights = np.array([0.3, 0.5, 0.2])  # Mixture weights
means = np.random.randn(K, D)  # Random means
covariances = np.array([np.eye(D) for _ in range(K)])  # Identity covariance matrices

log_likelihood = log_likelihood_gmm(X, weights, means, covariances)
print("Log-Likelihood:", log_likelihood)
