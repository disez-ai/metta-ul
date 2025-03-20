import numpy as np


def initialize_gmm(X, K, seed=42):
    """Initialize the GMM parameters randomly."""
    np.random.seed(seed)
    N, D = X.shape
    means = X[
        np.random.choice(N, K, replace=False)
    ]  # Randomly choose K data points as initial means
    covariances = np.repeat(
        np.cov(X, rowvar=False)[np.newaxis, :, :], K, axis=0
    ) + 1e-6 * np.eye(
        D
    )  # Add small noise for stability
    weights = np.ones(K) / K  # Equal weights initially
    return weights, means, covariances


def gaussian_pdf(X, means, covariances):
    """Compute the probability density function of a Gaussian for all components and data points."""
    N, D = X.shape
    K = means.shape[0]

    cov_inv = np.linalg.inv(covariances)  # (K, D, D)
    _, log_det_cov = np.linalg.slogdet(covariances)  # (K,)

    # Compute Mahalanobis distance
    X_centered = X[:, np.newaxis, :] - means  # (N, K, D)
    mahalanobis = np.einsum(
        "nkd,kde,nke->nk", X_centered, cov_inv, X_centered
    )  # (N, K)

    # Compute Gaussian probability
    log_prob = -0.5 * (mahalanobis + D * np.log(2 * np.pi) + log_det_cov)  # (N, K)

    return np.exp(log_prob)  # Convert log to probability


def log_likelihood(X, weights, means, covariances):
    """Compute the total log-likelihood of the data under the GMM."""
    pdfs = gaussian_pdf(X, means, covariances)  # (N, K)
    weighted_pdfs = pdfs * weights  # (N, K)
    return np.sum(np.log(np.sum(weighted_pdfs, axis=1)))  # Scalar


def gmm_em_recursive(
    X,
    K,
    weights,
    means,
    covariances,
    prev_log_likelihood=-np.inf,
    tol=1e-6,
    iteration=1,
    max_iters=100,
):
    """
    Recursive implementation of the EM algorithm for a Gaussian Mixture Model.

    Parameters:
    X : (N, D) array - Data points
    K : int - Number of Gaussian components
    weights : (K,) array - Mixture weights
    means : (K, D) array - Mean of each Gaussian component
    covariances : (K, D, D) array - Covariance matrices
    prev_log_likelihood : float - Log-likelihood from previous iteration
    tol : float - Convergence tolerance
    iteration : int - Current iteration number
    max_iters : int - Maximum number of EM iterations

    Returns:
    weights, means, covariances : Learned GMM parameters
    """
    if iteration > max_iters:
        return weights, means, covariances

    # E-Step: Compute responsibilities
    pdfs = gaussian_pdf(X, means, covariances)  # (N, K)
    weighted_pdfs = pdfs * weights  # (N, K)
    responsibilities = weighted_pdfs / np.sum(
        weighted_pdfs, axis=1, keepdims=True
    )  # (N, K)

    # M-Step: Update parameters
    Nk = np.sum(responsibilities, axis=0)  # (K,)
    weights = Nk / X.shape[0]  # (K,)
    means = (responsibilities.T @ X) / Nk[:, np.newaxis]  # (K, D)

    # Update covariances
    X_centered = X[:, np.newaxis, :] - means  # (N, K, D)
    covariances = (
        np.einsum("nk,nkd,nke->kde", responsibilities, X_centered, X_centered)
        / Nk[:, np.newaxis, np.newaxis]
    )  # (K, D, D)

    # Log-likelihood computation
    log_likelihood_value = log_likelihood(X, weights, means, covariances)
    print(f"Iteration {iteration}: Log-Likelihood = {log_likelihood_value:.6f}")

    # Convergence check
    if np.abs(log_likelihood_value - prev_log_likelihood) < tol:
        return weights, means, covariances

    # Recursive call for next iteration
    return gmm_em_recursive(
        X,
        K,
        weights,
        means,
        covariances,
        log_likelihood_value,
        tol,
        iteration + 1,
        max_iters,
    )


# Example usage:
N, D, K = 300, 2, 3  # 300 samples, 2D data, 3 Gaussian components
X = np.vstack(
    [np.random.randn(N // K, D) + np.array([i * 5, i * 5]) for i in range(K)]
)  # Generate synthetic clustered data

# Initialize parameters
weights, means, covariances = initialize_gmm(X, K)

# Train GMM using recursive EM
weights, means, covariances = gmm_em_recursive(X, K, weights, means, covariances)

print("\nFinal Parameters:")
print("Weights:", weights)
print("Means:\n", means)
print("Covariances:\n", covariances)
