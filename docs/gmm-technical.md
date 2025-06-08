# Gaussian Mixture Model (GMM) Clustering in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzedeh  
**Date:** May 21, 2025  
**Version:** 1.0

---

## Abstract
This report details an open‑source implementation of the Gaussian Mixture Model (GMM) clustering algorithm in the MeTTa language. It showcases MeTTa’s declarative and functional strengths by expressing the Expectation–Maximization steps—without native loops—using recursive function definitions and NumPy bindings. Performance considerations and complexity analyses are discussed in the context of future improvements.

## Introduction
Probabilistic clustering via GMM models each cluster as a multivariate Gaussian, allowing soft assignments and richer cluster shapes compared to KMeans. Embedding GMM in MeTTa:

- Extends MeTTa’s numeric capabilities with probabilistic models.  
- Provides a template for future statistical algorithm implementations.  
- Targets open‑source users in AI and Data Science.

MeTTa is a multi‑paradigm language for declarative computations over knowledge graphs. See http://www.metta-lang.dev for more information.

## Algorithm Overview
GMM uses the Expectation–Maximization (EM) algorithm to maximize data likelihood under a mixture of Gaussians:

1. **E‑Step:** Compute responsibilities $r_{ik} = p(z_k|x_i)$ via Gaussian PDFs and mixture weights.  
2. **M‑Step:** Update weights $\pi_k$, means $\mu_k$, and covariances $\Sigma_k$ based on responsibilities.  
3. Repeat until convergence or max iterations.

Mathematically:

- **Gaussian PDF:**
  ```math
  p_k(x) = \exp\bigl(-\tfrac12 (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k) - \tfrac{d}{2}\ln(2\pi) - \tfrac12 \ln|\Sigma_k|\bigr)
  ```
- **Log‑Likelihood:**
```math
  L = \sum_{i=1}^n \ln \Bigl( \sum_{k=1}^K \pi_k p_k(x_i) \Bigr)
```

## MeTTa Implementation Details
The GMM implementation uses MeTTa function definitions with NumPy bindings:

1. **Centering**  
   Subtract means from data via broadcasting.  
   - **Time Complexity:** O(n·k·d).  
   - **Memory:** O(n·k·d).

2. **Mahalanobis Term**  
   Compute $(x_i-\mu_k)^T \Sigma_k^{-1}(x_i-\mu_k)$ with `np.einsum`.  
   - **Time Complexity:** O(n·k·d^2).  
   - **Memory:** O(n·k).

3. **Gaussian PDF**  
   Evaluate the multivariate Gaussian density using the Mahalanobis term, determinant, and normalization.  
   - **Time Complexity:** O(n·k·d^2).  
   - **Memory:** O(n·k).

4. **Log‑Likelihood**  
   Sum log of weighted PDFs across data points.  
   - **Time Complexity:** O(n·k).  
   - **Memory:** O(n·k).

5. **Initialization (`gmm.init`)**  
   - Weights: uniform $\frac{1}{K}$.  
   - Means: random points via `np.choose`.  
   - Covariances: data covariance + small identity noise.  
   - **Time Complexity:** O(n·d^2).  
   - **Memory:** O(k·d^2).

6. **E‑Step (`gmm.e-step`)**  
   Compute responsibilities $r_{ik} = \pi_k p_k(x_i) / \sum_j \pi_j p_j(x_i)$.  
   - **Time Complexity:** O(n·k).  
   - **Memory:** O(n·k).

7. **M‑Step (`gmm.m-step`)**  
   Update:
   - $N_k = \sum_i r_{ik}$
   - $\pi_k = N_k / n$
   - $\mu_k = (1/N_k) \sum_i r_{ik} x_i$
   - $\Sigma_k = (1/N_k) \sum_i r_{ik} (x_i-\mu_k)(x_i-\mu_k)^T$  
   
   - **Time Complexity:** O(n·k·d^2).  
   - **Memory:** O(k·d^2 + n·k).

8. **Recursive EM Loop**  
   Repeat E‑ and M‑Steps up to `max_iter`.  
   - **Time Complexity:** O(max_iter · n · k · d^2).  
   - **Memory:** Tail‑calls reuse frames; peak O(n·k·d^2).

9. **Predict (`gmm.predict`)**  
   Assign cluster by highest responsibility.  
   - **Time Complexity:** O(n·k).  
   - **Memory:** O(n·k).


## Comparison with scikit‑learn
scikit‑learn provides `GaussianMixture` with C-optimized loops and multiple covariance options. Future work may include:

| Feature             | MeTTa Implementation       | scikit‑learn             |
|---------------------|----------------------------|--------------------------|
| Covariance types    | full only                  | full, tied, diag, spherical |
| Initialization      | random                     | k‑means++, random        |
| Convergence control | tol on log‑likelihood      | tol on log‑likelihood    |
| Parallelization     | single‑threaded            | single‑threaded          |

## Benchmark Setup
| Dataset        | MeTTa Time (s) | scikit-learn Time (s) | Silhouette | Calinski-Harabasz | Davies-Bouldin | Adjusted Rand Index | Normalized Mutual Info | Adjusted Mutual Info |
| -------------- | --------------- | ---------------------- | ---------- | ------------------ | --------------- | ------------------- | ---------------------- | -------------------- |
| varied         | 15.125          | <0.01                  | 0.58807    | 1198.59            | 0.68460         | 0.94682             | 0.91605                | 0.91574              |
| blobs          | 4.519           | <0.01                  | 0.65379    | 1422.54            | 0.47993         | 0.96444             | 0.94787                | 0.94768              |
| noisy_circles  | 149.801         | <0.01                  | 0.34909    | 281.92             | 1.19307         | -0.00199            | 0.00001                | -0.00144             |
| noisy_moon     | 78.848          | <0.01                  | 0.35369    | 297.61             | 0.88993         | 0.23313             | 0.33222                | 0.33114              |
| no-structure   | 151.017         | <0.01                  | 0.22208    | 181.17             | 1.60891         | 0                   | 0                      | 0                    |

## Usage Example
```metta
(import! &self metta_ul:cluster:gmm)

(let $params (gmm.fit X 3 50)
  (println! (gmm.predict X $params))
)
```

## Limitations & Future Work
- Support additional covariance structures.

## Conclusion
The GMM implementation demonstrates MeTTa’s ability to express probabilistic EM algorithms declaratively. While performance depends on NumPy backends, the clear, recursive definitions simplify future extensions.

## References
1. Dempster, Laird & Rubin (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm.  
2. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.  
3. MeTTa Language Specification. http://www.metta-lang.dev/spec
