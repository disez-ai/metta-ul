# KMeans Clustering in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzadeh  
**Date:** May 19, 2025  
**Version:** 1.0  

---

## Abstract
This report describes an open‑source implementation of the KMeans clustering algorithm in the MeTTa language. It demonstrates MeTTa’s meta­graph and functional capabilities by expressing key operations—assignment, update, and recursion—without native loop constructs. Benchmarks on six synthetic datasets show MeTTa’s runtime (~3.8 s) compared to scikit‑learn’s single‑threaded C backend (< 0.009 s), highlighting trade‑offs between expressiveness and performance.

## Introduction
Clustering partitions data into groups of similar points. KMeans is a widely used algorithm that minimizes within‑cluster variance. Embedding clustering in MeTTa:

- Expands MeTTa’s numeric and declarative expressiveness.  
- Serves as a reference for future ML algorithm implementations.  
- Targets open‑source users in AI and Data Science.

MeTTa: a multi‑paradigm language for declarative and functional computations over knowledge (meta)graphs. See http://www.metta-lang.dev for details.

## Algorithm Overview
KMeans seeks to minimize
```math
J = \sum_{i=1}^{k} \sum_{x\in C_i} \|x - \mu_i\|^2
```

where $\mu_i$ are centroids. We use Euclidean distance and continue untill either a fixed maximum of recursive iterations is reached or the change in cluster centers is less than some value in MeTTa. Recursion replaces loops due to MeTTa’s execution model.


## MeTTa Implementation Details

The MeTTa KMeans implementation uses a purely declarative, recursive style to express the core steps of the algorithm without explicit loops. Data is represented as NumPy arrays wrapped in MeTTa types, and centroids, assignments, and models are built via pattern‐matched function definitions.

1. **Initialization**  
   A fixed number of `k` centroids are sampled at random from the input matrix `X` using MeTTa’s binding to NumPy’s array‐indexing (`np.choose`).  
   - **Time Complexity:** O(k) to sample indices.  
   - **Memory Overhead:** O(k·d) to store the initial centroids.

2. **Assignment Step**  
   For each of the `n` data points, Euclidean distances to all `k` centroids are computed via a broadcasted norm operation. The index of the nearest centroid is converted into a one‐hot encoded assignment vector.  
   - **Time Complexity:** O(n·k·d) for distance calculations and argmin.  
   - **Memory Overhead:** O(n·k) for the one‐hot assignment matrix plus O(n·d) for intermediate expanded arrays.

3. **Update Step**  
   Centroids are recomputed by multiplying the assignment matrix by the data matrix (`assignments × X`) and normalizing by the cluster counts (sum over assignments).  
   - **Time Complexity:** O(n·k·d) for matrix multiplication and O(k·n) for sum reduction.  
   - **Memory Overhead:** O(k·d) for new centroids and O(k·n) for assignment reuse.

4. **Recursive Loop with Early Stopping**  
   The functions `assign` and `update` are composed in a recursion. Each recursive call passes the newly updated centroids to the next iteration.  
   - **Time Complexity (worst-case):** O(max_iter · n · k · d).  
   - **Time Complexity (average-case):** O(m · n · k · d), where m is the iteration count until tolerance is reached (m ≤ max_iter).  
   - **Memory Overhead:** O(n·k + k·d + n·d); recursion can use tail‑call optimization, so stack does not grow.

5. **Model Fit & Predict**  
   - `kmeans.fit(X, k, max_iter, tol)` performs recursion and returns centroids.  
   - `kmeans.predict(X, centers)` uses `argmax` on assignment matrix to output labels.  
   - **Predict Complexity:** O(n·k·d) for assignment plus O(n·k) for argmax.

The early‑stop mechanism often reduces iterations needed, improving average‑case performance while preserving worst‑case bounds.

## Comparison with scikit‑learn
Both implementations are single‑threaded. scikit‑learn’s KMeans uses optimized C loops and k‑means++ initialization by default.

| Feature               | MeTTa                  | scikit‑learn          |
|-----------------------|------------------------|-----------------------|
| Initialization        | random via `np.choose` | k‑means++             |
| Distance metric       | Euclidean              | Euclidean             |
| Control flow          | recursion              | loops (C backend)     |
| Parallelization       | single‑threaded        | single‑threaded       |
| Convergence criterion | tol on centroid shift  | tol on centroid shift |

## Benchmark Setup
- **Datasets** (500 samples each; seed=30 unless noted): `blobs`, `noisy_moons`, `noisy_circles`, `no_structure`, `aniso`, `varied`.  
- Synthetic generation uses scikit‑learn utilities (not shown).  
- Environment: CPU = 4 cores @ 3.0 GHz, Hyperon 0.2.2, NumPy 2.2.2, scikit‑learn 1.6.1.

## Results
**Performance comparison:**

| Dataset        | MeTTa Time (s) | scikit‑learn Time (s) | Silhouette | Calinski‑Harabasz | Davies‑Bouldin | ARI     | NMI    | AMI     |
| -------------- | -------------- | --------------------- | ---------- | ----------------- | -------------- | ------- | ------ | ------- |
| blobs          | 3.8121         | < 0.01                | 0.6542     | 1424.91           | 0.4793         | 0.9703  | 0.9544 | 0.9543  |
| noisy_moons    | 3.8136         | < 0.01                | 0.4956     | 690.81            | 0.8120         | 0.4834  | 0.3856 | 0.3848  |
| no_structure   | 3.7417         | < 0.01                | 0.3813     | 388.93            | 0.8689         | 0.0000  | 0.0000 | 0.0000  |
| varied         | 3.8929         | < 0.01                | 0.6395     | 1549.99           | 0.6104         | 0.7310  | 0.7345 | 0.7335  |
| noisy_circles  | 3.8889         | < 0.01                | 0.3471     | 281.73            | 1.1958         | −0.0017 | 0.0002 | −0.0013 |

MeTTa’s recursive overhead yields ~3.8 s runtimes, whereas scikit‑learn completes in under 0.01 s. Despite this gap, MeTTa’s implementation showcases the language’s ability to express complex numerical algorithms.

## Usage Example
```metta
(import! &self metta_ul)

(let $clusters (kmeans.fit X 3)
  (println! $clusters)
```
This snippet:
1. Imports the clustering module.  
2. Fits KMeans with `k=3`.  
3. Prints cluster labels.

## Limitations & Future Work
- **Initialization:** add k‑means++ seeding to improve convergence.

## Conclusion
This implementation verifies that MeTTa can express core ML algorithms declaratively. While performance lags optimized C libraries, the clarity and extensibility in MeTTa pave the way for further ML primitives in the language.

## References
1. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.  
2. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.  
3. MeTTa Language Specification. http://www.metta‑lang.dev/spec
