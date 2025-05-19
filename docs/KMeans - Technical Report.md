# KMeans Clustering in MeTTa

**Author:** Ramin Barati  
**Date:** May 19, 2025  
**Version:** 1.0  
**Repository:** [github.com/username/metta-kmeans](https://github.com/username/metta-kmeans)

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

where $\mu_i$ are centroids. We use Euclidean distance and a fixed maximum of 100 recursive iterations in MeTTa. Recursion replaces loops due to MeTTa’s execution model.

## MeTTa Implementation Details
All functions are defined with MeTTa type annotations and leverage NumPy calls:

- **Initialization**: random sampling via `(np.choose X k)`
- **Assignment** (`kmeans.assign`):
  ```metta
  (: kmeans.assign (-> (NPArray (n d)) (NPArray (k d)) (NPArray (k n))))
  (=
      (kmeans.assign X centroids)
      (np.transpose
          (np.one_hot
              (np.argmin
                  (np.linalg.norm
                      (np.sub
                          (np.expand_dims X 0)
                          (np.expand_dims centroids 1)
                      )
                      none
                      2
                  )
                  0
              )
              (np.shape centroids 0)
          )
      )
  )
  ```
- **Update** (`kmeans.update`):
  ```metta
  (: kmeans.update (-> (NPArray (n d)) (NPArray (k n)) (NPArray (k d))))
  (=
      (kmeans.update X assignments)
      (np.div
          (np.matmul assignments X)
          (np.sum assignments 1 none none True)
      )
  )
  ```
- **Recursion** (`kmeans.recursion`): applies assign & update up to `max-iter`:
  ```metta
  (: kmeans.recursion (-> (NPArray (n d)) (NPArray (k d)) Number (NPArray (k d))))
  (=
      (kmeans.recursion X centroids max-iter)
      (if (> max-iter 0)
          (kmeans.recursion
              X
              (kmeans.update X (kmeans.assign X centroids))
              (- max-iter 1)
          )
          centroids
      )
  )
  ```
- **Fit** (`kmeans.fit`): returns `KmeansModel` containing data, centers, and labels.

## Comparison with scikit‑learn
Both implementations are single‑threaded. scikit‑learn’s KMeans uses optimized C loops and k‑means++ initialization by default.

| Feature               | MeTTa                  | scikit‑learn          |
|-----------------------|------------------------|-----------------------|
| Initialization        | random via `np.choose` | k‑means++             |
| Distance metric       | Euclidean              | Euclidean             |
| Control flow          | recursion              | loops (C backend)     |
| Parallelization       | single‑threaded        | single‑threaded       |
| Convergence criterion | fixed max‑iter (100)   | tol on centroid shift |

## Benchmark Setup
- **Datasets** (500 samples each; seed=30 unless noted): `blobs`, `noisy_moons`, `noisy_circles`, `no_structure`, `aniso`, `varied`.  
- Synthetic generation uses scikit‑learn utilities (not shown).  
- Environment: CPU = 4 cores @ 3.0 GHz, Hyperon v0.2.2, NumPy 2.2.2, scikit‑learn 1.6.1.

## Results
**Performance comparison:**

| Dataset        | MeTTa Time (s) | scikit‑learn Time (s) | Silhouette | Calinski‑Harabasz | Davies‑Bouldin | ARI     | NMI    | AMI     |
| -------------- | -------------- | --------------------- | ---------- | ----------------- | -------------- | ------- | ------ | ------- |
| blobs          | 3.8121         | < 0.009               | 0.6542     | 1424.91           | 0.4793         | 0.9703  | 0.9544 | 0.9543  |
| noisy_moons    | 3.8136         | < 0.009               | 0.4956     | 690.81            | 0.8120         | 0.4834  | 0.3856 | 0.3848  |
| no_structure   | 3.7417         | < 0.009               | 0.3813     | 388.93            | 0.8689         | 0.0000  | 0.0000 | 0.0000  |
| varied         | 3.8929         | < 0.009               | 0.6395     | 1549.99           | 0.6104         | 0.7310  | 0.7345 | 0.7335  |
| noisy_circles  | 3.8889         | < 0.009               | 0.3471     | 281.73            | 1.1958         | −0.0017 | 0.0002 | −0.0013 |

MeTTa’s recursive overhead yields ~3.8 s runtimes, whereas scikit‑learn completes in under 0.009 s. Despite this gap, MeTTa’s implementation showcases the language’s ability to express complex numerical algorithms.

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
- **Recursion overhead:** fixed‑depth recursion incurs function-call costs.  
- **Initialization:** add k‑means++ seeding to improve convergence.  
- **Parallel recursion:** leverage MeTTa’s future runtime features for parallelism.  
- **Adaptive stopping:** monitor centroid movement for early exit.

## Conclusion
This implementation verifies that MeTTa can express core ML algorithms declaratively. While performance lags optimized C libraries, the clarity and extensibility in MeTTa pave the way for further ML primitives in the language.

## References
1. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.  
2. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.  
3. MeTTa Language Specification. http://www.metta‑lang.dev/spec
