# Agglomerative Clustering in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzadeh  
**Date:** May 21, 2025  
**Version:** 1.0  

---

## Abstract
This report presents an open‑source implementation of agglomerative (hierarchical) clustering in the MeTTa language. By leveraging MeTTa’s declarative recursion and linked‐list constructs, it performs bottom‐up cluster merging with selectable linkage criteria (single, complete, average). We analyze time and memory complexities at each step and compare design features to scikit‑learn’s `AgglomerativeClustering`.

## Introduction
Hierarchical clustering builds nested partitions by iteratively merging or splitting clusters. Agglomerative clustering starts with each point as its own cluster and merges the closest pair until `k` clusters remain. Implementing this in MeTTa:

- Demonstrates MeTTa’s capability to express non‐parametric, recursive algorithms.  
- Provides a template for tree‐based clustering methods.  
- Targets AI and Data Science open‑source contributors.

MeTTa is a multi‑paradigm language for functional and declarative computations over meta‑graphs. See http://www.metta‑lang.dev.

## Algorithm Overview
1. **Initialization**: Start with `n` singleton clusters.  
2. **Distance Matrix**: Compute all pairwise distances.  
3. **Linkage**: Define cluster distance via single (min), complete (max), or average metrics.  
4. **Merge Loop**: Repeatedly find and merge the two closest clusters until `k` remain.  
5. **Assignment**: Label each data point by its final cluster.

## MeTTa Implementation Details
The MeTTa code uses recursive definitions and NumPy bindings (`metta_ul`) plus a linked‐list module for cluster lists.

1. **Cluster Initialization** (`agglomerative.init-clusters`):
   - Recursively builds a `List` of singleton `PyList`s containing each index from `0` to `n-1`.
   - *Time:* O(n)  *Memory:* O(n)

2. **Distance Matrix Computation** (`agglomerative.distance-matrix`):
   - Computes an n-by-n symmetric matrix of Euclidean distances using `np.expand_dims` and `np.linalg.norm` over the last axis.
   - *Time:* O(n²·d)  *Memory:* O(n²)

3. **Linkage Distance** (`agglomerative.linkage-distance`):
   - For clusters C1, C2, extracts submatrix of pairwise distances and applies:
     - **single**: min  **complete**: max  **average**: mean
   - *Time:* O(|C1|·|C2|)  *Memory:* O(|C1|·|C2|)

4. **Closest Cluster Pair** (`agglomerative.closest-clusters`):
   - Recursively iterates all cluster pairs via two‐list traversal, tracking the minimum linkage distance.
   - *Time:* O(m²·p) where m=current cluster count, p=cluster‐pair cost; worst‐case O(n⁴) but decreases as clusters merge.  
   - *Memory:* negligible extra beyond clusters list.

5. **Merge Clusters** (`agglomerative.merge-clusters`):
   - Uses `closest-clusters` to identify clusters (c1, c2), concatenates their index lists, and updates the cluster `List`.
   - *Time & Memory:* dominated by `closest-clusters`, plus O(|c1|+|c2|) list operations.

6. **Recursive Merge Loop** (`agglomerative.recursion`):
   - Recurses until the number of clusters equals `k`, performing `n-k` merges.
   - *Time:* O((n–k)·n⁴) worst‑case; typical much lower.  
   - *Memory:* tail-call optimized, peak ~O(n²).

7. **Assignment** (`agglomerative.assign`):
   - Traverses final clusters and fills an assignment vector of length `n` with cluster indices.
   - *Time:* O(n)  *Memory:* O(n)

8. **Fit & Predict API** (`agglomerative.fit-predict`):
   - One‐line pipeline: `(agglomerative.fit-predict X k linkage)` returns an `np.array` of labels.
   - *Time:* combined complexities above.  *Memory:* O(n²).

## Comparison with scikit‑learn
scikit‑learn’s `AgglomerativeClustering` implements similar linkage strategies with optimized C loops and optional connectivity constraints.

| Feature             | MeTTa Implementation                         | scikit‑learn                       |
|---------------------|----------------------------------------------|-------------------------------------|
| Linkage types       | single, complete, average                    | ward, complete, average, single     |
| Connectivity        | not supported                                | graph‐based sparse                  |
| Initialization      | trivial (singletons)                         | trivial (singletons)                |
| Convergence control | fixed merges to reach `k`                    | same                               |
| Parallelization     | single‑threaded                              | single‑threaded                     |
| Memory trade‑off    | stores full distance matrix                  | stores condensed or sparse matrices |

## Benchmark Setup
*(To be populated with dataset-specific benchmarks.)*

## Usage Example
```metta
(import! &self metta_ul:cluster:agglomerative)

;; Fit and predict 4 clusters with average linkage
(let $labels (agglomerative.fit-predict X 4 "average")
    (println! $labels)
)
```

## Limitations & Future Work
- **High complexity**: worst-case merging is expensive; consider optimized search (e.g., priority queue).  
- **Connectivity constraints**: add sparse linkage for large datasets.  
- **Parallel merging**: explore MeTTa’s concurrent constructs.  

## Conclusion
This implementation illustrates MeTTa’s ability to encode hierarchical clustering declaratively. While performance and memory usage are constrained by O(n²) matrices and recursive merging, it provides a clear foundation for advanced clustering features.

## References
1. Johnson (1967). Hierarchical clustering schemes.  
2. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.  
3. MeTTa Language Specification. http://www.metta-lang.dev/spec
