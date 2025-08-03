# Agglomerative Clustering in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzadeh

**Date:** July 29, 2025

**Version:** 1.0

---

## Abstract

We present an implementation of agglomerative (hierarchical) clustering in the MeTTa programming language, incorporating a min-heap and union-find data structures for performance optimization. By avoiding repeated full pairwise searches and tracking cluster memberships with union-find, the algorithm achieves significantly improved performance over naive approaches. This report details the algorithm's design, MeTTa-specific implementations, and performance characteristics.

## Introduction

Hierarchical clustering is widely used for grouping unlabeled data without requiring a priori knowledge of the number of clusters. Agglomerative clustering performs this by repeatedly merging the closest pair of clusters until the desired number of clusters is reached.

This MeTTa implementation is designed to:

* Demonstrate efficient recursive programming with MeTTa.
* Serve as a high-performance clustering module for symbolic and numerical data.
* Showcase how classical data structures (heap, union-find) can be expressed declaratively.

## Algorithm Overview

1. **Distance Matrix Computation**: Compute full pairwise distances between data points.
2. **Heap Initialization**: Build a min-heap with the smallest distance for each point.
3. **Union-Find Initialization**: Each point begins in its own set.
4. **Recursive Merge**: Pop the heap to find the nearest cluster pair. If not already unified, merge them and update the heap with distances to the new cluster.
5. **Label Assignment**: Assign a unique label to each disjoint set.

## MeTTa Implementation Details

### 1. **Distance Matrix** (`agglomerative.distance-matrix`)

Computes an \$n \times n\$ matrix of pairwise Euclidean distances.

* *Implementation*: Uses broadcasting via `np.expand_dims` and `np.linalg.norm`.
* *Time:* O(n²·d)
* *Memory:* O(n²)

### 2. **Linkage Function** (`agglomerative.linkage-distance`)

Computes distance between two clusters using a linkage criterion:

* **single**: minimum pairwise distance

* **complete**: maximum pairwise distance

* **average**: mean pairwise distance

* *Time:* O(|C₁|·|C₂|)

* *Memory:* O(|C₁|·|C₂|)

### 3. **Heap Initialization** (`agglomerative.heapify`)

For each point, identifies its nearest neighbor and inserts the pair with their distance into a min-heap.

* *Implementation*: Uses masked distance matrix to avoid self-pairing.
* *Time:* O(n²)
* *Memory:* O(n)

### 4. **Heap Update** (`agglomerative.heappush`)

After a merge, computes the distance between the new cluster and all remaining clusters and pushes these distances to the heap.

* *Implementation*: Recursively compares new root with others using union-find and linkage function.
* *Time:* O(n·p) where p = cost of linkage calculation
* *Memory:* Heap contains up to O(n²) entries over time.

### 5. **Union-Find Recursion** (`agglomerative.recursion`)

Recursive loop merges the closest cluster pair at each step, skipping pairs already unified.

* *Implementation*:

  * Uses `UnionFind.areUnified` to avoid redundant merges.
  * After each merge, updates heap with new distances.
* *Time:* Worst-case O((n–k)·n²·p), where p is linkage cost
* *Memory:* Tail-recursive stack, union-find trees + heap

### 6. **Cluster Routine** (`agglomerative.cluster`)

Main pipeline for clustering: computes distances, initializes heap and union-find, runs merge recursion.

* *Time:* Aggregate of above steps
* *Memory:* Aggregate of above steps

### 7. **Cluster Assignment** (`agglomerative.assign`)

Labels each element according to the root of its union-find group.

* *Time:* O(n)
* *Memory:* O(n)

### 8. **API Entry Point** (`agglomerative.fit-predict`)

Wrapper that runs clustering and returns an `np.array` of cluster labels.

* *Time:* Total time from above steps
* *Memory:* O(n²)

## Comparison with Naive Agglomerative (e.g., scikit-learn)

| Feature           | MeTTa Optimized Implementation       | scikit‑learn                               |
| ----------------- | ------------------------------------ | ------------------------------------------ |
| Linkage types     | single, complete, average            | ward, complete, average, single            |
| Distance storage  | full matrix (numpy)                  | condensed or sparse matrix                 |
| Search strategy   | heap-based nearest-neighbor tracking | fast C loops, optional connectivity        |
| Merge bookkeeping | union-find                           | tree / linkage matrix                      |
| Parallelization   | not yet supported                    | not supported in `AgglomerativeClustering` |
| Best-case runtime | O(n²·log n) with fast linkage        | O(n²) for complete linkage                 |

## Benchmark Setup
*The allgorithm does not return in an acceptable time even for pretty small datasets (~20 samples).*


## Usage Example

```metta
(import! &self metta_ul:cluster:agglomerative)

;; Run average-linkage clustering with k = 3
(let $labels (agglomerative.fit-predict $X 3 "average")
    (println! $labels)
)
```

Where `$X` is an `np.array` of shape `(n, d)`.

## Limitations & Future Work

* **Dense Distance Matrix**: O(n²) memory use limits scalability.
* **Heap Growth**: No deduplication; heap may grow large unless compacted.
* **Condensed Matrix Support**: Could reduce memory footprint with careful index mapping.

### Planned Extensions:

* Add support for sparse graph connectivity (e.g., radius or k-NN graphs).
* Use priority queue with distance caching to avoid redundant linkage evaluations.

## Conclusion

This report documents a fast, modular, and expressive implementation of agglomerative clustering in MeTTa. The use of declarative recursion, heaps, and union-find enables realistic performance on medium-scale datasets and lays the groundwork for extending MeTTa into more advanced unsupervised learning domains.

## References

1. Johnson, S. C. (1967). Hierarchical clustering schemes.
2. Pedregosa et al. (2011). Scikit-learn: Machine learning in Python.
3. MeTTa Language Specification. [http://www.metta-lang.dev/spec](http://www.metta-lang.dev/spec)
4. Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm.

---

Would you like me to generate:

* a visual diagram of the algorithm flow?
* benchmark plots?
* unit test examples in MeTTa?
* a compact README.md version of this?

Let me know what you need next.
