# Spectral Clustering in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzadeh  
**Date:** May 24, 2025  
**Version:** 1.0  

---

## Abstract
This report presents an open‑source implementation of spectral clustering in the MeTTa language. By leveraging MeTTa's declarative approach and NumPy bindings, we express graph‑based clustering via eigendecomposition of the Laplacian matrix. Our implementation constructs similarity graphs with RBF kernels, computes normalized Laplacians, and extracts low‑dimensional embeddings for k‑means partitioning. Benchmarks show competitive clustering quality with runtimes of ~1.1s versus scikit‑learn's ~0.05s, demonstrating MeTTa's capability to express complex mathematical algorithms while highlighting areas for optimization.

## Introduction
Spectral clustering transforms data clustering into a graph partitioning problem, offering advantages over traditional methods for complex, non‑convex cluster shapes. Implementing spectral clustering in MeTTa:

- Demonstrates MeTTa's capability for linear algebra and eigendecomposition.  
- Extends MeTTa's ML toolkit with a graph‑based clustering method.  
- Provides a foundation for future spectral methods in MeTTa.  
- Targets open‑source users in AI and Data Science.

MeTTa is a multi‑paradigm language for declarative and functional computations over knowledge metagraphs. See http://www.metta‑lang.dev for details.

## Algorithm Overview
Spectral clustering operates on the principle that cluster structure can be revealed through the spectrum (eigenvalues and eigenvectors) of a graph Laplacian derived from data similarities.

Key steps:

1. **Similarity Graph Construction**: Create an affinity matrix $W$ using a Gaussian kernel
   ```math
   W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
   ```

2. **Laplacian Matrix**: Compute the normalized Laplacian $L_{sym}$
   ```math
   L_{sym} = I - D^{-1/2}WD^{-1/2}
   ```
   where $D$ is the diagonal degree matrix with $D_{ii} = \sum_j W_{ij}$

3. **Spectral Embedding**: Find the $k$ eigenvectors corresponding to the smallest eigenvalues of $L_{sym}$, forming a low‑dimensional embedding space.

4. **Clustering**: Apply k‑means to the normalized rows of the eigenvector matrix.

The algorithm exploits spectral graph theory: the multiplicity of the eigenvalue 0 equals the number of connected components, and the corresponding eigenvectors encode component membership.

## MeTTa Implementation Details

The MeTTa implementation utilizes NumPy bindings for matrix operations and eigendecomposition, with clear separation of algorithmic steps:

1. **Affinity Matrix Construction**  
   First computes squared norms and pairwise distances, then applies RBF kernel:
   ```metta
   (spectral-clustering.rbf-affinity-matrix $sqr-distance-matrix-X $rbf-kernel-sigma)
   ```
   - **Time Complexity:** O(n²·d) for distance matrix, O(n²) for kernel application.  
   - **Space Complexity:** O(n²) for the affinity matrix.

2. **Degree Matrix and Laplacian**  
   Computes node degrees by summing affinities, then builds the normalized Laplacian:
   ```metta
   (spectral-clustering.normalized-laplacian $W $inverse-degree-matrix-W)
   ```
   - **Time Complexity:** O(n²) for degree calculation, O(n³) for Laplacian normalization with matrix multiplication.  
   - **Space Complexity:** O(n²) for the Laplacian.

3. **Eigendecomposition**  
   Extracts eigenvalues and eigenvectors of the Laplacian:
   ```metta
   (spectral-clustering.eigh $X)
   ```
   - **Time Complexity:** O(n³) for full eigendecomposition.  
   - **Space Complexity:** O(n²) for eigenvectors.

4. **Spectral Embedding**  
   Selects and normalizes eigenvectors for the k smallest eigenvalues:
   ```metta
   (spectral-clustering.row-normalize 
     (spectral-clustering.spectral-embeddings $eigh-I $k))
   ```
   - **Time Complexity:** O(n·k) for selection and normalization.  
   - **Space Complexity:** O(n·k) for the embedding.

5. **Clustering**  
   Applies k‑means to the normalized embeddings:
   ```metta
   (kmeans.fit $embeddings $num-clusters $max-kmeans-iter 0.0001)
   ```
   - **Time Complexity:** O(t·n·k²) where t is iterations of k‑means.  
   - **Space Complexity:** O(n·k) for assignments.

6. **Pipeline Integration**  
   The complete pipeline combines these steps in a nested let‑expression structure:
   ```metta
   (spectral-clustering.fit $X $num-clusters $rbf-kernel-sigma $max-kmeans-iter)
   ```
   - **Overall Time Complexity:** O(n³) dominated by eigendecomposition.  
   - **Overall Space Complexity:** O(n²) dominated by the affinity and Laplacian matrices.

7. **Prediction**  
   Assigns new data to clusters based on nearest centroids:
   ```metta
   (spectral-clustering.predict ($embeddings $centroids) $num-clusters)
   ```
   - **Time Complexity:** O(n·k) for assignment calculation.  
   - **Space Complexity:** O(n·k) for assignment matrix.

The implementation handles the key algorithm steps declaratively, with eigendecomposition being the computational bottleneck for large datasets.

## Comparison with scikit‑learn

scikit‑learn's `SpectralClustering` offers more options and optimizations for larger datasets:

| Feature               | MeTTa Implementation                  | scikit‑learn                         |
|-----------------------|---------------------------------------|--------------------------------------|
| Affinity methods      | RBF kernel only                       | RBF, nearest neighbors, precomputed  |
| Laplacian types       | normalized symmetric                  | normalized, unnormalized, random walk|
| Eigensolvers          | full decomposition via NumPy          | ARPACK, LOBPCG, AMG (sparse options) |
| Scaling techniques    | none                                  | Nyström method for large datasets    |
| Final clustering      | k‑means only                          | k‑means or discretization            |
| Implementation        | declarative with NumPy bindings       | optimized C/Cython with specialized solvers |

The MeTTa implementation provides core functionality while scikit‑learn offers additional options for scalability and customization.

## Benchmark Setup
- **Datasets** (500 samples each, synthetic generation via scikit‑learn):
  - `noisy_moons`: Two interleaving half‑circles
  - `varied`: Varied variance blobs
  - `no_structure`: Random noise
- **Environment**: CPU = 4 cores @ 3.0 GHz, Hyperon 0.2.2, NumPy 2.2.2, scikit‑learn 1.6.1

## Results

Performance comparison:

| Dataset        | Time (s) | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI   | NMI   | AMI   |
| -------------- | -------- | ---------- | ----------------- | -------------- | ----- | ----- | ----- |
| no-structure   | 0.69     | 0.370      | 359.72            | 0.863          | 0     | 0     | 0     |
| blobs          | 1.57     | 0.457      | 268.15            | 0.594          | 0.568 | 0.729 | 0.728 |
| noisy_circles  | 1.18     | 0.114      | 0.0077            | 240.59         | 1.000 | 1.000 | 1.000 |
| noisy_moon     | 0.73     | 0.385      | 429.54            | 1.028          | 1.000 | 1.000 | 1.000 |
| varied         | 0.76     | 0.627      | 1474.97           | 0.642          | 0.843 | 0.828 | 0.827 |


## Usage Example

```metta
(import! &self metta_ul) 
(import! &self spectral-clustering)

(let 
    $labels 
    (spectral-clustering.predict $fit-outputs 2)
    (println! $labels)
)
```

This snippet:
1. Imports required modules
2. Fits spectral clustering with `k=2` clusters
3. Predicts cluster assignments
4. Prints the resulting labels

## Limitations & Future Work
- **Scalability**: The O(n²) memory requirement for the affinity matrix and O(n³) time complexity for eigendecomposition limit scaling to large datasets.
- **Affinity options**: Currently only supports RBF kernel; could add nearest‑neighbor and custom affinities.
- **Eigensolver efficiency**: Could integrate specialized solvers like ARPACK for large sparse matrices.
- **Hyperparameter selection**: Automatic sigma estimation for the RBF kernel would improve usability.
- **Alternative Laplacians**: Add unnormalized and random walk variants.

## Conclusion
The MeTTa implementation of spectral clustering demonstrates the language's capability to express complex numerical algorithms declaratively. Despite performance differences compared to optimized C/Cython code, the implementation achieves excellent clustering quality on non‑convex datasets. The clear functional decomposition allows for future extensions and optimizations while serving as a reference for spectral methods in MeTTa.

## References
1. Ng, Jordan, Weiss (2002). On Spectral Clustering: Analysis and an Algorithm.
2. von Luxburg (2007). A Tutorial on Spectral Clustering.
3. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.
4. MeTTa Language Specification. http://www.metta‑lang.dev/spec
