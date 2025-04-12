# metta_ul:cluster:spectral-clustering

## Overview
This module implements the Spectral Clustering algorithm. It uses the RBF (Radial Basis Function) kernel to construct an affinity matrix, computes the normalized graph Laplacian, and performs eigen-decomposition to extract spectral embeddings. These embeddings are then normalized and clustered using k-means. The algorithm is designed to cluster data that may not be linearly separable, leveraging the eigenstructure of the Laplacian.

## Function Definitions

### `spectral-clustering.square-norm`
Computes the square norm for each row (data point) in the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.

#### Returns:
- A vector where each element is the sum of squares of the components of the corresponding data point.

---

### `spectral-clustering.square-distance-matrix`
Computes the matrix of squared Euclidean distances between each pair of data points in the dataset.

#### Parameters:
- `$square-norm-X`: The vector of square norms computed from `$X`.
- `$X`: The dataset.

#### Returns:
- A square matrix where the entry at (i, j) is the squared Euclidean distance between data point *i* and *j*.

---

### `spectral-clustering.rbf-affinity-matrix`
Generates the RBF (Radial Basis Function) affinity matrix from a squared distance matrix.

#### Parameters:
- `$sqr-distance-matrix-X`: The matrix of squared distances between data points.
- `$rbf-kernel-sigma`: The sigma parameter for the RBF kernel.

#### Returns:
- A symmetric affinity matrix where each entry represents the similarity between a pair of data points based on the RBF kernel.

---

### `spectral-clustering.degree`
Computes the degree of each node (data point) based on the affinity matrix.

#### Parameters:
- `$W`: The affinity matrix.

#### Returns:
- A vector where each element is the sum of the corresponding row in `$W`.

---

### `spectral-clustering.inverse-degree-matrix`
Constructs the inverse degree matrix used for normalization.

#### Parameters:
- `$degree-W`: The vector of node degrees computed from `$W`.

#### Returns:
- A diagonal matrix where each diagonal element is the inverse square root of the corresponding degree.

---

### `spectral-clustering.normalized-laplacian`
Computes the normalized graph Laplacian from the affinity matrix and its inverse degree matrix.

#### Parameters:
- `$W`: The RBF affinity matrix.
- `$inverse-degree-matrix-W`: The inverse degree matrix computed from `$W`.

#### Returns:
- The normalized Laplacian matrix defined as *I - D^{-1/2} W D^{-1/2}*, where *I* is the identity matrix.

---

### `spectral-clustering.eigh`
Performs eigen-decomposition on a given matrix.

#### Parameters:
- `$X`: The matrix to decompose (typically the normalized Laplacian).

#### Returns:
- A tuple containing the eigenvalues and eigenvectors of `$X`.

---

### `spectral-clustering.eigenvalues`
Extracts the eigenvalues from the result of the eigen-decomposition.

#### Parameters:
- `$eigh-X`: The tuple returned from `spectral-clustering.eigh`.

#### Returns:
- A vector containing the eigenvalues.

---

### `spectral-clustering.eigenvectors`
Extracts the eigenvectors from the result of the eigen-decomposition.

#### Parameters:
- `$eigh-X`: The tuple returned from `spectral-clustering.eigh`.

#### Returns:
- A matrix whose columns correspond to the eigenvectors of `$X`.

---

### `spectral-clustering.eigval-top-k-index`
Finds the indices corresponding to the smallest *k* eigenvalues (after sorting).

#### Parameters:
- `$eigval-L`: The vector of eigenvalues.
- `$k`: The number of top eigenvalue indices to select.

#### Returns:
- A vector of indices for the top *k* eigenvalues.

---

### `spectral-clustering.spectral-embeddings`
Computes the spectral embeddings by selecting the top *k* eigenvectors based on their eigenvalues.

#### Parameters:
- `$eigh-I`: The eigen-decomposition result of the normalized Laplacian.
- `$k`: The number of clusters (and hence dimensions for the embeddings).

#### Returns:
- A matrix of spectral embeddings extracted from the selected eigenvectors.

---

### `spectral-clustering.row-normalize`
Normalizes each row of a matrix to have unit norm.

#### Parameters:
- `$X`: A matrix (such as the spectral embeddings).

#### Returns:
- The row-normalized version of `$X`.

---

### `spectral-clustering.cluster`
Clusters the spectral embeddings using the k-means algorithm.

#### Parameters:
- `$X`: The original dataset.
- `$num-clusters`: The desired number of clusters.
- `$rbf-kernel-sigma`: The sigma parameter for constructing the RBF affinity matrix.
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm.

#### Returns:
- The centroids obtained after clustering the row-normalized spectral embeddings with k-means.

---

### `spectral-clustering.fit`
Performs the full spectral clustering process on the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$num-clusters`: The desired number of clusters.
- `$rbf-kernel-sigma`: The sigma parameter for the RBF kernel (default example: `0.1`).
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm (default example: `100`).

#### Returns:
- The tuple containing spectral embeddings and the final centroids computed from clustering the spectral embeddings.

---

### `spectral-clustering.predict`
Predicts the cluster assignment for each data point in the dataset using the spectral embeddings and computed centroids.

#### Parameters:
- A tuple `($embeddings $centroids)`, where:
  - `$embeddings`: The spectral embedding matrix.
  - `$centroids`: The centroids obtained from the clustering step.
- `$num-clusters`: The number of clusters.

#### Returns:
- A vector of cluster labels, one for each data point, determined by assigning each point to the nearest centroid.

## Usage
To perform spectral clustering on a dataset `S` (an `NPArray` of shape `(n, d)`) with a specified number of clusters (e.g., 3), RBF kernel sigma (e.g., `0.1`), and a maximum of 100 iterations for k-means, you can use:
```metta
(=
    (embeddings-and-centroids)
    (spectral-clustering.fit (S) 3 0.1 100)
)
```
To predict the cluster labels for the dataset `S`, use:
```metta
(=
    (cluster-labels)
    (spectral-clustering.predict
        (embeddings-and-centroids)        
        3
    )    
)
```