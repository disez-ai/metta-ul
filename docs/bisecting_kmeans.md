# metta_ul:cluster:bisecting-kmeans

## Overview
This module implements the Bisecting K-means clustering algorithm using iterative splitting of clusters. The algorithm starts with the entire dataset as a single cluster and recursively bisects the cluster with the maximum Sum of Squared Errors (SSE) using standard k-means (with k=2) until a desired number of clusters is reached. The resulting hierarchical clustering structure captures the splits performed during the process.

## Function Definitions

### `bisecting-kmeans.compute-sse`
Computes the Sum of Squared Errors (SSE) for a given cluster.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$indices`: Indices of the data points belonging to the cluster.
- `$centers`: The center (mean) of the cluster.

#### Returns:
- The SSE value computed as the sum of squared differences between the data points and the cluster center.

---

### `bisecting-kmeans.compute-initial-cluster`
Computes the initial cluster for the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.

#### Returns:
- A tuple containing:
  - **Indices:** All data point indices in `$X`.
  - **Center:** The mean of the dataset.
  - **SSE:** The computed SSE for the cluster.
  - **Hierarchy:** `pyNone` (as no hierarchy is present initially).

---

### `bisecting-kmeans.get-cluster-indices`
Extracts the indices from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple `($indices $center $sse $hierarchy)` representing a cluster.

#### Returns:
- The indices of the data points in the cluster.

---

### `bisecting-kmeans.get-cluster-center`
Extracts the center of the cluster.

#### Parameters:
- `$cluster`: A tuple representing a cluster.

#### Returns:
- The center of the cluster.

---

### `bisecting-kmeans.get-cluster-sse`
Extracts the SSE value from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple representing a cluster.

#### Returns:
- The SSE value of the cluster.

---

### `bisecting-kmeans.get-cluster-hierarchy`
Extracts the hierarchical structure information from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple representing a cluster.

#### Returns:
- The hierarchy associated with the cluster.

---

### `bisecting-kmeans.find-max-cluster`
Finds the cluster with the maximum SSE from a list of clusters.

#### Parameters:
- A list of clusters, where each cluster is a tuple `($indices $center $sse $hierarchy)`.

#### Returns:
- The cluster tuple with the highest SSE value.

---

### `bisecting-kmeans.cluster-equal`
Checks whether two clusters are equal based on their indices, center, and SSE.

#### Parameters:
- Two cluster tuples: `($indices1 $center1 $sse1 $hierarchy1)` and `($indices2 $center2 $sse2 $hierarchy2)`.

#### Returns:
- `True` if both clusters are equal; otherwise, `False`.

---

### `bisecting-kmeans.remove-cluster`
Removes a target cluster from a list of clusters.

#### Parameters:
- `$clusters`: The list of current clusters.
- `$target`: The cluster tuple to be removed.

#### Returns:
- An updated list of clusters with the target cluster removed.

---

### `bisecting-kmeans.bisect-cluster`
Performs bisection on a given cluster using standard k-means with k=2.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$max-cluster`: The cluster to be bisected, represented as a tuple.
- `$max-iter`: The maximum number of iterations allowed for the k-means algorithm during the bisecting process.

#### Returns:
- A tuple containing two clusters obtained from splitting the input cluster. Each cluster is represented as `(indices, center, sse, hierarchy)`.

---

### `bisecting-kmeans.recursive-bisecting-kmeans`
Recursively applies bisecting k-means to further split clusters until the desired number of clusters is reached.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$clusters`: The current list of clusters.
- `$max-cluster`: The desired number of clusters.
- `$max-iter`: The maximum iterations for each bisecting step.
- `$hierarchy`: The current hierarchical clustering structure maintained as a MeTTa list.

#### Returns:
- An updated hierarchical clustering structure as a MeTTa list describing the clustering process.

---

### `bisecting-kmeans.fit`
Performs bisecting k-means clustering on the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$max-num-clusters`: The maximum (desired) number of clusters.
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm during the bisecting process.

#### Returns:
- A hierarchical clustering structure as a MeTTa list that describes the sequence of splits performed.

---

### `bisecting-kmeans.assign-point-to-cluster`
Assigns a single data point to the closest cluster based on Euclidean distance.

#### Parameters:
- `$point`: A single data point from the dataset.
- `$clusters`: A list of clusters.
- `$best-cluster-idx`: The current best cluster index for the point (initial value provided).
- `$best-distance`: The current best distance found (initially set to a high value, such as `pyINF`).
- `$cluster-idx`: The current index being evaluated.

#### Returns:
- The index of the cluster that is closest to the given data point.

---

### `bisecting-kmeans.assign-all-points`
Assigns all data points in the dataset to their closest clusters.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$clusters`: The list of clusters.
- `$point-idx`: The current index of the data point being processed.
- `$labels`: A list of cluster labels corresponding to the assignment of each data point.

#### Returns:
- A list of cluster labels indicating the assignment of each data point in `$X` to the nearest cluster.

---

### `bisecting-kmeans.predict`
Predicts the cluster membership for each data point based on the final hierarchical clustering structure.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$hierarchy`: The hierarchical clustering structure generated by `bisecting-kmeans.fit`.

#### Returns:
- A list of cluster labels indicating the assigned cluster for each data point in the dataset.

---

## Usage
To perform bisecting k-means clustering on a dataset `S` (an `NPArray` of shape `(n, d)`) with a maximum of 5 clusters and 100 maximum iterations for k-means:
```metta
(=
    (clustering-history)
    (bisecting-kmeans.fit (S) 5 100)
)
```
After clustering, you can predict the cluster assignments for S as follows:
```metta
(=
    (cluster-labels)
    (bisecting-kmeans.predict (S) (clustering-history))
)
```

## Notes

- The algorithm repeatedly splits clusters by applying standard k-means with k=2, 
targeting the cluster with the highest SSE for bisection.

- The clustering process maintains a hierarchical structure that records the 
splits performed, which can be used for further analysis or visualization.

- SSE is used as the criterion for selecting the cluster to bisect, 
ensuring that the cluster with the largest dispersion is split at each iteration.