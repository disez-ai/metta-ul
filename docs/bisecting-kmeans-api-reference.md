# metta_ul:cluster:bisecting-kmeans

## Overview
This module implements the Bisecting K-means clustering algorithm using iterative splitting of clusters. The algorithm starts with the entire dataset as a single cluster and recursively bisects the cluster with the maximum Sum of Squared Errors (SSE) using standard k-means (with k=2) until a desired number of clusters is reached. The resulting hierarchical clustering structure captures the splits performed during the process.

## Type Definitions

### Core Types
- `Cluster`: A tuple `($indices $center $sse $hierarchy)` representing a single cluster
- `(List Cluster)`: A typed list of clusters using `Cons` and `Nil` constructors
- `(List (List Cluster))`: Represents the hierarchical clustering structure


## Function Definitions

### `bisecting-kmeans.compute-sse`
Computes the Sum of Squared Errors (SSE) for a given cluster.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$indices`: Indices of the data points belonging to the cluster.
  - Type: `(NPArray ($M))`
- `$centers`: The center (mean) of the cluster.
  - Type: `(NPArray ($C))`

#### Returns:
- The SSE value computed as the sum of squared differences between the data points and the cluster center.
  - Type: `Number`

---

### `bisecting-kmeans.compute-initial-cluster`
Computes the initial cluster for the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
#### Returns:
- A list containing the initial cluster.
  - Type: `(List Cluster)
`

Where the `(List Cluster)
` is a list of clusters, and each cluster is a tuple containing:
1. **Indices:** All data point indices in `$X`.
   - Type: `(NPArray ($N $D))`
2. **Center:** The mean of the dataset.
   - Type: `(NPArray ($C))`
3. **SSE:** The computed SSE for the cluster.
   - Type: `Number`
4. **Hierarchy:** `pyNone` (as no hierarchy is present initially).
   - Type: `(List (List Cluster))`

---

### `bisecting-kmeans.get-cluster-indices`
Extracts the indices from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple `($indices $center $sse $hierarchy)` representing a cluster.
  - Type: `Cluster`

#### Returns:
- The indices of the data points in the cluster.
  - Type: `(NPArray ($M))`

---

### `bisecting-kmeans.get-cluster-center`
Extracts the center of the cluster.

#### Parameters:
- `$cluster`: A tuple representing a cluster.
  - Type: `Cluster`

#### Returns:
- The center of the cluster.
   - Type: `(NPArray ($C))`

---

### `bisecting-kmeans.get-cluster-sse`
Extracts the SSE value from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple representing a cluster.
  - Type: `Cluster`

#### Returns:
- The SSE value of the cluster.
  - Type: `Number`

---

### `bisecting-kmeans.get-cluster-hierarchy`
Extracts the hierarchical structure information from a cluster tuple.

#### Parameters:
- `$cluster`: A tuple representing a cluster.
  - Type: `Cluster`

#### Returns:
- The hierarchy associated with the cluster.
  - Type: `(List (List Cluster))`
---

### `bisecting-kmeans.find-max-cluster`
Finds the cluster with the maximum SSE from a list of clusters.

#### Parameters:
- A list of clusters, where each cluster is a tuple `($indices $center $sse $hierarchy)`.
  - Type: `(List Cluster)
`

#### Returns:
- The cluster tuple with the highest SSE value.
  - Type: `Cluster`

---

### `bisecting-kmeans.cluster-equal`
Checks whether two clusters are equal based on their indices, center, and SSE.

#### Parameters:
- Two cluster tuples: `($indices1 $center1 $sse1 $hierarchy1)` and `($indices2 $center2 $sse2 $hierarchy2)`.
  - Type: `Cluster Cluster`

#### Returns:
- `True` if both clusters are equal; otherwise, `False`.
  - Type: `Bool`
---

### `bisecting-kmeans.remove-cluster`
Removes a target cluster from a list of clusters.

#### Parameters:
- `$clusters`: The list of current clusters.
  - Type: `(List Cluster)
`
- `$target`: The cluster tuple to be removed.
  - Type: `Cluster`

#### Returns:
- An updated list of clusters with the target cluster removed.
  - Type: `Cluster`
---

### `bisecting-kmeans.bisect-cluster`
Performs bisection on a given cluster using standard k-means with k=2.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$max-cluster`: The cluster to be bisected, represented as a tuple.
  - Type: `Cluster`
- `$max-iter`: The maximum number of iterations allowed for the k-means algorithm during the bisecting process.
  - Type: `Number`

#### Returns:
- A tuple containing two clusters obtained from splitting the input cluster. Each cluster is represented as `(indices, center, sse, hierarchy)`.
  - Type: `(List Cluster)
`
---

### `bisecting-kmeans.recursive-bisecting-kmeans`
Recursively applies bisecting k-means to further split clusters until the desired number of clusters is reached.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$clusters`: The current list of clusters.
  - Type: `(List Cluster)
`
- `$max-num-clusters`: The desired number of clusters.
  - Type: `Number`
- `$max-iter`: The maximum iterations for each bisecting step.
  - Type: `Number`
- `$hierarchy`: The current hierarchical clustering structure maintained as a MeTTa list.
  - Type: `(List (List Cluster))`

#### Returns:
- An updated hierarchical clustering structure as a MeTTa list describing the clustering process.
  - Type: `(List (List Cluster))`
---

### `bisecting-kmeans.fit`
Performs bisecting k-means clustering on the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$max-num-clusters`: The maximum (desired) number of clusters.
  - Type: `Number`
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm during the bisecting process.
  - Type: `Number`

#### Returns:
- A hierarchical clustering structure as a MeTTa list that describes the sequence of splits performed.

---

### `bisecting-kmeans.assign-point-to-cluster`
Assigns a single data point to the closest cluster based on Euclidean distance.

#### Parameters:
- `$point`: A single data point from the dataset.
  - Type: `(NPArray ($D))`
- `$clusters`: A list of clusters.
  - Type: `(List Cluster)
`
- `$best-cluster-idx`: The current best cluster index for the point (initial value provided).
  - Type: `Number`
- `$best-distance`: The current best distance found (initially set to a high value, such as `pyINF`).
  - Type: `Number`
- `$cluster-idx`: The current index being evaluated.
  - Type: `Number`

#### Returns:
- The index of the cluster that is closest to the given data point.
  - Type: `Number`
---

### `bisecting-kmeans.assign-all-points`
Assigns all data points in the dataset to their closest clusters.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($D))`
- `$clusters`: The list of clusters.
  - Type: `(List Cluster)
`
- `$point-idx`: The current index of the data point being processed.
  - Type: `Number`
- `$labels`: A list of cluster labels corresponding to the assignment of each data point.
  - Type: `(NPArray ($N))`

#### Returns:
- A list of cluster labels indicating the assignment of each data point in `$X` to the nearest cluster.
  - Type: `(NPArray ($N))`
---

### `bisecting-kmeans.predict`
Predicts the cluster membership for each data point based on the final hierarchical clustering structure.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$hierarchy`: The hierarchical clustering structure generated by `bisecting-kmeans.fit`.
  - Type: `(List (List Cluster))`

#### Returns:
- A list of cluster labels indicating the assigned cluster for each data point in the dataset.
  - Type: `(NPArray ($N))`
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