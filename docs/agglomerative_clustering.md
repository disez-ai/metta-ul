# metta_ul:cluster:agglomerative

## Overview
This module implements agglomerative clustering using hierarchical merging. The algorithm starts with each data point as its own cluster and iteratively merges the closest clusters based on a chosen linkage criterion.

## Function Definitions

### `agglomerative.init-clusters`
Initializes clusters where each data point starts as its own cluster.

#### Parameters:
- `$n`: The number of data points (clusters initially).
    - Type: `Number`

#### Returns:
- A list where each data point is its own cluster.
    - Type: `(List PyList)`

### `agglomerative.distance-matrix`
Computes the pairwise Euclidean distance matrix for the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
    - Type: `(NPArray ($n $d))`

#### Returns:
- A square matrix where the entry at (i, j) represents the Euclidean distance between points i and j.
    - Type: `(NPArray ($n $n))`

### `agglomerative.linkage-distance`
Computes the distance between two clusters based on the specified linkage criterion.

#### Parameters:
- `$distance-matrix`: The precomputed distance matrix.
    - Type: `(NPArray ($n $n))`
- `$cluster1`: The first cluster.
    - Type: `PyList`
- `$cluster2`: The second cluster.
    - Type: `PyList`
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).
    - Type: `String`

#### Returns:
- The computed distance between the two clusters.
    - Type: `Number`

### `agglomerative.closest-clusters`
Finds the two clusters that are closest based on the specified linkage criterion.

#### Parameters:
- `$clusters`: The list of current clusters.
    - Type: `(List PyList)`
- `$distance-matrix`: The precomputed distance matrix.
    - Type: `(NPArray ($n $n))`
- `$linkage`: The linkage method to use.
    - Type: `String`
- `$min-distance`: The current minimum distance found.
    - Type: `Number`
- `$closest-pair`: The closest pair of clusters found so far.
    - Type: `(PyList PyList)`

#### Returns:
- A tuple containing the closest pair of clusters.

### `agglomerative.merge-clusters`
Merges the two clusters that are closest based on the specified linkage criterion.

#### Parameters:
- `$clusters`: The list of current clusters.
    - Type: `(List PyList)`
- `$distance-matrix`: The precomputed distance matrix.
    - Type: `(NPArray ($n $n))`
- `$linkage`: The linkage method to use.
    - Type: `String`

#### Returns:
- The updated clusters.
    - Type: `(List PyList)`

### `agglomerative.recursion`
The basic recursion step of the algorithm.

#### Parameters:
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).
    - Type: `String`
- `$clusters`: The list of current clusters.
    - Type: `(List PyList)`
- `$distance-matrix`: The precomputed distance matrix.
    - Type: `(NPArray ($n $n))`
- `$length`: The count of clusters in `$clusters`.
    - Type: `Number`

#### Returns:
- A clusters as a MeTTa list.
    - Type: `(List PyList)`

### `agglomerative`
Performs agglomerative clustering on a dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
    - Type: `(NPArray ($n $d))`
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).
    - Type: `String`

#### Returns:
- Clusters as a MeTTa list of Python lists of numbers which represent a row in `$X`.
    - Type: `(List PyList)`

### `agglomerative.assign`
Transform a MeTTa list of clusters into a Numpy array of assignments.

#### Parameters:
- `$clusters`: The list of clusters.
    - Type: `(List PyList)`
- `$assignment`: The current assignment.
    - Type: `(NPArray ($n))`
- `$index`: The index of the cluster in `$clusters`.
    - Type: `Number`

#### Returns:
- Assignments as a numpy array.
    - Type: `(NPArray ($n))`

### `agglomerative.fit-predict`
Clusters a numpy array of samples.
(: agglomerative.fit-predict (-> (NPArray ($n $d)) Number String (NPArray ($n))))
(=
    (agglomerative.fit-predict $X $k $linkage)

#### Parameters:
- `$X`: The list of clusters.
    - Type: `(NPArray ($n $d))`
- `$k`: The number of clusters.
    - Type: `Number`
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).
    - Type: `String`

#### Returns:
- Assignments as a numpy array.
    - Type: `(NPArray ($n))`

## Usage
To cluster a dataset `S` of type `(NPArray ($n $d))` with `"average"` linking:
```metta
(=
    (assignments)
    (agglomerative.fit-predict (S) "average")
)
```

## Notes
- The distance metric used is the Euclidean norm.
- The algorithm supports different linkage strategies to control the merging behavior.

