# metta_ul:cluster:agglomerative

## Overview
This module implements agglomerative clustering using hierarchical merging. The algorithm starts with each data point as its own cluster and iteratively merges the closest clusters based on a chosen linkage criterion.

## Function Definitions

### `agglomerative.init-clusters`
Initializes clusters where each data point starts as its own cluster.

#### Parameters:
- `$n`: The number of data points (clusters initially).

#### Returns:
- A list where each data point is its own cluster.

### `agglomerative.distance-matrix`
Computes the pairwise Euclidean distance matrix for the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.

#### Returns:
- A square matrix where the entry at (i, j) represents the Euclidean distance between points i and j.

### `agglomerative.linkage-distance`
Computes the distance between two clusters based on the specified linkage criterion.

#### Parameters:
- `$distance-matrix`: The precomputed distance matrix.
- `$cluster1`: The first cluster.
- `$cluster2`: The second cluster.
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).

#### Returns:
- The computed distance between the two clusters.

### `agglomerative.closest-clusters`
Finds the two clusters that are closest based on the specified linkage criterion.

#### Parameters:
- `$clusters`: The list of current clusters.
- `$distance-matrix`: The precomputed distance matrix.
- `$linkage`: The linkage method to use.
- `$min-distance`: The current minimum distance found.
- `$closest-pair`: The closest pair of clusters found so far.

#### Returns:
- A tuple containing the closest pair of clusters.

### `agglomerative.merge-clusters`
Merges the two clusters that are closest based on the specified linkage criterion.

#### Parameters:
- `$clusters`: The list of current clusters.
- `$distance-matrix`: The precomputed distance matrix.
- `$linkage`: The linkage method to use.

#### Returns:
- The updated clusters.

### `agglomerative.recursion`
The basic recursion step of the algorithm.

#### Parameters:
- `$linkage`: The linkage method to use.
- `$clusters`: The list of current clusters.
- `$distance-matrix`: The precomputed distance matrix.
- `$length`: The count of clusters in `$clusters`.

#### Returns:
- - A hierarchical clustering structure as a MeTTa list.

### `agglomerative`
Performs agglomerative clustering on a dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
- `$linkage`: The linkage method to use (`"single"`, `"complete"`, or `"average"`).

#### Returns:
- A hierarchical clustering structure as a MeTTa list, e.g. 
```metta
(:: 
    (:: [1] (:: [0] ()))
    (:: 
        (:: [1, 0] ()) 
        ()
    )
)
```
in which the numbers represent a row in `$X`.

## Usage
To cluster a dataset `S` of type `(NPArray (n, d))` with `"average"` linking:
```metta
(=
    (clustering-history)
    (agglomerative (S) "average")
)
```

## Notes
- The distance metric used is the Euclidean norm.
- The hierarchical clustering structure can be used to generate a dendrogram.
- The algorithm supports different linkage strategies to control the merging behavior.

