# `metta_ul:cluster:agglomerative`

## Overview

This module implements agglomerative clustering using a priority heap and a union-find structure to efficiently merge clusters until a desired number of clusters remain. It supports multiple linkage strategies: `"single"`, `"complete"`, and `"average"`.

## Function Definitions

### `agglomerative.distance-matrix`

Computes the pairwise Euclidean distance matrix for the dataset.

#### Parameters:

* `$X`: Input dataset as a 2D NumPy array.

  * Type: `(NPArray ($n $d))`

#### Returns:

* Pairwise distance matrix between points in `$X`.

  * Type: `(NPArray ($n $n))`

---

### `agglomerative.linkage-distance`

Calculates the distance between two clusters based on a specified linkage method.

#### Parameters:

* `$distance-matrix`: Precomputed pairwise distance matrix.

  * Type: `(NPArray ($n $n))`
* `$cluster1`: Indices of the first cluster.

  * Type: `(NPArray ($a))`
* `$cluster2`: Indices of the second cluster.

  * Type: `(NPArray ($b))`
* `$linkage`: Linkage method (`"single"`, `"complete"`, or `"average"`).

  * Type: `String`

#### Returns:

* Scalar distance between the two clusters.

  * Type: `Number`

---

### `agglomerative.heapify`

Initializes a heap with the smallest pairwise distances between data points.

#### Overload 1:

#### Parameters:

* `$distance-matrix`: Pairwise distance matrix.

  * Type: `(NPArray ($a $a))`

#### Returns:

* Initialized heap of distances.

  * Type: `(Heap (Number Number Number))`

#### Overload 2 (internal recursion):

#### Parameters:

* `$distance-matrix`: Pairwise distance matrix.

  * Type: `(NPArray ($a $a))`
* `$nn`: Array of closest neighbor indices.

  * Type: `(NPArray ($a))`
* `$i`: Index for recursion.

  * Type: `Number`
* `$heap`: Current heap.

  * Type: `(Heap (Number Number Number))`

#### Returns:

* Updated heap.

  * Type: `(Heap (Number Number Number))`

---

### `agglomerative.heappush`

Pushes new distance entries into the heap involving a newly formed cluster.

#### Overload 1:

#### Parameters:

* `$heap`: Current heap.

  * Type: `(Heap (Number Number Number))`
* `$distance-matrix`: Pairwise distance matrix.

  * Type: `(NPArray ($a $a))`
* `$linkage`: Linkage method.

  * Type: `String`
* `$uf`: UnionFind data structure.

  * Type: `UnionFind`
* `$new-root`: Root of the newly created cluster.

  * Type: `Number`

#### Returns:

* Updated heap.

  * Type: `(Heap (Number Number Number))`

#### Overload 2 (internal recursion):

#### Parameters:

* `$heap`: Current heap.

  * Type: `(Heap (Number Number Number))`
* `$distance-matrix`: Pairwise distance matrix.

  * Type: `(NPArray ($a $a))`
* `$linkage`: Linkage method.

  * Type: `String`
* `$uf`: UnionFind structure.

  * Type: `UnionFind`
* `$new-root`: Root of new cluster.

  * Type: `Number`
* `$roots`: All current roots.

  * Type: `(NPArray ($k))`
* `$k`: Index for recursion.

  * Type: `Number`

#### Returns:

* Updated heap.

  * Type: `(Heap (Number Number Number))`

---

### `agglomerative.recursion`

Recursively merges clusters until only `$k` clusters remain.

#### Parameters:

* `($parent $count)`: UnionFind structure holding cluster merges and count of current clusters.

  * Type: `UnionFind`
* `$distance-matrix`: Pairwise distance matrix.

  * Type: `(NPArray ($n $n))`
* `$heap`: Min-heap of cluster distances.

  * Type: `(Heap (Number Number Number))`
* `$linkage`: Linkage strategy to use.

  * Type: `String`
* `$k`: Desired number of clusters.

  * Type: `Number`

#### Returns:

* Updated UnionFind structure with merged clusters.

  * Type: `UnionFind`

---

### `agglomerative.cluster`

Performs the full agglomerative clustering procedure using a priority heap and union-find.

#### Parameters:

* `$X`: Input data points.

  * Type: `(NPArray ($n $d))`
* `$k`: Number of clusters to return.

  * Type: `Number`
* `$linkage`: Linkage method to use.

  * Type: `String`

#### Returns:

* Final UnionFind structure after merging.

  * Type: `UnionFind`

---

### `agglomerative.assign`

Assigns cluster indices to data points based on the final UnionFind structure.

#### Parameters:

* `$uf`: Final UnionFind structure.

  * Type: `UnionFind`
* `$assignment`: Array for storing assignment results.

  * Type: `(NPArray ($n))`
* `$roots`: List of cluster roots.

  * Type: `(NPArray ($k))`
* `$index`: Current index of the root being assigned.

  * Type: `Number`

#### Returns:

* Final cluster assignment array.

  * Type: `(NPArray ($n))`

---

### `agglomerative.fit-predict`

Clusters a dataset and returns an array of cluster assignments.

#### Parameters:

* `$X`: Dataset of shape `(n, d)`.

  * Type: `(NPArray ($n $d))`
* `$k`: Number of clusters to form.

  * Type: `Number`
* `$linkage`: Linkage method (`"single"`, `"complete"`, `"average"`).

  * Type: `String`

#### Returns:

* Cluster assignment for each point.

  * Type: `(NPArray ($n))`

---

## Usage

To cluster dataset `S` into 4 clusters using `"average"` linkage:

```metta
(=
    (assignments)
    (agglomerative.fit-predict S 4 "average")
)
```

---

## Notes

* This implementation is optimized using a heap for fast retrieval of the smallest cluster distances and a union-find structure to avoid merging already connected clusters.
* All distance computations are based on the Euclidean norm.
* The use of `np.ix_` allows efficient distance slicing between cluster indices.
* The recursive definitions are tail-optimized for stackless processing.
