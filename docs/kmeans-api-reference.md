# metta_ul:cluster:kmeans

## Overview
This module implements the K-Means clustering algorithm in MeTTa. K-Means is a popular unsupervised learning method for partitioning a dataset into `k` clusters based on feature similarity.

## Functions

### `kmeans.update`

#### Description:
This function updates the centroids based on the current cluster assignments. It computes the new centroids as the mean of all data points assigned to each cluster.

#### Parameters:
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$assignments`: One-hot encoded cluster assignment matrix.
    - Type: `(NPArray ($n $k))`

#### Returns:
- Updated cluster centroids.
    - Type: `(NPArray ($k $d))`

---

### `kmeans.assign`

#### Description:
Assigns each data point to the nearest centroid.

#### Parameters:
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$centroids`: Current centroids.
    - Type: `(NPArray ($k $d))`

#### Returns:
- A one-hot encoded assignment matrix indicating which cluster each point belongs to.
    - Type: `(NPArray ($k $n))`

---

### `kmeans.recursion`

#### Description:
Recursively updates the centroids until the maximum number of iterations is reached.

#### Parameters:
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$centroids`: Initial centroids.
    - Type: `(NPArray ($k $d))`
- `$max-iter`: Maximum number of iterations.
    - Type: `Number`

#### Returns:
- Final cluster centroids after convergence or reaching the iteration limit.
    - Type: `(NPArray ($k $d))`

---

### `kmeans.fit`

#### Description:
Main function to perform K-Means clustering. It initializes centroids randomly and iteratively updates them.

#### Parameters:
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $k))`
- `$k`: Number of clusters.
    - Type: `Number`
- `$max-iter` (optional, default: 100): Maximum number of iterations.
    - Type: `Number`
- `$tol` (optional, default: 0.0001): Tolerance on the shift of the centroids.
    - Type: `Number`

#### Returns:
- Final cluster centroids after completion of the algorithm.
    - Type: `(NPArray ($k $d))`

---

### `kmeans.predict`

#### Description:
Main function for predicting the labels of some test data after learning the centroids from the training data. It computes the assignments and then return the positions of the maximum assignments as the labels.

#### Parameters:
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $k))`
- `$centroids`: The position of centroids of the clusters.
    - Type: `(NPArray ($k $d))`

#### Returns:
- predicted labels.
    - Type: `(NPArray ($n))`

## Usage
To cluster a dataset `S` of type `(NPArray (n, d))` into 3 clusters with default settings:
```metta
(=
    (centroids)
    (kmeans.fit (S) 3)
)
```
To specify a maximum number of 50 iterations and a tolerance of 0.001:
```metta
(kmeans (S) 3 50 0.001)
```
To predict the labels of a dataset `X` of type `(NPArray (m, d))` using `centroids`:
```
(=
    (assignments)
    (kmeans.predict (X) (centroids))
)
```

## Notes
- The initial centroids are chosen randomly from the data points.
- The algorithm stops after reaching the iteration limit, but convergence is not guaranteed.
- The function `np.one_hot` is used to convert the assignment indices into a one-hot encoded matrix.

## Dependencies
This module uses NumPy functions within MeTTa:
- `np.div`: Element-wise division.
- `np.matmul`: Matrix multiplication.
- `np.sum`: Summation along a specified axis.
- `np.linalg.norm`: Computes the L2 norm.
- `np.sub`: Element-wise subtraction.
- `np.expand_dims`: Adds dimensions to an array.
- `np.choose`: Random selection of initial centroids.
- `np.one_hot`: Converts indices to one-hot encoding.
- `np.argmin`: Finds the index of the minimum value along an axis.
