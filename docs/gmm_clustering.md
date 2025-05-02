# metta_ul:cluster:gmm

## Overview
This module implements the Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm in MeTTa. GMM is a probabilistic model for representing normally distributed subpopulations within an overall dataset.

## Functions

### `gmm.center`
Centers the data points by subtracting the means.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$means`: Mean vectors of the clusters.
    - Type: `(NPArray ($k $d))`

**Returns:**
- Centered data matrix.
    - Type: `(NPArray ($n $k $d))`

---

### `gmm.mahalanobis-term`
Computes the Mahalanobis distance term for the Gaussian probability density function.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$means`: Mean vectors of the clusters.
    - Type: `(NPArray ($k $d))`
- `$covariances`: Covariance matrices for each cluster.
    - Type: `(NPArray ($k $d $d))`

**Returns:**
- Mahalanobis distance matrix.
    - Type: `(NPArray ($n $k))`

---

### `gmm.gaussian-pdf`
Computes the probability density function (PDF) for a multivariate Gaussian distribution.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$means`: Mean vectors of the clusters.
    - Type: `(NPArray ($k $d))`
- `$covariances`: Covariance matrices for each cluster.
    - Type: `(NPArray ($k $d $d))`

**Returns:**
- Matrix of Gaussian probabilities for each point and cluster.
    - Type: `(NPArray ($n $k))`

---

### `gmm.log-likelihood`
Computes the log-likelihood of the dataset given the current GMM parameters.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$weights`: Mixture component weights.
    - Type: `(NPArray ($k))`
- `$means`: Mean vectors of the clusters.
    - Type: `(NPArray ($k $d))`
- `$covariances`: Covariance matrices for each cluster.
    - Type: `(NPArray ($k $d $d))`

**Returns:**
- Log-likelihood scalar value.
    - Type: `Number`

---

### `gmm.init`
Initializes the GMM parameters (weights, means, and covariances).

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$k`: Number of Gaussian components.

**Returns:**
- Initial weights, means, and covariance matrices.
    - Type: `((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))`

---

### `gmm.e-step`
Performs the Expectation (E) step of the EM algorithm, computing responsibilities.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$weights`: Mixture component weights.
    - Type: `(NPArray ($k))`
- `$means`: Mean vectors of the clusters.
    - Type: `(NPArray ($k $d))`
- `$covariances`: Covariance matrices for each cluster.
    - Type: `(NPArray ($k $d $d))`

**Returns:**
- Responsibility matrix.
    - Type: `(NPArray ($n $k))`

---

### `gmm.m-step`
Performs the Maximization (M) step of the EM algorithm, updating parameters.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$responsibilities`: Responsibility matrix from the E-step.
    - Type: `(NPArray ($n $k))`

**Returns:**
- Updated weights, means, and covariance matrices.
    - Type: `((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))`

---

### `gmm.recursion`
Recursively applies the EM steps until the maximum number of iterations is reached.

**Parameters:**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `($weights, $means, $covariances)`: Current GMM parameters.
    - Type: `((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))`
- `$max-iter`: Maximum number of iterations.
    - Type: `Number`

**Returns:**
- Final weights, means, and covariance matrices.
    - Type: `((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))`

---

### `gmm`
Main function to train a GMM on a dataset.

**Parameters**
- `$X`: Data points as a matrix.
    - Type: `(NPArray ($n $d))`
- `$k`: Count of components.
    - Type: `Number`
- `$max-iter`: Maximum number of iterations.
    - Type: `Number`

**Returns:**
- Final weights, means, and covariance matrices.
    - Type: `((NPArray ($k)) (NPArray ($k $d)) (NPArray ($k $d $d)))`

## Usage
To cluster a dataset `S` of type `(NPArray ($n $d))` into 3 clusters with default settings:
```metta
(=
    (params)
    (gmm (S) 3)
)
```
To specify a maximum number of 50 iterations:
```metta
(gmm (S) 3 50)
```
To assign a dataset `X` of type `(NPArray ($m $d))` using `params`:
```
(=
    (assignments)
    (gmm.e-step (X) (params))
)
```

## Notes
- The GMM parameters are initialized using random means selected from the dataset and a slightly perturbed covariance matrix.
- The EM algorithm iteratively refines the parameters to maximize the log-likelihood of the data.
- Responsibilities indicate the probability of each data point belonging to each cluster.

## Dependencies
This module relies on the following NumPy operations within MeTTa:
- `np.sub`: Element-wise subtraction.
- `np.einsum`: Einstein summation notation for efficient matrix operations.
- `np.linalg.inv`: Inversion of covariance matrices.
- `np.linalg.slogabsdet`: Log determinant of covariance matrices.
- `np.mul`: Element-wise multiplication.
- `np.add`: Element-wise addition.
- `np.sum`: Summation along an axis.
- `np.log`: Logarithm function.
- `np.div`: Element-wise division.
- `np.exp`: Exponential function.
- `np.ones`: Creates an array of ones.
- `np.repeat`: Repeats an array along a specified axis.
- `np.eye`: Creates an identity matrix.
- `np.choose`: Selects random initial means from the dataset.
- `np.cov`: Computes the covariance matrix.

This implementation provides a full pipeline for training a Gaussian Mixture Model using the EM algorithm in MeTTa.

