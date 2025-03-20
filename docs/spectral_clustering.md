# Spectral Clustering

https://en.wikipedia.org/wiki/Spectral_clustering

The spectral clustering algorithm reduces the dimensionality of the input data by leveraging 
the eigenvalues and eigenvectors derived from the data's similarity matrix, and then applies 
standard clustering methods, such as k-means, on the reduced data to identify clusters.

Specifically, the algorithm constructs a graph structure using the similarity matrix of the data set, 
where each entry represents the similarity score between the corresponding pair of data points. 
It then employs the eigenvectors associated to the first k eigenvalues of the 
normalized graph Laplacian as input features for the k-means algorithm.


The data set is represented by the matrix X with shape [N, D], where N is the number of 
data points and D is the dimension of the input space. Our goal is to partition the data points
into K clusters.
Below is a clear, step-by-step outline of the basic spectral clustering algorithm:
1. Compute the similarity matrix W given the input data X. The similarity matrix W is a symmetric
matrix with the shape [N, N], where W[i, j] stores the similarity between data points X[i] and X[j].
A Gaussian (RBF) kernel is used for computing the similarity of data points X[i] and X[j], i.e., 
W[i, j] = exp(-|| X[i] - X[j] ||^2 / (2 * sigma^2)), where sigma is the scale parameter of the 
Gaussian kernel.
2. Using the similarity matrix, compute the normalized graph Laplacian L_norm:
L_norm = I - D^(-1/2) W D^(-1/2), where I is the identity matrix of shape [N, N], and D is the 
degree matrix with D[i, i] = \sum_j W[i, j].
3. Compute the spectral embeddings U as the eigenvectors of the k smallest eigenvalues of L. The
spectral embeddings matrix U has the shape [N, K]. 
4. Normalize the rows of U to have the sum equal to one.
5. Apply the k-means clustering algorithm to the spectral embeddings matrix U to find the K clusters.





